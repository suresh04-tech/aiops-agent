"""
app/agent/evaluators.py
───────────────────────
Contains helper functions for investigation that are not exposed as LLM tools.
"""

import json
import logging
from datetime import datetime, timedelta, timezone

from app.agent.rules import (
    _LOG_SIGNAL_RULES,
    _EC2_STATE_RULES,
    _ALB_REASON_RULES,
    _HIGH_CONFIDENCE_TRIAGE_MAP,
    RCA_CATEGORIES,
)

logger = logging.getLogger(__name__)

def extract_rca_signals(
    log_samples: list[str],
    ec2_state: str | None = None,
    alb_target_reasons: list[str] | None = None,
) -> dict:
    """
    RCA signal extractor (P2 + P9).

    Scans log samples, EC2 state, and ALB target reasons against rule tables.
    Returns classified signals with the highest-confidence primary root cause
    promoted to the top.

    This is called by Python (not the LLM) and the result is embedded into
    the initial investigation context so the agent starts with pre-classified
    signals.

    Returns:
    {
      "primary_root_cause": { rca_type, description, confidence, source } | None,
      "supporting_signals":  [...],
      "infra_symptoms":      [...],
      "downstream_impacts":  [...],
      "all_signals":         [...],
      "has_deterministic_rca": bool,
    }
    """
    all_signals: list[dict] = []

    # ── Scan log samples against rules ───────────────────────────────────────
    for sample in log_samples:
        for rule in _LOG_SIGNAL_RULES:
            if rule["pattern"].search(sample):
                all_signals.append({
                    "type":        rule["type"],
                    "rca_type":    rule["rca_type"],
                    "description": rule["description"],
                    "confidence":  rule["confidence"],
                    "source":      "log",
                    "evidence":    sample[:200],
                })

    # ── Check EC2 state ───────────────────────────────────────────────────────
    if ec2_state and ec2_state.lower() in _EC2_STATE_RULES:
        rule = _EC2_STATE_RULES[ec2_state.lower()]
        all_signals.append({
            "type":        rule["type"],
            "rca_type":    rule["rca_type"],
            "description": rule["description"],
            "confidence":  rule["confidence"],
            "source":      "ec2_state",
            "evidence":    f"EC2 state={ec2_state}",
        })

    # ── Check ALB target reasons ──────────────────────────────────────────────
    for reason in (alb_target_reasons or []):
        if reason in _ALB_REASON_RULES:
            rule = _ALB_REASON_RULES[reason]
            all_signals.append({
                "type":        rule["type"],
                "rca_type":    rule["rca_type"],
                "description": rule["description"],
                "confidence":  rule["confidence"],
                "source":      "alb_target_reason",
                "evidence":    f"ALB target reason={reason}",
            })

    # ── Deduplicate by rca_type (keep highest confidence) ────────────────────
    seen: dict[str, dict] = {}
    for sig in all_signals:
        key = sig["rca_type"]
        if key not in seen or sig["confidence"] > seen[key]["confidence"]:
            seen[key] = sig
    deduped = list(seen.values())

    primaries   = [s for s in deduped if s["type"] == "primary_root_cause"]
    supporting  = [s for s in deduped if s["type"] == "supporting_signal"]
    symptoms    = [s for s in deduped if s["type"] == "infra_symptom"]
    downstream  = [s for s in deduped if s["type"] == "downstream_impact"]

    # Best primary = highest confidence
    best_primary = max(primaries, key=lambda x: x["confidence"]) if primaries else None

    return {
        "primary_root_cause":    best_primary,
        "supporting_signals":    supporting,
        "infra_symptoms":        symptoms,
        "downstream_impacts":    downstream,
        "all_signals":           deduped,
        "has_strong_signal":     best_primary is not None and best_primary["confidence"] >= 0.85,
    }

def pre_triage_targets(targets: list[dict]) -> dict:
    """
    Provide an initial investigation hypothesis based on ALB target status.

    Returns:
      {
        "triage_result": None | {
            likely_issue, skip_metrics, skip_logs,
            confidence, summary, affected_instances,
            target_reason
        }
    }
    """
    if not targets:
        return {"triage_result": None}

    unhealthy = [t for t in targets if t.get("target_health") not in ("healthy", None)]
    if not unhealthy:
        return {"triage_result": None}

    reasons = {t.get("target_reason", "") for t in unhealthy}
    if len(reasons) != 1:
        return {"triage_result": None}

    reason    = list(reasons)[0]
    blueprint = _HIGH_CONFIDENCE_TRIAGE_MAP.get(reason)
    if not blueprint:
        return {"triage_result": None}

    return {
        "triage_result": {
            **blueprint,
            "target_reason":       reason,
            "affected_instances":  [t.get("instance_id") for t in unhealthy],
        }
    }

def evaluate_investigation_completion(
    candidate_rca: dict,
    evidence: list[str],
    investigation_state: dict
) -> tuple[bool, str]:
    """
    Evaluates whether an investigation starting from a symptom hypothesis has successfully
    identified a root cause.
    """
    evidence_text = " ".join(evidence).lower()

    if candidate_rca.get("rca_type") == "database_connectivity_failure":
        if investigation_state.get("network_validated"):
            if "security group blocks outbound" in evidence_text or "nacl blocks outbound" in evidence_text or "route table missing" in evidence_text:
                return True, "Investigation uncovered network blockage causing database connectivity failure."
            if "clear" in evidence_text and "network path" in evidence_text:
                # If network is clear, we might need DB tools next, but for now we note it's validated
                # We do not stop yet because the root cause is not found
                pass

    return False, ""


def has_sufficient_evidence(
    candidate_rca: dict | None,
    evidence: list[str],
    investigation_state: dict
) -> tuple[bool, str]:
    """
    Called after each tool-node execution to decide if we have enough evidence
    to stop the investigation loop without further LLM invokes.

    Returns (should_stop: bool, reason: str).
    """
    if not candidate_rca:
        return False, ""

    rca_type   = candidate_rca.get("rca_type", "")
    confidence = candidate_rca.get("confidence", 0)
    category   = RCA_CATEGORIES.get(rca_type, "investigate")

    if category == "investigate":
        return evaluate_investigation_completion(candidate_rca, evidence, investigation_state)

    # Instance stopped/terminated requires EC2 tool validation
    if rca_type in ("instance_stopped", "instance_terminated"):
        if investigation_state.get("compute_validated"):
            return True, f"EC2 instance is {rca_type.replace('instance_', '')} — corroborated by EC2 analysis."

    # High-confidence config/DB error confirmed by collected evidence
    high_conf_types = {
        "database_config_error",
        "iam_permission_error",  "aws_credentials_error",
        "disk_full",             "oom_kill",
    }
    if rca_type in high_conf_types and confidence >= 0.88:
        if len(evidence) > 0 or investigation_state.get("compute_validated"):
            return True, (
                f"High-confidence primary RCA '{rca_type}' ({confidence:.0%}) confirmed "
                f"by collected evidence."
            )

    # Network path investigation produced a definitive blocked-layer result
    network_rca_types = {"sg_blocked_port", "nacl_blocked_port", "route_table_missing"}
    if rca_type in network_rca_types and confidence >= 0.88:
        if investigation_state.get("network_validated"):
            return True, (
                f"Network path investigation confirmed '{rca_type}' as root cause ({confidence:.0%})."
            )

    # CloudTrail confirmed SG change = highest confidence possible
    if rca_type == "sg_change_caused_outage" and confidence >= 0.95:
        if investigation_state.get("change_history_validated"):
            return True, (
                "CloudTrail SG audit confirmed who changed the Security Group and when. "
                f"Confidence={confidence:.0%}."
            )

    return False, ""

def correlate_timeline(
    infra_events: list[dict],
    log_anchor_ts: str | None,
    down_time: str,
    window_minutes: int = 15,
) -> dict:
    """
    Links CloudTrail infrastructure events → first log error → ALB degradation
    in a causal timeline (P7).

    infra_events: list of CloudTrail high_risk_events dicts with 'event_time' key.
    log_anchor_ts: ISO timestamp of first error in logs (from log anchor).
    down_time: ISO timestamp of monitor alert.
    window_minutes: how many minutes before down_time to look for triggers.

    Returns:
    {
      "causal_chain": [
        { "timestamp": ..., "type": "infra_change|log_error|alb_degradation",
          "event": ..., "minutes_before_incident": float }
      ],
      "probable_trigger": { event } | None,
      "trigger_to_error_gap_minutes": float | None,
    }
    """
    try:
        down_dt = datetime.fromisoformat(down_time.replace("Z", "+00:00"))
    except Exception:
        return {"error": "Invalid down_time format"}

    cutoff = down_dt - timedelta(minutes=window_minutes)
    chain  = []

    # ── Infra changes ─────────────────────────────────────────────────────────
    for evt in (infra_events or []):
        raw_ts = evt.get("event_time") or evt.get("EventTime") or evt.get("time")
        if not raw_ts:
            continue
        try:
            ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
        except Exception:
            continue
        if ts >= cutoff:
            chain.append({
                "timestamp":                str(ts),
                "type":                     "infra_change",
                "event":                    evt.get("event_name") or evt.get("EventName", "unknown"),
                "detail":                   evt,
                "minutes_before_incident":  round((down_dt - ts).total_seconds() / 60, 1),
            })

    # ── First log error ───────────────────────────────────────────────────────
    if log_anchor_ts:
        try:
            anchor_dt = datetime.fromisoformat(log_anchor_ts.replace("Z", "+00:00"))
            chain.append({
                "timestamp":               str(anchor_dt),
                "type":                    "first_log_error",
                "event":                   "First error detected in application logs",
                "minutes_before_incident": round((down_dt - anchor_dt).total_seconds() / 60, 1),
            })
        except Exception:
            pass

    # ── ALB degradation = incident down_time ─────────────────────────────────
    chain.append({
        "timestamp":               str(down_dt),
        "type":                    "alb_degradation",
        "event":                   "Monitor alert fired / ALB reported unhealthy",
        "minutes_before_incident": 0,
    })

    chain.sort(key=lambda x: x["timestamp"])

    # ── Identify probable trigger (infra event closest before first log error) ─
    probable_trigger          = None
    trigger_to_error_gap_mins = None

    infra_changes = [c for c in chain if c["type"] == "infra_change"]
    log_errors    = [c for c in chain if c["type"] == "first_log_error"]

    if infra_changes and log_errors:
        first_log_ts = log_errors[0]["timestamp"]
        # Latest infra change before or at first log error
        before_error = [
            c for c in infra_changes if c["timestamp"] <= first_log_ts
        ]
        if before_error:
            probable_trigger = max(before_error, key=lambda x: x["timestamp"])
            try:
                trigger_dt           = datetime.fromisoformat(probable_trigger["timestamp"].replace("Z", "+00:00"))
                first_log_dt         = datetime.fromisoformat(first_log_ts.replace("Z", "+00:00"))
                trigger_to_error_gap_mins = round(
                    (first_log_dt - trigger_dt).total_seconds() / 60, 1
                )
            except Exception:
                pass

    return {
        "causal_chain":               chain,
        "probable_trigger":           probable_trigger,
        "trigger_to_error_gap_minutes": trigger_to_error_gap_mins,
    }

def find_similar_incidents(
    rca_type: str,
    monitor_type: str | None = None,
    limit: int = 3,
) -> list[dict]:
    """
    Query past incidents with the same rca_type / monitor_type for context (P8).
    Returns up to `limit` similar incidents with their root_cause and remediation.

    This is a simple keyword match — not vector similarity.
    Future: replace with pgvector cosine similarity on embedded RCA text.
    """
    try:
        from app.utils.db import get_db
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        id,
                        monitor_name,
                        monitor_type,
                        down_message,
                        analysis_result,
                        remediation_steps,
                        confidence_score,
                        analysis_completed_at
                    FROM meyiconnect.insight_incidents
                    WHERE analysis_status = 'completed'
                      AND analysis_result IS NOT NULL
                      AND (
                          analysis_result::text ILIKE %s
                          OR (monitor_type = %s AND %s IS NOT NULL)
                      )
                      AND analysis_completed_at IS NOT NULL
                    ORDER BY analysis_completed_at DESC
                    LIMIT %s
                    """,
                    (
                        f"%{rca_type}%",
                        monitor_type,
                        monitor_type,
                        limit,
                    ),
                )
                rows = cur.fetchall() or []

        results = []
        for row in rows:
            try:
                rca = json.loads(row.get("analysis_result") or "{}")
                results.append({
                    "incident_id":    row.get("id"),
                    "monitor_name":   row.get("monitor_name"),
                    "monitor_type":   row.get("monitor_type"),
                    "down_message":   row.get("down_message"),
                    "summary":        rca.get("summary", ""),
                    "confidence":     row.get("confidence_score"),
                    "resolved_at":    str(row.get("analysis_completed_at", "")),
                })
            except Exception:
                pass

        return results

    except Exception as exc:
        logger.warning(f"[SimilarIncidents] Lookup failed: {exc}")
        return []