"""
app/processor/process_incident.py
──────────────────────────────────
Orchestrator for AIOps incident investigation (v3).

Changes in v3
─────────────
P1 — Pre-triage hypothesis generation (guides investigation but does not skip it)
     entirely. RCA is built from deterministic signals. Zero Bedrock invokes.

P2 — Pre-signal extraction: extract_rca_signals() runs on any immediately
     available log samples / EC2 state / ALB reasons BEFORE the LLM starts.
     Result embedded in initial HumanMessage.

P4 — Strict state machine: _update_status() only called from Python.
     States: queued → triage_started → infra_analysis → logs_analysis →
             ai_reasoning → remediation_generated → completed | failed.
     LLM never touches status.

P7 — Temporal correlator: correlate_timeline() called in Python after
     resolve_incident_targets() + get_infra_events() pre-fetch.

P8 — Similar incident lookup: find_similar_incidents() called before
     the agent starts and embedded as context.

P5 — All DB writes (RCA, evidence, status) centralised here.
     No tool calls for persistence.
"""

import json
import logging
from datetime import datetime, timezone

from app.utils.db import get_db
from app.utils.aws_connector import AWSClientFactory, AWSAuthenticationError
from app.utils.invocation_logger import get_rca_logger, invocation_log_context
from app.agent.tools import (
    init_tools,
    resolve_incident_targets,
    get_ec2_analysis,
    get_alb_target_health,
)
from app.agent.evaluators import (
    pre_triage_targets,
    extract_rca_signals,
    correlate_timeline,
    find_similar_incidents,
)
from app.agent.rules import WORKFLOW_STATES
from app.agent.graph import run_agent_investigation

# Module-level fallback logger (used by helpers called before the invocation logger is ready)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# P4 — Strict state machine DB writer
# Only this function may update analysis_status / analysis_percent.
# LLM never calls this.
# ═══════════════════════════════════════════════════════════════════════════════

def _update_status(incident_id: str, status: str, percent: int | None = None, message: str | None = None) -> None:
    """
    Update incident status in DB.  Only called by Python — never by LLM.

    Valid states and their default percentages (WORKFLOW_STATES):
      queued                  0
      triage_started         10
      infra_analysis         30
      logs_analysis          60
      ai_reasoning           80
      remediation_generated  90
      completed             100
      failed                  0
    """
    pct = percent if percent is not None else WORKFLOW_STATES.get(status, 0)
    logger.info(f"[Status] {incident_id} → {status} ({pct}%)")
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                if message:
                    analysis_result_json = json.dumps({"error": message})
                    cur.execute(
                        """
                        UPDATE meyiconnect.insight_incidents
                        SET analysis_status  = %s,
                            analysis_percent = %s,
                            analysis_result  = %s,
                            updated_at       = NOW()
                        WHERE id = %s
                        """,
                        (status, pct, analysis_result_json, incident_id),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE meyiconnect.insight_incidents
                        SET analysis_status  = %s,
                            analysis_percent = %s,
                            updated_at       = NOW()
                        WHERE id = %s
                        """,
                        (status, pct, incident_id),
                    )
            conn.commit()
    except Exception as exc:
        logger.error(f"[StatusUpdate] Failed for {incident_id}: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# DB persistence helpers — all writes centralised here
# ═══════════════════════════════════════════════════════════════════════════════

def _save_rca(incident_id: str, structured: dict,
              ai_model: str = "langgraph-agent-v4") -> None:
    """
    Persist the 5-field RCA result from the agent.
    Stores the full structured dict in analysis_result.
    Sets analysis_status=completed, analysis_percent=100.

    Schema stored:
      probable_root_cause (str)
      confidence          (int 0-100)
      evidence            (list[str])
      dependency_impact   (list[str])
      recommended_actions (list[str])
    """
    probable_root_cause = structured.get("probable_root_cause", "")
    confidence          = structured.get("confidence", 50)

    # Normalise confidence: accept 0-100 int or 0.0-1.0 float
    try:
        confidence = float(confidence)
        if confidence <= 1.0:
            confidence = confidence * 100
        confidence = round(confidence, 2)
    except (TypeError, ValueError):
        confidence = 50.0

    # Build the analysis_result to store — the full 5-field schema
    analysis_result = {
        "probable_root_cause": probable_root_cause,
        "confidence":          int(confidence),
        "evidence":            structured.get("evidence", []),
        "dependency_impact":   structured.get("dependency_impact", []),
        "recommended_actions": structured.get("recommended_actions", []),
    }

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE meyiconnect.insight_incidents
                    SET
                        analysis_status       = 'completed',
                        analysis_percent      = 100,
                        analysis_result       = %s,
                        confidence_score      = %s,
                        ai_model_used         = %s,
                        analysis_completed_at = NOW(),
                        updated_at            = NOW()
                    WHERE id = %s
                    """,
                    (
                        json.dumps(analysis_result),
                        str(confidence),
                        ai_model,
                        incident_id,
                    ),
                )
                logger.info(f"[SaveRCA] Updated {cur.rowcount} row(s) for {incident_id}")
            conn.commit()
    except Exception as exc:
        logger.error(f"[SaveRCA] Failed for {incident_id}: {exc}")
        raise


def _save_evidence(incident_id: str, agent_result: dict) -> None:
    """
    Persist collected evidence, investigation findings, RCA signals, and tool calls.
    Non-fatal if it fails — RCA is already saved.
    """
    evidence_list = agent_result.get("evidence", [])
    investigation_findings = agent_result.get("investigation_findings", [])
    rca_signals = agent_result.get("rca_signals", {})
    tool_calls_made = agent_result.get("tool_calls_made", [])

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO meyiconnect.incident_evidence (
                        incident_id, evidence_text, investigation_findings,
                        rca_signals, tool_calls_made, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                    ON CONFLICT (incident_id) DO UPDATE
                        SET evidence_text          = EXCLUDED.evidence_text,
                            investigation_findings = EXCLUDED.investigation_findings,
                            rca_signals            = EXCLUDED.rca_signals,
                            tool_calls_made        = EXCLUDED.tool_calls_made,
                            updated_at             = NOW()
                    """,
                    (
                        incident_id,
                        json.dumps(evidence_list),
                        json.dumps(investigation_findings),
                        json.dumps(rca_signals),
                        json.dumps(tool_calls_made),
                    ),
                )
                logger.info(f"[SaveEvidence] Updated evidence for {incident_id}")
            conn.commit()
    except Exception as exc:
        logger.error(f"[SaveEvidence] Failed for {incident_id}: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# DB loaders
# ═══════════════════════════════════════════════════════════════════════════════

def _load_incident(incident_id: str) -> dict | None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM meyiconnect.insight_incidents WHERE id = %s LIMIT 1",
                (incident_id,),
            )
            return cur.fetchone()


def _load_project_by_tag(project_tag: str) -> dict | None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, tag,
                       aws_access_key_id,
                       aws_secret_access_key,
                       aws_region,
                       dependencies
                FROM meyiconnect.insight_projects
                WHERE tag = %s
                LIMIT 1
                """,
                (project_tag,),
            )
            return cur.fetchone()


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_incident(incident: dict | None, incident_id: str) -> str | None:
    if not incident:
        return f"Incident {incident_id} not found in DB"
    if not incident.get("down_time"):
        return f"Missing incident down_time"
    return None


def _validate_project(project: dict | None, project_tag: str, incident_id: str) -> str | None:
    if not project:
        return f"No project found with tag '{project_tag}' for incident {incident_id}"
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Deterministic pre-resolve (Python, no LLM invoke)
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_targets_deterministic() -> tuple[list[dict], dict | None, dict]:
    """
    Call resolve_incident_targets() in Python before the agent starts.
    Saves 1 LLM invoke vs letting the agent call it as its first tool.

    Returns (targets, triage_result, alb_meta).
    """
    try:
        result = resolve_incident_targets()
        targets       = result.get("targets", [])
        alb_meta      = result.get("alb_meta", {})
        triage_info   = pre_triage_targets(targets)
        triage_result = triage_info.get("triage_result")

        if triage_result:
            logger.info(
                f"[PreTriage] {triage_result['likely_issue']} | "
                f"reason={triage_result['target_reason']} | "
                f"confidence={triage_result['confidence']}"
            )
        else:
            logger.info("[PreTriage] No short-circuit — full investigation")

        return targets, triage_result, alb_meta

    except Exception as exc:
        logger.warning(f"[PreResolve] resolve_incident_targets failed: {exc}")
        return [], None, {}


# ═══════════════════════════════════════════════════════════════════════════════
# EC2 deterministic pre-enrichment
# ═══════════════════════════════════════════════════════════════════════════════

def _enrich_ec2_deterministic(targets: list[dict]) -> dict:
    """
    Pre-fetch EC2 analysis (state, status checks, metrics) for all resolved
    targets before the LLM starts.  Saves 1 LLM + Bedrock round-trip by
    embedding foundational EC2 facts in the initial HumanMessage.

    Returns the raw ec2_analysis dict as returned by get_ec2_analysis():
      {"instances": {"i-xxx": {details, status_checks, metrics}}, "findings": []}
    """
    instance_ids = [t["instance_id"] for t in targets if t.get("instance_id")]
    if not instance_ids:
        logger.info("[PreEnrich] No EC2 instance IDs to enrich")
        return {}
    try:
        result = get_ec2_analysis(instance_ids=instance_ids)
        logger.info(f"[PreEnrich] EC2 analysis fetched for {instance_ids}")
        return result
    except Exception as exc:
        logger.warning(f"[PreEnrich] EC2 analysis failed: {exc}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# ALB deterministic pre-enrichment
# ═══════════════════════════════════════════════════════════════════════════════

def _enrich_alb_deterministic(alb_meta: dict) -> dict:
    """
    Pre-fetch full ALB target health (per target-group breakdown) if the
    incident involves an ALB dependency.  Provides richer health data than
    the alb_meta already captured during dependency resolution.

    Returns the raw get_alb_target_health() result dict, or {} if no ALB.
    """
    # alb_arn is available from dependency resolution; fall back to dns
    alb_arn = alb_meta.get("alb_arn") or alb_meta.get("alb_dns")
    if not alb_arn:
        return {}
    try:
        result = get_alb_target_health(alb_dns_or_arn=alb_arn)
        logger.info(f"[PreEnrich] ALB health fetched for {alb_arn}")
        return result
    except Exception as exc:
        logger.warning(f"[PreEnrich] ALB health fetch failed: {exc}")
        return {}

# ═══════════════════════════════════════════════════════════════════════════════
# P2 — Pre-extract RCA signals before first LLM invoke
# ═══════════════════════════════════════════════════════════════════════════════

def _pre_extract_signals(
    targets: list[dict],
    triage_result: dict | None,
    incident_context: dict,
    ec2_analysis: dict | None = None,
) -> dict:
    """
    Run extract_rca_signals() on whatever we know BEFORE the LLM starts:
      - ALB target reasons from resolved targets
      - EC2 state from pre-fetched ec2_analysis (now available before LLM)
      - down_message text scanned for known error patterns

    Returns the signals dict to embed in the initial HumanMessage.
    """
    alb_reasons = [
        t.get("target_reason", "") for t in targets
        if t.get("target_reason")
    ]

    # Scan the down_message for known log patterns
    down_message = incident_context.get("down_message") or ""
    log_samples  = [down_message] if down_message else []

    # Extract real EC2 state from pre-fetched analysis (first instance wins as
    # primary signal; multi-instance scenarios are handled by correlate_instances)
    ec2_state: str | None = None
    if ec2_analysis:
        instances = ec2_analysis.get("instances", {})
        for iid, data in instances.items():
            state = data.get("details", {}).get("state")
            if state:
                ec2_state = state
                logger.info(f"[PreSignals] Using pre-fetched EC2 state={state} for {iid}")
                break

    signals = extract_rca_signals(
        log_samples=log_samples,
        ec2_state=ec2_state,
        alb_target_reasons=alb_reasons,
    )

    if signals.get("primary_root_cause"):
        logger.info(
            f"[PreSignals] primary_root_cause={signals['primary_root_cause']['rca_type']} "
            f"({signals['primary_root_cause']['confidence']:.0%})"
        )
    else:
        logger.info("[PreSignals] No primary root cause extracted pre-LLM")

    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def process_incident(payload: dict) -> None:
    """
    Process a single incident using the autonomous agentic investigation flow.
    Called by the worker thread pool with payload = {"incident_id": str}.
    """
    incident_id = payload.get("incident_id", "unknown")

    # ── Per-invocation file logger ─────────────────────────────────────────────
    # Creates: logs/rca/rca-<incident_id>-<timestamp>.log  (max 10 retained)
    base_log = get_rca_logger(incident_id)
    ctx = invocation_log_context(base_log)
    log = ctx.__enter__()

    log.info("=" * 60)
    log.info("INCIDENT PROCESSOR v3 STARTED")
    log.info("=" * 60)

    try:
        # ── 1. Load & validate incident ───────────────────────────────────────
        incident = _load_incident(incident_id)
        error    = _validate_incident(incident, incident_id)
        if error:
            log.error(f"[Validate] {error}")
            _update_status(incident_id, "failed", message=error)
            return
        incident = dict(incident)

        # ── 2. Load project ───────────────────────────────────────────────────
        project_tag = incident.get("project_tag")
        if not project_tag:
            log.error(f"[Project] No project_tag on incident {incident_id}")
            _update_status(incident_id, "failed", message=f"No project_tag on incident {incident_id}")
            return

        project = _load_project_by_tag(project_tag)
        error   = _validate_project(project, project_tag, incident_id)
        if error:
            log.error(f"[Project] {error}")
            _update_status(incident_id, "failed", message=error)
            return
        project = dict(project)

        log.info(
            f"[Load] incident={incident_id} | "
            f"project='{project.get('name')}' tag={project_tag} | "
            f"region={project.get('aws_region')}"
        )

        # ── 3. Build unified investigation context ────────────────────────────
        def _parse_json_field(val, fallback=None):
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except Exception:
                    return fallback or []
            return val or fallback or []

        raw_deps     = _parse_json_field(incident.get("dependencies"))
        project_deps = _parse_json_field(project.get("dependencies"))
        final_deps   = raw_deps if raw_deps else project_deps

        investigation_context = {
            "incident_id":        incident_id,
            "project_tag":        project_tag,
            "monitor_name":       incident.get("monitor_name"),
            "monitor_type":       incident.get("monitor_type"),
            "monitor_url":        incident.get("monitor_url"),
            "down_message":       incident.get("down_message"),
            "down_time": str(incident.get("down_time")),
            "aws_region":         project.get("aws_region"),
            "dependencies":       final_deps,
        }

        if not final_deps:
            log.error(f"[Validate] No dependencies for {incident_id}")
            _update_status(incident_id, "failed", message=f"This project has no attached dependencies")
            return

        # ── 4. Build AWS factory ──────────────────────────────────────────────
        try:
            aws_factory = AWSClientFactory(project_tag=project_tag)
        except AWSAuthenticationError as exc:
            log.error(f"[AWS] Authentication error: {exc}")
            _update_status(incident_id, "failed", message=str(exc))
            return
        except ValueError as exc:
            log.error(f"[AWS] Factory init failed: {exc}")
            _update_status(incident_id, "failed", message=f"Factory init failed: {exc}")
            return

        # ── 5. Inject tools context ───────────────────────────────────────────
        init_tools(aws_factory=aws_factory, incident_row=investigation_context)
        log.info(f"[Tools] Initialised for {incident_id}")

        # ── 6. P4: triage_started ─────────────────────────────────────────────
        _update_status(incident_id, "triage_started")

        # ── 7. Deterministic pre-resolve ─────────────────────────────────────────────────
        targets, triage_result, alb_meta = _resolve_targets_deterministic()

        # ── 7b. Deterministic EC2 enrichment ──────────────────────────────────────────
        ec2_analysis: dict = {}
        if targets:
            ec2_analysis = _enrich_ec2_deterministic(targets)
            log.info(f"[PreEnrich] EC2 enrichment complete: {list(ec2_analysis.get('instances', {}).keys())}")

        # ── 7c. Deterministic ALB enrichment (only if ALB dep) ─────────────────
        alb_health: dict = {}
        if alb_meta:
            alb_health = _enrich_alb_deterministic(alb_meta)
            if alb_health:
                log.info(f"[PreEnrich] ALB health enrichment complete: {alb_health.get('total_targets')} targets")

        # ── 8. P2: Pre-extract RCA signals ──────────────────────────────────────────
        rca_signals = _pre_extract_signals(
            targets, triage_result, investigation_context, ec2_analysis
        )

        # ── 9. P8: Similar incident lookup ────────────────────────────────────
        primary_rca_type = (
            rca_signals.get("primary_root_cause", {}) or {}
        ).get("rca_type")

        similar_incidents = find_similar_incidents(
            rca_type=primary_rca_type or "",
            monitor_type=incident.get("monitor_type"),
            limit=3,
        )
        if similar_incidents:
            log.info(f"[SimilarIncidents] Found {len(similar_incidents)} past incidents")

        # ── 10. Update Status before Analysis ──────────────────────────────────
        _update_status(incident_id, "infra_analysis")

        # ── 11. P7: Build temporal correlation (best-effort, pre-LLM) ─────────
        #
        # We don't have infra events yet (that requires a CloudTrail call)
        # but we can pre-build a skeleton timeline from triage timestamps.
        # The agent will enrich this via get_infra_events() if needed.
        #
        timeline = None
        try:
            timeline = correlate_timeline(
                infra_events=[],
                log_anchor_ts=None,
                down_time=investigation_context["down_time"],
            )
        except Exception as exc:
            log.warning(f"[Timeline] Pre-build failed: {exc}")

        # ── 12. Run the agent ──────────────────────────────────────────────────────
        log.info(
            f"[AgentInput] Starting investigation for {incident_id}. Inputs passed to LLM: "
            f"incident_context={investigation_context}, triage_result={triage_result}, "
            f"rca_signals={rca_signals}, timeline={timeline}, "
            f"similar_incidents={similar_incidents}, resolved_targets={targets}, "
            f"alb_meta={alb_meta}, ec2_analysis={ec2_analysis}, alb_health={alb_health}"
        )

        result = run_agent_investigation(
            incident_id=incident_id,
            incident_context=investigation_context,
            triage_result=triage_result,
            rca_signals=rca_signals,
            timeline=timeline,
            similar_incidents=similar_incidents,
            resolved_targets=targets,
            alb_meta=alb_meta,
            ec2_analysis=ec2_analysis,
            alb_health=alb_health,
        )

        structured = result.get("structured_result")

        log.info(
            f"[Agent] Finished | "
            f"tool_calls={result.get('tool_call_count')} | "
            f"messages={result.get('message_count')} | "
            f"structured={'yes' if structured else 'no'}"
        )

        # ── 13. Persist results — all DB writes happen here ─────────────────
        if structured:
            _update_status(incident_id, "remediation_generated",
                           WORKFLOW_STATES["remediation_generated"])
            _save_rca(incident_id, structured)   # also sets status=completed, percent=100
            _save_evidence(incident_id, result)
        else:
            log.error(f"[Agent] No structured result for {incident_id}.")
            _update_status(incident_id, "failed", message=f"No structured result for {incident_id}.")

        log.info("=" * 60)
        log.info(f"INCIDENT COMPLETED: {incident_id}")
        log.info("=" * 60)

    except AWSAuthenticationError as exc:
        log.error(f"[AWS] Authentication error during incident {incident_id}: {exc}")
        try:
            _update_status(incident_id, "failed", message=str(exc))
        except Exception:
            log.exception("[Fatal] Could not update status to failed")
    except Exception as exc:
        log.exception(f"[Fatal] process_incident failed for {incident_id}")
        try:
            _update_status(incident_id, "failed", message=f"Fatal error: {exc}")
        except Exception:
            log.exception("[Fatal] Could not update status to failed")
        raise
    finally:
        ctx.__exit__(None, None, None)