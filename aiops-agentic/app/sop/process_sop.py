"""
app/sop/process_sop.py
───────────────────────
Orchestrator for SOP generation — direct Markdown pipeline.

Pipeline:
  INCIDENT mode:
    load context → LLM → extract metadata → save DB

  PROMPT mode:
    guardrail check (Layer 1→2→3, zero tokens)
      → FAIL: mark invalid_prompt, store message, return
      → PASS: LLM → extract metadata → save DB

Status transitions:
  pending → generating → completed       (success)
  pending → invalid_prompt               (guardrail blocked — no LLM call made)
  pending → generating → failed          (unexpected error)
"""

import json
import logging
import re

from app.utils.db import get_db
from app.sop.generator import (
    generate_sop_from_incident,
    generate_sop_from_prompt,
    SOP_BEDROCK_MODEL,
)
from app.sop.guardrails import validate_sop_prompt

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DB helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _update_sop_status(db_id: str, status: str) -> None:
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE meyiconnect.insight_sops
                    SET status = %s, updated_at = NOW()
                    WHERE id = %s
                    """,
                    (status, db_id),
                )
            conn.commit()
        logger.info(f"[SOP-Status] {db_id} → {status}")
    except Exception as exc:
        logger.error(f"[SOP-Status] Failed to update {db_id}: {exc}")


def _mark_invalid_prompt(db_id: str, error_message: str) -> None:
    """Mark blocked by guardrail — no LLM call was made."""
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE meyiconnect.insight_sops
                    SET
                        status        = 'invalid_prompt',
                        steps = %s,
                        updated_at    = NOW()
                    WHERE id = %s
                    """,
                    (error_message, db_id),
                )
            conn.commit()
        logger.info(f"[SOP-Status] {db_id} → invalid_prompt (guardrail blocked)")
    except Exception as exc:
        logger.error(f"[SOP-Status] Failed to mark invalid_prompt for {db_id}: {exc}")


def _save_sop_content(
    db_id: str,
    content_md: str,
    title: str,
    alert_type: str,
    service: str,
    severity: str,
) -> None:
    allowed  = {"Critical", "High", "Medium", "Low"}
    if severity not in allowed:
        severity = "Medium"

    overview = _extract_overview(content_md)

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE meyiconnect.insight_sops
                    SET
                        steps         = %s,
                        description   = %s,
                        title         = %s,
                        alert_type    = %s,
                        service       = %s,
                        severity      = %s,
                        ai_model_used = %s,
                        status        = 'completed',
                        updated_at    = NOW()
                    WHERE id = %s
                    """,
                    (
                        content_md, overview, title,
                        alert_type, service, severity,
                        SOP_BEDROCK_MODEL, db_id,
                    ),
                )
                logger.info(f"[SOP-Save] Updated {cur.rowcount} row(s) for db_id={db_id}")
            conn.commit()
    except Exception as exc:
        logger.error(f"[SOP-Save] Failed for db_id={db_id}: {exc}")
        raise


def _load_incident_with_rca(incident_id: str) -> tuple[dict | None, dict | None, dict]:
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM meyiconnect.insight_incidents WHERE id = %s LIMIT 1",
                    (incident_id,),
                )
                row = cur.fetchone()
                cur.execute(
                    "SELECT * FROM meyiconnect.incident_evidence WHERE incident_id = %s LIMIT 1",
                    (incident_id,),
                )
                ev_row = cur.fetchone()

        if not row:
            logger.warning(f"[SOP-Load] Incident {incident_id} not found")
            return None, None, {}

        incident      = dict(row)
        evidence_data = dict(ev_row) if ev_row else {}

        raw = incident.get("analysis_result")
        if isinstance(raw, str):
            try:
                rca_result = json.loads(raw)
            except Exception:
                rca_result = {}
        elif isinstance(raw, dict):
            rca_result = raw
        else:
            rca_result = {}

        return incident, rca_result, evidence_data

    except Exception as exc:
        logger.error(f"[SOP-Load] DB error for incident {incident_id}: {exc}")
        return None, None, {}


# ═══════════════════════════════════════════════════════════════════════════════
# Markdown metadata extractors
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_title(md: str) -> str:
    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            return stripped[2:].strip()
    return "Generated SOP"


def _extract_overview(md: str) -> str:
    lines      = md.splitlines()
    in_section = False
    buffer     = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "## overview":
            in_section = True
            continue
        if in_section:
            if stripped.startswith("## "):
                break
            if stripped.startswith("|") or stripped.startswith("---"):
                continue
            if stripped:
                buffer.append(stripped)
    if buffer:
        return " ".join(buffer)
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("|"):
            return stripped
    return ""


def _extract_metadata_field(md: str, field: str) -> str:
    pattern = re.compile(
        rf"^\|\s*{re.escape(field)}\s*\|\s*(.+?)\s*\|",
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(md)
    if match:
        return re.sub(r"[*_`]", "", match.group(1)).strip()
    return ""


def _infer_severity_from_rca(rca_result: dict) -> str:
    confidence = rca_result.get("confidence", 50)
    if isinstance(confidence, (int, float)):
        if confidence >= 85: return "Critical"
        if confidence >= 65: return "High"
        if confidence >= 40: return "Medium"
    return "Low"


def _infer_severity_from_monitor(monitor_type: str) -> str:
    critical = {"CPUUtilization", "MemoryUtilization", "DiskUsage", "StatusCheckFailed"}
    high     = {"DatabaseConnections", "UnHealthyHostCount", "5xxErrorRate"}
    if monitor_type in critical: return "Critical"
    if monitor_type in high:     return "High"
    return "Medium"


def _resolve_severity(md: str, fallback: str) -> str:
    allowed = {"Critical", "High", "Medium", "Low"}
    from_md = _extract_metadata_field(md, "Severity").capitalize()
    return from_md if from_md in allowed else fallback


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def process_sop(payload: dict) -> None:
    """
    Full SOP generation pipeline.

    Payload (internal, built by the API route):
      { "id": db_id, "sop_id": "SOP-201", "incident_id": "<uuid>" }   ← INCIDENT
      { "id": db_id, "sop_id": "SOP-201", "prompt": "<text>" }        ← PROMPT
    """
    logger.info("=" * 60)
    logger.info("SOP PROCESSOR STARTED")
    logger.info("=" * 60)

    db_id       = payload.get("id")
    sop_id      = payload.get("sop_id", "unknown")
    incident_id = payload.get("incident_id")
    user_prompt = payload.get("prompt")

    if not db_id:
        logger.error("[SOP] db_id missing from payload — cannot update status")
        return

    if not incident_id and not user_prompt:
        logger.error(f"[SOP] db_id={db_id}: payload must have incident_id OR prompt")
        _update_sop_status(db_id, "failed")
        return

    try:
        # ── MODE A: Incident-based (no guardrail — context from DB) ───────
        if incident_id:
            logger.info(f"[SOP] Mode=INCIDENT db_id={db_id} incident_id={incident_id}")
            _update_sop_status(db_id, "generating")

            incident, rca_result, evidence_data = _load_incident_with_rca(incident_id)

            if incident is None:
                logger.error(f"[SOP] Incident {incident_id} not found — aborting")
                _update_sop_status(db_id, "failed")
                return

            fallback_severity = (
                _infer_severity_from_rca(rca_result)
                if rca_result
                else _infer_severity_from_monitor(incident.get("monitor_type", ""))
            )

            alert_type = incident.get("monitor_type", "")
            service    = incident.get("monitor_name", "")

            content_md = generate_sop_from_incident(
                sop_id=sop_id,
                incident=incident,
                rca_result=rca_result or {},
                evidence_data=evidence_data or {},
            )

        # ── MODE B: Prompt-based ───────────────────────────────────────────
        else:
            logger.info(f"[SOP] Mode=PROMPT db_id={db_id} prompt_len={len(user_prompt)}")

            # ── Guardrail check (zero tokens) ──────────────────────────────
            guard = validate_sop_prompt(user_prompt)
            guard.log(prompt_snippet=user_prompt)

            if not guard.passed:
                # Block before any Bedrock call — no token cost, no status=generating
                logger.warning(
                    f"[SOP-Guardrail] BLOCKED db_id={db_id} "
                    f"layer={guard.detail.get('layer', '?')} "
                    f"rule={guard.detail.get('rule', guard.detail.get('has_infra', '?'))}"
                )
                _mark_invalid_prompt(db_id, guard.message)
                return

            # ── Guardrail passed — mark generating and call LLM ───────────
            _update_sop_status(db_id, "generating")
            logger.info(f"[SOP-Guardrail] PASSED db_id={db_id} score={guard.detail}")

            fallback_severity = "Medium"
            alert_type        = ""
            service           = ""

            content_md = generate_sop_from_prompt(
                sop_id=sop_id,
                user_prompt=user_prompt,
            )

        # ── Extract metadata and save ──────────────────────────────────────
        title    = _extract_title(content_md)
        severity = _resolve_severity(content_md, fallback_severity)

        if not service:
            service = _extract_metadata_field(content_md, "Service")
        if not alert_type:
            alert_type = _extract_metadata_field(content_md, "Alert Type")

        logger.info(
            f"[SOP-Meta] title='{title}' severity={severity} "
            f"alert_type='{alert_type}' service='{service}'"
        )

        _save_sop_content(
            db_id=db_id,
            content_md=content_md,
            title=title,
            alert_type=alert_type,
            service=service,
            severity=severity,
        )

        logger.info("=" * 60)
        logger.info(f"SOP COMPLETED: db_id={db_id} title='{title}' severity={severity}")
        logger.info("=" * 60)

    except Exception:
        logger.exception(f"[SOP-Fatal] process_sop failed for db_id={db_id}")
        try:
            _update_sop_status(db_id, "failed")
        except Exception:
            logger.exception("[SOP-Fatal] Could not update status to failed")
        raise