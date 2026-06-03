"""
app/sop/process_sop.py
───────────────────────
Orchestrator for SOP generation — simplified direct Markdown pipeline.

Pipeline:
  1. Load context from DB (incident + RCA + evidence)
  2. Call generator.py  → content_md (Markdown string)
  3. Extract metadata from the Markdown (title, severity, etc.)
  4. Save content_md directly to DB (steps column)

No JSON parsing, no renderer, no validator — clean and simple.

Two modes:
  Mode A — INCIDENT: payload has incident_id → load full DB context → generate
  Mode B — PROMPT:   payload has prompt       → generate from text only

DB table: meyiconnect.insight_sops
  steps       TEXT   — full Markdown runbook (primary content column)
  description TEXT   — overview paragraph extracted from Markdown
  title, alert_type, service, severity, status — all updated on completion
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


def _save_sop_content(
    db_id: str,
    content_md: str,
    title: str,
    alert_type: str,
    service: str,
    severity: str,
) -> None:
    """
    Persist the Markdown runbook and metadata. Mark status = 'completed'.

    Columns updated:
      steps       — full Markdown runbook (what frontend renders)
      description — Overview section text (for list/preview views)
      title, alert_type, service, severity, ai_model_used
    """
    allowed = {"Critical", "High", "Medium", "Low"}
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
                        status        = 'completed',
                        updated_at    = NOW()
                    WHERE id = %s
                    """,
                    (
                        content_md,
                        overview,
                        title,
                        alert_type,
                        service,
                        severity,
                        db_id,
                    ),
                )
                logger.info(f"[SOP-Save] Updated {cur.rowcount} row(s) for db_id={db_id}")
            conn.commit()
    except Exception as exc:
        logger.error(f"[SOP-Save] Failed for db_id={db_id}: {exc}")
        raise


def _load_incident_with_rca(incident_id: str) -> tuple[dict | None, dict | None, dict]:
    """
    Load incident row + parse analysis_result JSON + load evidence row.
    Returns (incident_dict, rca_dict, evidence_data_dict).
    Returns (None, None, {}) if not found.
    """
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
    """Pull the first H1 heading as the runbook title."""
    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            return stripped[2:].strip()
    return "Generated SOP"


def _extract_overview(md: str) -> str:
    """
    Pull the paragraph immediately under the ## Overview heading.
    Used for the description column (list/preview views).
    Falls back to first non-heading paragraph if Overview section not found.
    """
    lines      = md.splitlines()
    in_section = False
    buffer     = []

    for line in lines:
        stripped = line.strip()

        if stripped.lower() == "## overview":
            in_section = True
            continue

        if in_section:
            # Stop at the next ## heading
            if stripped.startswith("## "):
                break
            # Skip the metadata table and empty lines at the start
            if stripped.startswith("|") or stripped.startswith("---"):
                continue
            if stripped:
                buffer.append(stripped)

    if buffer:
        return " ".join(buffer)

    # Fallback: first non-heading, non-empty paragraph
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("|"):
            return stripped

    return ""


def _extract_metadata_field(md: str, field: str) -> str:
    """
    Extract a value from the ## Metadata table by field name.
    Handles: | Severity | Critical | → "Critical"
    """
    pattern = re.compile(
        rf"^\|\s*{re.escape(field)}\s*\|\s*(.+?)\s*\|",
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(md)
    if match:
        # Strip markdown bold/italic markers
        value = re.sub(r"[*_`]", "", match.group(1)).strip()
        return value
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Severity helpers (fallback when LLM doesn't set it)
# ═══════════════════════════════════════════════════════════════════════════════

def _infer_severity_from_rca(rca_result: dict) -> str:
    confidence = rca_result.get("confidence", 50)
    if isinstance(confidence, (int, float)):
        if confidence >= 85:
            return "Critical"
        if confidence >= 65:
            return "High"
        if confidence >= 40:
            return "Medium"
    return "Low"


def _infer_severity_from_monitor(monitor_type: str) -> str:
    critical = {"CPUUtilization", "MemoryUtilization", "DiskUsage", "StatusCheckFailed"}
    high     = {"DatabaseConnections", "UnHealthyHostCount", "5xxErrorRate"}
    if monitor_type in critical:
        return "Critical"
    if monitor_type in high:
        return "High"
    return "Medium"


def _resolve_severity(md: str, fallback: str) -> str:
    """
    Try to get severity from the Metadata table in the generated Markdown.
    If not found or invalid, use the fallback.
    """
    allowed  = {"Critical", "High", "Medium", "Low"}
    from_md  = _extract_metadata_field(md, "Severity")

    # Normalize — LLM may write "**CRITICAL**" or "critical"
    from_md = from_md.capitalize()
    if from_md in allowed:
        return from_md

    logger.info(
        f"[SOP-Meta] Severity not found in Markdown metadata "
        f"(got: '{from_md}') — using fallback: {fallback}"
    )
    return fallback


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def process_sop(payload: dict) -> None:
    """
    Full SOP generation pipeline (direct Markdown):

      load context → LLM → Markdown string → extract metadata → save DB

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
        # ── 1. Mark as generating ──────────────────────────────────────────
        _update_sop_status(db_id, "generating")

        # ── 2. MODE A: Incident-based ──────────────────────────────────────
        if incident_id:
            logger.info(f"[SOP] Mode=INCIDENT db_id={db_id} incident_id={incident_id}")

            incident, rca_result, evidence_data = _load_incident_with_rca(incident_id)

            if incident is None:
                logger.error(f"[SOP] Incident {incident_id} not found — aborting")
                _update_sop_status(db_id, "failed")
                return

            if not rca_result:
                logger.warning(
                    f"[SOP] Incident {incident_id} has no RCA result — "
                    "proceeding with partial context"
                )

            fallback_severity = (
                _infer_severity_from_rca(rca_result)
                if rca_result
                else _infer_severity_from_monitor(incident.get("monitor_type", ""))
            )

            alert_type = incident.get("monitor_type", "")
            service    = incident.get("monitor_name", "")

            # ── LLM call → Markdown ────────────────────────────────────────
            content_md = generate_sop_from_incident(
                sop_id=sop_id,
                incident=incident,
                rca_result=rca_result or {},
                evidence_data=evidence_data or {},
            )

        # ── 2. MODE B: Prompt-based ────────────────────────────────────────
        else:
            logger.info(f"[SOP] Mode=PROMPT db_id={db_id} prompt_len={len(user_prompt)}")

            fallback_severity = "Medium"
            alert_type        = ""
            service           = ""

            # ── LLM call → Markdown ────────────────────────────────────────
            content_md = generate_sop_from_prompt(
                sop_id=sop_id,
                user_prompt=user_prompt,
            )

        # ── 3. Extract metadata from the generated Markdown ────────────────
        title    = _extract_title(content_md)
        severity = _resolve_severity(content_md, fallback_severity)

        # For prompt mode, try to extract service/alert_type from Markdown too
        if not service:
            service = _extract_metadata_field(content_md, "Service")
        if not alert_type:
            alert_type = _extract_metadata_field(content_md, "Alert Type")

        logger.info(
            f"[SOP-Meta] title='{title}' severity={severity} "
            f"alert_type='{alert_type}' service='{service}'"
        )

        # ── 4. Save Markdown directly to DB ───────────────────────────────
        _save_sop_content(
            db_id=db_id,
            content_md=content_md,
            title=title,
            alert_type=alert_type,
            service=service,
            severity=severity,
        )

        logger.info("=" * 60)
        logger.info(
            f"SOP COMPLETED: db_id={db_id} "
            f"title='{title}' "
            f"severity={severity}"
        )
        logger.info("=" * 60)

    except Exception:
        logger.exception(f"[SOP-Fatal] process_sop failed for db_id={db_id}")
        try:
            _update_sop_status(db_id, "failed")
        except Exception:
            logger.exception("[SOP-Fatal] Could not update status to failed")
        raise