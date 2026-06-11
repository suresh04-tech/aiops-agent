"""
app/sop/process_sop.py
───────────────────────
Orchestrator for SOP generation — direct Markdown pipeline.

Pipeline:
  ALERT mode:
    load context → LLM → extract metadata → save DB

  PROMPT mode:
    entity extraction  (regex, zero tokens)
      → resolve entities against DB (alert / source / project)
    guardrail check (Layer 1→2→3, zero tokens)
      → SKIP if at least one entity resolved (prompt has validated DB context)
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
    generate_sop_from_alert,
    generate_sop_from_prompt,
    SOP_BEDROCK_MODEL,
)
from app.sop.guardrails import validate_sop_prompt
from app.sop.entity_resolver import extract_entities, resolve_entities, ResolvedEntities
from app.utils.invocation_logger import get_sop_logger, invocation_log_context

# Module-level fallback logger
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DB helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _update_sop_status(db_id: str, status: str, error_response: str | None = None) -> None:
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                if error_response:
                    cur.execute(
                        """
                        UPDATE meyiconnect.insight_sops
                        SET status = %s, error_response = %s, updated_at = NOW()
                        WHERE id = %s
                        """,
                        (status, error_response, db_id),
                    )
                else:
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
                        error_response = %s,
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
    # ── Entity resolution fields (PROMPT mode) ──────────────────────────
    alert_id:   str | None = None,
    alert_name: str | None = None,
    source_id:  str | None = None,
    source:     str | None = None,
    project_id: str | None = None,
    project:    str | None = None,
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
                        updated_at    = NOW(),
                        alert_id      = %s,
                        alert_name    = %s,
                        source_id     = %s,
                        source        = %s,
                        project_id    = %s,
                        project       = %s
                    WHERE id = %s
                    """,
                    (
                        content_md, overview, title,
                        alert_type, service, severity,
                        SOP_BEDROCK_MODEL,
                        alert_id, alert_name,
                        source_id, source,
                        project_id, project,
                        db_id,
                    ),
                )
                logger.info(f"[SOP-Save] Updated {cur.rowcount} row(s) for db_id={db_id}")
            conn.commit()
    except Exception as exc:
        logger.error(f"[SOP-Save] Failed for db_id={db_id}: {exc}")
        raise


# def _load_incident_with_rca(incident_id: str) -> tuple[dict | None, dict | None, dict]:
    # try:
    #     with get_db() as conn:
    #         with conn.cursor() as cur:
    #             cur.execute(
    #                 "SELECT * FROM meyiconnect.insight_incidents WHERE id = %s LIMIT 1",
    #                 (incident_id,),
    #             )
    #             row = cur.fetchone()
    #             cur.execute(
    #                 "SELECT * FROM meyiconnect.incident_evidence WHERE incident_id = %s LIMIT 1",
    #                 (incident_id,),
    #             )
    #             ev_row = cur.fetchone()

    #     if not row:
    #         logger.warning(f"[SOP-Load] Incident {incident_id} not found")
    #         return None, None, {}

    #     incident      = dict(row)
    #     evidence_data = dict(ev_row) if ev_row else {}

    #     raw = incident.get("analysis_result")
    #     if isinstance(raw, str):
    #         try:
    #             rca_result = json.loads(raw)
    #         except Exception:
    #             rca_result = {}
    #     elif isinstance(raw, dict):
    #         rca_result = raw
    #     else:
    #         rca_result = {}

    #     return incident, rca_result, evidence_data

    # except Exception as exc:
    #     logger.error(f"[SOP-Load] DB error for incident {incident_id}: {exc}")
    #     return None, None, {}


def _load_alert_with_historical_rca(alert_id: str) -> tuple[dict | None, dict[str, dict]]:
    """
    Returns:
        current_alert: dict from insight_alerts (or None)
        historical_context: dict keyed by incident_id containing analysis_result, 
                            evidence_text, investigation_findings, rca_signals
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                # 1. Load current alert
                cur.execute(
                    """
                    SELECT id, connector_name, alert_name, description, additional_configuration
                    FROM meyiconnect.insight_alerts
                    WHERE id = %s
                    """,
                    (alert_id,)
                )
                alert_row = cur.fetchone()
                if not alert_row:
                    return None, {}
                current_alert = dict(alert_row)
                
                alert_name = current_alert.get("alert_name")
                if not alert_name:
                    return current_alert, {}

                # 2. Find similar alerts and their linked incidents
                cur.execute(
                    """
                    SELECT incident_id
                    FROM meyiconnect.insight_alerts
                    WHERE LOWER(alert_name) = LOWER(%s)
                      AND incident_id IS NOT NULL
                    """,
                    (alert_name,)
                )
                incident_ids = [row["incident_id"] for row in cur.fetchall()]
                
                if not incident_ids:
                    return current_alert, {}

                # 3. Load historical incidents with RCA (limit 10)
                cur.execute(
                    """
                    SELECT id, analysis_result
                    FROM meyiconnect.insight_incidents
                    WHERE id = ANY(%s::uuid[])
                      AND analysis_result IS NOT NULL
                    ORDER BY updated_at DESC
                    LIMIT 10
                    """,
                    (incident_ids,)
                )
                incident_rows = cur.fetchall()
                if not incident_rows:
                    return current_alert, {}

                rca_incident_ids = [row["id"] for row in incident_rows]
                
                # 4. Build merged context struct
                historical_context = {}
                latest_rca = None
                for idx, row in enumerate(incident_rows):
                    raw = row["analysis_result"]
                    if isinstance(raw, str):
                        try:
                            rca_json = json.loads(raw)
                        except Exception:
                            rca_json = {}
                    elif isinstance(raw, dict):
                        rca_json = raw
                    else:
                        rca_json = {}
                    
                    if idx == 0:
                        latest_rca = rca_json

                    historical_context[row["id"]] = {
                        "analysis_result": rca_json,
                        "evidence_text": [],
                        "investigation_findings": [],
                        "rca_signals": []
                    }

                # Attach the latest RCA to current_alert so we can infer severity easily later
                current_alert["_latest_rca"] = latest_rca

                # 5. Load evidence only for the ones with RCA
                cur.execute(
                    """
                    SELECT incident_id, evidence_text, investigation_findings, rca_signals
                    FROM meyiconnect.incident_evidence
                    WHERE incident_id = ANY(%s::uuid[])
                    """,
                    (rca_incident_ids,)
                )
                for ev_row in cur.fetchall():
                    iid = ev_row["incident_id"]
                    if iid in historical_context:
                        # parse json arrays
                        def _parse(val):
                            if isinstance(val, str):
                                try: return json.loads(val)
                                except: return [val] if val else []
                            return val if isinstance(val, list) else []
                        
                        historical_context[iid]["evidence_text"] = _parse(ev_row["evidence_text"])
                        historical_context[iid]["investigation_findings"] = _parse(ev_row["investigation_findings"])
                        historical_context[iid]["rca_signals"] = _parse(ev_row["rca_signals"])

                return current_alert, historical_context

    except Exception as exc:
        logger.error(f"[SOP-Load] DB error for alert {alert_id}: {exc}")
        return None, {}


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
      { "id": db_id, "sop_id": "SOP-201", "alert_id": "<uuid>" }      ← ALERT
      { "id": db_id, "sop_id": "SOP-201", "prompt": "<text>" }        ← PROMPT
    """
    db_id       = payload.get("id")
    sop_id      = payload.get("sop_id", "unknown")
    alert_id    = payload.get("alert_id")
    user_prompt = payload.get("prompt")

    # ── Per-invocation file logger ─────────────────────────────────────────────
    # Creates: logs/sop/sop-<sop_id>-<timestamp>.log  (max 10 retained)
    base_log = get_sop_logger(sop_id)
    ctx = invocation_log_context(base_log)
    log = ctx.__enter__()

    log.info("=" * 60)
    log.info("SOP PROCESSOR STARTED")
    log.info("=" * 60)

    if not db_id:
        log.error("[SOP] db_id missing from payload — cannot update status")
        return

    if not alert_id and not user_prompt:
        err_msg = "payload must have alert_id OR prompt"
        log.error(f"[SOP] db_id={db_id}: {err_msg}")
        _update_sop_status(db_id, "failed", error_response=err_msg)
        return

    try:
        # ── MODE A: Alert-based (no guardrail — context from DB) ──────────
        if alert_id:
            log.info(f"[SOP] Mode=ALERT db_id={db_id} alert_id={alert_id}")
            _update_sop_status(db_id, "generating")

            current_alert, historical_context = _load_alert_with_historical_rca(alert_id)

            if current_alert is None:
                err_msg = f"Alert {alert_id} not found — aborting"
                log.error(f"[SOP] {err_msg}")
                _update_sop_status(db_id, "failed", error_response=err_msg)
                return

            latest_rca = current_alert.pop("_latest_rca", None)
            
            fallback_severity = (
                _infer_severity_from_rca(latest_rca)
                if latest_rca
                else "Medium"
            )

            alert_type = current_alert.get("connector_name", "")
            service    = current_alert.get("alert_name", "")

            content_md = generate_sop_from_alert(
                sop_id=sop_id,
                current_alert=current_alert,
                historical_context=historical_context,
            )

        # ── MODE B: Prompt-based ───────────────────────────────────────────
        else:
            log.info(f"[SOP] Mode=PROMPT db_id={db_id} prompt_len={len(user_prompt)}")

            # ── Step 1: Entity extraction (regex, zero tokens) ─────────────
            log.info("[SOP-Entity] Extracting entities from prompt…")
            raw_entities = extract_entities(user_prompt)
            log.info(f"[SOP-Entity] Extracted: {raw_entities}")

            # ── Step 2: Entity resolution (DB lookups) ─────────────────────
            resolved: ResolvedEntities = resolve_entities(raw_entities, prompt=user_prompt)
            log.info(f"[SOP-Entity] Resolved: {resolved.as_log_dict()}")

            # ── Step 3: Guardrail check ────────────────────────────────────
            # Bypass guardrail when at least one entity resolved against the DB
            # (the prompt has validated context — no need to enforce keyword rules).
            if resolved.any_resolved:
                log.info(
                    f"[SOP-Guardrail] SKIPPED db_id={db_id} — "
                    f"entity resolved (alert_id={resolved.alert_id} "
                    f"source_id={resolved.source_id} project_id={resolved.project_id})"
                )
            else:
                guard = validate_sop_prompt(user_prompt)
                guard.log(prompt_snippet=user_prompt)

                if not guard.passed:
                    # Block before any Bedrock call — no token cost, no status=generating
                    log.warning(
                        f"[SOP-Guardrail] BLOCKED db_id={db_id} "
                        f"layer={guard.detail.get('layer', '?')} "
                        f"rule={guard.detail.get('rule', guard.detail.get('has_infra', '?'))}"
                    )
                    _mark_invalid_prompt(db_id, guard.message)
                    return

                log.info(f"[SOP-Guardrail] PASSED db_id={db_id} score={guard.detail}")

            # ── Step 4: Mark generating and call LLM ──────────────────────
            _update_sop_status(db_id, "generating")

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

        log.info(
            f"[SOP-Meta] title='{title}' severity={severity} "
            f"alert_type='{alert_type}' service='{service}'"
        )

        # Build entity kwargs — only populated for PROMPT mode
        entity_kwargs: dict = {}
        if user_prompt:   # PROMPT mode — resolved is always assigned above
            entity_kwargs = {
                "alert_id":   resolved.alert_id,
                "alert_name": resolved.alert_name,
                "source_id":  resolved.source_id,
                "source":     resolved.source,
                "project_id": resolved.project_id,
                "project":    resolved.project,
            }
            log.info(f"[SOP-Entity] Storing entity fields: {entity_kwargs}")


        _save_sop_content(
            db_id=db_id,
            content_md=content_md,
            title=title,
            alert_type=alert_type,
            service=service,
            severity=severity,
            **entity_kwargs,
        )

        log.info("=" * 60)
        log.info(f"SOP COMPLETED: db_id={db_id} title='{title}' severity={severity}")
        log.info("=" * 60)

    except Exception as exc:
        log.exception(f"[SOP-Fatal] process_sop failed for db_id={db_id}")
        try:
            _update_sop_status(db_id, "failed", error_response=str(exc))
        except Exception:
            logger.exception("[SOP-Fatal] Could not update status to failed")
        raise
    finally:
        ctx.__exit__(None, None, None)