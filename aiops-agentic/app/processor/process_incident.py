"""
processor/process_incident.py
──────────────────────────────
Entry point called by the worker.

OLD flow (replaced):
  Load incident → collect all data → build giant prompt → one Bedrock call → store

NEW agentic flow:
  Load incident → init tools with DB creds → run LangGraph ReAct agent
  The agent decides what to look at, calls tools iteratively, and stores
  its own results in DB via the store_rca_result tool.

What stayed the same
─────────────────────
• DB schema and queries (meyiconnect.insight_incidents, incident_logs)
• dependency_resolver.py       — agent calls it via resolve_incident_targets tool
• log_processor.py             — agent calls it via get_compressed_logs tool
• correlation_engine.py        — agent calls it via correlate_instances tool
• Cloudtrail_processor.py      — agent calls it via get_infra_events tool
• aws_connector.py             — used to build AWSClientFactory
• queue / worker               — unchanged, still calls process_incident(payload)

What changed
─────────────
• No more giant prompt builder (_build_prompt_multi deleted)
• No more direct Bedrock invocation (_invoke_bedrock deleted)
• No more fixed pipeline — agent decides investigation order
• Status updates happen INSIDE the agent via update_investigation_status tool
"""

import json
import logging
from datetime import datetime, timezone

from app.utils.db import get_db
from app.utils.aws_connector import AWSClientFactory
from app.agent.tools import init_tools
from app.agent.graph import run_agent_investigation

logger = logging.getLogger(__name__)


# ── Status constants (kept for fallback use) ──────────────────────────────────

STATUS_PROGRESS = {
    "queued":           5,
    "resolving_deps":   10,
    "fetching_ec2":     15,
    "fetching_metrics": 25,
    "fetching_logs":    40,
    "correlating":      65,
    "building_rca":     80,
    "completed":        100,
    "failed":           0,
}


def _update_status(incident_id: str, status: str) -> None:
    """Fallback direct status update (also called on failure path)."""
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE meyiconnect.insight_incidents
                    SET analysis_status  = %s,
                        analysis_percent = %s,
                        updated_at       = NOW()
                    WHERE id = %s
                    """,
                    (status, STATUS_PROGRESS.get(status, 0), incident_id),
                )
    except Exception as exc:
        logger.error(f"[StatusUpdate] Failed: {exc}")


def _load_incident(incident_id: str) -> dict | None:
    """Load incident row from DB."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM meyiconnect.insight_incidents WHERE id = %s LIMIT 1",
                (incident_id,),
            )
            return cur.fetchone()


def _validate_incident(incident: dict, incident_id: str) -> str | None:
    """Return error message if incident is not processable, else None."""
    if not incident:
        return f"Incident {incident_id} not found in DB"

    if not incident.get("incident_down_time"):
        return f"Missing incident_down_time for {incident_id}"

    raw_deps = incident.get("dependencies") or []
    if isinstance(raw_deps, str):
        try:
            raw_deps = json.loads(raw_deps)
        except Exception:
            raw_deps = []
    if not raw_deps:
        return f"No dependencies configured for {incident_id}"

    return None


# ── Main entry point ──────────────────────────────────────────────────────────

def process_incident(payload: dict) -> None:
    """
    Process a single incident using the autonomous agentic investigation flow.

    Called by the worker thread pool with payload = {"incident_id": str}.
    All incident data is loaded from the DB by the agent via tools.

    Flow:
      1. Load and validate incident from DB
      2. Build AWSClientFactory with connector credentials
      3. Initialise tools (inject factory + incident row)
      4. Run LangGraph agent — it handles everything from here
      5. Agent stores results in DB via store_rca_result tool
    """
    logger.info("=" * 60)
    logger.info("INCIDENT PROCESSOR STARTED (Agentic Mode)")
    logger.info("=" * 60)

    incident_id = payload.get("incident_id", "unknown")

    try:
        # ── Step 1: Load incident ─────────────────────────────────────────
        incident = _load_incident(incident_id)
        error    = _validate_incident(incident, incident_id)
        if error:
            logger.error(f"[Validate] {error}")
            _update_status(incident_id, "failed")
            return

        incident = dict(incident)
        logger.info(
            f"[Load] incident_id={incident_id} | "
            f"issue={incident.get('issue', '')[:60]} | "
            f"severity={incident.get('severity')}"
        )

        # ── Step 2: Build AWS factory ─────────────────────────────────────
        connector_id = incident.get("connector_id")
        try:
            aws_factory = AWSClientFactory(connector_id)
        except ValueError as exc:
            logger.error(f"[AWS] Connector init failed: {exc}")
            _update_status(incident_id, "failed")
            return

        # ── Step 3: Initialise tools ──────────────────────────────────────
        # Tools are module-level singletons — inject per-incident state here.
        # The agent will call init_tools output via the tool layer.
        init_tools(aws_factory=aws_factory, incident_row=incident)

        logger.info(f"[Tools] Initialised for incident {incident_id}")
        _update_status(incident_id, "resolving_deps")

        # ── Step 4: Run the agent ─────────────────────────────────────────
        # The agent drives the entire investigation from here:
        # it loads context, resolves targets, fetches EC2/metrics/logs/CloudTrail,
        # correlates, builds RCA, and stores results — all autonomously.
        result = run_agent_investigation(
            incident_id=incident_id,
            aws_factory=aws_factory,
        )

        logger.info(
            f"[Agent] Finished | "
            f"tool_calls={result.get('tool_call_count')} | "
            f"messages={result.get('message_count')}"
        )
        logger.info("=" * 60)
        logger.info(f"INCIDENT COMPLETED: {incident_id}")
        logger.info("=" * 60)

    except Exception:
        logger.exception(f"[Fatal] process_incident failed for {incident_id}")
        try:
            _update_status(incident_id, "failed")
        except Exception:
            logger.exception("[Fatal] Could not update status to failed")
        raise
