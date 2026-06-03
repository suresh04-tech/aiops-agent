"""
app/api/routes/sop.py
──────────────────────
Public /sop endpoints.

POST /sop/enqueue
    Two payload shapes accepted:
      { "sop_id": "SOP-201", "incident_id": "abc-123" }   ← INCIDENT mode
      { "sop_id": "SOP-201", "prompt": "We run a ..." }   ← PROMPT mode

    Creates the SOP row in DB (status=pending) then enqueues generation job.
    Returns 202 immediately — generation happens in the background worker.

GET /sop/stats
    Queue depth + counters.

GET /sop/{id}
    Poll generation status and retrieve the completed runbook.
"""
import time
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, model_validator

from app.sop.queue_manager import sop_queue_manager
from app.utils.db import get_db

logger = logging.getLogger(__name__)
router = APIRouter()

start = time.time()

# ── Request schemas ────────────────────────────────────────────────────────────

class SopEnqueueRequest(BaseModel):
    sop_id:      str
    incident_id: Optional[str] = None
    prompt:      Optional[str] = None

    @model_validator(mode="after")
    def check_exclusive(self) -> "SopEnqueueRequest":
        has_incident = bool(self.incident_id and self.incident_id.strip())
        has_prompt   = bool(self.prompt and self.prompt.strip())

        if has_incident and has_prompt:
            raise ValueError("Provide either incident_id OR prompt — not both.")
        if not has_incident and not has_prompt:
            raise ValueError("Provide either incident_id or prompt.")
        return self


# ── DB helpers ─────────────────────────────────────────────────────────────────


def _get_sop_row(db_id: str) -> dict | None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM meyiconnect.insight_sops WHERE id = %s LIMIT 1",
                (db_id,),
            )
            row = cur.fetchone()
    return dict(row) if row else None


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/enqueue", status_code=202)
async def enqueue_sop(body: SopEnqueueRequest):
    """
    Enqueue a SOP generation job.

    Mode A — Incident-based:
      { "incident_id": "<uuid>" }
      The worker loads the incident + RCA result from DB and generates a
      detailed, evidence-backed runbook.

    Mode B — Prompt-based:
      { "prompt": "We run a Flask app on ECS ..." }
      The worker uses the free-form description to generate a general runbook.
    """
    incident_id = body.incident_id.strip() if body.incident_id else None
    user_prompt = body.prompt.strip()      if body.prompt      else None
    req_sop_id  = body.sop_id.strip()

    mode = "incident" if incident_id else "prompt"

    # Verify the row exists and fetch the human-readable sop_id
    try:
        row = _get_sop_row(req_sop_id)
        if not row:
            raise HTTPException(status_code=404, detail=f"SOP record {req_sop_id} not found.")
        
        db_id = str(row["id"])
        sop_id_str = row["sop_id"]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[/sop/enqueue] DB lookup failed: {exc}")
        raise HTTPException(status_code=500, detail="Database lookup failed.")

    logger.info(f"[/sop/enqueue] Enqueuing db_id={db_id} sop_id={sop_id_str} mode={mode}")

    # Enqueue for background generation
    payload: dict = {"id": db_id, "sop_id": sop_id_str}
    if incident_id:
        payload["incident_id"] = incident_id
    else:
        payload["prompt"] = user_prompt

    await sop_queue_manager.enqueue(payload)

    return {
        "accepted":       True,
        "id":             db_id,
        "sop_id":         sop_id_str,
        "mode":           mode,
        "queue_position": sop_queue_manager.size,
        "message":        f"SOP generation queued. Poll GET /sop/{db_id} for status.",
        "poll_url":       f"/sop/{db_id}",
    }


@router.get("/stats")
async def sop_stats():
    """Return SOP queue depth and counters."""
    return sop_queue_manager.stats()


@router.get("/{id}")
async def get_sop(id: str):
    """
    Poll SOP generation status.

    Response fields:
      status      : draft | generating | completed | failed
      description : Generated description
      steps       : Markdown steps
      title       : Generated title
      severity    : Critical | High | Medium | Low
      alert_type  : e.g. CPUUtilization
      service     : e.g. Checkout API
      created_by  : e.g. AI
    """
    row = _get_sop_row(id)
    if not row:
        raise HTTPException(status_code=404, detail=f"SOP with id '{id}' not found.")

    return {
        "id":             row.get("id"),
        "sop_id":         row.get("sop_id"),
        "status":         row.get("status"),
        "title":          row.get("title"),
        "severity":       row.get("severity"),
        "alert_type":     row.get("alert_type"),
        "service":        row.get("service"),
        "created_by":     row.get("created_by"),
        "description":    row.get("description"),
        "steps":          row.get("steps"),
        "created_at":     str(row.get("created_at", "")),
        "updated_at":     str(row.get("updated_at", "")),
    }
