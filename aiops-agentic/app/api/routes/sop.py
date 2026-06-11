"""
app/api/routes/sop.py
──────────────────────
Public /sop endpoints.

POST /sop/enqueue
    Two payload shapes accepted:
      { "id": "<uuid>", "alert_id": "abc-123" }      ← ALERT mode
      { "id": "<uuid>", "prompt": "We run a ..." }   ← PROMPT mode

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
    id:          str
    alert_id:    Optional[str] = None
    prompt:      Optional[str] = None

    @model_validator(mode="after")
    def check_exclusive(self) -> "SopEnqueueRequest":
        has_alert  = bool(self.alert_id and self.alert_id.strip())
        has_prompt = bool(self.prompt and self.prompt.strip())

        if has_alert and has_prompt:
            raise ValueError("Provide either alert_id OR prompt — not both.")
        if not has_alert and not has_prompt:
            raise ValueError("Provide either alert_id or prompt.")
        return self


# ── DB helpers ─────────────────────────────────────────────────────────────────


def _get_sop_row(db_id: str) -> dict | None:
    try:
        uuid.UUID(str(db_id))
    except ValueError:
        return None

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

    Mode A — Alert-based:
      { "alert_id": "<uuid>" }
      The worker loads the alert + historical RCA result from DB and generates a
      detailed, evidence-backed runbook.

    Mode B — Prompt-based:
      { "prompt": "We run a Flask app on ECS ..." }
      The worker uses the free-form description to generate a general runbook.
    """
    alert_id    = body.alert_id.strip() if body.alert_id else None
    user_prompt = body.prompt.strip()   if body.prompt   else None
    req_db_id   = body.id.strip()

    mode = "alert" if alert_id else "prompt"

    # Verify the row exists and fetch the human-readable sop_id
    try:
        row = _get_sop_row(req_db_id)
        if not row:
            raise HTTPException(status_code=404, detail=f"SOP record {req_db_id} not found.")
        
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
    if alert_id:
        payload["alert_id"] = alert_id
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
