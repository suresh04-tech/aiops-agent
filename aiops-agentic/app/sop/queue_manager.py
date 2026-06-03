"""
app/sop/queue_manager.py
─────────────────────────
In-process asyncio queue for SOP generation jobs.
Mirrors app/queue/manager.py — same design, separate singleton.

Messages are plain dicts with shape:
  { "id": str, "sop_id": str, "incident_id": str }   ← INCIDENT mode
  { "id": str, "sop_id": str, "prompt": str }        ← PROMPT mode
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class SopQueueManager:
    def __init__(self, maxsize: int = 0):
        self._q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self._enqueued:  int = 0
        self._processed: int = 0
        self._failed:    int = 0

    # ── Producer ───────────────────────────────────────────────────────────

    async def enqueue(self, payload: dict[str, Any]) -> None:
        payload["_enqueued_at"] = datetime.now(timezone.utc).isoformat()
        await self._q.put(payload)
        self._enqueued += 1
        logger.info(
            f"[SOP-Queue] Enqueued sop_id={payload.get('sop_id')} | "
            f"queue_size={self._q.qsize()}"
        )

    # ── Consumer ───────────────────────────────────────────────────────────

    async def get(self) -> dict[str, Any]:
        return await self._q.get()

    def task_done(self) -> None:
        self._q.task_done()

    def mark_processed(self) -> None:
        self._processed += 1

    def mark_failed(self) -> None:
        self._failed += 1

    # ── Observability ──────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "pending":   self._q.qsize(),
            "enqueued":  self._enqueued,
            "processed": self._processed,
            "failed":    self._failed,
        }

    @property
    def size(self) -> int:
        return self._q.qsize()


# Singleton
sop_queue_manager = SopQueueManager()
