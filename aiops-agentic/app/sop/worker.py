"""
app/sop/worker.py
──────────────────
Async background SOP worker — mirrors processor/worker.py.

Picks up payloads from sop_queue_manager and calls process_sop()
in the same thread-pool so blocking Bedrock / DB calls don't
block the asyncio event loop.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from app.sop.queue_manager import SopQueueManager
from app.sop.process_sop import process_sop

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sop-worker")


async def start_sop_worker(manager: SopQueueManager) -> None:
    """
    Infinite loop: wait for a SOP payload → run processor in thread → repeat.
    CancelledError is the normal shutdown path.
    """
    logger.info("[SOP-Worker] SOP generation worker started.")

    while True:
        payload = await manager.get()
        sop_id = payload.get("sop_id", "unknown")
        logger.info(f"[SOP-Worker] Picked up sop_id={sop_id}")

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(_executor, process_sop, payload)
            manager.mark_processed()
            logger.info(f"[SOP-Worker] Completed sop_id={sop_id}")
        except Exception:
            manager.mark_failed()
            logger.exception(f"[SOP-Worker] Failed sop_id={sop_id}")
        finally:
            manager.task_done()
