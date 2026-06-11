import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import queue
from app.api.routes import sop as sop_route
from app.api.routes import classifier as classifier_route
from app.queue.manager import queue_manager
from app.processor.worker import start_worker
from app.sop.queue_manager import sop_queue_manager
from app.sop.worker import start_sop_worker
from app.classifier import predictor
from pathlib import Path
from logging.handlers import RotatingFileHandler

# ── Ensure all log directories exist before any handler is created ─────────────
for _log_subdir in ("logs", "logs/rca", "logs/sop", "logs/slm"):
    Path(_log_subdir).mkdir(parents=True, exist_ok=True)

# ── Root / shared log (application-wide, rolling) ─────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        RotatingFileHandler(
            "logs/aiops.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)

# ── SOP namespace logger (shared / rolling — separate from per-invocation logs) ─
# Per-invocation SOP logs go to logs/sop/sop-<id>-<ts>.log (via invocation_logger).
# This handler captures SOP-framework noise that isn't tied to one invocation.
sop_formatter    = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
sop_file_handler = RotatingFileHandler(
    "logs/sop.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5
)
sop_file_handler.setFormatter(sop_formatter)
sop_stream_handler = logging.StreamHandler()
sop_stream_handler.setFormatter(sop_formatter)

for logger_name in ["app.sop", "app.api.routes.sop", "app.entity_resolver"]:
    l = logging.getLogger(logger_name)
    l.setLevel(logging.INFO)
    l.propagate = False
    l.addHandler(sop_file_handler)
    l.addHandler(sop_stream_handler)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background workers on startup, cancel on shutdown."""
    logger.info("Starting incident processor worker...")
    worker_task = asyncio.create_task(start_worker(queue_manager))

    logger.info("Starting SOP generation worker...")
    sop_worker_task = asyncio.create_task(start_sop_worker(sop_queue_manager))

    # ── Load alert classifier model into memory (once) ────────────────────────
    logger.info("Loading alert classifier model...")
    predictor.load_model()
    if not predictor.is_model_loaded():
        logger.warning("Model not found on disk. Triggering auto-training...")
        try:
            from app.classifier.train import train_and_save
            train_and_save()
            predictor.load_model()
            logger.info("Auto-training complete.")
        except Exception as exc:
            logger.exception("Failed to auto-train model at startup: %s", exc)

    yield

    logger.info("Shutting down workers...")
    worker_task.cancel()
    sop_worker_task.cancel()
    for task in (worker_task, sop_worker_task):
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Incident Response API",
    version="2.0.0",
    description="Incident RCA processing service — Docker/FastAPI edition",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(queue.router,              prefix="/queue",      tags=["Queue"])
app.include_router(sop_route.router,          prefix="/sop",        tags=["SOP"])
app.include_router(classifier_route.router,   prefix="/classifier", tags=["Classifier"])


@app.get("/health", tags=["Health"])
async def health():
    stats = queue_manager.stats()
    sop_stats = sop_queue_manager.stats()
    return {
        "status":           "ok",
        "queue":            stats,
        "sop_queue":        sop_stats,
        "classifier_ready": predictor.is_model_loaded(),
    }