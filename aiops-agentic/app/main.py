import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import queue
from app.api.routes import sop as sop_route
from app.queue.manager import queue_manager
from app.processor.worker import start_worker
from app.sop.queue_manager import sop_queue_manager
from app.sop.worker import start_sop_worker
from logging.handlers import RotatingFileHandler

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

# ── SOP-specific logging ───────────────────────────────────────────────────────
sop_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
sop_file_handler = RotatingFileHandler(
    "logs/sop.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5
)
sop_file_handler.setFormatter(sop_formatter)
sop_stream_handler = logging.StreamHandler()
sop_stream_handler.setFormatter(sop_formatter)

for logger_name in ["app.sop", "app.api.routes.sop"]:
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

app.include_router(queue.router,      prefix="/queue", tags=["Queue"])
app.include_router(sop_route.router,  prefix="/sop",   tags=["SOP"])


@app.get("/health", tags=["Health"])
async def health():
    stats = queue_manager.stats()
    sop_stats = sop_queue_manager.stats()
    return {
        "status":     "ok",
        "queue":      stats,
        "sop_queue":  sop_stats,
    }
