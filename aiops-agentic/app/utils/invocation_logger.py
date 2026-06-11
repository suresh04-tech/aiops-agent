"""
app/utils/invocation_logger.py
───────────────────────────────
Centralized per-invocation file logger for AIOps.

Creates a separate log file for each invocation of:
  • RCA  (Root Cause Analysis)  → logs/rca/rca-<incident_id>-<timestamp>.log
  • SOP  (SOP Generation)       → logs/sop/sop-<sop_id>-<timestamp>.log
  • SLM  (Classifier / SLM)     → logs/slm/slm-<timestamp>.log

Retention:
  Each subdirectory keeps the latest MAX_LOG_FILES logs only.
  Older files are purged automatically after each new invocation.

Usage:
    from app.utils.invocation_logger import get_rca_logger, get_sop_logger, get_slm_logger

    logger = get_rca_logger(incident_id="abc-123")
    logger.info("Starting RCA...")
"""

import logging
import os
import re
import contextvars
from datetime import datetime, timezone
from pathlib import Path

current_invocation_id = contextvars.ContextVar("current_invocation_id", default=None)

class InvocationContextFilter(logging.Filter):
    def __init__(self, logger_name: str):
        super().__init__()
        self.logger_name = logger_name

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == self.logger_name:
            return True
        return current_invocation_id.get() == self.logger_name

class invocation_log_context:
    """
    Context manager to route root logger messages to the per-invocation
    file handler for the duration of a specific task.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.handler = logger.handlers[0] if logger.handlers else None
        self.token = None

    def __enter__(self):
        if self.handler:
            self.logger.removeHandler(self.handler)
            self.handler.addFilter(InvocationContextFilter(self.logger.name))
            logging.getLogger().addHandler(self.handler)
            self.token = current_invocation_id.set(self.logger.name)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handler:
            logging.getLogger().removeHandler(self.handler)
            for f in list(self.handler.filters):
                if isinstance(f, InvocationContextFilter):
                    self.handler.removeFilter(f)
            self.logger.addHandler(self.handler)
            if self.token:
                current_invocation_id.reset(self.token)

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_LOG_DIR   = Path("logs")
MAX_LOG_FILES  = 10                                  # keep last N per subdirectory
LOG_FORMAT     = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
DATE_FORMAT    = "%Y-%m-%d %H:%M:%S"

_SUBDIRS = {
    "rca": BASE_LOG_DIR / "rca",
    "sop": BASE_LOG_DIR / "sop",
    "slm": BASE_LOG_DIR / "slm",
}

# ── Internal helpers ───────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    """Create all log subdirectories if they don't exist."""
    for path in _SUBDIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    """UTC timestamp string safe for filenames: 20240612T143055Z"""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sanitize(value: str) -> str:
    """Strip characters that are unsafe in filenames."""
    return re.sub(r"[^\w\-]", "_", str(value))[:64]


def _purge_old_logs(directory: Path, prefix: str, keep: int = MAX_LOG_FILES) -> None:
    """
    Delete oldest log files in *directory* that start with *prefix*,
    keeping only the *keep* most recent ones (sorted by filename which
    embeds a timestamp, so lexicographic order == chronological order).
    """
    pattern = f"{prefix}*.log"
    files   = sorted(directory.glob(pattern))          # oldest first (lex sort)
    excess  = len(files) - keep
    for f in files[:excess]:
        try:
            f.unlink()
        except OSError:
            pass                                        # best-effort


def _build_file_handler(log_path: Path) -> logging.FileHandler:
    handler   = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    handler.setFormatter(formatter)
    return handler


def _make_logger(name: str, log_path: Path) -> logging.Logger:
    """
    Create (or return cached) a Logger that writes to *log_path*.

    Each invocation gets a unique logger name so Python's logging
    registry never reuses a stale handler from a prior invocation.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured — return as-is (idempotent within one process)
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = True                             # also flows to root (console / aiops.log)
    logger.addHandler(_build_file_handler(log_path))
    return logger


# ── Public API ─────────────────────────────────────────────────────────────────

def get_rca_logger(incident_id: str) -> logging.Logger:
    """
    Return a Logger that writes exclusively to:
        logs/rca/rca-<incident_id>-<timestamp>.log

    Purges oldest files so only MAX_LOG_FILES remain under logs/rca/.

    Example filenames:
        rca-abc123-20240612T143055Z.log
        rca-abc123-20240612T150102Z.log   ← new invocation for same incident
    """
    _ensure_dirs()
    safe_id   = _sanitize(incident_id)
    ts        = _timestamp()
    filename  = f"rca-{safe_id}-{ts}.log"
    log_path  = _SUBDIRS["rca"] / filename

    _purge_old_logs(_SUBDIRS["rca"], prefix="rca-")

    logger_name = f"rca.{safe_id}.{ts}"
    logger      = _make_logger(logger_name, log_path)
    logger.info(f"[RCA-Logger] Invocation log: {log_path}")
    return logger


def get_sop_logger(sop_id: str) -> logging.Logger:
    """
    Return a Logger that writes exclusively to:
        logs/sop/sop-<sop_id>-<timestamp>.log

    Purges oldest files so only MAX_LOG_FILES remain under logs/sop/.

    Example filenames:
        sop-SOP_201-20240612T143055Z.log
        sop-SOP_201-20240612T155900Z.log
    """
    _ensure_dirs()
    safe_id  = _sanitize(sop_id)
    ts       = _timestamp()
    filename = f"sop-{safe_id}-{ts}.log"
    log_path = _SUBDIRS["sop"] / filename

    _purge_old_logs(_SUBDIRS["sop"], prefix="sop-")

    logger_name = f"sop.{safe_id}.{ts}"
    logger      = _make_logger(logger_name, log_path)
    logger.info(f"[SOP-Logger] Invocation log: {log_path}")
    return logger


def get_slm_logger(label: str = "predict") -> logging.Logger:
    """
    Return a Logger that writes exclusively to:
        logs/slm/slm-<label>-<timestamp>.log

    Purges oldest files so only MAX_LOG_FILES remain under logs/slm/.

    Example filenames:
        slm-predict-20240612T143055Z.log
        slm-train-20240612T160000Z.log
    """
    _ensure_dirs()
    safe_label = _sanitize(label)
    ts         = _timestamp()
    filename   = f"slm-{safe_label}-{ts}.log"
    log_path   = _SUBDIRS["slm"] / filename

    _purge_old_logs(_SUBDIRS["slm"], prefix="slm-")

    logger_name = f"slm.{safe_label}.{ts}"
    logger      = _make_logger(logger_name, log_path)
    logger.info(f"[SLM-Logger] Invocation log: {log_path}")
    return logger