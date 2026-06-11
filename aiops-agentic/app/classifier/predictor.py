"""
classifier/predictor.py
───────────────────────
Loads the versioned sklearn pipeline once at startup (via the latest
symlink) and exposes predict().  Never touches train.py at runtime.

Confidence:
  - Returned as a percentage integer (0–100).
  - Results below CONFIDENCE_THRESHOLD_PCT are returned as "Unknown"
    to prevent bad low-confidence classifications reaching the frontend.
"""

import logging
from pathlib import Path

import joblib

from app.classifier.train import normalize   # reuse the same normalisation

logger = logging.getLogger(__name__)

_HERE        = Path(__file__).parent
_LATEST_LINK = _HERE / "models" / "model_latest.pkl"

CONFIDENCE_THRESHOLD_PCT = 70   # below this → "Unknown"

# Module-level singleton populated by load_model()
_pipeline        = None
_active_version  = None   # e.g. "model_v3.pkl"


def load_model() -> None:
    """
    Load the latest versioned pipeline into memory.
    Call ONCE from the FastAPI lifespan hook — NOT per request.
    """
    global _pipeline, _active_version

    if not _LATEST_LINK.exists():
        logger.warning(
            "No model found at %s — classifier unavailable. "
            "Run `python -m app.classifier.train` to generate one.",
            _LATEST_LINK,
        )
        return

    # Resolve symlink so we can log the actual version file
    resolved = _LATEST_LINK.resolve()
    _pipeline       = joblib.load(resolved)
    _active_version = resolved.name
    logger.info("Alert classifier loaded: %s", _active_version)


def is_model_loaded() -> bool:
    return _pipeline is not None


def get_active_version() -> str | None:
    return _active_version


def predict(raw_text: str) -> dict:
    """
    Predict the category for any free-form alert text / error message.

    The input is normalised with the same pipeline used during training,
    so raw messages like:
        "RequestError: getaddrinfo ENOTFOUND www.example.com"
        "Threshold Crossed: 1 datapoint [100.0] was greater than …"
    are handled correctly.

    Returns:
        {
            "category":   str,   # e.g. "Infra" or "Unknown"
            "confidence": int,   # 0-100 percentage
            "available":  bool,  # False when model not loaded
        }
    """
    if _pipeline is None:
        return {
            "category":   "Unknown",
            "confidence": 0,
            "available":  False,
        }

    cleaned = normalize(raw_text)
    print("Cleaned: ", cleaned)
    proba   = _pipeline.predict_proba([cleaned])[0]
    print("Proba: ", proba)
    classes = _pipeline.classes_
    print("Classes: ", classes)
    top_idx = int(proba.argmax())
    print("Top Index: ", top_idx)

    confidence_pct = round(float(proba[top_idx]) * 100)
    print("Confidence Percentage: ", confidence_pct)
    category = (
        classes[top_idx]
        if confidence_pct >= CONFIDENCE_THRESHOLD_PCT
        else "Unknown"
    )

    return {
        "category":   category,
        "confidence": confidence_pct,
        "available":  True,
    }