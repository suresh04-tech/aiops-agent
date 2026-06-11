"""
api/routes/classifier.py
────────────────────────
Endpoints consumed by the frontend team.

POST /classifier/predict
    Classify any raw alert text / error message.
    Input field is  `message`  (not down_message).

POST /classifier/feedback
    Store a human correction. Duplicate (message, category) pairs are
    silently ignored via ON CONFLICT DO NOTHING.

POST /classifier/retrain
    Trigger an immediate synchronous retrain + hot-swap.

GET  /classifier/status
    Model availability, active version, and training-set size.
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.classifier import predictor
from app.classifier.train import train_and_save
from app.utils.db import get_db

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    alert_id: str   # The ID used to query meyiconnect.insight_alerts


class PredictResponse(BaseModel):
    message:    str
    category:   str
    confidence: int    # percentage 0-100
    available:  bool


class FeedbackRequest(BaseModel):
    message:          str
    correct_category: str


class FeedbackResponse(BaseModel):
    stored:  bool
    message: str


# ── Routes ───────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictResponse)
async def classify_alert(body: PredictRequest):
    """
    Fetch an alert by ID, classify its description, and use additional_configuration
    as a fallback if the confidence is too low.
    """
    if not body.alert_id.strip():
        raise HTTPException(status_code=400, detail="alert_id must not be empty")

    description = ""
    additional_config = None

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT description, additional_configuration FROM meyiconnect.insight_alerts WHERE id = %s",
                    (body.alert_id.strip(),)
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail=f"Alert {body.alert_id} not found")
                
                description = row["description"] or ""
                additional_config = row["additional_configuration"]
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[classifier/predict] DB fetch error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch alert from DB")

    # Predict using the description
    result = predictor.predict(description)
    print("Result: ", result)

    # Fallback to additional_configuration if category is Unknown
    if result["category"] == "Unknown" and additional_config:
        try:
            config_str = (
                additional_config if isinstance(additional_config, str) 
                else json.dumps(additional_config)
            )
            fallback_result = predictor.predict(config_str)
            print("Fallback Result: ", fallback_result)

            if fallback_result["category"] != "Unknown":
                result["category"] = fallback_result["category"]
                result["confidence"] = fallback_result["confidence"]
        except Exception as exc:
            logger.warning("[classifier/predict] Fallback parsing error: %s", exc)

    logger.info(
        "[classifier/predict] id=%s conf=%d%% category=%s | desc=%r",
        body.alert_id, result["confidence"], result["category"], description[:120],
    )

    # Log prediction to DB (fire-and-forget; never fail the request on DB error)
    _log_prediction(description, result["category"], result["confidence"])

    return PredictResponse(message=description, **result)


@router.post("/feedback", response_model=FeedbackResponse)
async def store_feedback(body: FeedbackRequest):
    """
    Persist a human-corrected label.

    Duplicate (message, category) pairs are silently ignored so repeated
    feedback from the same alert cannot bias the training data.
    """
    if not body.message.strip() or not body.correct_category.strip():
        raise HTTPException(
            status_code=400,
            detail="message and correct_category must not be empty",
        )

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO meyiconnect.insight_alert_categories_training (down_message, category)
                    VALUES (%s, %s)
                    ON CONFLICT (down_message, category) DO NOTHING
                    """,
                    (body.message.strip(), body.correct_category.strip()),
                )
        logger.info(
            "[classifier/feedback] stored message=%r → category=%s",
            body.message[:80], body.correct_category,
        )
        return FeedbackResponse(
            stored=True,
            message="Feedback stored. It will be used in the next retraining cycle.",
        )
    except Exception as exc:
        logger.exception("[classifier/feedback] DB error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to store feedback")


@router.post("/retrain", status_code=202)
async def retrain():
    """
    Retrain from the latest training data and hot-swap the in-process model.

    A new versioned model_vN.pkl is written; the previous versions are
    kept (up to 3) for rollback.  No restart required.
    """
    try:
        model_path = train_and_save()
        predictor.load_model()
        logger.info("[classifier/retrain] hot-reloaded → %s", predictor.get_active_version())
        return {
            "retrained":      True,
            "active_version": predictor.get_active_version(),
            "message":        "Model retrained and reloaded successfully.",
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("[classifier/retrain] error: %s", exc)
        raise HTTPException(status_code=500, detail="Retraining failed")


@router.get("/status")
async def classifier_status():
    """Return model availability, active version, and training-set size."""
    loaded          = predictor.is_model_loaded()
    active_version  = predictor.get_active_version()
    training_count: Optional[int] = None

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS cnt FROM alert_categories_training")
                row = cur.fetchone()
                training_count = row["cnt"] if row else 0
    except Exception:
        pass

    return {
        "model_loaded":     loaded,
        "active_version":   active_version,
        "training_samples": training_count,
        "confidence_threshold_pct": predictor.CONFIDENCE_THRESHOLD_PCT,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _log_prediction(raw_message: str, category: str, confidence: int) -> None:
    """
    Persist every prediction to insight_classifier_predictions for later analysis.
    Failures are swallowed so they never affect the API response.
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO meyiconnect.insight_classifier_predictions
                        (down_message, predicted_category, confidence)
                    VALUES (%s, %s, %s)
                    """,
                    (raw_message[:1000], category, confidence),
                )
    except Exception as exc:
        logger.warning("[classifier] prediction log failed: %s", exc)