"""
app/entity_resolver/alert_resolver.py
──────────────────────────────────────
Resolves alert entities from user prompts via fuzzy matching
against the insight_alerts table (last 30 days).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from app.entity_resolver.db_queries import fetch_alerts
from app.entity_resolver.fuzzy_match import fuzzy_find

logger = logging.getLogger(__name__)


def resolve(prompt: str) -> Optional[Dict[str, object]]:
    """
    Fuzzy-match the normalised prompt against alert names from the DB.

    Args:
        prompt: Normalised user prompt text.

    Returns:
        {"alert_id": str, "alert_name": str, "score": int} if matched,
        else None.
    """
    candidates = fetch_alerts()
    if not candidates:
        logger.info("[AlertResolver] No alert candidates from DB")
        return None

    match = fuzzy_find(prompt, candidates)
    if match:
        name, entity_id, score = match
        logger.info(
            f"[AlertResolver] Matched alert '{name}' "
            f"(id={entity_id}, score={score})"
        )
        return {"alert_id": entity_id, "alert_name": name, "score": score}

    logger.info("[AlertResolver] No alert match found")
    return None
