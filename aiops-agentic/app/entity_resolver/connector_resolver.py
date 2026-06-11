"""
app/entity_resolver/connector_resolver.py
──────────────────────────────────────────
Resolves connector/source entities from user prompts via fuzzy matching
against the insight_connectors table.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from app.entity_resolver.db_queries import fetch_connectors
from app.entity_resolver.fuzzy_match import fuzzy_find

logger = logging.getLogger(__name__)


def resolve(prompt: str) -> Optional[Dict[str, object]]:
    """
    Fuzzy-match the normalised prompt against connector names from the DB.

    Args:
        prompt: Normalised user prompt text.

    Returns:
        {"connector_id": str, "connector_name": str, "score": int} if matched,
        else None.
    """
    candidates = fetch_connectors()
    if not candidates:
        logger.info("[ConnectorResolver] No connector candidates from DB")
        return None

    match = fuzzy_find(prompt, candidates)
    if match:
        name, entity_id, score = match
        logger.info(
            f"[ConnectorResolver] Matched connector '{name}' "
            f"(id={entity_id}, score={score})"
        )
        return {"connector_id": entity_id, "connector_name": name, "score": score}

    logger.info("[ConnectorResolver] No connector match found")
    return None
