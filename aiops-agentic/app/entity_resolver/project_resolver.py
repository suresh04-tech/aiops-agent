"""
app/entity_resolver/project_resolver.py
───────────────────────────────────────
Resolves project entities from user prompts via fuzzy matching
against the insight_projects table.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from app.entity_resolver.db_queries import fetch_projects
from app.entity_resolver.fuzzy_match import fuzzy_find

logger = logging.getLogger(__name__)


def resolve(prompt: str) -> Optional[Dict[str, object]]:
    """
    Fuzzy-match the normalised prompt against project names from the DB.

    Args:
        prompt: Normalised user prompt text.

    Returns:
        {"project_id": str, "project_name": str, "score": int} if matched,
        else None.
    """
    candidates = fetch_projects()
    if not candidates:
        logger.info("[ProjectResolver] No project candidates from DB")
        return None

    match = fuzzy_find(prompt, candidates)
    if match:
        name, entity_id, score = match
        logger.info(
            f"[ProjectResolver] Matched project '{name}' "
            f"(id={entity_id}, score={score})"
        )
        return {"project_id": entity_id, "project_name": name, "score": score}

    logger.info("[ProjectResolver] No project match found")
    return None
