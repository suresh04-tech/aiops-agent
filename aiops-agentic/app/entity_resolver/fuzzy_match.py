"""
app/entity_resolver/fuzzy_match.py
───────────────────────────────────
Fuzzy string matching utility using rapidfuzz.

Uses token_set_ratio which handles:
  - Word order differences  ("Cloud Watch" vs "AWS CloudWatch")
  - Partial matches          ("cloudwatch" vs "AWS CloudWatch")
  - Typos                    ("Cloud Wtach" vs "AWS CloudWatch")
"""

from __future__ import annotations

import re
import logging
from typing import Dict, Optional, Tuple

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# Default threshold — scores below this are considered non-matches.
# 85 prevents false positives like "Prometheus" matching "PostgreSQL" (~40).
DEFAULT_THRESHOLD = 85


def normalize(text: str) -> str:
    """
    Normalize text for fuzzy comparison.

    Steps:
      1. Lowercase
      2. Replace hyphens and underscores with spaces
      3. Strip non-alphanumeric characters (keep spaces)
      4. Collapse multiple spaces into one
      5. Strip leading/trailing whitespace
    """
    text = text.lower()
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fuzzy_find(
    query: str,
    candidates: Dict[str, str],
    threshold: int = DEFAULT_THRESHOLD,
) -> Optional[Tuple[str, str, int]]:
    """
    Find the best fuzzy match for *query* among *candidates*.

    Args:
        query:      The search string (e.g. text from the user prompt).
        candidates: Mapping of entity_name → entity_id.
        threshold:  Minimum score (0–100) to accept a match.

    Returns:
        (matched_name, matched_id, score) if best score >= threshold,
        else None.
    """
    if not query or not candidates:
        return None

    normalized_query = normalize(query)
    if not normalized_query:
        return None

    best_name: Optional[str] = None
    best_id: Optional[str] = None
    best_score: int = 0

    for name, entity_id in candidates.items():
        normalized_name = normalize(name)
        if not normalized_name:
            continue

        score = int(fuzz.token_set_ratio(normalized_query, normalized_name))

        if score > best_score:
            best_score = score
            best_name = name
            best_id = entity_id

    if best_score >= threshold and best_name is not None:
        logger.info(
            f"[FuzzyMatch] '{query}' → '{best_name}' (score={best_score})"
        )
        return (best_name, best_id, best_score)

    logger.debug(
        f"[FuzzyMatch] '{query}' → no match above threshold "
        f"(best_score={best_score}, threshold={threshold})"
    )
    return None
