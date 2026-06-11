"""
app/sop/entity_resolver.py
───────────────────────────
Pre-SOP entity extraction and database resolution layer.

Pipeline (PROMPT mode only):
  user_prompt
    → EntityResolverService.resolve()   — fuzzy matching (primary)
    → extract_entities() + DB lookups   — regex fallback (secondary)
    → ResolvedEntities                  — IDs + names for insight_sops storage

Supported entity types:
  alert     → meyiconnect.insight_alerts      (alert_name column)
  source    → meyiconnect.insight_connectors  (name column)
              "source" and "connector" are treated as synonyms
  project   → meyiconnect.insight_projects    (name column)

Resolution behaviour:
  • Found   → store id + name
  • Missing → store null id + provided name (generation continues — no error)

Public API:
  extract_entities(prompt: str)  -> dict[str, str | None]
  resolve_entities(entities: dict, prompt: str | None = None) -> ResolvedEntities
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.entity_resolver import db_queries
from app.entity_resolver.resolver import EntityResolverService, ResolvedEntities

logger = logging.getLogger(__name__)

# Shared service instance
_resolver_service = EntityResolverService()


# ═══════════════════════════════════════════════════════════════════════════════
# Regex extraction patterns (SECONDARY — kept as fallback)
# ═══════════════════════════════════════════════════════════════════════════════

# Each pattern captures everything after the keyword up to the next recognised
# keyword boundary or end of string.  We anchor on the keyword so order in the
# prompt doesn't matter.
#
# Keywords supported:
#   alert     → alert_name
#   source    → source  (synonym: connector)
#   project   → project

_STOP_KEYWORDS = r"(?:alert|source|connector|project|from|in|for|and)\b"

# Grab the value that follows a trigger keyword.
# Stops at the next stop-keyword or end of string.
_ALERT_RE   = re.compile(
    r"\balert\s+(?!name\b)([^\s](?:(?!(?:" + _STOP_KEYWORDS + r")).)*)".rstrip(),
    re.IGNORECASE,
)
_SOURCE_RE  = re.compile(
    r"\b(?:source|connector)\s+([^\s](?:(?!(?:" + _STOP_KEYWORDS + r")).)*)".rstrip(),
    re.IGNORECASE,
)
_PROJECT_RE = re.compile(
    r"\bproject\s+([^\s](?:(?!(?:" + _STOP_KEYWORDS + r")).)*)".rstrip(),
    re.IGNORECASE,
)


def extract_entities(prompt: str) -> dict[str, Optional[str]]:
    """
    Extract alert_name, source, and project from a free-form user prompt
    using regex patterns.

    This is the SECONDARY (fallback) extraction method.
    The primary method is fuzzy matching via EntityResolverService.

    Args:
        prompt: Raw user prompt string.

    Returns:
        dict with keys: alert_name, source, project.
        Any value may be None if the entity was not mentioned.

    Examples:
        "Create SOP for alert Aiops-test-1 from source AWS CloudWatch"
        → {"alert_name": "Aiops-test-1", "source": "AWS CloudWatch", "project": None}

        "Create SOP for project aiops-prod"
        → {"alert_name": None, "source": None, "project": "aiops-prod"}
    """
    alert_name = None
    source     = None
    project    = None

    m = _ALERT_RE.search(prompt)
    if m:
        alert_name = m.group(1).strip().rstrip(".,;")

    m = _SOURCE_RE.search(prompt)
    if m:
        source = m.group(1).strip().rstrip(".,;")

    m = _PROJECT_RE.search(prompt)
    if m:
        project = m.group(1).strip().rstrip(".,;")

    logger.info(
        f"[EntityExtract] alert_name={alert_name!r} "
        f"source={source!r} project={project!r}"
    )
    return {"alert_name": alert_name, "source": source, "project": project}


# ═══════════════════════════════════════════════════════════════════════════════
# DB lookup helpers (used by regex fallback path only)
# ═══════════════════════════════════════════════════════════════════════════════

def _lookup_alert(alert_name: str) -> Optional[str]:
    """
    Query insight_alerts for a matching alert_name (case-insensitive).

    Returns:
        The alert UUID (str) if found, else None.
    """
    return db_queries.get_alert_by_name(alert_name)


def _lookup_connector(source_name: str) -> Optional[str]:
    """
    Query insight_connectors for a matching name (case-insensitive).

    Returns:
        The connector UUID (str) if found, else None.
    """
    return db_queries.get_connector_by_name(source_name)


def _lookup_project(project_name: str) -> Optional[str]:
    """
    Query insight_projects for a matching name (case-insensitive).

    Returns:
        The project UUID (str) if found, else None.
    """
    return db_queries.get_project_by_name(project_name)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_entities(
    entities: dict[str, Optional[str]],
    prompt: str | None = None,
) -> ResolvedEntities:
    """
    Resolve entities from a user prompt.

    Strategy:
      1. PRIMARY — fuzzy matching via EntityResolverService (if prompt provided)
         Matches prompt text against all DB entities using rapidfuzz.
      2. FALLBACK — regex-extracted entities + exact DB lookups
         Used when fuzzy matching finds nothing, or prompt is not available.

    Args:
        entities: Output of extract_entities() — keys: alert_name, source, project.
        prompt:   Raw user prompt (needed for fuzzy matching). Optional for
                  backward compatibility.

    Returns:
        ResolvedEntities dataclass with all six fields populated.
    """
    # ── PRIMARY: Fuzzy matching ───────────────────────────────────────────
    if prompt:
        logger.info("[EntityResolve] Attempting fuzzy resolution (primary)…")
        fuzzy_result = _resolver_service.resolve(prompt)

        if fuzzy_result.has_any:
            logger.info(
                f"[EntityResolve] Fuzzy match succeeded: {fuzzy_result.as_log_dict()}"
            )
            return fuzzy_result

        logger.info(
            "[EntityResolve] Fuzzy match found nothing — "
            "falling back to regex + exact DB lookup"
        )

    # ── FALLBACK: Regex extraction + exact DB lookups ─────────────────────
    logger.info("[EntityResolve] Using regex fallback path")
    result = ResolvedEntities()

    # ── Alert ─────────────────────────────────────────────────────────────
    alert_name = entities.get("alert_name")
    if alert_name:
        result.alert_name = alert_name          # always store the provided name
        result.alert_id   = _lookup_alert(alert_name)

    # ── Source / Connector ────────────────────────────────────────────────
    source = entities.get("source")
    if source:
        result.source    = source               # always store the provided name
        result.source_id = _lookup_connector(source)

    # ── Project ───────────────────────────────────────────────────────────
    project = entities.get("project")
    if project:
        result.project    = project             # always store the provided name
        result.project_id = _lookup_project(project)

    # ── Summary log ───────────────────────────────────────────────────────
    logger.info(f"[EntityResolve] Final resolved entities: {result.as_log_dict()}")

    return result
