"""
app/entity_resolver/db_queries.py
──────────────────────────────────
Database fetch functions for entity resolution.

Each function queries the DB directly — no caching.
Alerts are filtered to the last 30 days to avoid matching stale entries.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from app.utils.db import get_db

logger = logging.getLogger(__name__)


def fetch_alerts() -> Dict[str, str]:
    """
    Fetch alert names and IDs from the last 30 days.

    Returns:
        dict mapping alert_name → id (as strings).
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, alert_name
                    FROM meyiconnect.insight_alerts
                    WHERE alert_name IS NOT NULL
                      AND created_at >= NOW() - INTERVAL '30 days'
                    """
                )
                rows = cur.fetchall()
        result = {row["alert_name"]: str(row["id"]) for row in rows}
        logger.info(f"[DBQuery] Fetched {len(result)} alerts (last 30 days)")
        return result
    except Exception as exc:
        logger.error(f"[DBQuery] Failed to fetch alerts: {exc}")
        return {}


def fetch_connectors() -> Dict[str, str]:
    """
    Fetch connector names and IDs.

    Returns:
        dict mapping connector name → id (as strings).
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name
                    FROM meyiconnect.insight_connectors
                    WHERE name IS NOT NULL
                    """
                )
                rows = cur.fetchall()
        result = {row["name"]: str(row["id"]) for row in rows}
        logger.info(f"[DBQuery] Fetched {len(result)} connectors")
        return result
    except Exception as exc:
        logger.error(f"[DBQuery] Failed to fetch connectors: {exc}")
        return {}


def fetch_projects() -> Dict[str, str]:
    """
    Fetch project names and IDs.

    Returns:
        dict mapping project name → id (as strings).
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name
                    FROM meyiconnect.insight_projects
                    WHERE name IS NOT NULL
                    """
                )
                rows = cur.fetchall()
        result = {row["name"]: str(row["id"]) for row in rows}
        logger.info(f"[DBQuery] Fetched {len(result)} projects")
        return result
    except Exception as exc:
        logger.error(f"[DBQuery] Failed to fetch projects: {exc}")
        return {}


def get_alert_by_name(alert_name: str) -> Optional[str]:
    """
    Lookup an alert ID by exact name (case-insensitive).
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id
                    FROM meyiconnect.insight_alerts
                    WHERE LOWER(alert_name) = LOWER(%s)
                    LIMIT 1
                    """,
                    (alert_name,)
                )
                row = cur.fetchone()
        if row:
            found_id = str(row["id"])
            logger.info(f"[DBQuery] alert '{alert_name}' → found id={found_id}")
            return found_id
        else:
            logger.info(f"[DBQuery] alert '{alert_name}' → not found")
            return None
    except Exception as exc:
        logger.error(f"[DBQuery] Failed to fetch alert by name '{alert_name}': {exc}")
        return None


def get_connector_by_name(connector_name: str) -> Optional[str]:
    """
    Lookup a connector ID by exact name (case-insensitive).
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id
                    FROM meyiconnect.insight_connectors
                    WHERE LOWER(name) = LOWER(%s)
                    LIMIT 1
                    """,
                    (connector_name,)
                )
                row = cur.fetchone()
        if row:
            found_id = str(row["id"])
            logger.info(f"[DBQuery] connector '{connector_name}' → found id={found_id}")
            return found_id
        else:
            logger.info(f"[DBQuery] connector '{connector_name}' → not found")
            return None
    except Exception as exc:
        logger.error(f"[DBQuery] Failed to fetch connector by name '{connector_name}': {exc}")
        return None


def get_project_by_name(project_name: str) -> Optional[str]:
    """
    Lookup a project ID by exact name (case-insensitive).
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id
                    FROM meyiconnect.insight_projects
                    WHERE LOWER(name) = LOWER(%s)
                    LIMIT 1
                    """,
                    (project_name,)
                )
                row = cur.fetchone()
        if row:
            found_id = str(row["id"])
            logger.info(f"[DBQuery] project '{project_name}' → found id={found_id}")
            return found_id
        else:
            logger.info(f"[DBQuery] project '{project_name}' → not found")
            return None
    except Exception as exc:
        logger.error(f"[DBQuery] Failed to fetch project by name '{project_name}': {exc}")
        return None
