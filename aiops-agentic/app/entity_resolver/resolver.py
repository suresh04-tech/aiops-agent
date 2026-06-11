"""
app/entity_resolver/resolver.py
────────────────────────────────
Main entity resolution service.

Pipeline:
  1. Normalise the user prompt (lowercase, strip punctuation, collapse spaces)
  2. Run fuzzy matching against DB entities (alert, connector, project)
  3. If fuzzy matching finds nothing, fall back to regex extraction + exact DB lookup

Each sub-resolver queries the DB directly — no in-memory cache.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Optional

from app.entity_resolver import alert_resolver
from app.entity_resolver import connector_resolver
from app.entity_resolver import project_resolver
from app.entity_resolver.fuzzy_match import normalize

logger = logging.getLogger(__name__)


@dataclass
class ResolvedEntities:
    """Resolved DB IDs and display names for the three entity types."""

    alert_id:   Optional[str] = None
    alert_name: Optional[str] = None

    source_id: Optional[str] = None
    source:    Optional[str] = None

    project_id: Optional[str] = None
    project:    Optional[str] = None

    @property
    def has_any(self) -> bool:
        """True if at least one entity was mentioned (regardless of DB match)."""
        return any([self.alert_name, self.source, self.project])

    @property
    def any_resolved(self) -> bool:
        """True if at least one entity was found in the database."""
        return any([self.alert_id, self.source_id, self.project_id])

    def as_log_dict(self) -> dict:
        return {
            "alert_id":   self.alert_id,
            "alert_name": self.alert_name,
            "source_id":  self.source_id,
            "source":     self.source,
            "project_id": self.project_id,
            "project":    self.project,
        }


class EntityResolverService:
    """
    Orchestrates fuzzy entity resolution across alerts, connectors, and projects.

    Usage:
        service = EntityResolverService()
        result = service.resolve("Create SOP for alert Aiops-test-1 from AWS CloudWatch")
    """

    @staticmethod
    def normalize_prompt(prompt: str) -> str:
        """
        Normalise the prompt for fuzzy matching.

        Input:  "Create SOP for alert Aiops-test-1. Issue detected in AWS CloudWatch."
        Output: "create sop for alert aiops test 1 issue detected in aws cloudwatch"
        """
        return normalize(prompt)

    def resolve(self, prompt: str) -> ResolvedEntities:
        """
        Resolve entities from a user prompt using fuzzy matching.

        Steps:
          1. Normalise the prompt
          2. Run each sub-resolver (alert, connector, project) against the DB
          3. Combine results into a ResolvedEntities dataclass

        Args:
            prompt: Raw user prompt string.

        Returns:
            ResolvedEntities with all matched fields populated.
        """
        result = ResolvedEntities()

        normalized = self.normalize_prompt(prompt)
        logger.info(f"[EntityResolver] Normalised prompt: {normalized!r}")

        # ── Alert ─────────────────────────────────────────────────────────
        alert_match = alert_resolver.resolve(normalized)
        if alert_match:
            result.alert_id   = alert_match["alert_id"]
            result.alert_name = alert_match["alert_name"]

        # ── Connector / Source ────────────────────────────────────────────
        connector_match = connector_resolver.resolve(normalized)
        if connector_match:
            result.source_id = connector_match["connector_id"]
            result.source    = connector_match["connector_name"]

        # ── Project ───────────────────────────────────────────────────────
        project_match = project_resolver.resolve(normalized)
        if project_match:
            result.project_id = project_match["project_id"]
            result.project    = project_match["project_name"]

        logger.info(f"[EntityResolver] Fuzzy resolved: {result.as_log_dict()}")
        return result
