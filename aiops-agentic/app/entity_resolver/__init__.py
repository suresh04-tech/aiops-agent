"""
app/entity_resolver
────────────────────
Fuzzy entity resolution service — matches user prompt text against
alerts, connectors, and projects stored in the database.
"""

from app.entity_resolver.resolver import EntityResolverService  # noqa: F401

__all__ = ["EntityResolverService"]
