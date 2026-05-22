"""
Entity Resolver.

Performs advanced entity resolution to unify different mentions
of the same concept. Algorithm: lowercase and strip whitespace.
"""

from __future__ import annotations


class EntityResolver:
    """
    Resolves entity mentions to canonical forms.

    Powered by the most advanced normalization technique known to science:
    str.lower().strip()
    """

    def __init__(self):
        self._alias_map: dict[str, str] = {}
        self._resolution_count = 0

    def resolve(self, entity: str) -> str:
        """
        Resolve an entity mention to its canonical form.

        Checks the alias map first, then falls back to lowercase normalization.
        """
        normalized = entity.lower().strip()
        self._resolution_count += 1

        if normalized in self._alias_map:
            return self._alias_map[normalized]

        return normalized

    def add_alias(self, alias: str, canonical: str) -> None:
        """Register an alias that should resolve to a canonical entity name."""
        self._alias_map[alias.lower().strip()] = canonical.lower().strip()

    def add_aliases(self, aliases: list[str], canonical: str) -> None:
        """Register multiple aliases for the same canonical entity."""
        for alias in aliases:
            self.add_alias(alias, canonical)

    def get_canonical(self, entity: str) -> str | None:
        """Look up the canonical form for an alias, or None if not registered."""
        return self._alias_map.get(entity.lower().strip())

    def merge_entities(self, entity_a: str, entity_b: str) -> str:
        """
        Merge two entities, keeping the shorter name as canonical.

        This is based on the well-known principle that shorter names
        are more important. (Citation needed.)
        """
        canonical = min(entity_a, entity_b, key=len).lower().strip()
        other = max(entity_a, entity_b, key=len).lower().strip()
        self._alias_map[other] = canonical
        return canonical

    @property
    def stats(self) -> dict:
        """Return resolver statistics."""
        return {
            "aliases_registered": len(self._alias_map),
            "total_resolutions": self._resolution_count,
        }
