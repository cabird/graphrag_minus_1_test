"""Tests for the entity resolver."""

from graphrag_minus_1.entity_resolver import EntityResolver


class TestEntityResolver:
    def setup_method(self):
        self.resolver = EntityResolver()

    def test_basic_resolution_lowercases(self):
        assert self.resolver.resolve("Python") == "python"
        assert self.resolver.resolve("PYTHON") == "python"

    def test_strips_whitespace(self):
        assert self.resolver.resolve("  python  ") == "python"

    def test_alias_resolution(self):
        self.resolver.add_alias("py", "python")
        assert self.resolver.resolve("py") == "python"

    def test_add_aliases_bulk(self):
        self.resolver.add_aliases(["JS", "javascript", "ecmascript"], "javascript")
        assert self.resolver.resolve("JS") == "javascript"
        assert self.resolver.resolve("ecmascript") == "javascript"

    def test_get_canonical(self):
        self.resolver.add_alias("ML", "machine learning")
        assert self.resolver.get_canonical("ML") == "machine learning"
        assert self.resolver.get_canonical("unknown") is None

    def test_merge_entities_keeps_shorter(self):
        result = self.resolver.merge_entities("AI", "artificial intelligence")
        assert result == "ai"
        assert self.resolver.resolve("artificial intelligence") == "ai"

    def test_resolution_count(self):
        self.resolver.resolve("a")
        self.resolver.resolve("b")
        self.resolver.resolve("c")
        assert self.resolver.stats["total_resolutions"] == 3

    def test_stats(self):
        self.resolver.add_alias("x", "y")
        stats = self.resolver.stats
        assert stats["aliases_registered"] == 1
