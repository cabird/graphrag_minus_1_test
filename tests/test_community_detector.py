"""Tests for the community detector."""

from graphrag_minus_1.graph_builder import KnowledgeGraph
from graphrag_minus_1.community_detector import CommunityDetector


class TestCommunityDetector:
    def setup_method(self):
        self.graph = KnowledgeGraph()
        self.detector = CommunityDetector(min_community_size=2)

    def test_detect_groups_by_first_letter(self):
        self.graph.add_entities(["apple", "avocado", "banana", "blueberry"])
        communities = self.detector.detect(self.graph)
        assert len(communities) == 2  # a-group and b-group

    def test_small_groups_go_to_overflow(self):
        self.graph.add_entities(["apple", "avocado", "banana", "blueberry", "cherry"])
        communities = self.detector.detect(self.graph)
        # cherry is alone in 'c', goes to miscellaneous
        misc = [c for c in communities if c.label == "miscellaneous"]
        assert len(misc) == 1

    def test_get_community_for_entity(self):
        self.graph.add_entities(["apple", "avocado"])
        self.detector.detect(self.graph)
        community = self.detector.get_community_for_entity("apple")
        assert community is not None
        assert "avocado" in community.members

    def test_get_related_entities(self):
        self.graph.add_entities(["apple", "avocado", "apricot"])
        self.detector.detect(self.graph)
        related = self.detector.get_related_entities("apple")
        assert "avocado" in related
        assert "apple" not in related  # exclude self

    def test_unknown_entity_returns_none(self):
        self.detector.detect(self.graph)
        assert self.detector.get_community_for_entity("nonexistent") is None

    def test_community_summary(self):
        self.graph.add_entities(["alpha", "aqua", "art", "ace", "aim"])
        communities = self.detector.detect(self.graph)
        big = [c for c in communities if c.size >= 5][0]
        assert "+2 more" in big.summary

    def test_inter_community_edges(self):
        self.graph.add_entities(["apple", "avocado", "banana", "blueberry"])
        # Create a cross-community edge
        self.graph.add_entities(["apple", "banana"])
        self.detector.detect(self.graph)
        cross = self.detector.inter_community_edges(self.graph)
        assert len(cross) > 0

    def test_stats(self):
        self.graph.add_entities(["x1", "x2", "y1", "y2"])
        self.detector.detect(self.graph)
        stats = self.detector.stats
        assert stats["communities_found"] >= 1
