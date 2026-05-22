"""Tests for the naive retriever."""

from graphrag_minus_1.graph_builder import KnowledgeGraph
from graphrag_minus_1.retriever import NaiveRetriever
from graphrag_minus_1.community_detector import CommunityDetector


class TestNaiveRetriever:
    def setup_method(self):
        self.graph = KnowledgeGraph()
        self.graph.add_entities(["python", "programming", "language", "snake", "reptile"])
        self.graph.add_entities(["python", "code", "development"])
        self.detector = CommunityDetector()
        self.detector.detect(self.graph)
        self.retriever = NaiveRetriever(
            graph=self.graph,
            community_detector=self.detector,
            top_k=3,
        )

    def test_retrieve_finds_matching_entity(self):
        results = self.retriever.retrieve("python")
        entity_names = [r.entity for r in results]
        assert "python" in entity_names

    def test_retrieve_returns_scores(self):
        results = self.retriever.retrieve("python")
        assert all(r.score > 0 for r in results)

    def test_retrieve_sorted_by_score(self):
        results = self.retriever.retrieve("python")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_respects_top_k(self):
        results = self.retriever.retrieve("python")
        assert len(results) <= 3

    def test_retrieve_includes_context(self):
        results = self.retriever.retrieve("python")
        python_result = next(r for r in results if r.entity == "python")
        assert len(python_result.context) > 0

    def test_retrieve_with_expansion(self):
        results = self.retriever.retrieve_with_expansion("python", expansion_depth=1)
        assert len(results) > 0

    def test_retrieve_no_match(self):
        results = self.retriever.retrieve("xylophone")
        # may still return results via embedding similarity
        if results:
            assert results[0].score < 0.5
