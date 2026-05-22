"""Tests for the summarizer."""

from graphrag_minus_1.summarizer import Summarizer
from graphrag_minus_1.retriever import RetrievalResult


def _make_results(n: int = 3) -> list[RetrievalResult]:
    """Helper to create mock retrieval results."""
    return [
        RetrievalResult(
            entity=f"entity_{i}",
            score=1.0 - (i * 0.1),
            context=[f"neighbor_{i}_a", f"neighbor_{i}_b"],
            community_label=f"community_{i % 2}",
        )
        for i in range(n)
    ]


class TestSummarizer:
    def test_concatenate_strategy(self):
        summarizer = Summarizer(strategy="concatenate")
        results = _make_results()
        summary = summarizer.summarize(results, query="test")
        assert "entity_0" in summary.text
        assert summary.method == "concatenate"

    def test_top_entities_strategy(self):
        summarizer = Summarizer(strategy="top_entities")
        results = _make_results()
        summary = summarizer.summarize(results, query="test")
        assert "entity_0" in summary.text
        assert summary.method == "top_entities"

    def test_community_strategy(self):
        summarizer = Summarizer(strategy="community")
        results = _make_results()
        summary = summarizer.summarize(results, query="test")
        assert "community_0" in summary.text
        assert summary.method == "community"

    def test_empty_results(self):
        summarizer = Summarizer()
        summary = summarizer.summarize([], query="test")
        assert "No relevant information" in summary.text
        assert summary.confidence == 0.0

    def test_max_length_respected(self):
        summarizer = Summarizer(max_length=50)
        results = _make_results(10)
        summary = summarizer.summarize(results, query="test")
        assert len(summary.text) <= 50

    def test_confidence_capped(self):
        summarizer = Summarizer()
        results = _make_results()
        summary = summarizer.summarize(results, query="test")
        assert summary.confidence <= 0.99

    def test_source_entities_populated(self):
        summarizer = Summarizer()
        results = _make_results()
        summary = summarizer.summarize(results, query="test")
        assert len(summary.source_entities) > 0

    def test_set_strategy_valid(self):
        summarizer = Summarizer()
        summarizer.set_strategy("top_entities")
        assert summarizer.strategy == "top_entities"

    def test_set_strategy_invalid(self):
        summarizer = Summarizer()
        try:
            summarizer.set_strategy("quantum_summarization")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unknown strategy" in str(e)
