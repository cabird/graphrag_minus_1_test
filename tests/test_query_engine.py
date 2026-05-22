"""Tests for the query engine (integration tests)."""

from graphrag_minus_1.query_engine import QueryEngine


SAMPLE_DOCS = [
    "Python is a versatile programming language created by Guido van Rossum. "
    "It supports multiple paradigms including object-oriented and functional programming.",

    "Knowledge graphs represent information as nodes and edges. "
    "They are used in search engines, recommendation systems, and AI applications.",

    "Retrieval augmented generation combines information retrieval with language models. "
    "RAG systems first retrieve relevant documents, then generate answers from them.",

    "Graph neural networks process graph-structured data. "
    "They learn node representations by aggregating information from neighboring nodes.",

    "Natural language processing enables computers to understand human language. "
    "Modern NLP uses transformer architectures and large language models.",
]


class TestQueryEngine:
    def setup_method(self):
        self.engine = QueryEngine(top_k=3)
        self.engine.index_batch(SAMPLE_DOCS)

    def test_index_document(self):
        engine = QueryEngine()
        doc_id = engine.index_document("Test document.")
        assert doc_id == "doc_0"

    def test_index_batch(self):
        engine = QueryEngine()
        ids = engine.index_batch(["Doc one.", "Doc two."])
        assert len(ids) == 2

    def test_query_returns_result(self):
        result = self.engine.query("What is Python?")
        assert result.answer is not None
        assert len(result.answer) > 0

    def test_query_has_confidence(self):
        result = self.engine.query("knowledge graphs")
        assert 0.0 <= result.confidence <= 1.0

    def test_query_has_sources(self):
        result = self.engine.query("programming language")
        assert isinstance(result.sources, list)

    def test_query_timing(self):
        result = self.engine.query("retrieval augmented generation")
        assert result.query_time_ms >= 0

    def test_query_with_expansion(self):
        result = self.engine.query("Python programming", expand=True)
        assert result.metadata["expanded"] is True

    def test_explain(self):
        explanation = self.engine.explain("knowledge graphs")
        assert "Query:" in explanation
        assert "Matching entities:" in explanation

    def test_graph_stats(self):
        stats = self.engine.get_graph_stats()
        assert stats["graph"]["node_count"] > 0
        assert stats["indexer"]["documents_indexed"] == 5
        assert stats["communities"]["communities_found"] >= 1

    def test_entity_alias(self):
        self.engine.add_entity_alias("py", "python")
        # After adding alias, the resolver should map py -> python
        resolved = self.engine.entity_resolver.resolve("py")
        assert resolved == "python"

    def test_query_history_tracked(self):
        self.engine.query("test query 1")
        self.engine.query("test query 2")
        assert self.engine._query_history[-1]["question"] == "test query 2"
        assert len(self.engine._query_history) >= 2

    def test_empty_graph_query(self):
        engine = QueryEngine()
        result = engine.query("anything")
        assert "No relevant information" in result.answer
        assert result.confidence == 0.0
