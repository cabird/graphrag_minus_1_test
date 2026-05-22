"""Tests for the document indexer."""

from graphrag_minus_1.graph_builder import KnowledgeGraph
from graphrag_minus_1.indexer import DocumentIndexer


class TestDocumentIndexer:
    def setup_method(self):
        self.graph = KnowledgeGraph()
        self.indexer = DocumentIndexer(self.graph)

    def test_index_document_returns_indexed_doc(self):
        doc = self.indexer.index_document("Python is a great programming language.")
        assert doc.doc_id == "doc_0"
        assert doc.original_text == "Python is a great programming language."
        assert doc.word_count == 6

    def test_index_document_extracts_entities(self):
        doc = self.indexer.index_document("Python is a great programming language.")
        assert "python" in doc.entities
        assert "programming" in doc.entities
        assert "language" in doc.entities
        # stop words excluded
        assert "is" not in doc.entities
        assert "a" not in doc.entities

    def test_index_document_populates_graph(self):
        self.indexer.index_document("Graphs contain nodes and edges.")
        assert len(self.graph.nodes) > 0

    def test_index_batch(self):
        docs = self.indexer.index_batch([
            "First document about cats.",
            "Second document about dogs.",
        ])
        assert len(docs) == 2
        assert docs[0].doc_id == "doc_0"
        assert docs[1].doc_id == "doc_1"

    def test_custom_doc_id(self):
        doc = self.indexer.index_document("Hello world.", doc_id="my-doc")
        assert doc.doc_id == "my-doc"

    def test_search_documents(self):
        self.indexer.index_document("Python is awesome.")
        self.indexer.index_document("Java is also cool.")
        results = self.indexer.search_documents("Python")
        assert len(results) == 1
        assert "Python" in results[0].original_text

    def test_search_case_insensitive(self):
        self.indexer.index_document("GraphRAG is interesting.")
        results = self.indexer.search_documents("graphrag")
        assert len(results) == 1

    def test_get_document(self):
        self.indexer.index_document("Test.", doc_id="test-1")
        assert self.indexer.get_document("test-1") is not None
        assert self.indexer.get_document("nonexistent") is None

    def test_sentences_split(self):
        doc = self.indexer.index_document("First sentence. Second sentence! Third?")
        assert len(doc.sentences) == 3

    def test_stats(self):
        self.indexer.index_document("Hello world.")
        self.indexer.index_document("Goodbye world.")
        stats = self.indexer.stats
        assert stats["documents_indexed"] == 2
        assert stats["total_words"] == 4
