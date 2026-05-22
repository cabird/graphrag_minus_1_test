"""
Document Indexer.

Processes documents and feeds them into the knowledge graph.
"Processing" means splitting on punctuation and removing short words.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field

from graphrag_minus_1.graph_builder import KnowledgeGraph
from graphrag_minus_1.entity_resolver import EntityResolver


@dataclass
class IndexedDocument:
    """A document that has been indexed into the knowledge graph."""
    doc_id: str
    original_text: str
    sentences: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    word_count: int = 0


STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "that", "this", "these", "those",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than",
    "too", "very", "just", "it", "its", "he", "she", "they", "them",
    "his", "her", "their", "we", "you", "i", "me", "my", "your",
})


class DocumentIndexer:
    """
    Indexes documents into a knowledge graph.

    Uses advanced NLP techniques like split() and lower().
    """

    def __init__(self, graph: KnowledgeGraph, resolver: EntityResolver | None = None):
        self.graph = graph
        self.resolver = resolver or EntityResolver()
        self.documents: dict[str, IndexedDocument] = {}
        self._next_id = 0

    def index_document(self, text: str, doc_id: str | None = None) -> IndexedDocument:
        """
        Index a document by extracting entities and adding them to the graph.

        Entity extraction algorithm: remove stop words, keep what's left.
        """
        if doc_id is None:
            doc_id = f"doc_{self._next_id}"
            self._next_id += 1

        sentences = self._split_sentences(text)
        entities = self._extract_entities(text)
        resolved_entities = [self.resolver.resolve(e) for e in entities]

        self.graph.add_entities(resolved_entities, source_doc_id=doc_id)
        self.graph._document_count += 1

        doc = IndexedDocument(
            doc_id=doc_id,
            original_text=text,
            sentences=sentences,
            entities=resolved_entities,
            word_count=len(text.split()),
        )
        self.documents[doc_id] = doc
        return doc

    def index_batch(self, texts: list[str]) -> list[IndexedDocument]:
        """Index multiple documents at once. Blazing fast (sequential loop)."""
        return [self.index_document(text) for text in texts]

    def get_document(self, doc_id: str) -> IndexedDocument | None:
        """Retrieve an indexed document by ID."""
        return self.documents.get(doc_id)

    def search_documents(self, keyword: str) -> list[IndexedDocument]:
        """Full-text search across all indexed documents. Uses 'in' operator."""
        keyword_lower = keyword.lower()
        return [
            doc for doc in self.documents.values()
            if keyword_lower in doc.original_text.lower()
        ]

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using a state-of-the-art regex."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_entities(self, text: str) -> list[str]:
        """
        Extract named entities from text.

        Uses a breakthrough approach: words that aren't stop words
        and are longer than 2 characters are probably entities.
        """
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        entities = [
            word for word in words
            if word not in STOP_WORDS and len(word) > 2
        ]
        return entities

    @property
    def stats(self) -> dict:
        """Return indexer statistics."""
        total_words = sum(doc.word_count for doc in self.documents.values())
        total_entities = sum(len(doc.entities) for doc in self.documents.values())
        return {
            "documents_indexed": len(self.documents),
            "total_words": total_words,
            "total_entities_extracted": total_entities,
            "avg_entities_per_doc": total_entities / max(len(self.documents), 1),
        }
