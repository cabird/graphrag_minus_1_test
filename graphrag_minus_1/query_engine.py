"""
Query Engine.

The main orchestrator that ties everything together into a
cohesive, if somewhat dubious, RAG pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone

from graphrag_minus_1.graph_builder import KnowledgeGraph
from graphrag_minus_1.indexer import DocumentIndexer
from graphrag_minus_1.retriever import NaiveRetriever
from graphrag_minus_1.summarizer import Summarizer, Summary
from graphrag_minus_1.embeddings import EmbeddingEngine
from graphrag_minus_1.community_detector import CommunityDetector
from graphrag_minus_1.entity_resolver import EntityResolver


@dataclass
class QueryResult:
    """The result of a query against the knowledge graph."""
    answer: str
    sources: list[str]
    confidence: float
    communities_consulted: int
    query_time_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class QueryEngine:
    """
    The main GraphRAG-1 query engine.

    Orchestrates the full pipeline: index → build graph → detect communities
    → embed → retrieve → summarize → pretend we know things.
    """

    def __init__(
        self,
        window_size: int = 3,
        top_k: int = 5,
        summary_strategy: str = "concatenate",
        max_summary_length: int = 280,
    ):
        self.entity_resolver = EntityResolver()
        self.graph = KnowledgeGraph(window_size=window_size)
        self.indexer = DocumentIndexer(self.graph, self.entity_resolver)
        self.embeddings = EmbeddingEngine()
        self.community_detector = CommunityDetector()
        self.summarizer = Summarizer(
            max_length=max_summary_length,
            strategy=summary_strategy,
        )
        self.retriever = NaiveRetriever(
            graph=self.graph,
            embedding_engine=self.embeddings,
            community_detector=self.community_detector,
            top_k=top_k,
        )
        self._query_history: list[dict] = []

    def index_document(self, text: str, doc_id: str | None = None) -> str:
        """Index a document and rebuild communities."""
        doc = self.indexer.index_document(text, doc_id)
        self.community_detector.detect(self.graph)
        return doc.doc_id

    def index_batch(self, texts: list[str]) -> list[str]:
        """Index multiple documents and rebuild communities once."""
        docs = self.indexer.index_batch(texts)
        self.community_detector.detect(self.graph)
        return [doc.doc_id for doc in docs]

    def query(self, question: str, expand: bool = False) -> QueryResult:
        """
        Query the knowledge graph and return a summarized answer.

        The confidence score is totally made up but always looks reasonable.
        """
        start = datetime.now(timezone.utc)

        if expand:
            results = self.retriever.retrieve_with_expansion(question)
        else:
            results = self.retriever.retrieve(question)

        summary = self.summarizer.summarize(results, query=question)

        communities = set()
        for r in results:
            if r.community_label:
                communities.add(r.community_label)

        elapsed = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        query_result = QueryResult(
            answer=summary.text,
            sources=summary.source_entities,
            confidence=summary.confidence,
            communities_consulted=len(communities),
            query_time_ms=round(elapsed, 2),
            metadata={
                "strategy": summary.method,
                "results_found": len(results),
                "expanded": expand,
            },
        )

        self._query_history.append({
            "question": question,
            "answer": summary.text,
            "confidence": summary.confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return query_result

    def add_entity_alias(self, alias: str, canonical: str) -> None:
        """Register an entity alias for resolution."""
        self.entity_resolver.add_alias(alias, canonical)

    def get_graph_stats(self) -> dict:
        """Get comprehensive statistics about the current state."""
        return {
            "graph": self.graph.stats,
            "indexer": self.indexer.stats,
            "communities": self.community_detector.stats,
            "resolver": self.entity_resolver.stats,
            "queries_executed": len(self._query_history),
        }

    def explain(self, question: str) -> str:
        """
        Provide a human-readable explanation of how a query would be processed.

        Useful for debugging why the system thinks "Python" and "snake" are related.
        """
        results = self.retriever.retrieve(question)
        lines = [
            f"Query: '{question}'",
            f"Matching entities: {len(results)}",
            "",
        ]

        for i, result in enumerate(results, 1):
            lines.append(f"  {i}. {result.entity} (score: {result.score:.3f})")
            lines.append(f"     Neighbors: {', '.join(result.context[:3]) or 'none'}")
            if result.community_label:
                lines.append(f"     Community: {result.community_label}")
            lines.append("")

        community_stats = self.community_detector.stats
        lines.append(f"Communities searched: {community_stats['communities_found']}")
        lines.append(f"Graph size: {self.graph.stats['node_count']} nodes, "
                      f"{self.graph.stats['edge_count']} edges")

        return "\n".join(lines)
