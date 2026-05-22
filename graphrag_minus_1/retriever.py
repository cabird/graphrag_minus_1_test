"""
Naive Retriever.

Retrieves relevant context from the knowledge graph for a given query.
Uses keyword matching, because who needs vector databases anyway?
"""

from __future__ import annotations
from dataclasses import dataclass

from graphrag_minus_1.graph_builder import KnowledgeGraph
from graphrag_minus_1.embeddings import EmbeddingEngine
from graphrag_minus_1.community_detector import CommunityDetector


@dataclass
class RetrievalResult:
    """A single retrieval result with score and context."""
    entity: str
    score: float
    context: list[str]
    community_label: str | None = None
    depth: int = 0


class NaiveRetriever:
    """
    Retrieves relevant subgraphs from the knowledge graph.

    Combines keyword matching with our world-class embedding similarity
    to find the most relevant nodes. Spoiler: it mostly just checks
    if the query words appear in node names.
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        embedding_engine: EmbeddingEngine | None = None,
        community_detector: CommunityDetector | None = None,
        top_k: int = 5,
    ):
        self.graph = graph
        self.embeddings = embedding_engine or EmbeddingEngine()
        self.community_detector = community_detector
        self.top_k = top_k

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """
        Retrieve the most relevant entities for a query.

        Scoring: keyword overlap (60%) + embedding similarity (30%) +
        node importance (10%). Very scientific weights.
        """
        query_terms = set(query.lower().split())
        results = []

        for name, node in self.graph.nodes.items():
            keyword_score = self._keyword_score(query_terms, name)
            embedding_score = self.embeddings.similarity(query, name)
            importance_score = min(node.importance_score / 10.0, 1.0)

            combined_score = (
                0.6 * keyword_score +
                0.3 * embedding_score +
                0.1 * importance_score
            )

            if combined_score > 0.05:
                neighbors = self.graph.get_neighbors(name)
                community_label = None
                if self.community_detector:
                    community = self.community_detector.get_community_for_entity(name)
                    community_label = community.label if community else None

                results.append(RetrievalResult(
                    entity=name,
                    score=combined_score,
                    context=neighbors,
                    community_label=community_label,
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:self.top_k]

    def retrieve_with_expansion(self, query: str, expansion_depth: int = 1) -> list[RetrievalResult]:
        """
        Retrieve with graph expansion — follow edges to find related context.

        This is our version of multi-hop reasoning. It hops once and gives up.
        """
        initial_results = self.retrieve(query)
        expanded_entities: set[str] = set()

        for result in initial_results:
            subgraph = self.graph.get_subgraph(result.entity, depth=expansion_depth)
            for entity in subgraph:
                expanded_entities.add(entity)

        # Score expanded entities
        query_terms = set(query.lower().split())
        expanded_results = []

        for entity in expanded_entities:
            if any(r.entity == entity for r in initial_results):
                continue

            node = self.graph.nodes.get(entity)
            if not node:
                continue

            score = self._keyword_score(query_terms, entity) * 0.5  # discount expanded
            if score > 0.01:
                expanded_results.append(RetrievalResult(
                    entity=entity,
                    score=score,
                    context=self.graph.get_neighbors(entity),
                    depth=1,
                ))

        all_results = initial_results + expanded_results
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:self.top_k]

    def _keyword_score(self, query_terms: set[str], entity: str) -> float:
        """Score based on keyword overlap. Revolutionary stuff."""
        entity_terms = set(entity.lower().split())
        if not query_terms:
            return 0.0
        overlap = query_terms & entity_terms
        return len(overlap) / len(query_terms)
