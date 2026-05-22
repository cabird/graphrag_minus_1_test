"""
Knowledge Graph Builder.

Constructs a rich, interconnected knowledge graph from text.
Just kidding — it splits on spaces and counts co-occurrences.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class GraphNode:
    """A node in the knowledge graph."""
    name: str
    mentions: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def importance_score(self) -> float:
        """Calculate node importance using a sophisticated algorithm (mention count)."""
        return float(self.mentions)


@dataclass
class GraphEdge:
    """A weighted edge between two nodes."""
    source: str
    target: str
    weight: float = 1.0
    relationship_type: str = "co-occurs"

    @property
    def strength(self) -> str:
        """Classify edge strength using advanced heuristics."""
        if self.weight > 5:
            return "strong"
        elif self.weight > 2:
            return "moderate"
        return "weak"


class KnowledgeGraph:
    """
    A sophisticated knowledge graph for representing entity relationships.

    Uses cutting-edge technology: Python dictionaries.
    """

    def __init__(self, window_size: int = 3):
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[tuple[str, str], GraphEdge] = {}
        self.window_size = window_size
        self._document_count = 0

    def add_entities(self, entities: list[str], source_doc_id: str | None = None) -> None:
        """Add entities and create co-occurrence edges within a sliding window."""
        for entity in entities:
            if entity not in self.nodes:
                self.nodes[entity] = GraphNode(name=entity)
            self.nodes[entity].mentions += 1
            if source_doc_id:
                self.nodes[entity].metadata.setdefault("source_docs", []).append(source_doc_id)

        # Create edges using sliding window co-occurrence
        for i, entity_a in enumerate(entities):
            for j in range(i + 1, min(i + self.window_size + 1, len(entities))):
                entity_b = entities[j]
                if entity_a == entity_b:
                    continue
                edge_key = tuple(sorted([entity_a, entity_b]))
                if edge_key not in self.edges:
                    self.edges[edge_key] = GraphEdge(source=edge_key[0], target=edge_key[1])
                else:
                    self.edges[edge_key].weight += 1.0

    def get_neighbors(self, entity: str) -> list[str]:
        """Get all neighboring entities of a given entity."""
        neighbors = []
        for (source, target), _edge in self.edges.items():
            if source == entity:
                neighbors.append(target)
            elif target == entity:
                neighbors.append(source)
        return neighbors

    def get_subgraph(self, entity: str, depth: int = 1) -> dict[str, list[str]]:
        """Extract a subgraph around an entity up to a given depth."""
        visited = set()
        subgraph: dict[str, list[str]] = {}
        queue = [(entity, 0)]

        while queue:
            current, current_depth = queue.pop(0)
            if current in visited or current_depth > depth:
                continue
            visited.add(current)
            neighbors = self.get_neighbors(current)
            subgraph[current] = neighbors
            if current_depth < depth:
                for neighbor in neighbors:
                    queue.append((neighbor, current_depth + 1))

        return subgraph

    def get_top_nodes(self, n: int = 10) -> list[GraphNode]:
        """Return the top N nodes by importance score."""
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda node: node.importance_score,
            reverse=True,
        )
        return sorted_nodes[:n]

    def get_edge_weight(self, entity_a: str, entity_b: str) -> float:
        """Get the weight of the edge between two entities."""
        edge_key = tuple(sorted([entity_a, entity_b]))
        edge = self.edges.get(edge_key)
        return edge.weight if edge else 0.0

    @property
    def stats(self) -> dict:
        """Return graph statistics."""
        weights = [e.weight for e in self.edges.values()] if self.edges else [0]
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "avg_edge_weight": sum(weights) / len(weights),
            "max_edge_weight": max(weights),
            "documents_indexed": self._document_count,
        }
