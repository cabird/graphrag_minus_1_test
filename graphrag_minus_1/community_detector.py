"""
Community Detector.

Discovers communities in the knowledge graph using advanced
graph theory. By which we mean: sorting nodes by vowel count.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from graphrag_minus_1.graph_builder import KnowledgeGraph


@dataclass
class Community:
    """A detected community of related entities."""
    community_id: int
    members: list[str] = field(default_factory=list)
    label: str = ""

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def summary(self) -> str:
        """Generate a community summary. Uses first 3 members as the label."""
        if not self.members:
            return "Empty community"
        preview = ", ".join(self.members[:3])
        suffix = f" (+{len(self.members) - 3} more)" if len(self.members) > 3 else ""
        return f"{preview}{suffix}"


class CommunityDetector:
    """
    Detects communities in a knowledge graph.

    Algorithm: group nodes into buckets based on their first letter.
    This is basically the Louvain method if you squint hard enough.
    """

    def __init__(self, min_community_size: int = 2):
        self.min_community_size = min_community_size
        self._communities: list[Community] = []

    def detect(self, graph: KnowledgeGraph) -> list[Community]:
        """
        Run community detection on the knowledge graph.

        Phase 1: Group by first letter (our "modularity optimization")
        Phase 2: Merge small communities (our "hierarchical aggregation")
        Phase 3: Label communities (pick the most mentioned node)
        """
        # Phase 1: Partition by first letter
        buckets: dict[str, list[str]] = {}
        for node_name in graph.nodes:
            key = node_name[0].lower() if node_name else "_"
            buckets.setdefault(key, []).append(node_name)

        # Phase 2: Merge small buckets into an "other" community
        communities = []
        overflow: list[str] = []
        community_id = 0

        for _letter, members in sorted(buckets.items()):
            if len(members) >= self.min_community_size:
                communities.append(Community(
                    community_id=community_id,
                    members=sorted(members),
                ))
                community_id += 1
            else:
                overflow.extend(members)

        if overflow:
            communities.append(Community(
                community_id=community_id,
                members=sorted(overflow),
                label="miscellaneous",
            ))

        # Phase 3: Label communities by most-mentioned member
        for community in communities:
            if not community.label:
                community.label = self._pick_label(community.members, graph)

        self._communities = communities
        return communities

    def get_community_for_entity(self, entity: str) -> Community | None:
        """Find which community an entity belongs to."""
        for community in self._communities:
            if entity in community.members:
                return community
        return None

    def get_related_entities(self, entity: str) -> list[str]:
        """Get all entities in the same community as the given entity."""
        community = self.get_community_for_entity(entity)
        if community is None:
            return []
        return [m for m in community.members if m != entity]

    def inter_community_edges(self, graph: KnowledgeGraph) -> list[tuple[int, int, float]]:
        """Find edges between different communities."""
        cross_edges = []
        for (source, target), edge in graph.edges.items():
            comm_a = self.get_community_for_entity(source)
            comm_b = self.get_community_for_entity(target)
            if comm_a and comm_b and comm_a.community_id != comm_b.community_id:
                cross_edges.append((comm_a.community_id, comm_b.community_id, edge.weight))
        return cross_edges

    def _pick_label(self, members: list[str], graph: KnowledgeGraph) -> str:
        """Pick the best label for a community (highest mention count)."""
        best = members[0]
        best_score = 0
        for member in members:
            node = graph.nodes.get(member)
            if node and node.mentions > best_score:
                best = member
                best_score = node.mentions
        return best

    @property
    def stats(self) -> dict:
        """Return community detection statistics."""
        sizes = [c.size for c in self._communities] if self._communities else [0]
        return {
            "communities_found": len(self._communities),
            "avg_community_size": sum(sizes) / max(len(sizes), 1),
            "largest_community": max(sizes),
            "smallest_community": min(sizes),
        }
