"""Tests for the knowledge graph builder."""

from graphrag_minus_1.graph_builder import KnowledgeGraph, GraphNode, GraphEdge


class TestGraphNode:
    def test_importance_score_equals_mentions(self):
        node = GraphNode(name="python", mentions=5)
        assert node.importance_score == 5.0

    def test_default_mentions_is_zero(self):
        node = GraphNode(name="test")
        assert node.mentions == 0
        assert node.importance_score == 0.0


class TestGraphEdge:
    def test_weak_edge(self):
        edge = GraphEdge(source="a", target="b", weight=1.0)
        assert edge.strength == "weak"

    def test_moderate_edge(self):
        edge = GraphEdge(source="a", target="b", weight=3.0)
        assert edge.strength == "moderate"

    def test_strong_edge(self):
        edge = GraphEdge(source="a", target="b", weight=6.0)
        assert edge.strength == "strong"


class TestKnowledgeGraph:
    def test_add_entities_creates_nodes(self):
        graph = KnowledgeGraph()
        graph.add_entities(["python", "programming", "language"])
        assert len(graph.nodes) == 3
        assert "python" in graph.nodes

    def test_add_entities_creates_edges_within_window(self):
        graph = KnowledgeGraph(window_size=2)
        graph.add_entities(["a", "b", "c", "d"])
        # a-b, a-c, b-c, b-d, c-d
        assert len(graph.edges) == 5

    def test_duplicate_entities_increment_mentions(self):
        graph = KnowledgeGraph()
        graph.add_entities(["python", "python", "python"])
        assert graph.nodes["python"].mentions == 3

    def test_get_neighbors(self):
        graph = KnowledgeGraph(window_size=1)
        graph.add_entities(["a", "b", "c"])
        neighbors = graph.get_neighbors("b")
        assert "a" in neighbors
        assert "c" in neighbors

    def test_get_subgraph(self):
        graph = KnowledgeGraph(window_size=1)
        graph.add_entities(["a", "b", "c", "d"])
        subgraph = graph.get_subgraph("b", depth=1)
        assert "b" in subgraph
        assert "a" in subgraph
        assert "c" in subgraph

    def test_get_top_nodes(self):
        graph = KnowledgeGraph()
        graph.add_entities(["rare"])
        graph.add_entities(["common", "common", "common"])
        top = graph.get_top_nodes(n=1)
        assert top[0].name == "common"

    def test_edge_weight_increases_on_cooccurrence(self):
        graph = KnowledgeGraph(window_size=3)
        graph.add_entities(["a", "b"])
        graph.add_entities(["a", "b"])
        assert graph.get_edge_weight("a", "b") == 2.0

    def test_stats(self):
        graph = KnowledgeGraph()
        graph.add_entities(["x", "y"])
        stats = graph.stats
        assert stats["node_count"] == 2
        assert stats["edge_count"] == 1
