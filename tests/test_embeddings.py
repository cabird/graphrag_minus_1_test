"""Tests for the embedding engine."""

from graphrag_minus_1.embeddings import EmbeddingEngine


class TestEmbeddingEngine:
    def setup_method(self):
        self.engine = EmbeddingEngine()

    def test_embed_returns_4d_vector(self):
        vec = self.engine.embed("hello world")
        assert len(vec) == 4

    def test_empty_text_returns_zeros(self):
        vec = self.engine.embed("")
        assert vec == [0.0, 0.0, 0.0, 0.0]

    def test_longer_text_has_larger_first_dimension(self):
        short = self.engine.embed("hi")
        long = self.engine.embed("this is a much longer piece of text")
        assert long[0] > short[0]

    def test_similarity_identical_texts(self):
        score = self.engine.similarity("hello world", "hello world")
        assert score > 0.99

    def test_similarity_different_texts(self):
        score = self.engine.similarity("a", "this is a very long different text indeed")
        assert score < 0.95  # should be notably different

    def test_similarity_symmetric(self):
        s1 = self.engine.similarity("cat", "dog")
        s2 = self.engine.similarity("dog", "cat")
        assert abs(s1 - s2) < 0.001

    def test_find_most_similar(self):
        candidates = ["python programming", "java code", "snake reptile"]
        ranked = self.engine.find_most_similar("python language", candidates)
        assert len(ranked) == 3
        # first result should have highest score
        assert ranked[0][1] >= ranked[1][1]

    def test_batch_embed(self):
        texts = ["hello", "world", "foo"]
        vectors = self.engine.batch_embed(texts)
        assert len(vectors) == 3
        assert all(len(v) == 4 for v in vectors)

    def test_cosine_similarity_zero_vector(self):
        score = EmbeddingEngine._cosine_similarity([0, 0], [1, 1])
        assert score == 0.0

    def test_euclidean_distance(self):
        dist = EmbeddingEngine._euclidean_distance([0, 0], [3, 4])
        assert abs(dist - 5.0) < 0.001
