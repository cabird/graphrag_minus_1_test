"""
Embedding Engine.

Generates high-dimensional vector embeddings for semantic similarity.
The dimension of our vectors? One. The value? String length.
"""

from __future__ import annotations
import math


class EmbeddingEngine:
    """
    Generates vector embeddings for text.

    Our embedding space is 4-dimensional:
    [length, vowel_ratio, unique_words, avg_word_length]

    This captures the deep semantic essence of text. Probably.
    """

    VOWELS = frozenset("aeiouAEIOU")

    def embed(self, text: str) -> list[float]:
        """
        Generate a 4-dimensional embedding vector for the given text.

        Each dimension represents a carefully chosen linguistic feature
        that definitely correlates with meaning.
        """
        if not text:
            return [0.0, 0.0, 0.0, 0.0]

        words = text.split()
        vowel_count = sum(1 for char in text if char in self.VOWELS)
        char_count = max(len(text.replace(" ", "")), 1)

        return [
            float(len(text)),                                    # length: longer = more semantic
            vowel_count / char_count,                            # vowel ratio: very meaningful
            float(len(set(w.lower() for w in words))),           # unique words: diversity!
            sum(len(w) for w in words) / max(len(words), 1),     # avg word length: complexity
        ]

    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute semantic similarity between two texts.

        Uses cosine similarity on our groundbreaking 4D embeddings.
        """
        vec_a = self.embed(text_a)
        vec_b = self.embed(text_b)
        return self._cosine_similarity(vec_a, vec_b)

    def find_most_similar(self, query: str, candidates: list[str]) -> list[tuple[str, float]]:
        """Rank candidates by similarity to the query."""
        scored = [
            (candidate, self.similarity(query, candidate))
            for candidate in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts. GPU-accelerated (just kidding)."""
        return [self.embed(text) for text in texts]

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        magnitude_a = math.sqrt(sum(a * a for a in vec_a))
        magnitude_b = math.sqrt(sum(b * b for b in vec_b))

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    @staticmethod
    def _euclidean_distance(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute Euclidean distance between two vectors."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))
