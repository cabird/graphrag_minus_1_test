"""
Summarizer.

Generates concise summaries from retrieved context.
Summarization strategy: take the first sentence and call it a day.
"""

from __future__ import annotations
from dataclasses import dataclass

from graphrag_minus_1.retriever import RetrievalResult


@dataclass
class Summary:
    """A generated summary with metadata."""
    text: str
    source_entities: list[str]
    confidence: float
    method: str = "naive"


class Summarizer:
    """
    Generates summaries from retrieved graph context.

    Supports multiple summarization strategies, all of which
    are variations of string concatenation.
    """

    def __init__(self, max_length: int = 280, strategy: str = "concatenate"):
        self.max_length = max_length
        self.strategy = strategy
        self._strategies = {
            "concatenate": self._concatenate_summary,
            "top_entities": self._top_entities_summary,
            "community": self._community_summary,
        }

    def summarize(self, results: list[RetrievalResult], query: str = "") -> Summary:
        """
        Generate a summary from retrieval results.

        Picks a strategy and hopes for the best.
        """
        if not results:
            return Summary(
                text="No relevant information found.",
                source_entities=[],
                confidence=0.0,
                method="empty",
            )

        strategy_fn = self._strategies.get(self.strategy, self._concatenate_summary)
        return strategy_fn(results, query)

    def _concatenate_summary(self, results: list[RetrievalResult], query: str) -> Summary:
        """
        Concatenation strategy: list top entities and their neighbors.

        This is basically how the first search engines worked, and
        look how well that turned out.
        """
        parts = []
        entities_used = []

        for result in results[:5]:
            neighbor_text = ", ".join(result.context[:3]) if result.context else "no connections"
            parts.append(f"{result.entity} (related to: {neighbor_text})")
            entities_used.append(result.entity)

        full_text = f"Regarding '{query}': " + "; ".join(parts)

        if len(full_text) > self.max_length:
            full_text = full_text[:self.max_length - 3] + "..."

        avg_score = sum(r.score for r in results[:5]) / min(len(results), 5)

        return Summary(
            text=full_text,
            source_entities=entities_used,
            confidence=min(avg_score * 1.5, 0.99),  # inflate slightly, cap at 0.99
            method="concatenate",
        )

    def _top_entities_summary(self, results: list[RetrievalResult], query: str) -> Summary:
        """Top entities strategy: just list the entity names. Minimalist."""
        entities = [r.entity for r in results[:5]]
        text = f"Key concepts for '{query}': {', '.join(entities)}"

        return Summary(
            text=text[:self.max_length],
            source_entities=entities,
            confidence=results[0].score if results else 0.0,
            method="top_entities",
        )

    def _community_summary(self, results: list[RetrievalResult], query: str) -> Summary:
        """Community strategy: group results by community label."""
        communities: dict[str, list[str]] = {}
        for result in results:
            label = result.community_label or "uncategorized"
            communities.setdefault(label, []).append(result.entity)

        parts = []
        for label, members in communities.items():
            parts.append(f"[{label}] {', '.join(members[:3])}")

        text = f"Communities for '{query}': " + " | ".join(parts)

        return Summary(
            text=text[:self.max_length],
            source_entities=[r.entity for r in results[:5]],
            confidence=sum(r.score for r in results) / max(len(results), 1),
            method="community",
        )

    def set_strategy(self, strategy: str) -> None:
        """Change the summarization strategy."""
        if strategy not in self._strategies:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {list(self._strategies.keys())}"
            )
        self.strategy = strategy
