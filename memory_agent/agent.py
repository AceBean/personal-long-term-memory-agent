from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from .nlp import answer_question, extract_keywords, generate_summary, tokenize
from .storage import MemoryItem, MemoryStore


@dataclass
class RetrievalResult:
    item: MemoryItem
    score: int


class MemoryAgent:
    """Implements the personal long-term memory agent workflow."""

    def __init__(self, storage_path: Path | str = "data/memory_pool.json"):
        self.store = MemoryStore(Path(storage_path))

    def ingest(
        self,
        content: str,
        *,
        content_type: str,
        title: str,
        source: str,
        timestamp: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
    ) -> MemoryItem:
        """Add new content while auto-tagging and timestamping."""

        timestamp = timestamp or datetime.utcnow().isoformat()
        tags = tags or extract_keywords(content)
        metadata = metadata or {"keywords": ", ".join(extract_keywords(content, 10))}

        item = MemoryItem(
            id=self.store.next_id(),
            content=content,
            content_type=content_type,
            title=title,
            source=source,
            timestamp=timestamp,
            tags=tags,
            metadata=metadata,
        )
        return self.store.add_item(item)

    def retrieve_by_keywords(self, query: str, limit: int = 5) -> List[RetrievalResult]:
        keywords = tokenize(query)
        items = self.store.search(keywords)
        results: List[RetrievalResult] = []
        for item in items[:limit]:
            score = sum(item.content.lower().count(kw) for kw in keywords)
            results.append(RetrievalResult(item=item, score=score))
        return results

    def retrieve_by_tags(self, tags: Iterable[str]) -> List[MemoryItem]:
        return self.store.filter_by_tags(tags)

    def timeline(self, start: Optional[str] = None, end: Optional[str] = None) -> List[MemoryItem]:
        return sorted(self.store.filter_by_time_range(start, end), key=lambda i: i.timestamp)

    def answer(self, question: str, search_limit: int = 5) -> str:
        results = self.retrieve_by_keywords(question, limit=search_limit)
        context = [r.item.content for r in results]
        return answer_question(question, context)

    def album(self, tags: Iterable[str]) -> str:
        items = self.retrieve_by_tags(tags)
        contents = [f"{item.title} ({item.timestamp}) - {item.content}" for item in items]
        if not contents:
            return "No items found for those tags."
        return generate_summary(contents, max_sentences=5)

    def summarize(self, limit: int = 5) -> str:
        items = sorted(self.store.items, key=lambda i: i.timestamp, reverse=True)[:limit]
        contents = [item.content for item in items]
        if not contents:
            return "No content available to summarize yet."
        return generate_summary(contents, max_sentences=5)
