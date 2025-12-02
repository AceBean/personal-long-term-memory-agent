from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class MemoryItem:
    """Represents an individual piece of content stored by the agent."""

    id: int
    content: str
    content_type: str
    title: str
    source: str
    timestamp: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


class MemoryStore:
    """Simple JSON-backed store for memory items."""

    def __init__(self, path: Path):
        self.path = path
        self._items: List[MemoryItem] = []
        self._load()

    @property
    def items(self) -> List[MemoryItem]:
        return list(self._items)

    def _load(self) -> None:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._items = []
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._items = [MemoryItem(**item) for item in data]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = [asdict(item) for item in self._items]
        self.path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_item(self, item: MemoryItem) -> MemoryItem:
        self._items.append(item)
        self._save()
        return item

    def next_id(self) -> int:
        return (max((item.id for item in self._items), default=0) + 1)

    def filter_by_tags(self, tags: Iterable[str]) -> List[MemoryItem]:
        tag_set = {tag.lower() for tag in tags}
        return [item for item in self._items if tag_set.intersection({t.lower() for t in item.tags})]

    def filter_by_time_range(self, start: Optional[str], end: Optional[str]) -> List[MemoryItem]:
        def in_range(ts: str) -> bool:
            after_start = start is None or ts >= start
            before_end = end is None or ts <= end
            return after_start and before_end

        return [item for item in self._items if in_range(item.timestamp)]

    def search(self, keywords: Iterable[str]) -> List[MemoryItem]:
        keyword_set = {kw.lower() for kw in keywords}
        scored: List[tuple[int, MemoryItem]] = []
        for item in self._items:
            text = " ".join([item.title, item.content, " ".join(item.tags)]).lower()
            score = sum(text.count(kw) for kw in keyword_set)
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored]

    def by_id(self, item_id: int) -> Optional[MemoryItem]:
        return next((item for item in self._items if item.id == item_id), None)
