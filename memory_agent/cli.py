from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .agent import MemoryAgent


def print_items(items) -> None:
    for item in items:
        print(f"[{item.id}] {item.title} ({item.content_type}) @ {item.timestamp}")
        print(f"tags: {', '.join(item.tags)}")
        if item.metadata:
            print(f"metadata: {json.dumps(item.metadata, ensure_ascii=False)}")
        print(f"source: {item.source}")
        print(f"content: {item.content}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Personal long-term memory agent")
    parser.add_argument(
        "--storage",
        type=Path,
        default=Path("data/memory_pool.json"),
        help="Path to the JSON storage file",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest new content")
    ingest.add_argument("title", help="Title for the content")
    ingest.add_argument("content", help="Content body text")
    ingest.add_argument("source", help="Source description (e.g., file, device)")
    ingest.add_argument("content_type", help="Type of content (photo, video, note, etc.)")
    ingest.add_argument("--tags", nargs="*", default=None, help="Optional tags")
    ingest.add_argument("--timestamp", help="Optional ISO timestamp")
    ingest.add_argument("--metadata", help="Optional metadata JSON string")

    search = subparsers.add_parser("search", help="Keyword-based retrieval")
    search.add_argument("query", help="Search keywords")
    search.add_argument("--limit", type=int, default=5)

    qa = subparsers.add_parser("qa", help="Answer questions from stored memories")
    qa.add_argument("question", help="Question to answer")

    timeline = subparsers.add_parser("timeline", help="Chronological report")
    timeline.add_argument("--start", help="Start ISO timestamp")
    timeline.add_argument("--end", help="End ISO timestamp")

    album = subparsers.add_parser("album", help="Generate a tag-focused album/summary")
    album.add_argument("tags", nargs="+", help="Tags to include")

    summarize = subparsers.add_parser("summarize", help="Summarize latest items")
    summarize.add_argument("--limit", type=int, default=5)

    return parser


def load_metadata(raw: str | None) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise SystemExit("Metadata must be valid JSON")


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    agent = MemoryAgent(storage_path=args.storage)

    if args.command == "ingest":
        metadata = load_metadata(args.metadata)
        item = agent.ingest(
            content=args.content,
            content_type=args.content_type,
            title=args.title,
            source=args.source,
            tags=args.tags,
            timestamp=args.timestamp,
            metadata=metadata,
        )
        print(f"Stored item {item.id} with tags: {', '.join(item.tags)}")
    elif args.command == "search":
        results = agent.retrieve_by_keywords(args.query, limit=args.limit)
        for result in results:
            item = result.item
            print(f"score={result.score} :: {item.title} [{item.id}] -> {item.content}")
    elif args.command == "qa":
        answer = agent.answer(args.question)
        print(answer)
    elif args.command == "timeline":
        items = agent.timeline(args.start, args.end)
        print_items(items)
    elif args.command == "album":
        summary = agent.album(args.tags)
        print(summary)
    elif args.command == "summarize":
        summary = agent.summarize(limit=args.limit)
        print(summary)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
