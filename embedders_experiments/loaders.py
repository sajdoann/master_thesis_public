from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_docs(path: Path) -> List[Dict[str, object]]:
    """Load documents from a JSONL file, preserving id, source, and text."""
    docs: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append({
                "docID": obj.get("docID"),
                "source": obj.get("source"),
                "text": obj.get("text"),
            })
    return docs


def load_docs_stream(path: Path) -> Iterable[Dict[str, object]]:
    """Stream documents from a JSONL file one by one to reduce memory usage."""
    import json
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            yield {
                "docID": obj.get("docID"),
                "source": obj.get("source"),
                "text": obj.get("text"),
            }

def load_queries(path: Path) -> List[Tuple[str, object]]:
    """Load queries from a JSONL file as (query_text, docID)."""
    queries: List[Tuple[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            queries.append((obj["query"], obj["docID"]))
    return queries
