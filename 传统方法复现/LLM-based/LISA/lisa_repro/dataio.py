from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_document_text(row: Dict, mode: str = "full_text") -> str:
    title = (row.get("title") or "").strip()
    abstract = (row.get("abstract") or "").strip()
    full_text = (row.get("text") or "").strip()
    keywords = row.get("keywords") or []
    author_keywords = row.get("author_keywords") or []
    keywords_plus = row.get("keywords_plus") or []
    if mode == "full_text":
        if full_text:
            return full_text
        pieces = [p for p in [title, abstract] if p]
        return "\n\n".join(pieces).strip()
    if mode == "title_abstract_keywords":
        merged = []
        for item in list(keywords) + list(author_keywords) + list(keywords_plus):
            if item and item not in merged:
                merged.append(item)
        pieces = [p for p in [title, abstract, "; ".join(merged)] if p]
    else:
        pieces = [p for p in [title, abstract] if p]
    return "\n\n".join(pieces).strip()


def truncate_for_prompt(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()
