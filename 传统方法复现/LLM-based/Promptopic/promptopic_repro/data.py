from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
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


def document_text(row: Dict, mode: str) -> str:
    text = (row.get("text") or "").strip()
    title = (row.get("title") or "").strip()
    abstract = (row.get("abstract") or "").strip()
    if mode == "full_text" and text:
        return text
    return "\n\n".join(part for part in [title, abstract] if part).strip()


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()

