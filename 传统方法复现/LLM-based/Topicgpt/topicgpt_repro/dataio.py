from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_document_text(row: Dict, mode: str = "title_abstract") -> str:
    title = (row.get("title") or "").strip()
    abstract = (row.get("abstract") or "").strip()
    keywords = row.get("keywords") or []
    author_keywords = row.get("author_keywords") or []
    keywords_plus = row.get("keywords_plus") or []

    if mode == "title_abstract_keywords":
        kw = []
        for item in list(keywords) + list(author_keywords) + list(keywords_plus):
            if item and item not in kw:
                kw.append(item)
        kw_text = "; ".join(kw)
        pieces = [p for p in [title, abstract, kw_text] if p]
    else:
        pieces = [p for p in [title, abstract] if p]
    return "\n\n".join(pieces).strip()

