from __future__ import annotations

from typing import Iterable, Optional


def bar(items: Iterable, total: Optional[int] = None, desc: str = ""):
    try:
        from tqdm import tqdm

        return tqdm(items, total=total, desc=desc)
    except Exception:
        return items

