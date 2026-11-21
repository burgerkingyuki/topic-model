from __future__ import annotations

from typing import Iterable, Optional


def progress(iterable: Iterable, total: Optional[int] = None, desc: str = ""):
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable

