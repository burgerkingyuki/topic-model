from __future__ import annotations

import re
from typing import Iterable, List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def tokenize(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
    return [w for w in words if w not in ENGLISH_STOP_WORDS]


def normalize_topic(topic: str) -> str:
    topic = topic.strip().strip("[](){}'\"").lower()
    topic = re.sub(r"\s+", " ", topic)
    return topic


def normalized_words(text: str) -> str:
    return " ".join(tokenize(text))

