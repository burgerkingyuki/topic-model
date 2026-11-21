from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .client import ChatApi
from .config import ApiConfig, PromptTopicConfig
from .data import document_text
from .preprocess import normalize_topic, normalized_words
from .progress import bar
from .prompts import word_filter_messages


def documents_by_final_topic(rows: List[dict], generated_by_doc: Dict[str, dict], mapping: Dict[str, str], mode: str) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    for row in rows:
        topics = generated_by_doc.get(row["id"], {}).get("topics", [])
        doc = document_text(row, mode)
        seen = set()
        for topic in topics:
            final_topic = mapping.get(normalize_topic(topic))
            if final_topic and final_topic not in seen:
                groups[final_topic].append(doc)
                seen.add(final_topic)
    return dict(groups)


def ctfidf_words(groups: Dict[str, List[str]], top_n: int) -> Dict[str, List[str]]:
    labels = list(groups)
    docs = [normalized_words(" ".join(groups[label])) for label in labels]
    if not docs:
        return {}
    vectorizer = TfidfVectorizer(max_features=20000)
    matrix = vectorizer.fit_transform(docs)
    vocab = np.asarray(vectorizer.get_feature_names_out())
    result = {}
    for i, label in enumerate(labels):
        row = matrix[i].toarray()[0]
        top = row.argsort()[::-1][:top_n]
        result[label] = [str(vocab[j]) for j in top if row[j] > 0]
    return result


def _parse_word_list(text: str) -> List[str]:
    out = []
    for part in text.replace("\n", ",").split(","):
        word = part.strip(" -'\"")
        if word:
            out.append(word)
    return out


def filter_representative_words(words_by_topic: Dict[str, List[str]], api_cfg: ApiConfig, cfg: PromptTopicConfig) -> Dict[str, List[str]]:
    if not cfg.use_llm_word_filter:
        return {topic: words[: cfg.representation_top_n] for topic, words in words_by_topic.items()}
    client = ChatApi(
        api_key=api_cfg.api_key,
        base_url=api_cfg.base_url,
        model=api_cfg.model_filter,
        temperature=api_cfg.temperature,
        max_tokens=api_cfg.max_tokens,
        timeout=api_cfg.timeout,
        retry_limit=api_cfg.retry_limit,
    )
    try:
        filtered = {}
        for topic, words in bar(list(words_by_topic.items()), total=len(words_by_topic), desc="PromptTopic word filtering"):
            result = client.ask(word_filter_messages(topic, words[: cfg.ctfidf_top_n], cfg.representation_top_n), model=api_cfg.model_filter)
            parsed = _parse_word_list(result.text)
            filtered[topic] = parsed[: cfg.representation_top_n] if parsed else words[: cfg.representation_top_n]
        return filtered
    finally:
        client.close()

