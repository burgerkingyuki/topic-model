from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .client import ChatApi
from .config import ApiConfig, PromptTopicConfig
from .data import document_text
from .preprocess import normalize_topic, normalized_words
from .progress import bar
from .prompts import pbm_merge_messages


def topic_frequencies(generated_by_doc: Dict[str, dict]) -> Counter:
    counts = Counter()
    for item in generated_by_doc.values():
        for topic in item.get("topics", []):
            norm = normalize_topic(topic)
            if norm:
                counts[norm] += 1
    return counts


def initial_topic_mapping(counts: Counter) -> Dict[str, str]:
    return {topic: topic for topic in counts}


def collapse_pbm(
    counts: Counter,
    target_size: int,
    api_cfg: ApiConfig,
    cfg: PromptTopicConfig,
) -> Dict[str, str]:
    mapping = initial_topic_mapping(counts)
    client = ChatApi(
        api_key=api_cfg.api_key,
        base_url=api_cfg.base_url,
        model=api_cfg.model_collapse,
        temperature=api_cfg.temperature,
        max_tokens=api_cfg.max_tokens,
        timeout=api_cfg.timeout,
        retry_limit=api_cfg.retry_limit,
    )
    try:
        while len(set(mapping.values())) > target_size:
            current_topics = Counter(mapping.values())
            sorted_topics = [t for t, _ in counts.most_common() if t in current_topics]
            if len(sorted_topics) <= target_size:
                break
            topic_to_merge = sorted_topics[-1]
            candidate_window = [t for t in sorted_topics[: cfg.pbm_window_size] if t != topic_to_merge]
            if not candidate_window:
                break
            response = client.ask(pbm_merge_messages(topic_to_merge, candidate_window), model=api_cfg.model_collapse)
            chosen = normalize_topic(response.text)
            if chosen not in candidate_window:
                chosen = "miscellaneous"
            for original, assigned in list(mapping.items()):
                if assigned == topic_to_merge:
                    mapping[original] = chosen
        return mapping
    finally:
        client.close()


def _topic_documents(rows: List[dict], generated_by_doc: Dict[str, dict], mapping: Dict[str, str], mode: str) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    for row in rows:
        generated = generated_by_doc.get(row["id"], {})
        topics = generated.get("topics", [])
        doc = document_text(row, mode)
        for topic in topics:
            assigned = mapping.get(normalize_topic(topic))
            if assigned:
                groups[assigned].append(doc)
    return groups


def _top_words_by_topic(topic_docs: Dict[str, List[str]], top_n: int) -> Dict[str, List[str]]:
    labels = list(topic_docs)
    docs = [normalized_words(" ".join(topic_docs[label])) for label in labels]
    if not docs:
        return {}
    vectorizer = TfidfVectorizer(max_features=10000)
    matrix = vectorizer.fit_transform(docs)
    vocab = np.asarray(vectorizer.get_feature_names_out())
    out = {}
    for i, label in enumerate(labels):
        row = matrix[i].toarray()[0]
        top = row.argsort()[::-1][:top_n]
        out[label] = [str(vocab[j]) for j in top if row[j] > 0]
    return out


def _word_similarity(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(set(a) & set(b))
    return inter / max(1, min(len(set(a)), len(set(b))))


def collapse_wsm(
    rows: List[dict],
    generated_by_doc: Dict[str, dict],
    start_mapping: Dict[str, str],
    target_size: int,
    cfg: PromptTopicConfig,
) -> Dict[str, str]:
    mapping = dict(start_mapping)
    while len(set(mapping.values())) > target_size:
        topic_docs = _topic_documents(rows, generated_by_doc, mapping, cfg.generation_mode)
        top_words = _top_words_by_topic(topic_docs, cfg.ctfidf_top_n)
        labels = list(top_words)
        best_pair = None
        best_score = -1.0
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                score = _word_similarity(top_words[labels[i]], top_words[labels[j]])
                if score > best_score:
                    best_score = score
                    best_pair = (labels[i], labels[j])
        if best_pair is None:
            break
        keep, drop = best_pair
        if best_score <= 0 and len(set(mapping.values())) <= target_size:
            break
        for original, assigned in list(mapping.items()):
            if assigned == drop:
                mapping[original] = keep
    return mapping


def collapse_topics(rows: List[dict], generated_by_doc: Dict[str, dict], api_cfg: ApiConfig, cfg: PromptTopicConfig) -> Dict[str, str]:
    counts = topic_frequencies(generated_by_doc)
    strategy = cfg.collapse_strategy.lower()
    if strategy == "pbm":
        return collapse_pbm(counts, cfg.num_topics, api_cfg, cfg)
    if strategy == "wsm":
        return collapse_wsm(rows, generated_by_doc, initial_topic_mapping(counts), cfg.num_topics, cfg)
    if strategy == "hybrid":
        intermediate = max(cfg.num_topics, min(cfg.intermediate_topics, len(counts)))
        pbm_mapping = collapse_pbm(counts, intermediate, api_cfg, cfg)
        return collapse_wsm(rows, generated_by_doc, pbm_mapping, cfg.num_topics, cfg)
    raise ValueError(f"Unknown collapse_strategy: {cfg.collapse_strategy}")

