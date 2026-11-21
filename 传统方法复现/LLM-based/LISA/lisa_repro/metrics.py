from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def harmonic_purity(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return 0.0
    cluster_to_true = defaultdict(Counter)
    true_to_cluster = defaultdict(Counter)
    for t, p in zip(y_true, y_pred):
        cluster_to_true[p][t] += 1
        true_to_cluster[t][p] += 1

    def purity(mapping):
        return sum(max(counter.values()) for counter in mapping.values()) / len(y_true)

    p = purity(cluster_to_true)
    ip = purity(true_to_cluster)
    return 0.0 if p + ip == 0 else 2 * p * ip / (p + ip)


def clustering_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    return {
        "p1": float(harmonic_purity(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
    }


def topic_quality(texts_by_topic: Dict[int, List[str]], top_n: int = 10) -> Dict[str, float]:
    corpus = []
    labels = []
    for topic_id, texts in texts_by_topic.items():
        corpus.extend(texts)
        labels.extend([topic_id] * len(texts))
    if not corpus:
        return {"diversity": 0.0, "vocab_size": 0.0}
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform(corpus)
    return {"diversity": float(len(set(vec.get_feature_names_out())) / max(1, len(vec.get_feature_names_out()))), "vocab_size": float(X.shape[1])}

