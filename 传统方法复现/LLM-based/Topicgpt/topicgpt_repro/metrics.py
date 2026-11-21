from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List

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
        total = 0
        for key, counter in mapping.items():
            total += max(counter.values())
        return total / len(y_true)

    p = purity(cluster_to_true)
    ip = purity(true_to_cluster)
    if p + ip == 0:
        return 0.0
    return 2 * p * ip / (p + ip)


def evaluate_clustering(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    return {
        "harmonic_purity": harmonic_purity(y_true, y_pred),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
    }

