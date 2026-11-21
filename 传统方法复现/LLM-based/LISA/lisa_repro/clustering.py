from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering

try:
    import hdbscan
except Exception:  # pragma: no cover
    hdbscan = None


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int, method: str = "kmeans", seed: int = 42) -> np.ndarray:
    if len(embeddings) == 0:
        return np.array([], dtype=int)
    n_clusters = max(1, min(n_clusters, len(embeddings)))
    method = method.lower()
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
        return model.fit_predict(embeddings)
    if method == "spectral":
        model = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors", random_state=seed)
        return model.fit_predict(embeddings)
    if method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        return model.fit_predict(embeddings)
    if method == "hdbscan":
        if hdbscan is None:
            raise RuntimeError("hdbscan is not installed")
        model = hdbscan.HDBSCAN(min_cluster_size=max(2, len(embeddings) // max(2, n_clusters)))
        labels = model.fit_predict(embeddings)
        if labels.min() < 0:
            labels = relabel_noise(labels)
        return labels
    raise ValueError(f"Unknown clustering method: {method}")


def relabel_noise(labels: np.ndarray) -> np.ndarray:
    labels = labels.copy()
    next_label = labels.max() + 1 if labels.size else 0
    for idx, value in enumerate(labels):
        if value < 0:
            labels[idx] = next_label
            next_label += 1
    return labels


def hungarian_mapping(doc_cluster_ids: np.ndarray, topic_cluster_ids: np.ndarray) -> dict[int, int]:
    doc_labels = sorted(set(int(x) for x in doc_cluster_ids.tolist()))
    topic_labels = sorted(set(int(x) for x in topic_cluster_ids.tolist()))
    if not doc_labels or not topic_labels:
        return {}
    matrix = np.zeros((len(doc_labels), len(topic_labels)), dtype=int)
    doc_index = {label: i for i, label in enumerate(doc_labels)}
    topic_index = {label: i for i, label in enumerate(topic_labels)}
    for d, t in zip(doc_cluster_ids, topic_cluster_ids):
        matrix[doc_index[int(d)], topic_index[int(t)]] += 1
    row_ind, col_ind = linear_sum_assignment(-matrix)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[doc_labels[r]] = topic_labels[c]
    return mapping

