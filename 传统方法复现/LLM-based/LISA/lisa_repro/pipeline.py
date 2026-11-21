from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .assignment import assign_documents
from .clustering import cluster_embeddings, hungarian_mapping
from .config import LLMConfig, PipelineConfig
from .dataio import build_document_text, load_jsonl, write_jsonl
from .embeddings import Embedder
from .generation import generate_topic_words, summarize_topic_cluster
from .metrics import clustering_metrics
from .progress import progress
from .schemas import ClusterTopic
from .tpn import train_tpn


def _compact_labels(labels: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    unique = sorted(set(int(x) for x in labels.tolist()))
    mapping = {old: new for new, old in enumerate(unique)}
    return np.asarray([mapping[int(x)] for x in labels], dtype=int), mapping


def _build_neighbors(embeddings: np.ndarray, k: int) -> np.ndarray:
    if len(embeddings) < 2:
        return np.zeros((len(embeddings), 1), dtype=int)
    k = max(1, min(k, len(embeddings) - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    return indices[:, 1:]


def _summarize_topics(
    rows: List[dict],
    topic_words,
    topic_cluster_ids: np.ndarray,
    llm_cfg: LLMConfig,
    use_llm: bool,
) -> List[ClusterTopic]:
    topic_clusters: Dict[int, List] = {}
    for item, cluster_id in zip(topic_words, topic_cluster_ids):
        topic_clusters.setdefault(int(cluster_id), []).append(item)
    summaries = []
    cluster_ids = sorted(topic_clusters)
    for cluster_id in progress(cluster_ids, total=len(cluster_ids), desc="LiSA topic summarization"):
        summaries.append(
            summarize_topic_cluster(
                cluster_id,
                topic_clusters[cluster_id],
                llm_cfg,
                use_llm=use_llm,
            )
        )
    return summaries


def run_lisa(llm_cfg: LLMConfig, pipe_cfg: PipelineConfig) -> Path:
    random.seed(pipe_cfg.random_seed)
    np.random.seed(pipe_cfg.random_seed)

    rows = load_jsonl(pipe_cfg.input_path)
    if not rows:
        raise ValueError(f"No rows found in {pipe_cfg.input_path}")

    rng = random.Random(pipe_cfg.random_seed)
    if pipe_cfg.sample_size > 0 and pipe_cfg.sample_size < len(rows):
        sample_indices = sorted(rng.sample(range(len(rows)), pipe_cfg.sample_size))
        active_rows = [rows[i] for i in sample_indices]
    else:
        sample_indices = list(range(len(rows)))
        active_rows = list(rows)

    embedder = Embedder(pipe_cfg.embed_model_name)

    # Stage 1: topic word generation
    topic_words = generate_topic_words(active_rows, llm_cfg, pipe_cfg)
    topic_texts = [f"{item.topic_word}: {item.description}".strip() for item in topic_words]
    doc_texts = [build_document_text(row, pipe_cfg.generation_mode) for row in active_rows]

    doc_embeddings = embedder.encode(doc_texts, batch_size=pipe_cfg.batch_size)
    topic_embeddings = embedder.encode(topic_texts, batch_size=pipe_cfg.batch_size)

    # Stage 2: separate clustering spaces
    doc_cluster_ids = cluster_embeddings(doc_embeddings, pipe_cfg.num_topics, pipe_cfg.document_cluster_method, pipe_cfg.random_seed)
    topic_cluster_ids = cluster_embeddings(topic_embeddings, pipe_cfg.num_topics, pipe_cfg.topic_cluster_method, pipe_cfg.random_seed)
    doc_cluster_ids, _ = _compact_labels(doc_cluster_ids)
    topic_cluster_ids, _ = _compact_labels(topic_cluster_ids)

    # Stage 2: summarize clustered topic words into topic representations
    candidate_topics = _summarize_topics(rows, topic_words, topic_cluster_ids, llm_cfg, pipe_cfg.use_llm_summarize)
    candidate_topics = sorted(candidate_topics, key=lambda x: x.cluster_id)

    # Map document clusters to topic clusters with Hungarian matching
    paired_doc_clusters = np.asarray([doc_cluster_ids[i] for i in range(len(active_rows))], dtype=int)
    paired_topic_clusters = topic_cluster_ids
    if len(paired_doc_clusters) != len(paired_topic_clusters):
        paired_doc_clusters = paired_doc_clusters[: len(paired_topic_clusters)]
    cluster_map = hungarian_mapping(paired_doc_clusters, paired_topic_clusters)

    # Stage 3: collaborative enhancement
    tpn = train_tpn(
        doc_embeddings=doc_embeddings,
        topic_embeddings=topic_embeddings,
        num_topics=max(pipe_cfg.num_topics, len(candidate_topics)),
        hidden_dims=pipe_cfg.hidden_dims,
        epochs=pipe_cfg.train_epochs,
        batch_size=pipe_cfg.train_batch_size,
        lr=pipe_cfg.tpn_learning_rate,
        tau=pipe_cfg.tau,
        alpha=pipe_cfg.alpha,
        mu_neighbors=pipe_cfg.mu_neighbors,
        seed=pipe_cfg.random_seed,
    )

    doc_neighbors = _build_neighbors(doc_embeddings, pipe_cfg.lambda_neighbors)
    topic_neighbors = _build_neighbors(topic_embeddings, pipe_cfg.lambda_neighbors)

    enriched = assign_documents(
        rows=active_rows,
        candidate_topics=candidate_topics,
        llm_cfg=llm_cfg,
        pipe_cfg=pipe_cfg,
        doc_cluster_ids=doc_cluster_ids,
        topic_cluster_ids=topic_cluster_ids,
        mapped_doc_clusters=cluster_map,
        doc_probs=tpn.doc_probs,
        topic_probs=tpn.topic_probs,
        doc_neighbor_indices=doc_neighbors,
        topic_neighbor_indices=topic_neighbors,
        topic_cluster_lookup={t.cluster_id: t for t in candidate_topics},
    )

    write_jsonl(pipe_cfg.output_path, enriched)

    # Optional report for quick checks
    if enriched and "label" in enriched[0]:
        y_true = [str(row.get("label", "")) for row in enriched]
        y_pred = [str(row["lisa"]["final_topic_id"]) for row in enriched]
        report = clustering_metrics(y_true, y_pred)
        report_path = pipe_cfg.output_path.with_suffix(".report.json")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return pipe_cfg.output_path
