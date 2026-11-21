from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import numpy as np

from .config import LLMConfig, PipelineConfig
from .dataio import build_document_text, truncate_for_prompt
from .llm_client import OpenAICompatibleClient
from .progress import progress
from .prompts import assignment_prompt
from .schemas import ClusterTopic


def _assign_chunk(
    chunk: List[dict],
    llm_cfg: dict,
    generation_mode: str,
    max_document_chars: int,
) -> List[dict]:
    client = OpenAICompatibleClient(**llm_cfg)
    try:
        outputs = []
        for job in chunk:
            row = job["row"]
            candidate_topics = job["candidate_topics"]
            document = truncate_for_prompt(build_document_text(row, generation_mode), max_document_chars)
            result = client.chat(
                [
                    {"role": "system", "content": "Follow the user instruction exactly."},
                    {"role": "user", "content": assignment_prompt(document, candidate_topics)},
                ],
                model=llm_cfg["model"],
            )
            chosen_topic = result.content.strip().strip('"').strip("'")
            outputs.append(
                {
                    "doc_id": row["id"],
                    "topic_word": chosen_topic,
                    "description": "",
                    "reason": "",
                    "quote": "",
                    "raw_response": result.raw,
                }
            )
        return outputs
    finally:
        client.close()


def _chunk(seq, size):
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


def build_candidate_topics(topic_pool: List[ClusterTopic]) -> List[dict]:
    return [
        {"topic_id": t.cluster_id, "topic_word": t.topic_word, "description": t.description}
        for t in topic_pool
    ]


def _normalize_topic_word(text: str) -> str:
    return text.strip().strip(".。,:;，；").lower()


def assign_documents(
    rows: List[dict],
    candidate_topics: List[ClusterTopic],
    llm_cfg: LLMConfig,
    pipe_cfg: PipelineConfig,
    doc_cluster_ids: np.ndarray,
    topic_cluster_ids: np.ndarray,
    mapped_doc_clusters: Dict[int, int],
    doc_probs: np.ndarray,
    topic_probs: np.ndarray,
    doc_neighbor_indices: np.ndarray,
    topic_neighbor_indices: np.ndarray,
    topic_cluster_lookup: Dict[int, ClusterTopic],
) -> List[dict]:
    if not rows:
        return []

    topic_map = {t.cluster_id: t for t in candidate_topics}
    worker_cfg = {
        "api_key": llm_cfg.api_key,
        "base_url": llm_cfg.base_url,
        "model": llm_cfg.model_assign,
        "temperature": llm_cfg.temperature,
        "max_tokens": llm_cfg.max_tokens,
        "timeout": llm_cfg.timeout,
        "retry_limit": llm_cfg.retry_limit,
    }

    doc_by_id = {row["id"]: i for i, row in enumerate(rows)}
    results: List[dict] = []
    jobs: List[dict] = []
    for row in rows:
        i = doc_by_id[row["id"]]
        doc_cluster = int(doc_cluster_ids[i])
        topic_cluster = int(topic_cluster_ids[i])
        mapped_cluster = int(mapped_doc_clusters.get(doc_cluster, topic_cluster))
        neighbor_clusters = set()
        for idx in doc_neighbor_indices[i]:
            neighbor_clusters.add(int(mapped_doc_clusters.get(int(doc_cluster_ids[int(idx)]), int(topic_cluster_ids[int(idx)]))))
        for idx in topic_neighbor_indices[i]:
            neighbor_clusters.add(int(topic_cluster_ids[int(idx)]))
        candidate_list = [topic_map[c] for c in neighbor_clusters if c in topic_map]
        if not candidate_list:
            candidate_list = list(candidate_topics)
        jobs.append({"row": row, "candidate_topics": build_candidate_topics(candidate_list)})

    chunks = _chunk(jobs, max(1, pipe_cfg.batch_size))
    with ProcessPoolExecutor(max_workers=pipe_cfg.assignment_workers) as ex:
        futures = [
            ex.submit(_assign_chunk, chunk, worker_cfg, pipe_cfg.generation_mode, pipe_cfg.max_document_chars)
            for chunk in chunks
        ]
        llm_outputs: Dict[str, dict] = {}
        for fut in progress(futures, total=len(futures), desc="LiSA LLM assignment"):
            for item in fut.result():
                llm_outputs[item["doc_id"]] = item

    for row in progress(rows, total=len(rows), desc="LiSA output serialization"):
        i = doc_by_id[row["id"]]
        doc_cluster = int(doc_cluster_ids[i])
        topic_cluster = int(topic_cluster_ids[i])
        mapped_cluster = int(mapped_doc_clusters.get(doc_cluster, topic_cluster))
        top_doc_prob = doc_probs[i]
        top_topic_prob = topic_probs[i]
        blended = 0.5 * top_doc_prob + 0.5 * top_topic_prob
        blended_cluster = int(np.argmax(blended))
        confidence = float(blended.max())

        neighbor_clusters = set()
        for idx in doc_neighbor_indices[i]:
            neighbor_clusters.add(int(mapped_doc_clusters.get(int(doc_cluster_ids[int(idx)]), int(topic_cluster_ids[int(idx)]))))
        for idx in topic_neighbor_indices[i]:
            neighbor_clusters.add(int(topic_cluster_ids[int(idx)]))

        direct_topic = topic_map.get(mapped_cluster) or topic_map.get(blended_cluster) or candidate_topics[0]
        llm_item = llm_outputs.get(row["id"], {})
        if mapped_cluster == topic_cluster:
            mode = "direct"
            final_topic = direct_topic
            reason = "document-cluster and topic-cluster are consistent"
            quote = llm_item.get("quote", "")
        elif confidence >= pipe_cfg.low_confidence_threshold:
            mode = "enhanced"
            final_topic = topic_map.get(blended_cluster) or direct_topic
            reason = "TPN confidence is high"
            quote = llm_item.get("quote", "")
        elif pipe_cfg.use_llm_assignment:
            mode = "llm"
            llm_prompt_item = llm_item
            chosen_word = llm_prompt_item.get("topic_word", "") or direct_topic.topic_word
            normalized_choice = _normalize_topic_word(chosen_word)
            final_topic = next(
                (t for t in candidate_topics if _normalize_topic_word(t.topic_word) == normalized_choice),
                direct_topic,
            )
            reason = llm_prompt_item.get("reason", "")
            quote = llm_prompt_item.get("quote", "")
        else:
            mode = "fallback"
            final_topic = direct_topic
            reason = "fallback to mapped cluster"
            quote = llm_item.get("quote", "")

        results.append(
            {
                **row,
                "final_topic": {
                    "topic_id": final_topic.cluster_id,
                    "topic": final_topic.topic_word,
                    "description": final_topic.description,
                    "keywords": [],
                },
                "lisa": {
                    "candidate_topic_word": llm_item.get("topic_word", ""),
                    "candidate_description": llm_item.get("description", ""),
                    "document_cluster": doc_cluster,
                    "topic_cluster": topic_cluster,
                    "mapped_cluster": mapped_cluster,
                    "confidence": confidence,
                    "final_topic_id": final_topic.cluster_id,
                    "final_topic_word": final_topic.topic_word,
                    "final_topic_description": final_topic.description,
                    "assignment_mode": mode,
                    "reason": reason,
                    "quote": quote,
                    "raw_assignment": llm_item.get("raw_response", {}),
                },
            }
        )
    return results
