from __future__ import annotations

import sys
from itertools import combinations
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import LLMConfig, PipelineConfig
from .llm_client import OpenAICompatibleClient, parse_json_object
from .prompts import refinement_prompt
from .schemas import Topic


def _normalize_topic_text(topic: Topic) -> str:
    return f"{topic.label}. {topic.description}".strip()


def _build_candidate_pairs(
    topics: List[Topic],
    model: SentenceTransformer,
    threshold: float,
) -> List[dict]:
    if len(topics) < 2:
        return []
    texts = [_normalize_topic_text(t) for t in topics]
    emb = model.encode(texts, normalize_embeddings=True)
    pairs = []
    pair_id = 1
    for i, j in combinations(range(len(topics)), 2):
        sim = float(np.dot(emb[i], emb[j]))
        if sim >= threshold:
            pairs.append(
                {
                    "pair_id": pair_id,
                    "topic_a": {
                        "topic_id": topics[i].topic_id,
                        "label": topics[i].label,
                        "description": topics[i].description,
                    },
                    "topic_b": {
                        "topic_id": topics[j].topic_id,
                        "label": topics[j].label,
                        "description": topics[j].description,
                    },
                    "similarity": sim,
                }
            )
            pair_id += 1
    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return pairs[:50]


def _build_candidate_pairs_tfidf(topics: List[Topic], threshold: float) -> List[dict]:
    if len(topics) < 2:
        return []
    texts = [_normalize_topic_text(t) for t in topics]
    vec = TfidfVectorizer().fit_transform(texts)
    sims = cosine_similarity(vec)
    pairs = []
    pair_id = 1
    for i, j in combinations(range(len(topics)), 2):
        sim = float(sims[i, j])
        if sim >= threshold:
            pairs.append(
                {
                    "pair_id": pair_id,
                    "topic_a": {
                        "topic_id": topics[i].topic_id,
                        "label": topics[i].label,
                        "description": topics[i].description,
                    },
                    "topic_b": {
                        "topic_id": topics[j].topic_id,
                        "label": topics[j].label,
                        "description": topics[j].description,
                    },
                    "similarity": sim,
                }
            )
            pair_id += 1
    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return pairs[:50]


def refine_topics(
    topics: List[Topic],
    llm_cfg: LLMConfig,
    pipe_cfg: PipelineConfig,
) -> List[Topic]:
    if len(topics) <= 1:
        return topics

    use_embedder = True
    try:
        embedder = SentenceTransformer(pipe_cfg.embed_model_name)
    except Exception:
        embedder = None
        use_embedder = False

    initial_pairs = (
        _build_candidate_pairs(topics, embedder, pipe_cfg.merge_similarity_threshold)
        if use_embedder and embedder is not None
        else _build_candidate_pairs_tfidf(topics, pipe_cfg.merge_similarity_threshold)
    )
    if not initial_pairs:
        return topics

    client = OpenAICompatibleClient(
        api_key=llm_cfg.api_key,
        base_url=llm_cfg.base_url,
        model=llm_cfg.model_refine,
        temperature=llm_cfg.temperature,
        max_tokens=llm_cfg.max_tokens,
        timeout=llm_cfg.timeout,
        retry_limit=llm_cfg.retry_limit,
    )
    try:
        updated: Dict[str, Topic] = {t.topic_id: t for t in topics}
        changed = False
        rounds = max(1, pipe_cfg.max_topic_rounds)
        pbar = tqdm(
            total=rounds,
            desc="Refine rounds",
            file=sys.stderr,
            unit="round",
            ncols=80,
            dynamic_ncols=True,
        )
        try:
            for round_idx in range(rounds):
                if len(updated) <= 1:
                    pbar.set_postfix_str(f"topics={len(updated)} (converged: <=1)")
                    break
                pairs = (
                    _build_candidate_pairs(list(updated.values()), embedder, pipe_cfg.merge_similarity_threshold)
                    if use_embedder and embedder is not None
                    else _build_candidate_pairs_tfidf(list(updated.values()), pipe_cfg.merge_similarity_threshold)
                )
                if not pairs:
                    pbar.set_postfix_str(f"topics={len(updated)} (no candidate pairs)")
                    break
                merge_map: Dict[str, str] = {}
                pair_by_id = {int(pair["pair_id"]): pair for pair in pairs}
                for start in range(0, len(pairs), 5):
                    prompt = refinement_prompt(pairs[start : start + 5])
                    result = client.chat_json(
                        [{"role": "system", "content": "Return JSON only."}, {"role": "user", "content": prompt}],
                        model=llm_cfg.model_refine,
                    )
                    payload = parse_json_object(result.content)
                    decisions = payload.get("decisions") or []
                    for decision in decisions:
                        if not decision.get("merge"):
                            continue
                        pair_id = int(decision.get("pair_id", 0))
                        pair = pair_by_id.get(pair_id)
                        if pair is None:
                            continue
                        a = pair["topic_a"]["topic_id"]
                        b = pair["topic_b"]["topic_id"]
                        keep = a
                        drop = b
                        merge_map[drop] = keep
                        keep_topic = updated.get(keep)
                        drop_topic = updated.get(drop)
                        if keep_topic and drop_topic:
                            changed = True
                            keep_topic.frequency += drop_topic.frequency
                            for doc_id in drop_topic.source_doc_ids:
                                if doc_id not in keep_topic.source_doc_ids:
                                    keep_topic.source_doc_ids.append(doc_id)
                            keep_topic.label = str(decision.get("merged_label") or keep_topic.label).strip()
                            keep_topic.description = str(decision.get("merged_description") or keep_topic.description).strip()
                    if merge_map:
                        break
                if not merge_map:
                    pbar.set_postfix_str(f"topics={len(updated)} (no LLM merges)")
                    pbar.update(rounds - round_idx)
                    break
                updated = {k: v for k, v in updated.items() if k not in merge_map}
                pbar.update(1)
                pbar.set_postfix_str(f"topics={len(updated)}, pairs_candidates={len(pairs)}")
        finally:
            pbar.close()

        refined = list(updated.values())
        refined = [t for t in refined if t.frequency >= pipe_cfg.low_frequency_removal_threshold]
        if not refined and changed:
            refined = list(updated.values())
        if not refined:
            refined = topics

        for idx, topic in enumerate(refined, start=1):
            topic.topic_id = f"T{idx:03d}"
        return refined
    finally:
        client.close()
