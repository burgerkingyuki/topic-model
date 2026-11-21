from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import List, Sequence

from .config import LLMConfig, PipelineConfig
from .dataio import build_document_text, truncate_for_prompt
from .llm_client import OpenAICompatibleClient, parse_json_object
from .progress import progress
from .prompts import generation_prompt, summarize_prompt
from .schemas import ClusterTopic, GeneratedTopicWord


def _chunk(seq: Sequence[dict], size: int) -> List[List[dict]]:
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


def _generate_chunk(
    chunk: List[dict],
    llm_cfg: dict,
    generation_mode: str,
    max_document_chars: int,
) -> List[dict]:
    client = OpenAICompatibleClient(**llm_cfg)
    try:
        outputs = []
        for row in chunk:
            document = truncate_for_prompt(build_document_text(row, generation_mode), max_document_chars)
            result = client.chat_json(
                [
                    {"role": "system", "content": "Return JSON only."},
                    {"role": "user", "content": generation_prompt(document)},
                ],
                model=llm_cfg["model"],
            )
            payload = parse_json_object(result.content)
            outputs.append(
                {
                    "doc_id": row["id"],
                    "topic_word": str(payload.get("topic_word", "")).strip(),
                    "description": str(payload.get("description", "")).strip(),
                    "raw_response": result.raw,
                }
            )
        return outputs
    finally:
        client.close()


def generate_topic_words(rows: List[dict], llm_cfg: LLMConfig, pipe_cfg: PipelineConfig) -> List[GeneratedTopicWord]:
    if not rows:
        return []
    chunks = _chunk(rows, max(1, pipe_cfg.batch_size))
    worker_cfg = {
        "api_key": llm_cfg.api_key,
        "base_url": llm_cfg.base_url,
        "model": llm_cfg.model_generate,
        "temperature": llm_cfg.temperature,
        "max_tokens": llm_cfg.max_tokens,
        "timeout": llm_cfg.timeout,
        "retry_limit": llm_cfg.retry_limit,
    }
    outputs: List[GeneratedTopicWord] = []
    with ProcessPoolExecutor(max_workers=pipe_cfg.generation_workers) as ex:
        futures = [
            ex.submit(_generate_chunk, chunk, worker_cfg, pipe_cfg.generation_mode, pipe_cfg.max_document_chars)
            for chunk in chunks
        ]
        for fut in progress(futures, total=len(futures), desc="LiSA topic generation"):
            for item in fut.result():
                outputs.append(GeneratedTopicWord(**item))
    return outputs


def summarize_topic_cluster(
    cluster_id: int,
    topic_words: List[GeneratedTopicWord],
    llm_cfg: LLMConfig,
    use_llm: bool = True,
) -> ClusterTopic:
    if not topic_words:
        return ClusterTopic(cluster_id=cluster_id, topic_word=f"Topic {cluster_id}", description="")
    if not use_llm:
        first = topic_words[0]
        return ClusterTopic(
            cluster_id=cluster_id,
            topic_word=first.topic_word,
            description=first.description,
            source_topic_ids=[t.doc_id for t in topic_words],
            frequency=len(topic_words),
        )

    client = OpenAICompatibleClient(
        api_key=llm_cfg.api_key,
        base_url=llm_cfg.base_url,
        model=llm_cfg.model_summarize,
        temperature=llm_cfg.temperature,
        max_tokens=llm_cfg.max_tokens,
        timeout=llm_cfg.timeout,
        retry_limit=llm_cfg.retry_limit,
    )
    try:
        candidate_payload = [f"{t.topic_word}: {t.description}" for t in topic_words]
        result = client.chat(
            [
                {"role": "system", "content": "Follow the user instruction exactly."},
                {"role": "user", "content": summarize_prompt(candidate_payload)},
            ],
            model=llm_cfg.model_summarize,
        )
        topic_word = result.content.strip().strip('"').strip("'")
        return ClusterTopic(
            cluster_id=cluster_id,
            topic_word=topic_word or topic_words[0].topic_word,
            description="",
            source_topic_ids=[t.doc_id for t in topic_words],
            frequency=len(topic_words),
        )
    finally:
        client.close()
