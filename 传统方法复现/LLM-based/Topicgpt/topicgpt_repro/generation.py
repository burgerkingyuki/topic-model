from __future__ import annotations

import sys
from typing import List

from tqdm import tqdm

from .config import LLMConfig, PipelineConfig
from .dataio import build_document_text
from .llm_client import OpenAICompatibleClient, parse_json_object
from .prompts import generation_prompt
from .schemas import GenerationResult, Topic


def generate_topics(
    sample_rows: List[dict],
    demo_topics: List[Topic],
    llm_cfg: LLMConfig,
    pipe_cfg: PipelineConfig,
) -> GenerationResult:
    if not sample_rows:
        return GenerationResult(topics=[], trace=[])

    # The paper starts from a small initial set S. These seed topics must be
    # visible in the authoritative pool used by every subsequent round.
    topics: List[Topic] = [
        Topic(
            topic_id=topic.topic_id,
            label=topic.label,
            description=topic.description,
            source_doc_ids=[],
            frequency=0,
        )
        for topic in demo_topics
    ]
    trace: List[dict] = []

    client = OpenAICompatibleClient(
        api_key=llm_cfg.api_key,
        base_url=llm_cfg.base_url,
        model=llm_cfg.model_generate,
        temperature=llm_cfg.temperature,
        max_tokens=llm_cfg.max_tokens,
        timeout=llm_cfg.timeout,
        retry_limit=llm_cfg.retry_limit,
    )
    try:
        pbar = tqdm(
            total=len(sample_rows),
            desc="Gen docs",
            file=sys.stderr,
            unit="doc",
            ncols=80,
            dynamic_ncols=True,
        )
        try:
            for row in sample_rows:
                doc_text = build_document_text(row, pipe_cfg.generation_mode)
                pool_before = [topic.topic_id for topic in topics]
                try:
                    prompt = generation_prompt(doc_text, demo_topics, topics)
                    result = client.chat_json(
                        [
                            {"role": "system", "content": "Return JSON only."},
                            {"role": "user", "content": prompt},
                        ],
                        model=llm_cfg.model_generate,
                    )
                    payload = parse_json_object(result.content)
                    item = {"source_id": row["id"], "payload": payload, "raw": result.raw}
                    action = (payload.get("action") or "").strip()
                    if action == "existing_topic":
                        existing_id = str(payload.get("existing_topic_id") or "").strip()
                        existing_topic = next(
                            (topic for topic in topics if topic.topic_id == existing_id),
                            None,
                        )
                        if existing_topic is None:
                            # Invalid IDs are not allowed to mutate the pool.
                            item["validation_error"] = "existing_topic_id_not_in_current_pool"
                        else:
                            existing_topic.frequency += 1
                            existing_topic.source_doc_ids.append(item["source_id"])
                    elif action == "new_topic" and payload.get("topic"):
                        topic_info = payload["topic"]
                        label = str(topic_info.get("label", "")).strip()
                        description = str(topic_info.get("description", "")).strip()
                        if not label:
                            item["validation_error"] = "new_topic_label_is_empty"
                        else:
                            new_id = f"T{len(topics) + 1:03d}"
                            topics.append(
                                Topic(
                                    topic_id=new_id,
                                    label=label,
                                    description=description,
                                    source_doc_ids=[item["source_id"]],
                                    frequency=1,
                                )
                            )
                    else:
                        if action not in {"none", "existing_topic", "new_topic"}:
                            item["validation_error"] = "unknown_action"
                    item["topic_pool_before"] = pool_before
                    item["topic_pool_after"] = [topic.topic_id for topic in topics]
                    item["topic_pool_size_after"] = len(topics)
                    trace.append(item)
                except Exception as exc:
                    trace.append(
                        {
                            "source_id": row["id"],
                            "skipped": True,
                            "skip_reason": f"generation_parse_failed: {exc}",
                            "topic_pool_before": pool_before,
                            "topic_pool_after": [topic.topic_id for topic in topics],
                            "topic_pool_size_after": len(topics),
                        }
                    )
                pbar.update(1)
                pbar.set_postfix_str(f"topics={len(topics)}")
        finally:
            pbar.close()
    finally:
        client.close()
    return GenerationResult(topics=topics, trace=trace)
