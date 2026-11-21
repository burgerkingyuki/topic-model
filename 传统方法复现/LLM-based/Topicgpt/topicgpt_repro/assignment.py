from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from typing import List

from tqdm import tqdm

from .config import LLMConfig, PipelineConfig
from .dataio import build_document_text
from .llm_client import OpenAICompatibleClient, parse_json_object
from .prompts import assignment_correction_prompt, assignment_prompt
from .schemas import Topic, TopicAssignment


def _chunk(seq, size):
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


def _validate_assignment_payload(
    payload: dict,
    topics: List[Topic],
    document: str,
    max_assignments: int,
) -> tuple[List[TopicAssignment], List[str]]:
    errors: List[str] = []
    topic_map = {topic.topic_id: topic for topic in topics}
    raw_assignments = payload.get("assignments")
    if not isinstance(raw_assignments, list):
        return [], ["assignments must be a JSON list"]
    if len(raw_assignments) > max_assignments:
        errors.append(f"too_many_assignments: maximum is {max_assignments}")

    assignments: List[TopicAssignment] = []
    seen_ids = set()
    for index, item in enumerate(raw_assignments[:max_assignments], start=1):
        if not isinstance(item, dict):
            errors.append(f"assignment_{index}_is_not_an_object")
            continue
        topic_id = str(item.get("topic_id") or "").strip()
        label = str(item.get("label") or "").strip()
        description = str(item.get("description") or "").strip()
        quote = str(item.get("quote") or "").strip()
        reason = str(item.get("reason") or "").strip()
        topic = topic_map.get(topic_id)
        if topic is None:
            errors.append(f"assignment_{index}_uses_invalid_topic_id")
            continue
        if topic_id in seen_ids:
            errors.append(f"assignment_{index}_duplicates_topic_id")
            continue
        if label != topic.label:
            errors.append(f"assignment_{index}_label_does_not_match_topic_id")
            continue
        if not quote or quote not in document:
            errors.append(f"assignment_{index}_quote_is_not_a_verbatim_document_substring")
            continue
        seen_ids.add(topic_id)
        assignments.append(
            TopicAssignment(
                topic_id=topic_id,
                label=label,
                description=description,
                quote=quote,
                reason=reason,
            )
        )
    if raw_assignments and not assignments:
        errors.append("no_valid_assignments_after_validation")
    return assignments, errors


def _assign_one(row, topics, client, pipe_cfg, llm_cfg):
    document = build_document_text(row, pipe_cfg["generation_mode"])
    max_assignments = pipe_cfg["max_assignments_per_doc"]
    max_attempts = max(0, int(pipe_cfg.get("assignment_self_correction_retries", 10)))
    messages = [
        {"role": "system", "content": "Return JSON only."},
        {
            "role": "user",
            "content": assignment_prompt(
                document,
                topics,
                max_assignments,
                pipe_cfg.get("assignment_demonstrations", []),
            ),
        },
    ]
    attempts = []
    last_payload = {}
    last_errors = []
    try:
        for attempt in range(max_attempts + 1):
            result = client.chat_json(messages, model=llm_cfg["model"])
            payload = parse_json_object(result.content)
            assignments, errors = _validate_assignment_payload(
                payload, topics, document, max_assignments
            )
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "payload": payload,
                    "raw": result.raw,
                    "errors": errors,
                }
            )
            if not errors:
                return {
                    "id": row["id"],
                    "assignments": assignments,
                    "raw": result.raw,
                    "self_correction_attempts": attempts,
                    "self_correction_succeeded": True,
                }
            last_payload = payload
            last_errors = errors
            messages = [
                {"role": "system", "content": "Return JSON only."},
                {
                    "role": "user",
                    "content": assignment_correction_prompt(
                        document,
                        topics,
                        max_assignments,
                        last_payload,
                        last_errors,
                    ),
                },
            ]
    except Exception as exc:
        last_errors = [f"assignment_pipeline_failed: {exc}"]
    return {
        "id": row["id"],
        "assignments": [],
        "raw": attempts[-1]["raw"] if attempts else {},
        "self_correction_attempts": attempts,
        "self_correction_succeeded": False,
        "self_correction_errors": last_errors,
    }


def _assign_chunk(chunk, topics_snapshot, llm_cfg, pipe_cfg):
    client = OpenAICompatibleClient(**llm_cfg)
    try:
        topics = [Topic(**item) for item in topics_snapshot]
        outputs = []
        for row in chunk:
            outputs.append(_assign_one(row, topics, client, pipe_cfg, llm_cfg))
        return outputs
    finally:
        client.close()


def assign_topics(
    rows: List[dict],
    topics: List[Topic],
    llm_cfg: LLMConfig,
    pipe_cfg: PipelineConfig,
) -> List[dict]:
    if not rows:
        return []

    chunks = _chunk(rows, max(1, pipe_cfg.batch_size))
    worker_cfg = {
        "api_key": llm_cfg.api_key,
        "base_url": llm_cfg.base_url,
        "model": llm_cfg.model_assign,
        "temperature": llm_cfg.temperature,
        "max_tokens": llm_cfg.max_tokens,
        "timeout": llm_cfg.timeout,
        "retry_limit": llm_cfg.retry_limit,
    }
    pipe_dict = {
        "generation_mode": pipe_cfg.generation_mode,
        "max_assignments_per_doc": pipe_cfg.max_assignments_per_doc,
        "assignment_self_correction_retries": pipe_cfg.assignment_self_correction_retries,
        "assignment_demonstrations": pipe_cfg.assignment_demonstrations or [],
    }
    topic_snapshot = [asdict(t) for t in topics]
    outputs_by_id = {}
    with ProcessPoolExecutor(max_workers=pipe_cfg.assignment_workers) as ex:
        futures = [
            ex.submit(_assign_chunk, chunk, topic_snapshot, worker_cfg, pipe_dict)
            for chunk in chunks
        ]
        pbar = tqdm(
            total=len(futures),
            desc="Assign chunks",
            file=sys.stderr,
            unit="chunk",
            ncols=80,
            dynamic_ncols=True,
        )
        try:
            for fut in futures:
                for item in fut.result():
                    outputs_by_id[item["id"]] = item
                pbar.update(1)
                pbar.set_postfix_str(f"done={len(outputs_by_id)}/{len(rows)}")
        finally:
            pbar.close()

    enriched = []
    topic_map = {t.topic_id: t for t in topics}
    for row in rows:
        item = outputs_by_id.get(row["id"], {"assignments": []})
        assignments = []
        for a in item["assignments"]:
            topic = topic_map.get(a.topic_id)
            if topic is None:
                continue
            assignments.append(
                {
                    "topic_id": topic.topic_id,
                    "label": a.label or topic.label,
                    "description": a.description or topic.description,
                    "quote": a.quote,
                    "reason": a.reason,
                }
            )
        enriched.append(
            {
                **row,
                "final_topic": (
                    {
                        "topic_id": assignments[0]["topic_id"],
                        "topic": assignments[0]["label"],
                        "description": assignments[0]["description"],
                        "keywords": [],
                    }
                    if assignments
                    else None
                ),
                "topicgpt": {
                    "assigned_topics": assignments,
                    "topic_pool_size": len(topics),
                    "raw_assignment": item.get("raw", {}),
                    "self_correction_attempts": item.get("self_correction_attempts", []),
                    "self_correction_succeeded": item.get("self_correction_succeeded", False),
                    "self_correction_errors": item.get("self_correction_errors", []),
                },
            }
        )
    return enriched
