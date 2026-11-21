from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Sequence

from .client import ChatApi, parse_topic_list
from .config import ApiConfig, PromptTopicConfig
from .data import document_text, truncate_text
from .progress import bar
from .prompts import topic_generation_messages


def _chunks(items: Sequence[dict], size: int) -> List[List[dict]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _generate_chunk(chunk: List[dict], api: dict, cfg: dict) -> List[dict]:
    client = ChatApi(**api)
    try:
        outputs = []
        for row in chunk:
            article = truncate_text(document_text(row, cfg["generation_mode"]), cfg["max_document_chars"])
            result = client.ask(
                topic_generation_messages(
                    article,
                    cfg["demonstrations"],
                    cfg["max_topics_per_document"],
                ),
                model=api["model"],
            )
            topics = parse_topic_list(result.text)
            max_topics = cfg["max_topics_per_document"]
            if max_topics > 0:
                topics = topics[:max_topics]
            outputs.append({"id": row["id"], "topics": topics, "raw": result.raw})
        return outputs
    finally:
        client.close()


def generate_document_topics(rows: List[dict], api_cfg: ApiConfig, cfg: PromptTopicConfig) -> Dict[str, dict]:
    api = {
        "api_key": api_cfg.api_key,
        "base_url": api_cfg.base_url,
        "model": api_cfg.model_generate,
        "temperature": api_cfg.temperature,
        "max_tokens": api_cfg.max_tokens,
        "timeout": api_cfg.timeout,
        "retry_limit": api_cfg.retry_limit,
    }
    common = {
        "generation_mode": cfg.generation_mode,
        "max_document_chars": cfg.max_document_chars,
        "max_topics_per_document": cfg.max_topics_per_document,
        "demonstrations": cfg.demonstrations,
    }
    chunks = _chunks(rows, max(1, cfg.batch_size))
    generated: Dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=cfg.generation_workers) as pool:
        futures = [pool.submit(_generate_chunk, chunk, api, common) for chunk in chunks]
        for fut in bar(futures, total=len(futures), desc="PromptTopic topic generation"):
            for item in fut.result():
                generated[item["id"]] = item
    return generated
