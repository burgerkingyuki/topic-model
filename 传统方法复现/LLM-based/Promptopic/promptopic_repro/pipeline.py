from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from .collapse import collapse_topics
from .config import ApiConfig, PromptTopicConfig
from .data import read_jsonl, write_jsonl
from .generation import generate_document_topics
from .preprocess import normalize_topic
from .representation import ctfidf_words, documents_by_final_topic, filter_representative_words


def _assign_final_topics(generated_topics: List[str], mapping: Dict[str, str]) -> List[str]:
    out = []
    seen = set()
    for topic in generated_topics:
        final_topic = mapping.get(normalize_topic(topic))
        if final_topic and final_topic not in seen:
            out.append(final_topic)
            seen.add(final_topic)
    return out


def run_promptopic(api_cfg: ApiConfig, cfg: PromptTopicConfig) -> Path:
    rows = read_jsonl(cfg.input_path)
    if not rows:
        raise ValueError(f"No rows found in {cfg.input_path}")

    generated_by_doc = generate_document_topics(rows, api_cfg, cfg)
    topic_mapping = collapse_topics(rows, generated_by_doc, api_cfg, cfg)
    grouped_docs = documents_by_final_topic(rows, generated_by_doc, topic_mapping, cfg.generation_mode)
    raw_words = ctfidf_words(grouped_docs, cfg.ctfidf_top_n)
    representative_words = filter_representative_words(raw_words, api_cfg, cfg)

    topic_ids = {topic: idx for idx, topic in enumerate(sorted(set(topic_mapping.values())))}
    enriched = []
    for row in rows:
        generated = generated_by_doc.get(row["id"], {"topics": [], "raw": {}})
        final_topics = _assign_final_topics(generated.get("topics", []), topic_mapping)
        final_topic_records = [
            {
                "topic_id": topic_ids[topic],
                "topic": topic,
                "description": None,
                "keywords": representative_words.get(topic, []),
            }
            for topic in final_topics
        ]
        enriched.append(
            {
                **row,
                "final_topic": final_topic_records[0] if final_topic_records else None,
                "promptopic": {
                    "generated_topics": generated.get("topics", []),
                    "final_topics": [
                        {
                            **item,
                            "representative_words": item["keywords"],
                        }
                        for item in final_topic_records
                    ],
                    "collapse_strategy": cfg.collapse_strategy,
                    "raw_generation": generated.get("raw", {}),
                },
            }
        )

    write_jsonl(cfg.output_path, enriched)

    report_path = cfg.output_path.with_suffix(".report.json")
    topic_counts = Counter()
    for row in enriched:
        for topic in row["promptopic"]["final_topics"]:
            topic_counts[topic["topic"]] += 1
    report = {
        "documents": len(rows),
        "unique_generated_topics": len(topic_mapping),
        "final_topics": len(topic_ids),
        "topic_counts": dict(topic_counts),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return cfg.output_path
