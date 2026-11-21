from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model_generate: str
    model_assign: str
    model_refine: str
    temperature: float = 0.0
    max_tokens: int = 300
    timeout: float = 120.0
    retry_limit: int = 5


@dataclass
class PipelineConfig:
    input_path: Path
    output_path: Path
    sample_size: int = 1000
    generation_workers: int = 4
    assignment_workers: int = 6
    batch_size: int = 16
    merge_similarity_threshold: float = 0.5
    low_frequency_removal_threshold: int = 5
    generation_mode: str = "title_abstract"
    max_topic_rounds: int = 3
    max_assignments_per_doc: int = 1
    assignment_self_correction_retries: int = 10
    assignment_demonstrations: Optional[List[Dict[str, Any]]] = None
    seed_topics_path: Optional[Path] = None
    embed_model_name: str = "all-MiniLM-L6-v2"


def load_pipeline_config(path: Path) -> PipelineConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PipelineConfig(
        input_path=Path(payload["input_path"]),
        output_path=Path(payload["output_path"]),
        sample_size=int(payload.get("sample_size", 1000)),
        generation_workers=int(payload.get("generation_workers", 4)),
        assignment_workers=int(payload.get("assignment_workers", 6)),
        batch_size=int(payload.get("batch_size", 16)),
        merge_similarity_threshold=float(payload.get("merge_similarity_threshold", 0.5)),
        low_frequency_removal_threshold=int(payload.get("low_frequency_removal_threshold", 5)),
        generation_mode=str(payload.get("generation_mode", "title_abstract")),
        max_topic_rounds=int(payload.get("max_topic_rounds", 3)),
        max_assignments_per_doc=int(payload.get("max_assignments_per_doc", 1)),
        assignment_self_correction_retries=int(payload.get("assignment_self_correction_retries", 10)),
        assignment_demonstrations=list(payload.get("assignment_demonstrations", [])),
        seed_topics_path=Path(payload["seed_topics_path"]) if payload.get("seed_topics_path") else None,
        embed_model_name=str(payload.get("embed_model_name", "all-MiniLM-L6-v2")),
    )


def load_llm_config() -> LLMConfig:
    load_dotenv()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    if not api_key:
        raise ValueError("LLM_API_KEY is missing in .env")
    if not base_url:
        raise ValueError("LLM_BASE_URL is missing in .env")
    return LLMConfig(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        model_generate=os.getenv("LLM_MODEL_GENERATE", "gpt-4o"),
        model_assign=os.getenv("LLM_MODEL_ASSIGN", "gpt-4o-mini"),
        model_refine=os.getenv("LLM_MODEL_REFINE", "gpt-4o-mini"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "300")),
        timeout=float(os.getenv("LLM_TIMEOUT", "120")),
        retry_limit=int(os.getenv("LLM_RETRY_LIMIT", "5")),
    )
