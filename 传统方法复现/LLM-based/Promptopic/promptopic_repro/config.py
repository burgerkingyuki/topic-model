from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv


@dataclass
class ApiConfig:
    api_key: str
    base_url: str
    model_generate: str
    model_collapse: str
    model_filter: str
    temperature: float
    max_tokens: int
    timeout: float
    retry_limit: int


@dataclass
class PromptTopicConfig:
    input_path: Path
    output_path: Path
    generation_mode: str
    max_document_chars: int
    generation_workers: int
    batch_size: int
    max_topics_per_document: int
    num_topics: int
    intermediate_topics: int
    collapse_strategy: str
    pbm_window_size: int
    ctfidf_top_n: int
    representation_top_n: int
    use_llm_word_filter: bool
    random_seed: int
    demonstrations: List[Dict]


def load_api_config() -> ApiConfig:
    load_dotenv()
    key = os.getenv("LLM_API_KEY", "").strip()
    url = os.getenv("LLM_BASE_URL", "").strip()
    if not key:
        raise ValueError("LLM_API_KEY is missing in .env")
    if not url:
        raise ValueError("LLM_BASE_URL is missing in .env")
    return ApiConfig(
        api_key=key,
        base_url=url.rstrip("/"),
        model_generate=os.getenv("LLM_MODEL_GENERATE", "Meta-Llama-13B"),
        model_collapse=os.getenv("LLM_MODEL_COLLAPSE", "Meta-Llama-13B"),
        model_filter=os.getenv("LLM_MODEL_FILTER", "Meta-Llama-13B"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "256")),
        timeout=float(os.getenv("LLM_TIMEOUT", "120")),
        retry_limit=int(os.getenv("LLM_RETRY_LIMIT", "5")),
    )


def load_promptopic_config(path: Path) -> PromptTopicConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return PromptTopicConfig(
        input_path=Path(data["input_path"]),
        output_path=Path(data["output_path"]),
        generation_mode=str(data.get("generation_mode", "full_text")),
        max_document_chars=int(data.get("max_document_chars", 12000)),
        generation_workers=int(data.get("generation_workers", 4)),
        batch_size=int(data.get("batch_size", 16)),
        max_topics_per_document=int(data.get("max_topics_per_document", 0)),
        num_topics=int(data.get("num_topics", 20)),
        intermediate_topics=int(data.get("intermediate_topics", 400)),
        collapse_strategy=str(data.get("collapse_strategy", "hybrid")),
        pbm_window_size=int(data.get("pbm_window_size", 50)),
        ctfidf_top_n=int(data.get("ctfidf_top_n", 100)),
        representation_top_n=int(data.get("representation_top_n", 10)),
        use_llm_word_filter=bool(data.get("use_llm_word_filter", True)),
        random_seed=int(data.get("random_seed", 42)),
        demonstrations=list(data.get("demonstrations", [])),
    )
