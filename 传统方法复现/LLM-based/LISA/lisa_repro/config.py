from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model_generate: str
    model_summarize: str
    model_assign: str
    temperature: float = 0.0
    max_tokens: int = 256
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
    document_cluster_method: str = "kmeans"
    topic_cluster_method: str = "kmeans"
    num_topics: int = 20
    generation_mode: str = "full_text"
    embed_model_name: str = "all-MiniLM-L6-v2"
    max_document_chars: int = 12000
    lambda_neighbors: int = 5
    mu_neighbors: int = 15
    low_confidence_threshold: float = 0.55
    train_epochs: int = 20
    train_batch_size: int = 128
    hidden_dims: List[int] = None
    tpn_learning_rate: float = 1e-3
    tau: float = 1.5
    alpha: float = 0.8
    use_llm_generation: bool = True
    use_llm_summarize: bool = True
    use_llm_assignment: bool = True
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]


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
        model_summarize=os.getenv("LLM_MODEL_SUMMARIZE", "gpt-4o"),
        model_assign=os.getenv("LLM_MODEL_ASSIGN", "gpt-4o-mini"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "256")),
        timeout=float(os.getenv("LLM_TIMEOUT", "120")),
        retry_limit=int(os.getenv("LLM_RETRY_LIMIT", "5")),
    )


def load_pipeline_config(path: Path) -> PipelineConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PipelineConfig(
        input_path=Path(payload["input_path"]),
        output_path=Path(payload["output_path"]),
        sample_size=int(payload.get("sample_size", 1000)),
        generation_workers=int(payload.get("generation_workers", 4)),
        assignment_workers=int(payload.get("assignment_workers", 6)),
        batch_size=int(payload.get("batch_size", 16)),
        document_cluster_method=str(payload.get("document_cluster_method", "kmeans")),
        topic_cluster_method=str(payload.get("topic_cluster_method", "kmeans")),
        num_topics=int(payload.get("num_topics", 20)),
        generation_mode=str(payload.get("generation_mode", "full_text")),
        embed_model_name=str(payload.get("embed_model_name", "all-MiniLM-L6-v2")),
        max_document_chars=int(payload.get("max_document_chars", 12000)),
        lambda_neighbors=int(payload.get("lambda_neighbors", 5)),
        mu_neighbors=int(payload.get("mu_neighbors", 15)),
        low_confidence_threshold=float(payload.get("low_confidence_threshold", 0.55)),
        train_epochs=int(payload.get("train_epochs", 20)),
        train_batch_size=int(payload.get("train_batch_size", 128)),
        hidden_dims=list(payload.get("hidden_dims", [256, 256])),
        tpn_learning_rate=float(payload.get("tpn_learning_rate", 1e-3)),
        tau=float(payload.get("tau", 1.5)),
        alpha=float(payload.get("alpha", 0.8)),
        use_llm_generation=bool(payload.get("use_llm_generation", True)),
        use_llm_summarize=bool(payload.get("use_llm_summarize", True)),
        use_llm_assignment=bool(payload.get("use_llm_assignment", True)),
        random_seed=int(payload.get("random_seed", 42)),
    )
