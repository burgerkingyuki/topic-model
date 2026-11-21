from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from .assignment import assign_topics
from .config import LLMConfig, PipelineConfig
from .dataio import load_jsonl, write_jsonl
from .generation import generate_topics
from .refinement import refine_topics
from .schemas import Topic


def _default_seed_topics() -> List[Topic]:
    return [
        Topic(topic_id="E001", label="Quantum algorithms", description="Methods for solving problems using quantum circuits."),
        Topic(topic_id="E002", label="Quantum information", description="Topics about quantum states, entanglement, and information processing."),
    ]


def run_topicgpt(llm_cfg: LLMConfig, pipe_cfg: PipelineConfig) -> Path:
    t0 = time.perf_counter()
    rows = load_jsonl(pipe_cfg.input_path)
    if not rows:
        raise ValueError(f"No rows found in {pipe_cfg.input_path}")

    rng = random.Random(42)
    sample_size = min(pipe_cfg.sample_size, len(rows))
    sample = rng.sample(rows, sample_size) if sample_size < len(rows) else list(rows)
    demo_topics = _default_seed_topics()

    total_docs = len(rows)
    total_sample = len(sample)
    print(f"[TopicGPT] Starting pipeline")
    print(f"  Input: {total_docs} documents ({pipe_cfg.input_path})")
    print(f"  Sample for generation: {total_sample} docs")
    print(f"  Output: {pipe_cfg.output_path}", flush=True)

    # ------ Stage 1: Generation ------
    t1 = time.perf_counter()
    print(
        f"\n[Stage 1/3] Topic generation (sample={total_sample}, "
        "strict sequential topic-pool updates)", flush=True,
    )
    generation = generate_topics(sample, demo_topics, llm_cfg, pipe_cfg)
    print(
        f"[Stage 1/3] Done: {len(generation.topics)} topics found in "
        f"{len(generation.trace)} document traces ({time.perf_counter() - t1:.1f}s)", flush=True,
    )

    # ------ Stage 2: Refinement ------
    t2 = time.perf_counter()
    initial_n = len(generation.topics)
    print(
        f"\n[Stage 2/3] Topic refinement ({initial_n} initial topics, "
        f"rounds={pipe_cfg.max_topic_rounds}, threshold={pipe_cfg.merge_similarity_threshold})", flush=True,
    )
    refined_topics = refine_topics(generation.topics, llm_cfg, pipe_cfg)
    print(
        f"[Stage 2/3] Done: {initial_n} -> {len(refined_topics)} topics "
        f"({time.perf_counter() - t2:.1f}s)", flush=True,
    )

    # ------ Stage 3: Assignment ------
    t3 = time.perf_counter()
    print(
        f"\n[Stage 3/3] Topic assignment ({total_docs} docs, batch_size={pipe_cfg.batch_size}, "
        f"workers={pipe_cfg.assignment_workers}, topic_count={len(refined_topics)})", flush=True,
    )
    enriched = assign_topics(rows, refined_topics, llm_cfg, pipe_cfg)
    print(
        f"[Stage 3/3] Done: {len(enriched)} documents processed ({time.perf_counter() - t3:.1f}s)", flush=True,
    )

    write_jsonl(pipe_cfg.output_path, enriched)
    total_elapsed = time.perf_counter() - t0
    print(
        f"\n[TopicGPT] Pipeline finished in {total_elapsed:.1f}s. "
        f"Output: {pipe_cfg.output_path}", flush=True,
    )
    return pipe_cfg.output_path
