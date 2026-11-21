from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_llm_config, load_pipeline_config
from .pipeline import run_topicgpt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TopicGPT reproduction pipeline")
    parser.add_argument("--config", default="config.json", help="Pipeline config JSON path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    llm_cfg = load_llm_config()
    pipe_cfg = load_pipeline_config(Path(args.config))
    out = run_topicgpt(llm_cfg, pipe_cfg)
    print(str(out))


if __name__ == "__main__":
    main()
