from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_api_config, load_promptopic_config
from .pipeline import run_promptopic


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PromptTopic reproduction pipeline")
    p.add_argument("--config", default="config.json")
    return p


def main() -> None:
    args = parser().parse_args()
    api = load_api_config()
    cfg = load_promptopic_config(Path(args.config))
    print(run_promptopic(api, cfg))


if __name__ == "__main__":
    main()

