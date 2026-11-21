"""Run the complete LDA workflow: model, metric preprocessing, metrics, outputs."""

import argparse
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
NEW_VERSION_DIR = THIS_DIR.parent
PROJECT_DIR = NEW_VERSION_DIR.parent
sys.path.insert(0, str(NEW_VERSION_DIR))
sys.path.insert(0, str(PROJECT_DIR))


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Run LDA and calculate all metrics.")
    parser.add_argument("--config", default=str(THIS_DIR / "config.json"))
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Skip model training and calculate metrics from the existing model_output.jsonl.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_jsonl = output_dir / "model_output.jsonl"

    from LDA import lda_topic_mining
    from common_metrics import evaluate_model

    if not args.metrics_only:
        lda_topic_mining(
            config["input_file"],
            str(model_jsonl),
            num_topics=int(config.get("num_topics", 50)),
            top_words_per_topic=int(config.get("top_words_per_topic", 10)),
            top_words_per_doc=int(config.get("top_words_per_doc", 5)),
            random_state=int(config.get("random_state", 42)),
            passes=int(config.get("passes", 10)),
            alpha=config.get("alpha", "auto"),
            no_below=int(config.get("no_below", 2)),
            no_above=float(config.get("no_above", 0.8)),
        )
    elif not model_jsonl.is_file():
        raise FileNotFoundError(f"Model output does not exist: {model_jsonl}")
    summary = evaluate_model("LDA", model_jsonl, config)
    print("\nLDA workflow completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
