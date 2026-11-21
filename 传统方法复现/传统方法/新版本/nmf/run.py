"""Run the complete NMF workflow: model, metric preprocessing, metrics, outputs."""

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
    parser = argparse.ArgumentParser(description="Run NMF and calculate all metrics.")
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

    from common_metrics import evaluate_model
    from nmf import run_nmf_topic_modeling

    if not args.metrics_only:
        run_nmf_topic_modeling(
            input_file=config["input_file"],
            output_file=str(model_jsonl),
            n_topics=int(config.get("num_topics", 50)),
            top_words_per_topic=int(config.get("top_words_per_topic", 5)),
            metric_top_words=int(config.get("top_words_for_metrics", 10)),
            random_state=int(config.get("random_state", 42)),
            max_features=int(config.get("max_features", 5000)),
            min_df=int(config.get("min_df", 2)),
            max_df=float(config.get("max_df", 0.90)),
            alpha_w=float(config.get("alpha_w", 0.0)),
            l1_ratio=float(config.get("l1_ratio", 0.01)),
            init=config.get("init", "nndsvd"),
            max_iter=int(config.get("max_iter", 1000)),
            tol=float(config.get("tol", 1e-4)),
        )
    elif not model_jsonl.is_file():
        raise FileNotFoundError(f"Model output does not exist: {model_jsonl}")
    summary = evaluate_model("NMF", model_jsonl, config)
    print("\nNMF workflow completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
