"""Run the complete BERTopic workflow: model, metric preprocessing, metrics, outputs."""

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
    parser = argparse.ArgumentParser(description="Run BERTopic and calculate all metrics.")
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

    from bert_topic import bertopic_topic_mining
    from common_metrics import evaluate_model

    if not args.metrics_only:
        bertopic_topic_mining(
            config["input_file"],
            str(model_jsonl),
            num_topics=int(config.get("num_topics", 50)),
            embedding_model_path=config["embedding_model_path"],
            umap_n_neighbors=int(config.get("umap_n_neighbors", 15)),
            umap_n_components=int(config.get("umap_n_components", 5)),
            umap_min_dist=float(config.get("umap_min_dist", 0.0)),
            hdbscan_min_cluster_size=int(config.get("hdbscan_min_cluster_size", 5)),
            hdbscan_min_samples=int(config.get("hdbscan_min_samples", 1)),
            hdbscan_cluster_selection_epsilon=float(config.get("hdbscan_cluster_selection_epsilon", 0.1)),
            vectorizer_min_df=int(config.get("vectorizer_min_df", 1)),
            vectorizer_max_df=float(config.get("vectorizer_max_df", 0.95)),
            vectorizer_max_features=int(config.get("vectorizer_max_features", 5000)),
            top_n_words=int(config.get("top_n_words", 10)),
        )
    elif not model_jsonl.is_file():
        raise FileNotFoundError(f"Model output does not exist: {model_jsonl}")
    summary = evaluate_model("BERTopic", model_jsonl, config)
    print("\nBERTopic workflow completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
