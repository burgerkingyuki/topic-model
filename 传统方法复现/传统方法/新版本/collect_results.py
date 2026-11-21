"""Collect metrics from multiple datasets, models, and topic counts into one CSV."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


BASE_COLUMNS = [
    "数据集",
    "模型",
    "K",
    "匹配成功数",
    "未匹配数",
    "匹配成功率",
    "NPMI_Avg",
    "NPMI_Max",
    "Semantic_Similarity_Mean",
    "Semantic_Similarity_Max",
    "CV",
    "CV_NPMI",
    "TD",
    "CV_native",
    "CV_NPMI_native",
    "TD_native",
    "CV_document_tfidf",
    "CV_NPMI_document_tfidf",
    "TD_document_tfidf",
    "Soft Jaccard",
    "relevance_mean_best_topic",
    "relevance_max",
    "relevance_mean_all_pairs",
    "relevance_mean_best_sentence",
    "KL_mean",
]


def find_nearest_config(metrics_path, root_dir):
    current = metrics_path.parent
    root_dir = root_dir.resolve()
    while True:
        candidate = current / "config.json"
        if candidate.is_file():
            return candidate
        if current == root_dir or current.parent == current:
            return None
        current = current.parent


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dataset_name(config):
    configured_name = config.get("dataset_name")
    if configured_name:
        return configured_name
    input_file = config.get("input_file", "")
    return Path(input_file).stem if input_file else ""


def topic_count(config):
    for key in ("num_topics", "n_topics", "topics", "k"):
        if key in config:
            return config[key]
    return ""


def infer_from_path(metrics_path, root_dir):
    """Infer model, dataset, and K from paths such as nmf/output/ICLR/k_20_topics."""
    relative_parts = metrics_path.resolve().relative_to(root_dir.resolve()).parts[:-1]
    directories = list(relative_parts)
    lower_directories = [directory.lower() for directory in directories]
    output_index = next(
        (index for index, directory in enumerate(lower_directories) if directory in {"output", "outputs", "result", "results"}),
        None,
    )

    model = directories[output_index - 1] if output_index and output_index > 0 else (directories[0] if directories else "")
    dataset = directories[output_index + 1] if output_index is not None and output_index + 1 < len(directories) else ""

    k_value = ""
    for directory in reversed(directories):
        match = re.search(r"(?:^|[_-])k[_-]?(\d+)(?:[_-]|$)", directory, re.IGNORECASE)
        if not match:
            match = re.search(r"(\d+)[_-]?topics?$", directory, re.IGNORECASE)
        if match:
            k_value = int(match.group(1))
            break

    return model, dataset, k_value


def collect(root_dir):
    rows = []
    for metrics_path in root_dir.rglob("metrics.json"):
        try:
            metrics = load_json(metrics_path)
            config_path = find_nearest_config(metrics_path, root_dir)
            config = load_json(config_path) if config_path else {}
            path_model, path_dataset, path_k = infer_from_path(metrics_path, root_dir)
            rows.append(
                {
                    "数据集": path_dataset or dataset_name(config),
                    "模型": metrics.get("模型", path_model or config.get("model_name", "")),
                    "K": path_k if path_k != "" else topic_count(config),
                    **metrics,
                    "结果文件": str(metrics_path),
                    "配置文件": str(config_path) if config_path else "",
                }
            )
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipped {metrics_path}: {exc}")

    if not rows:
        return pd.DataFrame(columns=BASE_COLUMNS + ["结果文件", "配置文件"])

    frame = pd.DataFrame(rows)
    extra_columns = [
        column
        for column in frame.columns
        if column not in BASE_COLUMNS and column not in {"结果文件", "配置文件"}
    ]
    ordered_columns = [
        column for column in BASE_COLUMNS if column in frame.columns
    ] + sorted(extra_columns) + ["结果文件", "配置文件"]
    frame = frame.reindex(columns=ordered_columns)
    return frame.sort_values(["数据集", "模型", "K"], kind="stable").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Collect all metrics.json files under a root directory into a plotting CSV."
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="Root directory containing model result directories.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Default: <root>/plot_metrics.csv",
    )
    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")
    output_path = Path(args.output).resolve() if args.output else root_dir / "plot_metrics.csv"

    frame = collect(root_dir)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Collected {len(frame)} result row(s).")
    print(f"Saved plotting CSV to: {output_path}")


if __name__ == "__main__":
    main()
