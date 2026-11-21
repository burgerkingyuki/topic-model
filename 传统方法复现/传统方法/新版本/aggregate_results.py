"""Aggregate per-model metric outputs into one CSV."""

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_MODELS = ["lda", "nmf", "bertopic"]
SUMMARY_COLUMNS = [
    "模型",
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
    "KL_mean",
]


def main():
    parser = argparse.ArgumentParser(description="Aggregate model metrics into summary.csv.")
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    model_names = [item.strip() for item in args.models.split(",") if item.strip()]
    rows = []
    missing = []
    for model_name in model_names:
        metrics_path = base_dir / model_name / "output" / "metrics.json"
        if not metrics_path.is_file():
            missing.append(str(metrics_path))
            continue
        with open(metrics_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append({column: metrics.get(column, "") for column in SUMMARY_COLUMNS})

    output_path = Path(args.output) if args.output else base_dir / "summary.csv"
    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Summary saved to: {output_path}")
    if missing:
        print("Missing metrics files:")
        for path in missing:
            print(f"- {path}")


if __name__ == "__main__":
    main()
