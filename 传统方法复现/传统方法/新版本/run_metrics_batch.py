"""Recalculate metrics for existing model outputs in a configured order.

Edit metrics_batch_config.json, then run:
    python run_metrics_batch.py --config metrics_batch_config.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from common_metrics import evaluate_model


MODEL_LABELS = {
    "lda": "LDA",
    "nmf": "NMF",
    "bertopic": "BERTopic",
    "bert_topic": "BERTopic",
    "topicgpt": "TopicGPT",
    "ctm": "CTM",
}


PATH_FIELDS = {
    "input_file",
    "metadata_file",
    "output_dir",
    "model_jsonl",
    "embedding_model_path",
}


REQUIRED_FIELDS = {
    "model_name",
    "input_file",
    "metadata_file",
    "output_dir",
    "embedding_model_path",
}


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def resolve_path(value, base_dir):
    if value is None or value == "":
        return value
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def merge_config(defaults, job, config_dir):
    merged = dict(defaults)
    merged.update(job)
    for field in PATH_FIELDS:
        if field in merged:
            merged[field] = resolve_path(merged[field], config_dir)
    if "model_jsonl" not in merged or not merged["model_jsonl"]:
        output_dir = Path(merged.get("output_dir", ""))
        merged["model_jsonl"] = str(output_dir / "model_output.jsonl")
    if "output_dir" not in merged or not merged["output_dir"]:
        merged["output_dir"] = str(Path(merged["model_jsonl"]).parent)
    return merged


def validate_job(job, index):
    missing = [field for field in REQUIRED_FIELDS if not job.get(field)]
    if missing:
        raise ValueError(f"Job #{index} is missing required fields: {', '.join(missing)}")

    for field in ("input_file", "metadata_file", "model_jsonl"):
        path = Path(job[field])
        if not path.is_file():
            raise FileNotFoundError(f"Job #{index} {field} does not exist: {path}")

    embedding_model_path = Path(job["embedding_model_path"])
    if not embedding_model_path.exists():
        raise FileNotFoundError(
            f"Job #{index} embedding_model_path does not exist: {embedding_model_path}"
        )


def normalize_model_name(value):
    key = str(value).strip().lower()
    return MODEL_LABELS.get(key, str(value).strip())


def write_batch_outputs(rows, failures, output_csv, output_json):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    report = {
        "total_jobs": len(rows) + len(failures),
        "successful_jobs": len(rows),
        "failed_jobs": len(failures),
        "failures": failures,
    }
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def run_batch(config_path):
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent
    batch_config = load_json(config_path)

    defaults = batch_config.get("defaults", {})
    jobs = batch_config.get("jobs", [])
    if not jobs:
        raise ValueError("No jobs configured. Add items under the 'jobs' list.")

    continue_on_error = bool(batch_config.get("continue_on_error", True))
    summary_csv = Path(resolve_path(batch_config.get("summary_csv", "batch_metrics_summary.csv"), config_dir))
    report_json = Path(resolve_path(batch_config.get("report_json", "batch_metrics_report.json"), config_dir))

    summaries = []
    failures = []
    runnable_jobs = [job for job in jobs if job.get("enabled", True)]
    total = len(runnable_jobs)

    for position, raw_job in enumerate(runnable_jobs, start=1):
        job = merge_config(defaults, raw_job, config_dir)
        job_name = job.get("name") or f"job_{position}"
        model_name = normalize_model_name(job["model_name"])
        started = time.time()

        print(f"\n[{position}/{total}] Recalculating metrics: {job_name} ({model_name})")
        try:
            validate_job(job, position)
            summary = evaluate_model(model_name, job["model_jsonl"], job)
            elapsed_seconds = round(time.time() - started, 2)
            reference_summary_path = (
                Path(job["output_dir"])
                / "metrics"
                / "reference_metrics_summary.json"
            )
            reference_summary = {}
            if reference_summary_path.is_file():
                reference_summary = load_json(reference_summary_path)
            row = {
                "job_name": job_name,
                "model_name": model_name,
                "model_jsonl": job["model_jsonl"],
                "output_dir": job["output_dir"],
                "elapsed_seconds": elapsed_seconds,
                **summary,
                **{
                    f"reference_{key}": value
                    for key, value in reference_summary.items()
                    if not isinstance(value, (dict, list))
                },
            }
            summaries.append(row)
            print(f"[{position}/{total}] Done in {elapsed_seconds}s: {job_name}")
        except Exception as exc:
            failure = {
                "job_name": job_name,
                "model_name": model_name,
                "error": str(exc),
            }
            failures.append(failure)
            print(f"[{position}/{total}] FAILED: {job_name}: {exc}")
            if not continue_on_error:
                break

    write_batch_outputs(summaries, failures, summary_csv, report_json)
    print(f"\nBatch summary written to: {summary_csv}")
    print(f"Batch report written to: {report_json}")

    if failures:
        print("\nFailed jobs:")
        for failure in failures:
            print(f"- {failure['job_name']} ({failure['model_name']}): {failure['error']}")
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description="Recalculate metrics from existing model_output.jsonl files.")
    parser.add_argument(
        "--config",
        default=str(THIS_DIR / "metrics_batch_config.json"),
        help="Path to the batch metrics config JSON.",
    )
    args = parser.parse_args()
    raise SystemExit(run_batch(args.config))


if __name__ == "__main__":
    main()
