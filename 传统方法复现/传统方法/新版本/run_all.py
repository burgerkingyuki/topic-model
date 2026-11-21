"""Run all per-model workflows and write the final summary CSV."""

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MODELS = ["lda", "nmf", "bertopic"]


def main():
    parser = argparse.ArgumentParser(description="Run selected model workflows and aggregate results.")
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    model_names = [item.strip() for item in args.models.split(",") if item.strip()]
    failures = []
    for model_name in model_names:
        run_script = base_dir / model_name / "run.py"
        print(f"\n===== Running {model_name} =====")
        result = subprocess.run([sys.executable, str(run_script)], cwd=str(base_dir))
        if result.returncode != 0:
            failures.append(model_name)

    aggregate_script = base_dir / "aggregate_results.py"
    subprocess.run(
        [sys.executable, str(aggregate_script), "--base-dir", str(base_dir), "--models", ",".join(model_names)],
        cwd=str(base_dir),
        check=False,
    )
    if failures:
        print(f"\nCompleted with failed workflows: {', '.join(failures)}")
        sys.exit(1)
    print("\nAll workflows completed.")


if __name__ == "__main__":
    main()
