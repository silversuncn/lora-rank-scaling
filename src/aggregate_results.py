from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-run metrics into a results table.")
    parser.add_argument("--input_root", type=str, default="artifacts/final_runs")
    parser.add_argument("--output_json", type=str, default="artifacts/final_runs/results.json")
    parser.add_argument("--output_csv", type=str, default="artifacts/final_runs/results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = resolve_path(args.input_root).resolve()
    metrics_files = sorted(input_root.glob("*/metrics.json"))

    rows = []
    for metrics_file in metrics_files:
        payload = json.loads(metrics_file.read_text())
        row = {
            "run_dir": payload.get("run_dir"),
            "method": payload.get("config", {}).get("method"),
            "model_name": payload.get("config", {}).get("model_name"),
            "task_name": payload.get("config", {}).get("task_name"),
            "seed": payload.get("config", {}).get("seed"),
            "train_subset_size": payload.get("config", {}).get("train_subset_size"),
            "primary_metric_name": payload.get("primary_metric_name"),
            "primary_metric": payload.get("eval_metrics", {}).get("eval_primary_metric"),
            "trainable_parameters": payload.get("parameter_stats", {}).get("trainable_parameters"),
            "trainable_percentage": payload.get("parameter_stats", {}).get("trainable_percentage"),
        }
        for key, value in payload.get("eval_metrics", {}).items():
            row[key] = value
        rows.append(row)

    dataframe = pd.DataFrame(rows)
    output_json = resolve_path(args.output_json).resolve()
    output_csv = resolve_path(args.output_csv).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2, sort_keys=True))
    dataframe.to_csv(output_csv, index=False)
    print(output_json)
    print(output_csv)


if __name__ == "__main__":
    main()
