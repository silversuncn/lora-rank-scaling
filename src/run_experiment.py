from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if "--offline" in sys.argv:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import ExperimentConfig, resolve_project_path, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single fine-tuning experiment.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--train_subset_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_root", type=str, default="artifacts/final_runs")
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--probe_batch_size", type=int, default=8)
    parser.add_argument("--probe_max_steps", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use only locally cached Hugging Face assets.")
    parser.add_argument("--json_out", type=str, default=None, help="Optional extra JSON summary path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    arg_dict = vars(args).copy()
    json_out = arg_dict.pop("json_out")
    arg_dict.pop("offline")
    config = ExperimentConfig(**arg_dict)
    result = run_experiment(config)
    if json_out:
        output_path = resolve_project_path(json_out).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
