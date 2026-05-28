from __future__ import annotations

import json
import numbers
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

from .constants import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_MAX_LENGTH,
    DEFAULT_TRAIN_SUBSET,
    SUPPORTED_METHODS,
    SUPPORTED_MODELS,
    SUPPORTED_TASKS,
    TASK_PRIMARY_METRIC,
    TASK_TO_KEYS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE_CANDIDATES = ["q_lin", "v_lin", "query", "value", "query_proj", "value_proj"]


@dataclass
class ExperimentConfig:
    method: str
    model_name: str
    task_name: str
    train_subset_size: int = DEFAULT_TRAIN_SUBSET
    max_length: int = DEFAULT_MAX_LENGTH
    seed: int = 42
    output_root: str = "artifacts/final_runs"
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    logging_steps: int = 10
    eval_strategy: str = "epoch"
    save_strategy: str = "no"
    lora_r: int = DEFAULT_LORA_R
    lora_alpha: int = DEFAULT_LORA_ALPHA
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    disable_tqdm: bool = False
    use_cpu: bool = False

    def validate(self) -> None:
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method: {self.method}")
        if self.task_name not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {self.task_name}")
        if self.model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {self.model_name}")
        if self.train_subset_size <= 0:
            raise ValueError("train_subset_size must be positive")


class SequenceClassificationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "-")


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_run_dir(config: ExperimentConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.method}__{_safe_model_name(config.model_name)}__{config.task_name}__n{config.train_subset_size}__r{config.lora_r}__s{config.seed}"
    return resolve_project_path(config.output_root).resolve() / run_name


def prepare_datasets(config: ExperimentConfig, tokenizer):
    sentence1_key, sentence2_key = TASK_TO_KEYS[config.task_name]
    raw = load_dataset("glue", config.task_name)

    train_split = raw["train"].shuffle(seed=config.seed)
    subset_size = min(config.train_subset_size, len(train_split))
    train_split = train_split.select(range(subset_size))
    eval_split = raw["validation"]

    def tokenize_fn(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True, max_length=config.max_length)
        return tokenizer(
            examples[sentence1_key],
            examples[sentence2_key],
            truncation=True,
            max_length=config.max_length,
        )

    train_dataset = train_split.map(tokenize_fn, batched=True)
    eval_dataset = eval_split.map(tokenize_fn, batched=True)

    keep_columns = {"input_ids", "attention_mask", "label"}
    if "token_type_ids" in train_dataset.column_names:
        keep_columns.add("token_type_ids")

    remove_train = [col for col in train_dataset.column_names if col not in keep_columns]
    remove_eval = [col for col in eval_dataset.column_names if col not in keep_columns]
    train_dataset = train_dataset.remove_columns(remove_train)
    eval_dataset = eval_dataset.remove_columns(remove_eval)
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")
    return train_dataset, eval_dataset


def get_num_labels(task_name: str) -> int:
    return 2 if task_name != "stsb" else 1


def count_parameters(model: torch.nn.Module) -> Dict[str, float]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_percentage": float(100.0 * trainable / total),
    }


def infer_lora_target_modules(model: torch.nn.Module) -> list[str]:
    suffixes: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(name.endswith(candidate) for candidate in TARGET_MODULE_CANDIDATES):
            suffix = name.split(".")[-1]
            if suffix not in suffixes:
                suffixes.append(suffix)
    if not suffixes:
        raise ValueError("Could not find target attention linear modules for LoRA injection.")
    return suffixes


def prepare_lora(model: torch.nn.Module, config: ExperimentConfig) -> torch.nn.Module:
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=infer_lora_target_modules(model),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    return get_peft_model(model, lora_config)


def prepare_model_for_method(
    config: ExperimentConfig,
    model: torch.nn.Module,
    train_dataset,
    tokenizer,
) -> Tuple[torch.nn.Module, Dict[str, object]]:
    notes: Dict[str, object] = {}

    if config.method == "lora":
        model = prepare_lora(model, config)
        notes["adapter_strategy"] = "uniform_lora"
        return model, notes

    raise ValueError(f"Unknown method: {config.method}")


def build_compute_metrics(task_name: str):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if task_name == "stsb":
            predictions = np.squeeze(logits)
            raise ValueError("stsb regression is not supported by this scaffold.")

        predictions = np.argmax(logits, axis=-1)
        metric_result = {
            "accuracy": float(accuracy_score(labels, predictions)),
        }
        if task_name == "mrpc":
            metric_result["f1"] = float(f1_score(labels, predictions))
        if task_name == "cola":
            metric_result["matthews_correlation"] = float(matthews_corrcoef(labels, predictions))
        metric_result["primary_metric"] = float(metric_result[TASK_PRIMARY_METRIC[task_name]])
        metric_result[f"eval_{TASK_PRIMARY_METRIC[task_name]}"] = float(metric_result[TASK_PRIMARY_METRIC[task_name]])
        metric_result["eval_primary_metric"] = float(metric_result["primary_metric"])
        return metric_result

    return compute_metrics


def build_training_arguments(config: ExperimentConfig, run_dir: Path) -> TrainingArguments:
    use_cuda = torch.cuda.is_available() and not config.use_cpu
    return TrainingArguments(
        output_dir=str(run_dir / "trainer_output"),
        do_train=True,
        do_eval=True,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=max(config.learning_rate, 2e-4),
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        report_to="none",
        optim="adamw_torch",
        seed=config.seed,
        data_seed=config.seed,
        disable_tqdm=config.disable_tqdm,
        bf16=bool(use_cuda),
        fp16=False,
        use_cpu=not use_cuda,
        remove_unused_columns=False,
    )


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def run_experiment(config: ExperimentConfig) -> Dict[str, object]:
    config.validate()
    set_seed(config.seed)

    run_dir = build_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=get_num_labels(config.task_name),
    )

    model, method_notes = prepare_model_for_method(config, model, train_dataset, tokenizer)
    parameter_stats = count_parameters(model)
    training_args = build_training_arguments(config, run_dir)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = SequenceClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=build_compute_metrics(config.task_name),
    )

    train_output = trainer.train()
    eval_metrics = trainer.evaluate()

    config_payload = asdict(config)
    result = {
        "config": config_payload,
        "parameter_stats": parameter_stats,
        "method_notes": method_notes,
        "train_metrics": {k: float(v) for k, v in train_output.metrics.items() if isinstance(v, numbers.Number)},
        "eval_metrics": {k: float(v) for k, v in eval_metrics.items() if isinstance(v, numbers.Number)},
        "primary_metric_name": TASK_PRIMARY_METRIC[config.task_name],
        "run_dir": str(run_dir),
    }

    _write_json(run_dir / "config.json", config_payload)
    _write_json(run_dir / "metrics.json", result)

    model_save_dir = run_dir / "model"
    trainer.save_model(str(model_save_dir))
    tokenizer.save_pretrained(str(model_save_dir))

    return result
