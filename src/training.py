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
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

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
from .gradaware_lora import (
    _summarize_layer_ranks,
    apply_lora_with_rank_pattern,
    compute_gradaware_rank_pattern,
    compute_topheavy_rank_pattern,
    probe_gradient_norms,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    probe_batch_size: int = 8
    probe_max_steps: int | None = None
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


def freeze_all_parameters(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def prepare_bitfit(model: torch.nn.Module) -> torch.nn.Module:
    freeze_all_parameters(model)
    for name, param in model.named_parameters():
        if name.endswith("bias"):
            param.requires_grad = True
    classifier_keywords = ("classifier", "score", "pre_classifier")
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in classifier_keywords):
            param.requires_grad = True
    return model


def prepare_model_for_method(
    config: ExperimentConfig,
    model: torch.nn.Module,
    train_dataset,
    tokenizer,
) -> Tuple[torch.nn.Module, Dict[str, object]]:
    notes: Dict[str, object] = {}

    if config.method == "full_ft":
        return model, notes

    if config.method == "bitfit":
        model = prepare_bitfit(model)
        notes["adapter_strategy"] = "train_bias_only"
        return model, notes

    if config.method == "lora":
        model = apply_lora_with_rank_pattern(
            model,
            rank_pattern=None,
            base_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )
        notes["adapter_strategy"] = "uniform_lora"
        return model, notes

    if config.method == "topheavy_lora":
        rank_pattern, layer_ranks = compute_topheavy_rank_pattern(model, base_r=config.lora_r)
        model = apply_lora_with_rank_pattern(
            model,
            rank_pattern=rank_pattern,
            base_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )
        notes["adapter_strategy"] = "topheavy_lora"
        notes["layer_ranks"] = layer_ranks
        notes["layer_rank_budget"] = _summarize_layer_ranks(layer_ranks)
        return model, notes

    if config.method == "gradaware_lora":
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        probe_loader = DataLoader(
            train_dataset,
            batch_size=config.probe_batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        device = torch.device("cuda" if torch.cuda.is_available() and not config.use_cpu else "cpu")
        model.to(device)
        gradient_norms = probe_gradient_norms(
            model,
            probe_loader,
            device=device,
            max_steps=config.probe_max_steps,
        )
        rank_pattern, layer_ranks = compute_gradaware_rank_pattern(
            model,
            gradient_norms=gradient_norms,
            base_r=config.lora_r,
        )
        model = apply_lora_with_rank_pattern(
            model,
            rank_pattern=rank_pattern,
            base_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )
        notes["adapter_strategy"] = "gradaware_lora"
        notes["probe_gradient_norms"] = gradient_norms
        notes["layer_ranks"] = layer_ranks
        notes["layer_rank_budget"] = _summarize_layer_ranks(layer_ranks)
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
        learning_rate=config.learning_rate if config.method == "full_ft" else max(config.learning_rate, 2e-4),
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
