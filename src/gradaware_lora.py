from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model

from .constants import DEFAULT_LORA_ALPHA, DEFAULT_LORA_DROPOUT, DEFAULT_LORA_R


TARGET_MODULE_CANDIDATES = ["q_lin", "v_lin", "query", "value", "query_proj", "value_proj"]


# PEFT exposes per-module rank overrides through rank_pattern, so this scaffold uses
# exact module-name rank assignments rather than a custom LoRA implementation.
# The budget is conserved in integer "rank units" across all targeted attention
# linear modules, which is a practical approximation of per-layer proportional ranks.

def get_transformer_layer_stack(model: torch.nn.Module):
    if hasattr(model, "distilbert"):
        return model.distilbert.transformer.layer
    if hasattr(model, "bert"):
        return model.bert.encoder.layer
    if hasattr(model, "roberta"):
        return model.roberta.encoder.layer
    raise ValueError(f"Unsupported model architecture: {model.__class__.__name__}")


def iter_named_target_linear_modules(model: torch.nn.Module) -> Iterable[Tuple[str, torch.nn.Module]]:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(name.endswith(candidate) for candidate in TARGET_MODULE_CANDIDATES):
            yield name, module


def get_target_module_suffixes(model: torch.nn.Module) -> List[str]:
    suffixes = []
    for name, _ in iter_named_target_linear_modules(model):
        suffix = name.split(".")[-1]
        if suffix not in suffixes:
            suffixes.append(suffix)
    if not suffixes:
        raise ValueError("Could not find target attention linear modules for LoRA injection.")
    return suffixes


def infer_layer_index(module_name: str) -> int:
    parts = module_name.split(".")
    for idx, part in enumerate(parts):
        if part == "layer" and idx + 1 < len(parts):
            next_part = parts[idx + 1]
            if next_part.isdigit():
                return int(next_part)
    raise ValueError(f"Could not infer transformer layer index from module name: {module_name}")


def _distribute_integer_values(weights: List[float], total: int, minimum_value: int = 1) -> List[int]:
    if total < len(weights) * minimum_value:
        raise ValueError("Total is too small for the requested minimum allocation.")

    adjusted = [max(0.0, float(w)) for w in weights]
    if sum(adjusted) == 0.0:
        adjusted = [1.0 for _ in weights]

    remaining = total - len(weights) * minimum_value
    allocations = [minimum_value for _ in weights]
    raw = [remaining * (w / sum(adjusted)) for w in adjusted]
    increments = [math.floor(x) for x in raw]

    for index, inc in enumerate(increments):
        allocations[index] += inc

    leftover = remaining - sum(increments)
    if leftover > 0:
        remainders = sorted(
            enumerate([raw_i - math.floor(raw_i) for raw_i in raw]),
            key=lambda item: item[1],
            reverse=True,
        )
        for index, _ in remainders[:leftover]:
            allocations[index] += 1
    return allocations


def _build_rank_pattern(model: torch.nn.Module, layer_ranks: Dict[int, int]) -> Dict[str, int]:
    rank_pattern = {}
    for module_name, _ in iter_named_target_linear_modules(model):
        layer_idx = infer_layer_index(module_name)
        if layer_idx in layer_ranks:
            rank_pattern[module_name] = int(layer_ranks[layer_idx])
    return rank_pattern


def compute_topheavy_rank_pattern(
    model: torch.nn.Module,
    base_r: int = DEFAULT_LORA_R,
) -> Tuple[Dict[str, int], Dict[int, Dict[str, int]]]:
    layer_stack = get_transformer_layer_stack(model)
    num_layers = len(layer_stack)
    target_modules = list(iter_named_target_linear_modules(model))
    if not target_modules:
        raise ValueError("No target modules found for TopHeavy-LoRA.")

    modules_by_layer = defaultdict(list)
    for module_name, module in target_modules:
        modules_by_layer[infer_layer_index(module_name)].append((module_name, module))

    minimum_layer_budget = min(len(modules_by_layer[layer_idx]) for layer_idx in range(num_layers))
    total_rank_units = base_r * len(target_modules)
    layer_weights = [float(index + 1) for index in range(num_layers)]
    layer_rank_units = _distribute_integer_values(
        layer_weights,
        total_rank_units,
        minimum_value=minimum_layer_budget,
    )

    rank_pattern = {}
    layer_ranks = {}
    for layer_idx in range(num_layers):
        module_names = [name for name, _ in modules_by_layer[layer_idx]]
        per_module_ranks = _distribute_integer_values(
            [1.0] * len(module_names),
            layer_rank_units[layer_idx],
            minimum_value=1,
        )
        layer_ranks[layer_idx] = {module_name: rank for module_name, rank in zip(module_names, per_module_ranks)}
        for module_name, rank in layer_ranks[layer_idx].items():
            rank_pattern[module_name] = int(rank)
    return rank_pattern, layer_ranks


@torch.no_grad()
def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def probe_gradient_norms(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    max_steps: int | None = None,
) -> Dict[int, float]:
    model.train()
    grads_by_layer = defaultdict(float)
    step_count = 0

    for batch in dataloader:
        step_count += 1
        inputs = _move_batch_to_device(batch, device)
        model.zero_grad(set_to_none=True)
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

        for module_name, module in iter_named_target_linear_modules(model):
            if module.weight.grad is None:
                continue
            layer_idx = infer_layer_index(module_name)
            grads_by_layer[layer_idx] += module.weight.grad.detach().norm(p=2).item()

        if max_steps is not None and step_count >= max_steps:
            break

    model.zero_grad(set_to_none=True)
    return dict(grads_by_layer)


def compute_gradaware_rank_pattern(
    model: torch.nn.Module,
    gradient_norms: Dict[int, float],
    base_r: int = DEFAULT_LORA_R,
) -> Tuple[Dict[str, int], Dict[int, Dict[str, int]]]:
    layer_stack = get_transformer_layer_stack(model)
    num_layers = len(layer_stack)
    target_modules = list(iter_named_target_linear_modules(model))
    if not target_modules:
        raise ValueError("No target attention modules found for GradAware-LoRA.")

    modules_by_layer = defaultdict(list)
    for module_name, module in target_modules:
        modules_by_layer[infer_layer_index(module_name)].append((module_name, module))

    minimum_layer_budget = min(len(modules_by_layer[layer_idx]) for layer_idx in range(num_layers))
    total_rank_units = base_r * len(target_modules)
    layer_weights = [math.sqrt(max(gradient_norms.get(layer_idx, 0.0), 0.0)) for layer_idx in range(num_layers)]
    layer_rank_units = _distribute_integer_values(
        layer_weights,
        total_rank_units,
        minimum_value=minimum_layer_budget,
    )

    rank_pattern = {}
    layer_ranks = {}
    for layer_idx in range(num_layers):
        module_names = [name for name, _ in modules_by_layer[layer_idx]]
        per_module_ranks = _distribute_integer_values(
            [1.0] * len(module_names),
            layer_rank_units[layer_idx],
            minimum_value=1,
        )
        layer_ranks[layer_idx] = {module_name: rank for module_name, rank in zip(module_names, per_module_ranks)}
        for module_name, rank in layer_ranks[layer_idx].items():
            rank_pattern[module_name] = int(rank)
    return rank_pattern, layer_ranks



def _summarize_layer_ranks(layer_ranks: Dict[int, Dict[str, int]]) -> Dict[int, int]:
    summary = {}
    for layer_idx, module_ranks in layer_ranks.items():
        summary[int(layer_idx)] = int(sum(module_ranks.values()))
    return summary


def apply_lora_with_rank_pattern(
    model: torch.nn.Module,
    rank_pattern: Dict[str, int] | None = None,
    base_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
):
    target_modules = get_target_module_suffixes(model)
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=base_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        rank_pattern=rank_pattern or {},
    )
    return get_peft_model(model, config)
