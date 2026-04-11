from __future__ import annotations

# Paper 4: LoRA Rank Scaling Laws
# Only LoRA method — this study focuses on rank dimension
SUPPORTED_METHODS = ["lora"]

SUPPORTED_TASKS = ["sst2", "mrpc", "cola", "qnli", "rte"]

SUPPORTED_MODELS = [
    "bert-base-uncased",
    "roberta-base",
    "microsoft/deberta-v3-base",
]

# LoRA rank values to sweep
LORA_RANKS = [2, 4, 8, 16, 32, 64]

# Sample sizes for scaling law experiments
SAMPLE_SIZES = [50, 100, 200, 500, 1000, 2000, 5000]

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
}

TASK_PRIMARY_METRIC = {
    "cola": "matthews_correlation",
    "mrpc": "f1",
    "qnli": "accuracy",
    "rte": "accuracy",
    "sst2": "accuracy",
}

DEFAULT_TRAIN_SUBSET = 500
DEFAULT_MAX_LENGTH = 256
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.0

# Seeds for reproducibility
SEEDS = [42, 123, 456, 789, 1024]
