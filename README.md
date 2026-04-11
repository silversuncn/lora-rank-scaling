# LoRA Rank Scaling: Post-hoc Rank-Sensitivity Analysis

This repository contains the experimental data and source code for the paper:

> **Choosing LoRA Rank in a Fixed Benchmark: Post-hoc Rank-Sensitivity Analysis on GLUE-style Tasks**
> Yaowen Sun, Qian Zhang, Hai Du — Naval Submarine Academy, Qingdao, China

## Overview

This paper investigates a practical question: which LoRA rank should you actually use? We provide a statistically grounded post-hoc analysis of **1,680 fine-tuning runs** using repeated-measures ANOVA, matched-seed paired tests with Holm correction, bootstrap confidence intervals, and effect sizes.

## Repository Structure

```
├── src/                          # Source code
│   ├── training.py               # LoRA fine-tuning loop
│   ├── run_experiment.py         # Single-run experiment driver
│   ├── constants.py              # Hyperparameter definitions
│   ├── gradaware_lora.py         # GradAware-LoRA implementation
│   ├── aggregate_results.py      # Result aggregation
│   ├── statistical_analysis.py   # ANOVA + pairwise tests
│   ├── posthoc_rank_analysis.py  # Post-hoc rank-sensitivity analysis
│   ├── fit_scaling_law.py        # Log-linear scaling fits
│   ├── generate_figures.py       # Figure generation
│   └── plot_scaling.py           # Scaling curve plots
├── data/                         # Cleaned experimental data
│   ├── results.csv               # 1,680 rows (1,680 unique configs; 0 duplicates)
│   ├── results_aggregated.csv    # Aggregated statistics by condition
│   ├── anova_rank.csv            # Per-cell ANOVA results
│   ├── pairwise_rank_tests.csv   # Matched-seed pairwise comparisons
│   ├── scaling_law_fits.csv      # Fitted scaling-law parameters
│   ├── analysis_report.json      # Summary analysis report
│   └── summary.json              # Experiment summary metadata
└── figures/                      # All paper figures (PNG)
    ├── fig1_scaling_curves_grid.png
    ├── fig2_optimal_rank_heatmap.png
    ├── fig3_rank_winrate_by_n.png
    ├── fig4_effectsize_best_vs_r2.png
    ├── fig5_taskwise_optimal_rank_trends.png
    └── fig6_complexity_vs_optimal_rank.png
```

## Experimental Setup

### Grid Design (1,680 runs)

| Dimension | Values |
|---|---|
| **Models** | BERT-base, RoBERTa-base |
| **Tasks** | SST-2, MRPC, QNLI, RTE (GLUE) |
| **LoRA Ranks** | 2, 4, 8, 16, 32, 64 |
| **Sample Sizes** | 50, 100, 200, 500, 1,000, 2,000, 5,000 |
| **Seeds** | 42, 123, 456, 789, 1024 |

**Full grid**: 2 models × 4 tasks × 6 ranks × 7 sample sizes × 5 seeds = **1,680 runs** (0 duplicates, 0 missing).

### Data Format

`data/results.csv` columns:

| Column | Description |
|---|---|
| `model` | Pretrained model identifier |
| `task` | GLUE task name |
| `n` | Training sample size |
| `rank` | LoRA rank |
| `seed` | Random seed |
| `epochs` | Training epochs |
| `accuracy` | Evaluation accuracy |
| `trainable_params` | Number of trainable parameters |
| `trainable_pct` | Percentage of trainable parameters |

## Hardware & Environment

All experiments were conducted on a single workstation:

| Component | Specification |
|---|---|
| CPU | Intel Core i9-12900K (16C/24T) |
| RAM | 128 GB DDR5 |
| GPU | NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM) |
| OS | Ubuntu 22.04 (WSL2) |

### Software Versions

| Package | Version |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| Transformers | 5.4.0 |
| PEFT | 0.18.1 |
| Datasets | 4.8.4 |
| scikit-learn | 1.8.0 |

## Key Results

1. **Rank matters, but not universally**: 29 of 56 valid inferential cells are significant after FDR correction.
2. **Mid-to-high ranks dominate**: Ranks 32 and 64 each win 17 unique-optimum cells; rank 16 wins 14.
3. **Rank 32 is the strongest cost-aware default**: Lowest mean regret (0.0102 accuracy points) relative to the best observed rank, at 9.18× the parameter cost of rank 2.

## Requirements

```
torch>=2.11.0
transformers>=5.4.0
peft>=0.18.1
datasets>=4.8.0
scikit-learn>=1.8.0
scipy
matplotlib
seaborn
numpy
pandas
```

## Reproducing the Analysis

```bash
# Install dependencies
pip install torch transformers datasets peft scipy pandas matplotlib seaborn scikit-learn

# Run a single experiment
python src/run_experiment.py --model bert-base-uncased --task sst2 --rank 16 --n 1000 --seed 42

# Aggregate results
python src/aggregate_results.py

# Statistical analysis (ANOVA + pairwise tests)
python src/statistical_analysis.py

# Post-hoc rank analysis
python src/posthoc_rank_analysis.py

# Generate figures
python src/generate_figures.py
```

## Citation

```bibtex
@article{sun2026lorarankscaling,
  title={Choosing LoRA Rank in a Fixed Benchmark: Post-hoc Rank-Sensitivity Analysis on Encoder-Only Architectures},
  author={Sun, Yaowen and Zhang, Qian and Du, Hai},
  year={2026}
}
```

## License

This repository is provided for academic reproducibility purposes. Please cite the paper if you use this code or data.
