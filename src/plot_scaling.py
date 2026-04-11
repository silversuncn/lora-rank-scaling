#!/usr/bin/env python3
"""plot_scaling.py — Generate publication figures for PEFT scaling law paper.

Generates: Fig 1 (Scaling curves), Fig 2 (Crossover), Fig 3 (Heatmap),
           Fig 4 (GradAware delta), Fig 5 (Decoder-only validation).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Method labels defined inline below

# Style constants — will fall back to defaults if constants not found
METHOD_LABELS = {
    "gradaware_lora": "GradAware-LoRA",
    "lora": "LoRA",
    "topheavy_lora": "TopHeavy-LoRA",
    "bitfit": "BitFit",
    "full_ft": "Full FT",
}
METHOD_COLORS = {
    "gradaware_lora": "#1b9e77",
    "lora": "#7570b3",
    "topheavy_lora": "#d95f02",
    "bitfit": "#66a61e",
    "full_ft": "#e7298a",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams.update({"font.size": 11, "figure.dpi": 150})


def load_data(results_path):
    """Load and prepare results dataframe."""
    df = pd.DataFrame(json.loads(Path(results_path).read_text()))
    return df


def fig1_scaling_curves(df, fits_data, output_dir):
    """Fig 1: Scaling curves with power law fits."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    methods = df["method"].unique()
    for method in methods:
        subset = df[df["method"] == method]
        grouped = subset.groupby("train_subset_size")["primary_metric"].agg(["mean", "std"])
        x = grouped.index.values
        y = grouped["mean"].values
        yerr = grouped["std"].values
        
        label = METHOD_LABELS.get(method, method)
        color = METHOD_COLORS.get(method, None)
        ax.errorbar(x, y, yerr=yerr, marker="o", label=label, color=color,
                    capsize=3, markersize=5, linewidth=1.5)
        
        # Overlay fitted curve if available
        if fits_data and method in fits_data.get("fits", {}):
            fit = fits_data["fits"][method]
            x_fit = np.linspace(min(x), max(x), 100)
            y_fit = 1.0 - (fit["a"] * np.power(x_fit, -fit["b"]) + fit["c"])
            ax.plot(x_fit, y_fit, "--", color=color, alpha=0.5, linewidth=1)

    ax.set_xscale("log")
    ax.set_xlabel("Training Samples (N)")
    ax.set_ylabel("Performance (Primary Metric)")
    ax.set_title("PEFT Scaling Curves with Power Law Fits")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_scaling_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig1_scaling_curves.png")


def fig2_crossover(df, fits_data, output_dir):
    """Fig 2: Efficiency crossover points."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for method in ["lora", "gradaware_lora", "bitfit", "topheavy_lora"]:
        subset = df[df["method"] == method]
        if subset.empty:
            continue
        grouped = subset.groupby("train_subset_size")["primary_metric"].mean()
        label = METHOD_LABELS.get(method, method)
        color = METHOD_COLORS.get(method, None)
        ax.plot(grouped.index, grouped.values, marker="o", label=label, color=color,
                markersize=5, linewidth=1.5)
    
    # Full FT as reference
    ft_subset = df[df["method"] == "full_ft"]
    if not ft_subset.empty:
        grouped_ft = ft_subset.groupby("train_subset_size")["primary_metric"].mean()
        ax.plot(grouped_ft.index, grouped_ft.values, marker="s", label="Full FT",
                color=METHOD_COLORS.get("full_ft", "gray"), linewidth=2, linestyle="--")
    
    # Mark crossover points
    if fits_data:
        for key, n_cross in fits_data.get("crossovers", {}).items():
            ax.axvline(x=n_cross, linestyle=":", alpha=0.5, color="gray")
            ax.annotate(f"N={n_cross:.0f}", xy=(n_cross, ax.get_ylim()[0]),
                       fontsize=8, ha="center", va="bottom", rotation=90)

    ax.set_xscale("log")
    ax.set_xlabel("Training Samples (N)")
    ax.set_ylabel("Performance")
    ax.set_title("Efficiency Crossover: PEFT vs Full Fine-Tuning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_crossover.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig2_crossover.png")


def fig3_heatmap(df, output_dir):
    """Fig 3: Parameter efficiency heatmap."""
    pivot = df.groupby(["method", "train_subset_size"]).agg(
        perf=("primary_metric", "mean"),
        params=("trainable_percentage", "mean"),
    ).reset_index()
    pivot["efficiency"] = pivot["perf"] / (pivot["params"] + 1e-6)
    
    heatmap_data = pivot.pivot(index="method", columns="train_subset_size", values="efficiency")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    im = ax.imshow(heatmap_data.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in heatmap_data.index])
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Method")
    ax.set_title("Parameter Efficiency (Performance / Trainable %)")
    plt.colorbar(im, ax=ax, label="Efficiency Score")
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig3_heatmap.png")


def fig4_gradaware_delta(df, output_dir):
    """Fig 4: GradAware-LoRA advantage over LoRA by sample size."""
    lora_perf = df[df["method"] == "lora"].groupby("train_subset_size")["primary_metric"].agg(["mean", "std"])
    ga_perf = df[df["method"] == "gradaware_lora"].groupby("train_subset_size")["primary_metric"].agg(["mean", "std"])
    
    common = lora_perf.index.intersection(ga_perf.index)
    if len(common) == 0:
        print("  Skipping fig4: no common sample sizes")
        return
    
    delta = ga_perf.loc[common, "mean"] - lora_perf.loc[common, "mean"]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    colors = ["#1b9e77" if d >= 0 else "#e7298a" for d in delta.values]
    ax.bar(range(len(common)), delta.values * 100, color=colors, edgecolor="white")
    ax.set_xticks(range(len(common)))
    ax.set_xticklabels(common)
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Δ Performance (GradAware - LoRA) %")
    ax.set_title("GradAware-LoRA Advantage vs Standard LoRA")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_gradaware_delta.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig4_gradaware_delta.png")


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures.")
    parser.add_argument("--results", type=str, default="results.json")
    parser.add_argument("--fits", type=str, default="scaling_law_params.json")
    parser.add_argument("--output_dir", type=str, default="figures")
    args = parser.parse_args()
    
    results_path = PROJECT_ROOT / args.results
    if not results_path.exists():
        results_path = PROJECT_ROOT / "artifacts" / "final_runs" / args.results
    
    fits_path = PROJECT_ROOT / args.fits
    fits_data = json.loads(fits_path.read_text()) if fits_path.exists() else None
    
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from {results_path}")
    df = load_data(results_path)
    print(f"  {len(df)} runs loaded")
    
    print("Generating figures...")
    fig1_scaling_curves(df, fits_data, output_dir)
    fig2_crossover(df, fits_data, output_dir)
    fig3_heatmap(df, output_dir)
    fig4_gradaware_delta(df, output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
