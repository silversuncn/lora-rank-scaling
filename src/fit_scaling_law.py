#!/usr/bin/env python3
"""fit_scaling_law.py — Power law fitting for PEFT scaling experiments.

Fits E = a * N^(-b) + c for each method and computes N_crossover.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, brentq


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def power_law(N, a, b, c):
    """Error rate model: E = a * N^(-b) + c."""
    return a * np.power(N, -b) + c


def fit_method(df, method, metric_col="primary_metric"):
    """Fit power law for a single method across sample sizes."""
    subset = df[df["method"] == method]
    grouped = subset.groupby("train_subset_size")[metric_col].agg(["mean", "std", "count"])
    grouped = grouped.sort_index()

    x = grouped.index.values.astype(float)
    y = 1.0 - grouped["mean"].values  # convert accuracy to error rate

    if len(x) < 3:
        return None

    try:
        popt, pcov = curve_fit(
            power_law, x, y,
            p0=[1.0, 0.5, 0.05],
            bounds=([0, 0, 0], [100, 5, 1]),
            maxfev=10000,
        )
        y_pred = power_law(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "method": method,
            "a": float(popt[0]),
            "b": float(popt[1]),
            "c": float(popt[2]),
            "r2": float(r2),
            "n_points": len(x),
            "sample_sizes": x.tolist(),
            "error_rates": y.tolist(),
            "fitted_values": y_pred.tolist(),
            "perr": np.sqrt(np.diag(pcov)).tolist(),
        }
    except Exception as e:
        print(f"  Warning: fitting failed for {method}: {e}")
        return None


def find_crossover(fit_a, fit_b, search_range=(10, 50000)):
    """Find N where two methods have equal error rate."""
    if fit_a is None or fit_b is None:
        return None
    try:
        f = lambda N: (power_law(N, fit_a["a"], fit_a["b"], fit_a["c"])
                     - power_law(N, fit_b["a"], fit_b["b"], fit_b["c"]))
        # Check if sign change exists
        f_lo, f_hi = f(search_range[0]), f(search_range[1])
        if f_lo * f_hi > 0:
            return None  # No crossover in range
        n_cross = brentq(f, search_range[0], search_range[1])
        return float(n_cross)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Fit power law scaling curves.")
    parser.add_argument("--results", type=str, default="results.json")
    parser.add_argument("--output", type=str, default="scaling_law_params.json")
    args = parser.parse_args()

    results_path = PROJECT_ROOT / args.results
    if not results_path.exists():
        # Try artifacts directory
        results_path = PROJECT_ROOT / "artifacts" / "final_runs" / args.results
    
    print(f"Loading results from {results_path}")
    df = pd.DataFrame(json.loads(results_path.read_text()))
    print(f"  Loaded {len(df)} runs")

    # Fit each method
    methods = df["method"].unique()
    fits = {}
    for method in methods:
        print(f"  Fitting {method}...")
        result = fit_method(df, method)
        if result:
            fits[method] = result
            print(f"    a={result['a']:.4f}, b={result['b']:.4f}, c={result['c']:.4f}, R²={result['r2']:.4f}")
        else:
            print(f"    SKIPPED (insufficient data)")

    # Find crossover points
    crossovers = {}
    if "full_ft" in fits:
        for method in methods:
            if method != "full_ft" and method in fits:
                n_cross = find_crossover(fits[method], fits["full_ft"])
                if n_cross is not None:
                    crossovers[f"{method}_vs_full_ft"] = n_cross
                    print(f"  Crossover {method} vs full_ft: N = {n_cross:.0f}")

    output = {
        "fits": fits,
        "crossovers": crossovers,
        "metadata": {
            "total_runs": len(df),
            "methods": list(methods),
            "formula": "E = a * N^(-b) + c",
        },
    }

    output_path = PROJECT_ROOT / args.output
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
