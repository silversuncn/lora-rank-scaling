#!/usr/bin/env python3
"""统计分析 + Scaling Law 拟合 + LaTeX 表格生成"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "artifacts" / "results.csv")
OUT = ROOT / "artifacts"
print(f"=== Statistical Analysis on {len(df)} runs ===\n")

MODEL_NAMES = {'bert-base-uncased':'BERT','roberta-base':'RoBERTa','microsoft/deberta-v3-base':'DeBERTa-v3'}

# ============================================================
# 1. ANOVA: rank 对 accuracy 的显著性
# ============================================================
print("1. One-way ANOVA: effect of rank on accuracy")
anova_results = []
for m in sorted(df.model.unique()):
    for t in sorted(df.task.unique()):
        sub = df[(df.model==m) & (df.task==t)]
        groups = [sub[sub['rank']==r].accuracy.values for r in sorted(sub['rank'].unique())]
        f_stat, p_val = stats.f_oneway(*groups)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        anova_results.append({
            'model': MODEL_NAMES.get(m,m), 'task': t.upper(),
            'F': round(f_stat, 2), 'p': f'{p_val:.2e}', 'sig': sig
        })
        print(f"  {MODEL_NAMES.get(m,m):10s} {t:5s}: F={f_stat:8.2f}, p={p_val:.2e} {sig}")
pd.DataFrame(anova_results).to_csv(OUT / "anova_rank.csv", index=False)

# ============================================================
# 2. Scaling Law 拟合: acc = a * log(n) + b (对数线性)
# ============================================================
print("\n2. Scaling Law fitting: acc = a * log(n) + b")
def log_linear(x, a, b):
    return a * np.log(x) + b

def power_law(x, a, b, c):
    return a * np.power(x, b) + c

fit_results = []
for m in sorted(df.model.unique()):
    for t in sorted(df.task.unique()):
        for r in sorted(df['rank'].unique()):
            sub = df[(df.model==m) & (df.task==t) & (df['rank']==r)]
            agg = sub.groupby('n').accuracy.mean().reset_index()
            x, y = agg.n.values.astype(float), agg.accuracy.values
            try:
                popt, _ = curve_fit(log_linear, x, y, maxfev=5000)
                y_pred = log_linear(x, *popt)
                r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
                fit_results.append({
                    'model': MODEL_NAMES.get(m,m), 'task': t.upper(), 'rank': r,
                    'a': round(popt[0], 4), 'b': round(popt[1], 4),
                    'R2': round(r2, 4), 'equation': f'acc = {popt[0]:.4f}*ln(n) + {popt[1]:.4f}'
                })
            except Exception:
                pass
fit_df = pd.DataFrame(fit_results)
fit_df.to_csv(OUT / "scaling_law_fits.csv", index=False)
print(f"  Fitted {len(fit_df)} curves, mean R²={fit_df.R2.mean():.4f}")
print(f"  Best fit: {fit_df.loc[fit_df.R2.idxmax()].to_dict()}")

# ============================================================
# 3. Pairwise rank comparison: is r=64 significantly better than r=2?
# ============================================================
print("\n3. Pairwise rank comparison (paired t-test, r_high vs r_low)")
pairs = [(2,64),(2,32),(4,32),(8,32),(16,64)]
pair_results = []
for r_lo, r_hi in pairs:
    for m in sorted(df.model.unique()):
        lo = df[(df.model==m) & (df['rank']==r_lo)].groupby(['task','n','seed']).accuracy.mean().reset_index()
        hi = df[(df.model==m) & (df['rank']==r_hi)].groupby(['task','n','seed']).accuracy.mean().reset_index()
        merged = lo.merge(hi, on=['task','n','seed'], suffixes=('_lo','_hi'))
        if len(merged) > 0:
            t_stat, p_val = stats.ttest_rel(merged.accuracy_hi, merged.accuracy_lo)
            diff = (merged.accuracy_hi - merged.accuracy_lo).mean()
            pair_results.append({
                'model': MODEL_NAMES.get(m,m), 'r_lo': r_lo, 'r_hi': r_hi,
                'mean_diff': round(diff, 4), 't': round(t_stat, 2), 'p': f'{p_val:.2e}',
                'sig': "***" if p_val<0.001 else ("**" if p_val<0.01 else ("*" if p_val<0.05 else "ns"))
            })
pd.DataFrame(pair_results).to_csv(OUT / "pairwise_rank_tests.csv", index=False)
print(f"  {len(pair_results)} pairwise comparisons done")

# ============================================================
# 4. LaTeX 表格：主结果表（model × task × rank, n=1000 为例）
# ============================================================
print("\n4. Generating LaTeX tables...")
agg = df.groupby(['model','task','rank','n']).agg(
    acc_mean=('accuracy','mean'), acc_std=('accuracy','std')
).reset_index()

# Main results table (n=1000)
latex_lines = []
latex_lines.append(r"\begin{table}[h]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Accuracy by Model, Task, and LoRA Rank (n=1000)}")
latex_lines.append(r"\label{tab:main_results}")
ranks = sorted(df['rank'].unique())
cols = "l l " + " ".join(["c"] * len(ranks))
latex_lines.append(r"\begin{tabular}{" + cols + "}")
latex_lines.append(r"\toprule")
header = "Model & Task & " + " & ".join([f"r={r}" for r in ranks]) + r" \\"
latex_lines.append(header)
latex_lines.append(r"\midrule")
for m in sorted(df.model.unique()):
    for t in ['sst2','mrpc','qnli','rte','cola']:
        cells = []
        best_acc = 0
        for r in ranks:
            sub = agg[(agg.model==m) & (agg.task==t) & (agg['rank']==r) & (agg.n==1000)]
            if len(sub) > 0:
                acc = sub.acc_mean.values[0]
                std = sub.acc_std.values[0]
                best_acc = max(best_acc, acc)
                cells.append((acc, std, r))
            else:
                cells.append((0, 0, r))
        row_cells = []
        for acc, std, r in cells:
            s = f"{acc:.3f}" + r"{\scriptsize$\pm$" + f"{std:.3f}" + "}"
            if acc == best_acc and acc > 0:
                s = r"\textbf{" + s + "}"
            row_cells.append(s)
        line = f"{MODEL_NAMES.get(m,m)} & {t.upper()} & " + " & ".join(row_cells) + r" \\"
        latex_lines.append(line)
    latex_lines.append(r"\midrule")
latex_lines[-1] = r"\bottomrule"
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\end{table}")
latex_text = "\n".join(latex_lines)
(OUT / "table_main_results.tex").write_text(latex_text)
print(f"  -> table_main_results.tex")

# Scaling law fit table
latex2 = []
latex2.append(r"\begin{table}[h]")
latex2.append(r"\centering")
latex2.append(r"\caption{Log-Linear Scaling Law Fits: $\text{acc} = a \cdot \ln(n) + b$}")
latex2.append(r"\label{tab:scaling_fits}")
latex2.append(r"\begin{tabular}{l l r r r r}")
latex2.append(r"\toprule")
latex2.append(r"Model & Task & Rank & $a$ & $b$ & $R^2$ \\")
latex2.append(r"\midrule")
for _, row in fit_df.iterrows():
    latex2.append(f"{row.model} & {row.task} & {row['rank']} & {row.a:.4f} & {row.b:.4f} & {row.R2:.3f}" + r" \\")
latex2.append(r"\bottomrule")
latex2.append(r"\end{tabular}")
latex2.append(r"\end{table}")
(OUT / "table_scaling_fits.tex").write_text("\n".join(latex2))
print(f"  -> table_scaling_fits.tex")

# ============================================================
# 5. 综合分析报告
# ============================================================
print("\n5. Generating analysis report...")
report = {
    "experiment_summary": {
        "total_runs": len(df),
        "models": list(df.model.unique()),
        "tasks": list(df.task.unique()),
        "ranks": sorted(df['rank'].unique().tolist()),
        "sample_sizes": sorted(df.n.unique().tolist()),
        "seeds": sorted(df.seed.unique().tolist()),
    },
    "key_findings": {
        "rank_effect_significant": all(r['sig'] != 'ns' for r in anova_results),
        "anova_all_significant": sum(1 for r in anova_results if r['sig'] != 'ns'),
        "anova_total": len(anova_results),
        "scaling_law_mean_R2": round(fit_df.R2.mean(), 4),
        "scaling_law_median_R2": round(fit_df.R2.median(), 4),
        "best_overall_rank": int(df.groupby('rank').accuracy.mean().idxmax()),
        "rank_accuracy_ranking": {int(r): round(v, 4) for r, v in df.groupby('rank').accuracy.mean().sort_values(ascending=False).items()},
    },
    "per_model_best_rank": {
        MODEL_NAMES.get(m,m): int(df[df.model==m].groupby('rank').accuracy.mean().idxmax())
        for m in df.model.unique()
    },
    "per_task_best_rank": {
        t: int(df[df.task==t].groupby('rank').accuracy.mean().idxmax())
        for t in df.task.unique()
    },
}
(OUT / "analysis_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(f"  -> analysis_report.json")

print("\n=== Statistical Analysis Complete ===")
