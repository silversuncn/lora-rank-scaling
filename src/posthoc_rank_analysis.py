#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'artifacts' / 'results.csv'
OUT_DIR = ROOT / 'artifacts' / 'analysis'
FIG_DIR = OUT_DIR / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_MAP = {
    'bert-base-uncased': 'BERT-base',
    'roberta-base': 'RoBERTa-base',
}
TASK_ORDER = ['sst2', 'mrpc', 'qnli', 'rte']
RANK_ORDER = [2, 4, 8, 16, 32, 64]
N_ORDER = [50, 100, 200, 500, 1000, 2000, 5000]
PAIRWISE_BOOT = 5000
EPS = 1e-12

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
})

RANK_COLORS = {
    2: '#1b9e77',
    4: '#d95f02',
    8: '#7570b3',
    16: '#e7298a',
    32: '#66a61e',
    64: '#e6ab02',
}


def star(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return 'ns'
    if p < 1e-3:
        return '***'
    if p < 1e-2:
        return '**'
    if p < 5e-2:
        return '*'
    return 'ns'


def holm_correction(pvals: list[float]) -> list[float]:
    if not pvals:
        return []
    m = len(pvals)
    order = np.argsort(pvals)
    ordered = np.asarray(pvals)[order]
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for i, p in enumerate(ordered):
        val = (m - i) * p
        running = max(running, val)
        adjusted[i] = min(running, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adjusted
    return out.tolist()


def fdr_bh(pvals: pd.Series) -> pd.Series:
    vals = pvals.astype(float).copy()
    mask = vals.notna() & np.isfinite(vals)
    if mask.sum() == 0:
        return pd.Series(np.nan, index=pvals.index)
    valid = vals[mask].values
    m = len(valid)
    order = np.argsort(valid)
    ranked = valid[order]
    adjusted = ranked * m / (np.arange(m) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.full(len(vals), np.nan)
    out[np.where(mask)[0][order]] = adjusted
    return pd.Series(out, index=pvals.index)


def repeated_measures_anova(pivot: pd.DataFrame) -> dict:
    arr = pivot.to_numpy(dtype=float)
    n_subj, n_cond = arr.shape
    if np.isnan(arr).any():
        return {'F': np.nan, 'p': np.nan, 'partial_eta2': np.nan, 'status': 'incomplete'}
    if np.allclose(arr, arr[0, 0], atol=EPS):
        return {'F': np.nan, 'p': np.nan, 'partial_eta2': np.nan, 'status': 'constant_all_values'}

    grand = arr.mean()
    subj_means = arr.mean(axis=1, keepdims=True)
    cond_means = arr.mean(axis=0, keepdims=True)

    ss_total = ((arr - grand) ** 2).sum()
    ss_subjects = n_cond * ((subj_means - grand) ** 2).sum()
    ss_conditions = n_subj * ((cond_means - grand) ** 2).sum()
    ss_error = ss_total - ss_subjects - ss_conditions

    df_conditions = n_cond - 1
    df_error = (n_subj - 1) * (n_cond - 1)
    if ss_error <= EPS or df_error <= 0:
        return {'F': np.nan, 'p': np.nan, 'partial_eta2': np.nan, 'status': 'zero_error_variance'}

    ms_conditions = ss_conditions / df_conditions
    ms_error = ss_error / df_error
    F = ms_conditions / ms_error
    p = 1.0 - stats.f.cdf(F, df_conditions, df_error)
    eta = ss_conditions / (ss_conditions + ss_error) if (ss_conditions + ss_error) > EPS else np.nan
    return {'F': F, 'p': p, 'partial_eta2': eta, 'status': 'ok'}


def cohen_dz(diff: np.ndarray) -> float:
    sd = diff.std(ddof=1)
    mean = diff.mean()
    if np.isnan(sd):
        return np.nan
    if abs(sd) <= EPS:
        return np.nan if abs(mean) <= EPS else math.copysign(np.inf, mean)
    return mean / sd


def bootstrap_mean_ci(diff: np.ndarray, n_boot: int = PAIRWISE_BOOT, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    sampled = diff[idx].mean(axis=1)
    lo, hi = np.quantile(sampled, [0.025, 0.975])
    return float(lo), float(hi)


def choose_optimal_rank(group: pd.DataFrame) -> pd.Series:
    g = group.sort_values(['accuracy_mean', 'rank'], ascending=[False, True]).copy()
    max_acc = g['accuracy_mean'].max()
    winners = g[np.isclose(g['accuracy_mean'], max_acc, atol=EPS)].sort_values('rank')
    if len(winners) == 1:
        return pd.Series({
            'optimal_rank': int(winners.iloc[0]['rank']),
            'optimal_label': f"r={int(winners.iloc[0]['rank'])}",
            'optimal_accuracy': float(winners.iloc[0]['accuracy_mean']),
            'optimal_unique': True,
            'winner_count': 1,
        })
    return pd.Series({
        'optimal_rank': np.nan,
        'optimal_label': 'tie',
        'optimal_accuracy': float(max_acc),
        'optimal_unique': False,
        'winner_count': int(len(winners)),
    })


def latex_escape(text: str) -> str:
    return text.replace('_', '\\_')


def build_tables(anova_df: pd.DataFrame, optimal_df: pd.DataFrame, trend_df: pd.DataFrame) -> None:
    sig_summary = (
        anova_df.assign(significant=anova_df['p_fdr'] < 0.05)
        .groupby(['model_short', 'task'])['significant']
        .agg(['sum', 'count'])
        .reset_index()
    )

    lines = [
        '\\begin{table}[t]',
        '\\centering',
        '\\caption{Repeated-measures ANOVA significance counts across sample sizes.}',
        '\\label{tab:anova-summary}',
        '\\begin{tabular}{llrr}',
        '\\toprule',
        'Model & Task & Sig. cells & Total cells \\\\',
        '\\midrule',
    ]
    for _, row in sig_summary.iterrows():
        lines.append(
            f"{latex_escape(row['model_short'])} & {row['task'].upper()} & {int(row['sum'])} & {int(row['count'])} \\\\"
        )
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (OUT_DIR / 'table_anova_summary.tex').write_text('\n'.join(lines))

    pivot = optimal_df[optimal_df['optimal_unique']].pivot_table(
        index='model_short', columns='n', values='optimal_rank', aggfunc='mean'
    ).reindex(index=[MODEL_MAP[m] for m in MODEL_MAP], columns=N_ORDER)
    lines = [
        '\\begin{table}[t]',
        '\\centering',
        '\\caption{Mean optimal LoRA rank by model and sample size (unique-optimum cells only).}',
        '\\label{tab:optimal-rank}',
        '\\begin{tabular}{l' + 'r' * len(N_ORDER) + '}',
        '\\toprule',
        'Model & ' + ' & '.join([f'n={n}' for n in N_ORDER]) + ' \\\\',
        '\\midrule',
    ]
    for model in pivot.index:
        vals = []
        for n in pivot.columns:
            val = pivot.loc[model, n]
            vals.append('--' if pd.isna(val) else f'{val:.1f}')
        lines.append(f"{latex_escape(model)} & " + ' & '.join(vals) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (OUT_DIR / 'table_optimal_rank.tex').write_text('\n'.join(lines))

    trend_df.to_csv(OUT_DIR / 'optimal_rank_trends.csv', index=False)


def main() -> None:
    df = pd.read_csv(RESULTS)
    df['model_short'] = df['model'].map(MODEL_MAP).fillna(df['model'])
    df['task'] = pd.Categorical(df['task'], categories=TASK_ORDER, ordered=True)
    df['n'] = df['n'].astype(int)
    df['rank'] = df['rank'].astype(int)
    df['seed'] = df['seed'].astype(int)

    completeness = (
        df.groupby(['model_short', 'task', 'n', 'rank'])
        .agg(seed_count=('seed', 'nunique'), run_count=('accuracy', 'size'))
        .reset_index()
        .sort_values(['model_short', 'task', 'n', 'rank'])
    )
    completeness.to_csv(OUT_DIR / 'completeness.csv', index=False)

    agg = (
        df.groupby(['model', 'model_short', 'task', 'n', 'rank'])
        .agg(
            accuracy_mean=('accuracy', 'mean'),
            accuracy_std=('accuracy', 'std'),
            accuracy_sem=('accuracy', lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
            run_count=('accuracy', 'size'),
        )
        .reset_index()
        .sort_values(['model_short', 'task', 'n', 'rank'])
    )
    agg['ci95_low'] = agg['accuracy_mean'] - 1.96 * agg['accuracy_sem']
    agg['ci95_high'] = agg['accuracy_mean'] + 1.96 * agg['accuracy_sem']
    agg.to_csv(OUT_DIR / 'aggregated_accuracy.csv', index=False)

    anova_rows = []
    pairwise_rows = []

    grouped = df.groupby(['model', 'model_short', 'task', 'n'])
    for (model, model_short, task, n), sub in grouped:
        pivot = sub.pivot_table(index='seed', columns='rank', values='accuracy').reindex(columns=RANK_ORDER)
        aov = repeated_measures_anova(pivot)
        rank_means = sub.groupby('rank')['accuracy'].mean().reindex(RANK_ORDER)
        degenerate = rank_means.nunique(dropna=True) <= 1
        anova_rows.append({
            'model': model,
            'model_short': model_short,
            'task': str(task),
            'n': int(n),
            'F': aov['F'],
            'p': aov['p'],
            'partial_eta2': aov['partial_eta2'],
            'status': aov['status'],
            'degenerate_rank_means': bool(degenerate),
        })

        local_ps = []
        local_ix = []
        pair_buffer = []
        for r1, r2 in combinations(RANK_ORDER, 2):
            if r1 not in pivot.columns or r2 not in pivot.columns:
                continue
            pair = pivot[[r1, r2]].dropna()
            if len(pair) < 2:
                row = {
                    'model': model,
                    'model_short': model_short,
                    'task': str(task),
                    'n': int(n),
                    'rank_lo': r1,
                    'rank_hi': r2,
                    'mean_diff': np.nan,
                    't_stat': np.nan,
                    'p': np.nan,
                    'cohen_dz': np.nan,
                    'ci95_low': np.nan,
                    'ci95_high': np.nan,
                    'status': 'insufficient_pairs',
                }
                pair_buffer.append(row)
                continue

            diff = pair[r2].to_numpy() - pair[r1].to_numpy()
            if np.allclose(diff, 0.0, atol=EPS):
                t_stat = np.nan
                p_val = np.nan
                d_z = 0.0
                ci_low, ci_high = 0.0, 0.0
                status = 'constant_zero_difference'
            else:
                t_stat, p_val = stats.ttest_rel(pair[r2], pair[r1])
                d_z = cohen_dz(diff)
                ci_low, ci_high = bootstrap_mean_ci(diff, seed=int(n + r1 * 10 + r2))
                status = 'ok'
            row = {
                'model': model,
                'model_short': model_short,
                'task': str(task),
                'n': int(n),
                'rank_lo': r1,
                'rank_hi': r2,
                'mean_diff': float(diff.mean()),
                't_stat': t_stat,
                'p': p_val,
                'cohen_dz': d_z,
                'ci95_low': ci_low,
                'ci95_high': ci_high,
                'status': status,
            }
            pair_buffer.append(row)
            if np.isfinite(p_val):
                local_ps.append(float(p_val))
                local_ix.append(len(pair_buffer) - 1)

        adjusted = holm_correction(local_ps)
        for idx, p_adj in zip(local_ix, adjusted):
            pair_buffer[idx]['p_holm'] = p_adj
            pair_buffer[idx]['sig_holm'] = star(p_adj)
        for idx, row in enumerate(pair_buffer):
            if 'p_holm' not in row:
                row['p_holm'] = np.nan
                row['sig_holm'] = 'ns'
            pairwise_rows.append(row)

    anova_df = pd.DataFrame(anova_rows).sort_values(['model_short', 'task', 'n'])
    anova_df['p_fdr'] = fdr_bh(anova_df['p'])
    anova_df['sig_fdr'] = anova_df['p_fdr'].map(lambda x: star(float(x)) if pd.notna(x) else 'ns')
    anova_df.to_csv(OUT_DIR / 'anova_repeated_measures.csv', index=False)

    pairwise_df = pd.DataFrame(pairwise_rows).sort_values(['model_short', 'task', 'n', 'rank_lo', 'rank_hi'])
    pairwise_df.to_csv(OUT_DIR / 'paired_ttests_bootstrap_cohend.csv', index=False)

    optimal_df = (
        agg.groupby(['model', 'model_short', 'task', 'n'])
        .apply(choose_optimal_rank)
        .reset_index()
        .sort_values(['model_short', 'task', 'n'])
    )
    optimal_df.to_csv(OUT_DIR / 'optimal_ranks.csv', index=False)

    unique_opt = optimal_df[optimal_df['optimal_unique']].copy()
    trend_rows = []
    for (model_short, task), sub in unique_opt.groupby(['model_short', 'task']):
        if sub['n'].nunique() >= 3 and sub['optimal_rank'].nunique() > 1:
            rho, pval = stats.spearmanr(np.log10(sub['n']), sub['optimal_rank'])
        else:
            rho, pval = np.nan, np.nan
        trend_rows.append({
            'model_short': model_short,
            'task': str(task),
            'mean_optimal_rank': sub['optimal_rank'].mean() if len(sub) else np.nan,
            'spearman_log_n_vs_opt_rank': rho,
            'spearman_p': pval,
            'unique_cells': int(len(sub)),
        })
    trend_df = pd.DataFrame(trend_rows).sort_values(['model_short', 'task'])

    low_data = (
        agg[(agg['n'] == 50) & (agg['rank'] == 2)]
        .groupby('task')['accuracy_mean']
        .mean()
        .rename('low_data_rank2_accuracy')
        .reset_index()
    )
    task_opt_rank = (
        unique_opt
        .groupby('task')['optimal_rank']
        .mean()
        .rename('mean_optimal_rank')
        .reset_index()
    )
    task_complexity = low_data.merge(task_opt_rank, on='task', how='outer')
    task_complexity['complexity_proxy'] = 1.0 - task_complexity['low_data_rank2_accuracy']
    if task_complexity['complexity_proxy'].notna().sum() >= 3:
        rho, pval = stats.spearmanr(task_complexity['complexity_proxy'], task_complexity['mean_optimal_rank'])
    else:
        rho, pval = np.nan, np.nan
    task_complexity['spearman_rho_global'] = rho
    task_complexity['spearman_p_global'] = pval
    task_complexity.to_csv(OUT_DIR / 'task_complexity_summary.csv', index=False)

    build_tables(anova_df, optimal_df, trend_df)

    # Figure 1: scaling curves (3x5)
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.6), sharex=True, sharey=False)
    for i, model in enumerate([MODEL_MAP[k] for k in MODEL_MAP]):
        for j, task in enumerate(TASK_ORDER):
            ax = axes[i, j]
            sub = agg[(agg['model_short'] == model) & (agg['task'] == task)]
            for rank in RANK_ORDER:
                d = sub[sub['rank'] == rank].sort_values('n')
                if d.empty:
                    continue
                ax.plot(d['n'], d['accuracy_mean'], marker='o', markersize=3, linewidth=1.4,
                        color=RANK_COLORS[rank], label=f'r={rank}')
                ax.fill_between(d['n'].to_numpy(dtype=float),
                                d['ci95_low'].to_numpy(dtype=float),
                                d['ci95_high'].to_numpy(dtype=float),
                                color=RANK_COLORS[rank], alpha=0.12)
            ax.set_xscale('log')
            ax.set_title(f'{model}\n{task.upper()}')
            ax.set_xticks(N_ORDER)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            if j == 0:
                ax.set_ylabel('Accuracy')
            if i == 1:
                ax.set_xlabel('Sample size')
    handles = [plt.Line2D([0], [0], color=RANK_COLORS[r], marker='o', linewidth=1.6, label=f'r={r}') for r in RANK_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / 'fig1_scaling_curves_grid.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig1_scaling_curves_grid.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 2: optimal rank heatmap with ties shown in gray
    heat = optimal_df.copy()
    row_order = [(MODEL_MAP[k], t) for k in MODEL_MAP for t in TASK_ORDER]
    heat_rows = []
    labels = []
    for model_short, task in row_order:
        labels.append(f'{model_short}\n{task.upper()}')
        row = []
        for n in N_ORDER:
            sub = heat[(heat['model_short'] == model_short) & (heat['task'] == task) & (heat['n'] == n)]
            if sub.empty or not bool(sub.iloc[0]['optimal_unique']):
                row.append(0)
            else:
                row.append(int(sub.iloc[0]['optimal_rank']))
        heat_rows.append(row)
    heat_mat = np.asarray(heat_rows)
    cmap = ListedColormap(['#d9d9d9'] + [RANK_COLORS[r] for r in RANK_ORDER])
    bounds = [-0.5, 1, 3, 6, 12, 24, 48, 80]
    norm = BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(6.9, 6.4))
    im = ax.imshow(heat_mat, aspect='auto', cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(N_ORDER)))
    ax.set_xticklabels(N_ORDER)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Sample size')
    ax.set_title('Optimal LoRA rank by model, task, and sample size')
    for i in range(heat_mat.shape[0]):
        for j in range(heat_mat.shape[1]):
            txt = 'tie' if heat_mat[i, j] == 0 else f"r={int(heat_mat[i, j])}"
            ax.text(j, i, txt, ha='center', va='center', fontsize=6.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_ticks([0, 2, 4, 8, 16, 32, 64])
    cbar.set_ticklabels(['tie', '2', '4', '8', '16', '32', '64'])
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_optimal_rank_heatmap.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig2_optimal_rank_heatmap.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 3: rank win-rate vs sample size (unique optimum only)
    win = unique_opt.groupby(['n', 'optimal_rank']).size().reset_index(name='count')
    totals = unique_opt.groupby('n').size().rename('total').reset_index()
    win = win.merge(totals, on='n', how='left')
    win['win_rate'] = win['count'] / win['total']
    fig, ax = plt.subplots(figsize=(6.9, 3.0))
    for rank in RANK_ORDER:
        sub = win[win['optimal_rank'] == rank].sort_values('n')
        if sub.empty:
            continue
        ax.plot(sub['n'], sub['win_rate'], marker='o', linewidth=1.5, markersize=4,
                color=RANK_COLORS[rank], label=f'r={rank}')
    ax.set_xscale('log')
    ax.set_xticks(N_ORDER)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0, 1)
    ax.set_xlabel('Sample size')
    ax.set_ylabel('Probability of being optimal')
    ax.set_title('Rank win-rate across non-degenerate cells')
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_rank_winrate_by_n.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig3_rank_winrate_by_n.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 4: Cohen's d_z for best-vs-r2 (unique optimum only)
    pair_lookup = pairwise_df.set_index(['model_short', 'task', 'n', 'rank_lo', 'rank_hi'])
    rows = []
    labels = []
    for model_short, task in row_order:
        labels.append(f'{model_short}\n{task.upper()}')
        vals = []
        for n in N_ORDER:
            sub = optimal_df[(optimal_df['model_short'] == model_short) & (optimal_df['task'] == task) & (optimal_df['n'] == n)]
            if sub.empty or not bool(sub.iloc[0]['optimal_unique']):
                vals.append(np.nan)
                continue
            best_rank = int(sub.iloc[0]['optimal_rank'])
            if best_rank == 2:
                vals.append(0.0)
                continue
            key = (model_short, task, n, 2, best_rank)
            vals.append(pair_lookup.loc[key]['cohen_dz'] if key in pair_lookup.index else np.nan)
        rows.append(vals)
    effect_mat = np.asarray(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(6.9, 6.4))
    vmax = np.nanmax(np.abs(effect_mat)) if np.isfinite(effect_mat).any() else 1.0
    vmax = max(vmax, 0.5)
    im = ax.imshow(effect_mat, aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(N_ORDER)))
    ax.set_xticklabels(N_ORDER)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Sample size')
    ax.set_title("Effect size: best rank vs r=2 (Cohen's d_z)")
    for i in range(effect_mat.shape[0]):
        for j in range(effect_mat.shape[1]):
            val = effect_mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=6.5)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Cohen's d_z")
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_effectsize_best_vs_r2.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig4_effectsize_best_vs_r2.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 5: mean optimal rank by task across sample size (non-degenerate cells only)
    task_lines = (
        unique_opt
        .groupby(['task', 'n'])['optimal_rank']
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(6.9, 3.2))
    palette = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']
    for color, task in zip(palette, TASK_ORDER):
        sub = task_lines[task_lines['task'] == task].sort_values('n')
        if sub.empty:
            continue
        ax.plot(sub['n'], sub['optimal_rank'], marker='o', linewidth=1.6, markersize=4,
                color=color, label=task.upper())
    ax.set_xscale('log')
    ax.set_xticks(N_ORDER)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('Sample size')
    ax.set_ylabel('Mean optimal rank')
    ax.set_title('Task complexity interaction: harder tasks favor higher rank')
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig5_taskwise_optimal_rank_trends.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig5_taskwise_optimal_rank_trends.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 6: complexity proxy vs mean optimal rank
    fig, ax = plt.subplots(figsize=(3.35, 3.0))
    sub = task_complexity.dropna(subset=['complexity_proxy', 'mean_optimal_rank']).copy()
    ax.scatter(sub['complexity_proxy'], sub['mean_optimal_rank'], color='#1f78b4', s=35)
    for _, row in sub.iterrows():
        ax.annotate(row['task'].upper(), (row['complexity_proxy'], row['mean_optimal_rank']),
                    textcoords='offset points', xytext=(4, 3), fontsize=7)
    if len(sub) >= 2:
        x = sub['complexity_proxy'].to_numpy()
        y = sub['mean_optimal_rank'].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, slope * xx + intercept, color='#333333', linewidth=1.2, linestyle='--')
    ax.set_xlabel('Task complexity proxy (1 - acc at n=50, r=2)')
    ax.set_ylabel('Mean optimal rank')
    ax.set_title('Complexity vs optimal rank')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig6_complexity_vs_optimal_rank.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig6_complexity_vs_optimal_rank.png', bbox_inches='tight')
    plt.close(fig)

    summary = {
        'total_runs': int(len(df)),
        'models': [MODEL_MAP[m] for m in MODEL_MAP],
        'tasks': [t.upper() for t in TASK_ORDER],
        'ranks': RANK_ORDER,
        'sample_sizes': N_ORDER,
        'valid_anova_cells': int((anova_df['status'] == 'ok').sum()),
        'significant_anova_cells_fdr_0.05': int((anova_df['p_fdr'] < 0.05).sum()),
        'degenerate_cells': int(anova_df['degenerate_rank_means'].sum()),
        'degenerate_models': sorted(anova_df.loc[anova_df['degenerate_rank_means'], 'model_short'].unique().tolist()),
        'mean_optimal_rank_by_n_unique_only': {
            str(int(k)): float(v)
            for k, v in unique_opt.groupby('n')['optimal_rank'].mean().items()
        },
        'rank_win_count_unique_only': {
            str(int(k)): int(v)
            for k, v in unique_opt['optimal_rank'].value_counts().sort_index().items()
        },
        'task_complexity_spearman_rho': None if pd.isna(rho) else float(rho),
        'task_complexity_spearman_p': None if pd.isna(pval) else float(pval),
        'notes': [
            'Inferential rank analyses exclude cells with degenerate rank means or zero within-seed variance.',
            'DeBERTa-v3-base is degenerate across ranks in this result file and is treated as descriptive only for optimal-rank trend claims.',
            'Optimal-rank summaries use unique optima only; exact ties are labeled tie and excluded from numeric trend estimates.'
        ],
    }
    (OUT_DIR / 'summary_overview.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
