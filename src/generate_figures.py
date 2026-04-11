#!/usr/bin/env python3
"""自动生成 LoRA Rank Scaling Laws 论文图表"""
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox_inches': 'tight', 'figure.figsize': (10, 6),
})

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "artifacts" / "results.csv")
agg = df.groupby(['model','task','rank','n']).agg(
    acc_mean=('accuracy','mean'), acc_std=('accuracy','std'),
    count=('accuracy','count')
).reset_index()
print(f"Loaded {len(df)} runs, {len(agg)} groups")

RANK_COLORS = {2:'#e74c3c', 4:'#e67e22', 8:'#f1c40f', 16:'#2ecc71', 32:'#3498db', 64:'#9b59b6'}
MODEL_NAMES = {'bert-base-uncased':'BERT', 'roberta-base':'RoBERTa', 'microsoft/deberta-v3-base':'DeBERTa-v3'}
TASK_ORDER = ['sst2','mrpc','qnli','rte','cola']

# ============================================================
# Figure 1: Scaling Curves — Accuracy vs Sample Size per Rank
# (One subplot per model×task = 15 subplots)
# ============================================================
print("Fig 1: Scaling curves...")
models = sorted(df.model.unique())
tasks = TASK_ORDER
fig, axes = plt.subplots(len(models), len(tasks), figsize=(20, 4*len(models)), squeeze=False)
for i, m in enumerate(models):
    for j, t in enumerate(tasks):
        ax = axes[i][j]
        sub = agg[(agg.model==m) & (agg.task==t)]
        for r in sorted(sub['rank'].unique()):
            d = sub[sub['rank']==r].sort_values('n')
            ax.errorbar(d.n, d.acc_mean, yerr=d.acc_std, marker='o', ms=4,
                       label=f'r={r}', color=RANK_COLORS.get(r,'gray'), capsize=2, linewidth=1.5)
        ax.set_xscale('log')
        ax.set_xlabel('Training Samples')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{MODEL_NAMES.get(m,m)} — {t.upper()}')
        if i==0 and j==0:
            ax.legend(loc='lower right', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.0)
fig.suptitle('LoRA Rank Scaling Laws: Accuracy vs Sample Size', fontsize=16, y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig1_scaling_curves.png")
fig.savefig(FIG_DIR / "fig1_scaling_curves.pdf")
plt.close()
print("  -> fig1_scaling_curves.png/pdf")

# ============================================================
# Figure 2: Rank Effect — Accuracy vs Rank per Sample Size
# (One subplot per model, lines = sample sizes)
# ============================================================
print("Fig 2: Rank effect...")
fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5), squeeze=False)
SIZE_COLORS = {50:'#bdc3c7',100:'#95a5a6',200:'#7f8c8d',500:'#e67e22',1000:'#e74c3c',2000:'#2ecc71',5000:'#3498db'}
for i, m in enumerate(models):
    ax = axes[0][i]
    sub = agg[agg.model==m].groupby(['rank','n']).agg(acc_mean=('acc_mean','mean')).reset_index()
    for n in sorted(sub.n.unique()):
        d = sub[sub.n==n].sort_values('rank')
        ax.plot(d['rank'], d.acc_mean, marker='s', ms=5, label=f'n={n}',
                color=SIZE_COLORS.get(n,'gray'), linewidth=1.5)
    ax.set_xlabel('LoRA Rank')
    ax.set_ylabel('Accuracy (avg across tasks)')
    ax.set_title(MODEL_NAMES.get(m,m))
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(sub['rank'].unique()))
plt.suptitle('Effect of LoRA Rank on Accuracy (averaged across tasks)', fontsize=14)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig2_rank_effect.png")
fig.savefig(FIG_DIR / "fig2_rank_effect.pdf")
plt.close()
print("  -> fig2_rank_effect.png/pdf")

# ============================================================
# Figure 3: Heatmaps — Rank × Sample Size for each Model×Task
# ============================================================
print("Fig 3: Heatmaps...")
fig, axes = plt.subplots(len(models), len(tasks), figsize=(20, 4*len(models)), squeeze=False)
ranks_sorted = sorted(df['rank'].unique())
sizes_sorted = sorted(df.n.unique())
for i, m in enumerate(models):
    for j, t in enumerate(tasks):
        ax = axes[i][j]
        sub = agg[(agg.model==m) & (agg.task==t)]
        matrix = np.full((len(ranks_sorted), len(sizes_sorted)), np.nan)
        for _, row in sub.iterrows():
            ri = ranks_sorted.index(row['rank'])
            si = sizes_sorted.index(row.n)
            matrix[ri][si] = row.acc_mean
        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0.45, vmax=0.95)
        ax.set_xticks(range(len(sizes_sorted)))
        ax.set_xticklabels(sizes_sorted, fontsize=8, rotation=45)
        ax.set_yticks(range(len(ranks_sorted)))
        ax.set_yticklabels(ranks_sorted)
        ax.set_xlabel('Sample Size')
        ax.set_ylabel('LoRA Rank')
        ax.set_title(f'{MODEL_NAMES.get(m,m)} — {t.upper()}', fontsize=11)
        # Annotate cells
        for ri in range(len(ranks_sorted)):
            for si in range(len(sizes_sorted)):
                if not np.isnan(matrix[ri][si]):
                    ax.text(si, ri, f'{matrix[ri][si]:.2f}', ha='center', va='center', fontsize=7)
plt.suptitle('Accuracy Heatmap: LoRA Rank × Sample Size', fontsize=16, y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig3_heatmaps.png")
fig.savefig(FIG_DIR / "fig3_heatmaps.pdf")
plt.close()
print("  -> fig3_heatmaps.png/pdf")

# ============================================================
# Figure 4: Model Comparison — same rank, same task, different models
# ============================================================
print("Fig 4: Model comparison...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
for idx, r in enumerate([2, 8, 64]):
    for row, t in enumerate(['sst2', 'qnli']):
        ax = axes[row][idx]
        for m in models:
            sub = agg[(agg.model==m) & (agg.task==t) & (agg['rank']==r)].sort_values('n')
            ax.errorbar(sub.n, sub.acc_mean, yerr=sub.acc_std, marker='o', ms=4,
                       label=MODEL_NAMES.get(m,m), capsize=2, linewidth=1.5)
        ax.set_xscale('log')
        ax.set_xlabel('Training Samples')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{t.upper()}, r={r}')
        ax.legend()
        ax.grid(True, alpha=0.3)
plt.suptitle('Model Comparison at Different LoRA Ranks', fontsize=14)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig4_model_comparison.png")
fig.savefig(FIG_DIR / "fig4_model_comparison.pdf")
plt.close()
print("  -> fig4_model_comparison.png/pdf")

# ============================================================
# Figure 5: Task Complexity — accuracy across tasks for fixed rank×n
# ============================================================
print("Fig 5: Task complexity...")
fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5), squeeze=False)
for i, m in enumerate(models):
    ax = axes[0][i]
    sub = agg[(agg.model==m) & (agg.n==1000)]  # fix n=1000
    for r in sorted(sub['rank'].unique()):
        d = sub[sub['rank']==r]
        task_accs = [d[d.task==t].acc_mean.values[0] if len(d[d.task==t])>0 else 0 for t in tasks]
        ax.bar(np.arange(len(tasks)) + r/100, task_accs, width=0.12,
               label=f'r={r}', color=RANK_COLORS.get(r,'gray'))
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t.upper() for t in tasks])
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{MODEL_NAMES.get(m,m)} (n=1000)')
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3, axis='y')
plt.suptitle('Task Complexity × LoRA Rank Interaction', fontsize=14)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig5_task_complexity.png")
fig.savefig(FIG_DIR / "fig5_task_complexity.pdf")
plt.close()
print("  -> fig5_task_complexity.png/pdf")

# ============================================================
# Figure 6: Optimal Rank — for each task×model, which rank is best?
# ============================================================
print("Fig 6: Optimal rank decision matrix...")
fig, ax = plt.subplots(figsize=(12, 5))
results = []
for m in models:
    for t in tasks:
        for n in sizes_sorted:
            sub = agg[(agg.model==m) & (agg.task==t) & (agg.n==n)]
            if len(sub) > 0:
                best = sub.loc[sub.acc_mean.idxmax()]
                results.append({'model':MODEL_NAMES.get(m,m),'task':t.upper(),'n':n,
                              'best_rank':int(best['rank']),'best_acc':best.acc_mean})
opt_df = pd.DataFrame(results)
# Pivot: rows = model+task, cols = n
pivot_data = []
for m in [MODEL_NAMES.get(x,x) for x in models]:
    for t in [x.upper() for x in tasks]:
        row = {'config': f'{m}/{t}'}
        for n in sizes_sorted:
            sub = opt_df[(opt_df.model==m) & (opt_df.task==t) & (opt_df.n==n)]
            if len(sub) > 0:
                row[f'n={n}'] = int(sub.best_rank.values[0])
        pivot_data.append(row)
opt_pivot = pd.DataFrame(pivot_data).set_index('config')
im = ax.imshow(opt_pivot.values.astype(float), aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(opt_pivot.columns)))
ax.set_xticklabels(opt_pivot.columns, rotation=45, fontsize=9)
ax.set_yticks(range(len(opt_pivot.index)))
ax.set_yticklabels(opt_pivot.index, fontsize=9)
for i in range(len(opt_pivot.index)):
    for j in range(len(opt_pivot.columns)):
        ax.text(j, i, f'r={int(opt_pivot.values[i][j])}', ha='center', va='center', fontsize=7)
ax.set_title('Optimal LoRA Rank by Task × Model × Sample Size')
plt.colorbar(im, label='Best Rank')
plt.tight_layout()
fig.savefig(FIG_DIR / "fig6_optimal_rank.png")
fig.savefig(FIG_DIR / "fig6_optimal_rank.pdf")
plt.close()
print("  -> fig6_optimal_rank.png/pdf")

print(f"\nAll figures saved to {FIG_DIR}/")
print(f"Total: {len(list(FIG_DIR.glob('*.png')))} PNG + {len(list(FIG_DIR.glob('*.pdf')))} PDF")
