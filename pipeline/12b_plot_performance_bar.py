import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =====================================================================
# 12b_plot_performance_bar.py —— 多模型性能对比条形图
#
# 数据源:
#   tables/table2_model_performance.csv   —— 7 个 ML 模型内部测试集性能
#   tables/table5_qsofa_windows.csv       —— qSOFA-6h AUC/AUPRC
#   tables/table5_sofa_cutoffs.csv        —— SOFA 连续得分在正文报告 AUC 0.672/AUPRC 0.345
#                                           （AGENT_PROJECT_CONTROL.md L80 口径）
# 输出: figures/model_performance_bar.png
#
# 说明: SOFA/qSOFA 是评分工具，无 F3 阈值操作点（Precision/Recall 口径与 ML 不同），
#       因此条形图只对比 AUC 与 AUPRC 两个判别指标；Precision/Recall 仅画 7 ML。
# =====================================================================

perf = pd.read_csv('tables/table2_model_performance.csv')

# 模型顺序：按 AUC 降序（读图自然）
perf = perf.sort_values('AUC', ascending=True)

# 临床评分（内部测试集，正文口径）
SOFA_AUC, SOFA_AUPRC = 0.672, 0.345
QSOFA_AUC, QSOFA_AUPRC = 0.573, 0.223

models = perf['Model'].tolist() + ['SOFA', 'qSOFA-6h']
auc = perf['AUC'].tolist() + [SOFA_AUC, QSOFA_AUC]
auprc = perf['AUPRC'].tolist() + [SOFA_AUPRC, QSOFA_AUPRC]
auc_lo = perf['AUC_95CI_low'].tolist() + [np.nan, np.nan]
auc_hi = perf['AUC_95CI_high'].tolist() + [np.nan, np.nan]
prec = perf['Precision'].tolist() + [np.nan, np.nan]
rec = perf['Recall'].tolist() + [np.nan, np.nan]

n = len(models)
ypos = np.arange(n)

# 颜色：7 ML 蓝色系，SOFA 棕、qSOFA 青（与现有图一致）
colors = []
for m in models:
    if m == 'SOFA':
        colors.append('#8C564B')
    elif m == 'qSOFA-6h':
        colors.append('#17BECF')
    else:
        colors.append('#3498db')

fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))

# ---- 面板 A: AUC ----
ax = axes[0]
bars = ax.barh(ypos, auc, color=colors, edgecolor='black', linewidth=0.5, height=0.62)
for i, (lo, hi) in enumerate(zip(auc_lo, auc_hi)):
    if not np.isnan(lo):
        ax.plot([lo, hi], [i, i], color='black', lw=1.2)
        ax.plot([lo, lo], [i-0.15, i+0.15], color='black', lw=1.0)
        ax.plot([hi, hi], [i-0.15, i+0.15], color='black', lw=1.0)
for i, v in enumerate(auc):
    ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
ax.set_yticks(ypos)
ax.set_yticklabels(models, fontsize=10)
ax.set_xlim(0.5, 0.85)
ax.set_xlabel('AUC (internal test set)', fontsize=12)
ax.set_title('(A) Discrimination — AUC', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.axvline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)

# ---- 面板 B: AUPRC ----
ax = axes[1]
bars = ax.barh(ypos, auprc, color=colors, edgecolor='black', linewidth=0.5, height=0.62)
for i, v in enumerate(auprc):
    ax.text(v + 0.004, i, f'{v:.3f}', va='center', fontsize=9)
ax.set_yticks(ypos)
ax.set_yticklabels(models, fontsize=10)
ax.set_xlim(0, 0.55)
ax.set_xlabel('AUPRC (internal test set)', fontsize=12)
ax.set_title('(B) Discrimination — AUPRC', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# ---- 面板 C: Precision/Recall（仅 7 ML，F3 阈值操作点）----
ax = axes[2]
ml_idx = np.arange(len(perf))
w = 0.36
bars1 = ax.barh(ml_idx - w/2, prec[:len(perf)], height=w, color='#2ecc71',
                edgecolor='black', linewidth=0.5, label='Precision')
bars2 = ax.barh(ml_idx + w/2, rec[:len(perf)], height=w, color='#e74c3c',
                edgecolor='black', linewidth=0.5, label='Recall')
for i in ml_idx:
    ax.text(prec[i] + 0.005, i - w/2, f'{prec[i]:.2f}', va='center', fontsize=8.5)
    ax.text(rec[i] + 0.005, i + w/2, f'{rec[i]:.2f}', va='center', fontsize=8.5)
ax.set_yticks(ml_idx)
ax.set_yticklabels(perf['Model'].tolist(), fontsize=10)
ax.set_xlim(0, 1.05)
ax.set_xlabel('Value at F3-optimal threshold', fontsize=12)
ax.set_title('(C) Precision & Recall (ML models)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/model_performance_bar.png', dpi=300, bbox_inches='tight')
print('已保存: figures/model_performance_bar.png')
