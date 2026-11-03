import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# 12a_plot_missing_diagnosis.py —— 缺失值诊断图（TRIPOD+AI 条目11）
#
# 数据源: tables/table4_missing_rate.csv（make_table4.py 产物）
# 输出:   figures/missing_diagnosis.png
#
# 左面板：14 个 HCV 变量的「患者×小时」单元缺失率（%）
# 右面板：完全无观测患者比例（%）+ 每患者平均观测小时数（次轴）
#   —— 直观支撑"三级插补"（固定生理值 / 患者均值 / 最近邻）的写法
# =====================================================================

df = pd.read_csv('tables/table4_missing_rate.csv')

# 去掉 Weight/Height 两个静态变量（单元缺失为 "-"）
df = df[df['Missing cells (%)'] != '-'].copy()
df['cell_pct'] = df['Missing cells (%)'].astype(float)
df['pat_pct'] = df['Patients completely missing (%)'].astype(float)
df['mean_obs'] = df['Mean obs per patient (of 48h)'].astype(float)

# 按单元缺失率排序（高缺失在最上，读图自然）
df = df.sort_values('cell_pct')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

# ---- 左面板：单元缺失率 ----
colors = ['#c0392b' if p >= 50 else '#e67e22' if p >= 20 else '#27ae60' for p in df['cell_pct']]
bars = ax1.barh(df['Variable'], df['cell_pct'], color=colors, edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Missing cells (%) — patient×hour level', fontsize=12)
ax1.set_title('(A) Missingness of hourly variables', fontsize=13, fontweight='bold')
for b, p in zip(bars, df['cell_pct']):
    ax1.text(b.get_width() + 1, b.get_y() + b.get_height()/2, f'{p:.1f}%',
             va='center', fontsize=10)
ax1.set_xlim(0, 100)
ax1.grid(axis='x', alpha=0.3)

# ---- 右面板：完全无观测患者比例（柱）+ 平均观测数（线）----
ax2b = ax2.twinx()
bars2 = ax2.barh(df['Variable'], df['pat_pct'], color='#8e44ad',
                 edgecolor='black', linewidth=0.5, alpha=0.75, label='Completely unobserved (%)')
line = ax2b.plot(df['mean_obs'], df['Variable'], 'o-', color='#2980b9',
                 linewidth=2, markersize=7, label='Mean obs. hours per patient')
ax2.set_xlabel('Patients completely unobserved (%)', fontsize=12)
ax2b.set_xlabel('', fontsize=12)
ax2b.set_xlabel('Mean observed hours (of 48)', fontsize=12, color='#2980b9')
ax2b.tick_params(axis='x', colors='#2980b9')
ax2.set_title('(B) Patient-level missingness & observation density', fontsize=13, fontweight='bold')
for b, p in zip(bars2, df['pat_pct']):
    if p > 0.3:
        ax2.text(b.get_width() + 0.5, b.get_y() + b.get_height()/2, f'{p:.1f}%',
                 va='center', fontsize=9)
# 数值低时在条外标注，避免遮挡
for b, p in zip(bars2, df['pat_pct']):
    if p <= 0.3:
        ax2.text(b.get_width() + 0.5, b.get_y() + b.get_height()/2, f'{p:.2f}%',
                 va='center', fontsize=8, color='#555')
# 线值标注
for x, y in zip(df['mean_obs'], df['Variable']):
    ax2b.annotate(f'{x:.0f}', (x, y), textcoords='offset points',
                  xytext=(6, 0), va='center', fontsize=9, color='#2980b9')
ax2.set_xlim(0, max(df['pat_pct']) * 1.35)
ax2b.set_xlim(0, 48)

# 图例
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2b.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc='lower right', fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig('figures/missing_diagnosis.png', dpi=300, bbox_inches='tight')
print('已保存: figures/missing_diagnosis.png')
