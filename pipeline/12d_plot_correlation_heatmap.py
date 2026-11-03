import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from split_utils import grouped_tts

# =====================================================================
# 12d_plot_correlation_heatmap.py —— 预测变量相关性热图 + VIF
#
# 目的: 回应审稿人对"14 变量是否存在共线性/冗余"的常见质疑
#       （TRIPOD+AI 条目 12b：变量处理透明化）
#
# 展示: 18 个建模变量（14 HCV 的 mean 统计量 + 4 静态）的 Pearson
#       相关矩阵 + 右上角标注 VIF。
#   VIF 在训练集上计算（防泄漏红线：标准性与 VIF 均只 fit 训练集）。
#
# 输入: sepsis_training_data.npz
# 输出: figures/correlation_heatmap.png
# =====================================================================

SEED = 42
np.random.seed(SEED)

d = np.load('sepsis_training_data.npz')
X_hcv = d['X_hcv'].astype(np.float32)
X_sv = d['X_sv'].astype(np.float32)
y = d['y'].astype(np.int64)

CLINICAL = {
    'heart_rate':   'Heart Rate',
    'sbp':          'Systolic BP',
    'dbp':          'Diastolic BP',
    'mbp':          'Mean BP',
    'resp_rate':    'Respiratory Rate',
    'spo2':         'SpO2',
    'temperature':  'Temperature',
    'glucose':      'Glucose',
    'fio2':         'FiO2',
    'ph':           'pH',
    'gcs':          'GCS total',
    'gcs_eyes':     'GCS eyes',
    'gcs_motor':    'GCS motor',
    'gcs_verbal':   'GCS verbal',
}
SV_CLINICAL = ['Age', 'Gender (M)', 'Weight', 'Height']

# ---- 只用 mean 统计量（48h 均值），共 14+4=18 个代表变量 ----
feat_names = []
feat_parts = []
for j, var in enumerate(CLINICAL):
    feat_parts.append(X_hcv[:, j, :].mean(axis=1))
    feat_names.append(CLINICAL[var])
for j, name in enumerate(SV_CLINICAL):
    feat_parts.append(X_sv[:, j])
    feat_names.append(name)

X = np.column_stack(feat_parts).astype(np.float32)

tr_idx, va_idx, te_idx = grouped_tts(d['subject_id'].astype(np.int64), y)

# ---- 训练集上算相关性 + VIF（防泄漏）----
X_tr = X[tr_idx]
corr = pd.DataFrame(X_tr, columns=feat_names).corr()

def vif(df):
    """训练集 VIF（标准方法：对每列回归其余列，VIF=1/(1-R²)）"""
    Xs = df.values.astype(float)
    out = {}
    for i, col in enumerate(df.columns):
        ycol = Xs[:, i]
        Xothers = np.delete(Xs, i, axis=1)
        lr = LinearRegression().fit(Xothers, ycol)
        r2 = lr.score(Xothers, ycol)
        out[col] = 1.0 / (1.0 - r2) if r2 < 0.9999 else np.inf
    return pd.Series(out)

vif_s = vif(pd.DataFrame(X_tr, columns=feat_names))

# ---- 画图 ----
fig = plt.figure(figsize=(13, 12), layout='constrained')
gs = fig.add_gridspec(2, 1, height_ratios=[8, 1], hspace=0.15)
ax = fig.add_subplot(gs[0])
im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)

ax.set_xticks(np.arange(len(feat_names)))
ax.set_yticks(np.arange(len(feat_names)))
ax.set_xticklabels(feat_names, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(feat_names, fontsize=9)

# 单元格数值
for i in range(len(feat_names)):
    for j in range(len(feat_names)):
        v = corr.values[i, j]
        color = 'white' if abs(v) > 0.6 else 'black'
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7, color=color)

cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('Pearson correlation (training set)', fontsize=11)
ax.set_title('Pairwise correlation of model predictors (48-h means)',
             fontsize=14, fontweight='bold')

# 下方 VIF 面板（多行文本，避免长文本拉伸画布）
axv = fig.add_subplot(gs[1])
axv.axis('off')
vif_sorted = vif_s.sort_values(ascending=False)
# 分两行显示
n = len(vif_sorted)
half = (n + 1) // 2
line1 = '    '.join(f'{c}: {v:.1f}' for c, v in vif_sorted.iloc[:half].items())
line2 = '    '.join(f'{c}: {v:.1f}' for c, v in vif_sorted.iloc[half:].items())
vif_text = f'VIF (training set, n={len(X_tr):,}):\n{line1}\n{line2}'
axv.text(0.01, 0.5, vif_text, fontsize=10, ha='left', va='center',
         bbox=dict(boxstyle='round', facecolor='#fef9e7', edgecolor='grey'))
axv.set_xlim(0, 1)

plt.savefig('figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print('已保存: figures/correlation_heatmap.png')
print('VIF (训练集):')
print(vif_sorted.to_string())
