import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap
from split_utils import grouped_tts

# =====================================================================
# 08_shap_summary.py v2 —— RandomForest 的 SHAP Summary（论文级）
#
# 与 v1 的区别（对齐期刊论文标准）：
#   1. 特征聚合：48h 时序 → 每变量取 mean/min/max 统计量
#      （对齐 Nature Sci Rep / medRxiv 外验的做法，蜜蜂图清晰可读）
#   2. 临床命名：技术列名 → 临床可读名（Feature labels rendered with
#      clinical nomenclature）
#   3. 绘图用官方推荐 API: shap.plots.beeswarm()
#
# 输出: shap_summary.png + shap_summary_bar.png + shap_values.npz
# =====================================================================

SEED = 42
np.random.seed(SEED)

# ---------- 1. 加载原始 48h 数据 ----------
d = np.load('sepsis_training_data.npz')
X_hcv = d['X_hcv'].astype(np.float32)   # [N, 14, 48]
X_sv = d['X_sv'].astype(np.float32)     # [N, 4]
y = d['y'].astype(np.int64)
N = len(y)

# ---------- 2. 聚合特征：每个 HCV 变量 → mean/min/max ----------
# 临床命名映射（专业标准）
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
    'gcs':          'GCS Score',
    'gcs_eyes':     'GCS Eyes',
    'gcs_motor':    'GCS Motor',
    'gcs_verbal':   'GCS Verbal',
}

feat_names = []
feat_parts = []
for j, var in enumerate(CLINICAL):
    name = CLINICAL[var]
    series = X_hcv[:, j, :]                       # [N, 48]
    feat_parts.append(series.mean(axis=1))        # mean
    feat_names.append(f'{name} (mean)')
    feat_parts.append(series.min(axis=1))         # min
    feat_names.append(f'{name} (min)')
    feat_parts.append(series.max(axis=1))         # max
    feat_names.append(f'{name} (max)')

# 静态变量（临床命名）
SV_CLINICAL = ['Age', 'Gender (M)', 'Weight', 'Height']
for j, name in enumerate(SV_CLINICAL):
    feat_parts.append(X_sv[:, j])
    feat_names.append(name)

X = np.column_stack(feat_parts).astype(np.float32)   # [N, 42+4=46]
print(f'聚合特征: {X.shape}（14 变量 × 3 统计量 + 4 静态 = 46 个临床特征）')

# ---------- 3. 划分（受试者级分组，同 06）----------
tr_idx, va_idx, te_idx = grouped_tts(d['subject_id'].astype(np.int64), y)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X[tr_idx])
X_te = scaler.transform(X[te_idx])
y_tr, y_te = y[tr_idx], y[te_idx]

# ---------- 4. 重新训练 RF（同 06 超参）----------
rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                            class_weight='balanced', n_jobs=-1, random_state=SEED)
rf.fit(X_tr, y_tr)
print('RF 训练完成')

# ---------- 5. TreeExplainer ----------
explainer = shap.TreeExplainer(rf)
print('计算 SHAP 值（测试集 3742 例 × 46 特征，约 30-60 秒）...')
shap_values = explainer.shap_values(X_te)

if isinstance(shap_values, list):
    sv = shap_values[1]                     # 死亡类
else:
    sv = shap_values[..., 1] if shap_values.ndim == 3 else shap_values

# 保存 SHAP 值（后续 Waterfall 等复用）
np.savez('shap_values.npz', shap_values=sv, y_true=y_te)

# ---------- 6. Summary Plot（蜜蜂图）----------
shap.summary_plot(sv, X_te, feature_names=feat_names,
                  max_display=25, show=False)
plt.tight_layout()
plt.savefig('figures/shap_summary.png', dpi=300, bbox_inches='tight')
print('已保存: shap_summary.png')

# ---------- 7. Bar Plot（重要性柱状图）----------
plt.figure()
shap.summary_plot(sv, X_te, feature_names=feat_names,
                  max_display=25, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('figures/shap_summary_bar.png', dpi=300, bbox_inches='tight')
print('已保存: shap_summary_bar.png')

# ---------- 8. 打印 Top-15 ----------
mean_abs = np.abs(sv).mean(axis=0)
top_idx = np.argsort(mean_abs)[::-1][:15]
print('\nTop-15 特征（临床命名，按 mean|SHAP| 排序）:')
for i in top_idx:
    print(f'  {feat_names[i]:30s} mean|SHAP| = {mean_abs[i]:.5f}')