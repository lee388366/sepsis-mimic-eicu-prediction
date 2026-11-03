import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap
from split_utils import grouped_tts

# =====================================================================
# 08_shap_dependence.py —— SHAP 依赖图（Top-3 特征）
#
# 与 08_shap_summary.py 完全同源：同 seed、同聚合方式、同 RF 超参
# 输出: shap_dependence.png（3 个子图，每个 top 特征一张）
# =====================================================================

SEED = 42
np.random.seed(SEED)

# ---------- 1-4. 与 08 完全相同的特征构造 + 训练 ----------
d = np.load('sepsis_training_data.npz')
X_hcv = d['X_hcv'].astype(np.float32)
X_sv = d['X_sv'].astype(np.float32)
y = d['y'].astype(np.int64)
N = len(y)

CLINICAL = ['Heart Rate', 'Systolic BP', 'Diastolic BP', 'Mean BP', 'Respiratory Rate',
            'SpO2', 'Temperature', 'Glucose', 'FiO2', 'pH', 'GCS Score',
            'GCS Eyes', 'GCS Motor', 'GCS Verbal']
feat_names = []
feat_parts = []
feat_names = []
feat_parts = []
for j, name in enumerate(CLINICAL):
    series = X_hcv[:, j, :]                      # 直接用索引 j
    feat_parts += [series.mean(1), series.min(1), series.max(1)]
    feat_names += [f'{name} (mean)', f'{name} (min)', f'{name} (max)']

SV = ['Age', 'Gender (M)', 'Weight', 'Height']
for j, name in enumerate(SV):
    feat_parts.append(X_sv[:, j])
    feat_names.append(name)
X = np.column_stack(feat_parts).astype(np.float32)

tr_idx, va_idx, te_idx = grouped_tts(d['subject_id'].astype(np.int64), y)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X[tr_idx])
X_te = scaler.transform(X[te_idx])
y_tr, y_te = y[tr_idx], y[te_idx]

rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                            class_weight='balanced', n_jobs=-1, random_state=SEED)
rf.fit(X_tr, y_tr)

# ---------- 5. 算 SHAP ----------
explainer = shap.TreeExplainer(rf)
shap_values = explainer(X_te)          # (n, n_feat, 2)
sv = shap_values[:, :, 1].values       # 死亡类 SHAP，取 .values 变数组
# 用原始（未标准化）特征值做 x 轴显示
X_te_raw = X[te_idx]

# ---------- 6. 画 Top-3 特征的依赖图 ----------
# 先算 mean|SHAP| 找 top-3
mean_abs = np.abs(sv).mean(axis=0)
top3 = np.argsort(mean_abs)[::-1][:3]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, fi in zip(axes, top3):
    shap.dependence_plot(fi, sv, X_te_raw, feature_names=feat_names,
                         ax=ax, show=False)
    ax.set_title(f'{feat_names[fi]}\nmean|SHAP| = {mean_abs[fi]:.4f}', fontsize=11)
plt.tight_layout()
plt.savefig('figures/shap_dependence.png', dpi=300, bbox_inches='tight')
print('已保存: shap_dependence.png')
print('Top-3 特征:', [feat_names[i] for i in top3])