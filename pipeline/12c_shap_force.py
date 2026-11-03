import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap
from split_utils import grouped_tts

# =====================================================================
# 12c_shap_force.py —— SHAP Force Plot（力图，论文级，个体解释闭环）
#
# 与 09_shap_waterfall.py 同源同患者：同一 RF、同一 seed、同一测试集。
# 瀑布图展示"特征如何把概率推离基线"；力图展示"这个患者被判死/判活
# 的完整贡献结构"，两者互为补充（TRIPOD+AI 条目 12e / 可解释性叙事）。
#
# 输入: sepsis_training_data.npz
# 输出: figures/shap_force.png
# =====================================================================

SEED = 42
np.random.seed(SEED)

SHOW_DECEASED = True   # 与 09 保持一致：展示"判死"病例
PATIENT_INDEX = None
MAX_DISPLAY = 15

# ---------- 1. 加载原始 48h 数据 ----------
d = np.load('sepsis_training_data.npz')
X_hcv = d['X_hcv'].astype(np.float32)
X_sv = d['X_sv'].astype(np.float32)
y = d['y'].astype(np.int64)
N = len(y)

# ---------- 2. 聚合特征（与 08/09 完全一致）----------
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
    series = X_hcv[:, j, :]
    feat_parts.append(series.mean(axis=1))
    feat_names.append(f'{name} (mean)')
    feat_parts.append(series.min(axis=1))
    feat_names.append(f'{name} (min)')
    feat_parts.append(series.max(axis=1))
    feat_names.append(f'{name} (max)')

SV_CLINICAL = ['Age', 'Gender (M)', 'Weight', 'Height']
for j, name in enumerate(SV_CLINICAL):
    feat_parts.append(X_sv[:, j])
    feat_names.append(name)

X = np.column_stack(feat_parts).astype(np.float32)

# ---------- 3. 划分 + 标准化 ----------
tr_idx, va_idx, te_idx = grouped_tts(d['subject_id'].astype(np.int64), y)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X[tr_idx])
X_te = scaler.transform(X[te_idx])
y_tr, y_te = y[tr_idx], y[te_idx]

# ---------- 4. 重新训练 RF（同 06/08/09 超参）----------
rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                            class_weight='balanced', n_jobs=-1, random_state=SEED)
rf.fit(X_tr, y_tr)

# ---------- 5. 选患者（与 09 完全一致的选法）----------
if PATIENT_INDEX is not None:
    i = PATIENT_INDEX
else:
    prob_death = rf.predict_proba(X_te)[:, 1]
    if SHOW_DECEASED:
        grp = np.where(y_te == 1)[0]
    else:
        grp = np.where(y_te == 0)[0]
    i = grp[np.argmax(prob_death[grp])]

print(f'展示患者: 测试集下标 {i} | 真实标签 = {"死亡" if y_te[i] == 1 else "存活"}')

# ---------- 6. SHAP + 力图 ----------
explainer = shap.TreeExplainer(rf)
shap_values = explainer(X_te)
sv_i = shap_values[i, :, 1]
sv_i.display_data = X[te_idx[i]]
sv_i.feature_names = feat_names

base = float(shap_values.base_values[i, 1] if np.ndim(shap_values.base_values) == 2
             else shap_values.base_values[i])
f_x = float(base + sv_i.values.sum())
print(f'基线 E[f(X)] = {base:.4f} → 该患者 f(x) = {f_x:.4f}')

# ---------- 7. 画力图（force plot，显示 top 特征）----------
# force plot 横轴是概率，红色推高、蓝色推低
shap.force_plot(
    base, sv_i.values, sv_i.display_data,
    feature_names=feat_names, matplotlib=True,
    show=False, figsize=(14, 4),
    text_rotation=30,
)
plt.gcf().set_size_inches(14, 4.5)
plt.tight_layout()
plt.savefig('figures/shap_force.png', dpi=300, bbox_inches='tight')
print('已保存: figures/shap_force.png')
