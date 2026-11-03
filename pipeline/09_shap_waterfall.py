import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap
from split_utils import grouped_tts

# =====================================================================
# 09_shap_waterfall.py —— SHAP Waterfall Plot（瀑布图，论文级）
#
# 单个患者：从基线概率出发，每个特征把预测往左/右推多少。
# 展示"这个死亡患者为什么被判死"。
#
# 输入: sepsis_training_data.npz（与 08 完全同源，同 seed、同聚合方式）
# 输出: shap_waterfall.png
#
# 画法依据：SHAP 官方 GitHub（issue #2268 / #3172）对 sklearn
#   RandomForestClassifier 的标准做法 ——
#   explainer(X) 返回 (n_samples, n_features, n_classes) 的 Explanation，
#   用 [i, :, 1] 切片取第 i 个患者的"死亡类"，直接传给 waterfall()。
# =====================================================================

SEED = 42
np.random.seed(SEED)

# 测试集里挑患者的方式：
#   SHOW_DECEASED = True  → 实际死亡且模型预测概率最高（最典型的"判死"病例）
#   SHOW_DECEASED = False → 实际存活且预测概率最高（最典型的"判活"病例）
SHOW_DECEASED = True
PATIENT_INDEX = None   # 想指定某患者时填测试集下标（如 0），否则留 None
MAX_DISPLAY = 15

# ---------- 1. 加载原始 48h 数据 ----------
d = np.load('sepsis_training_data.npz')
X_hcv = d['X_hcv'].astype(np.float32)   # [N, 14, 48]
X_sv = d['X_sv'].astype(np.float32)     # [N, 4]
y = d['y'].astype(np.int64)
N = len(y)

# ---------- 2. 聚合特征（与 08 完全一致）----------
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

SV_CLINICAL = ['Age', 'Gender (M)', 'Weight', 'Height']
for j, name in enumerate(SV_CLINICAL):
    feat_parts.append(X_sv[:, j])
    feat_names.append(name)

X = np.column_stack(feat_parts).astype(np.float32)   # [N, 46]

# ---------- 3. 划分 + 标准化（受试者级分组，与 08 一致）----------
tr_idx, va_idx, te_idx = grouped_tts(d['subject_id'].astype(np.int64), y)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X[tr_idx])
X_te = scaler.transform(X[te_idx])
y_tr, y_te = y[tr_idx], y[te_idx]

# ---------- 4. 重新训练 RF（同 06/08 超参）----------
rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                            class_weight='balanced', n_jobs=-1, random_state=SEED)
rf.fit(X_tr, y_tr)
print('RF 训练完成')

# ---------- 5. 选一个代表性患者 ----------
if PATIENT_INDEX is not None:
    i = PATIENT_INDEX
else:
    prob_death = rf.predict_proba(X_te)[:, 1]
    if SHOW_DECEASED:
        grp = np.where(y_te == 1)[0]                    # 实际死亡
    else:
        grp = np.where(y_te == 0)[0]                    # 实际存活
    i = grp[np.argmax(prob_death[grp])]

print(f'展示患者: 测试集下标 {i} | 真实标签 = {"死亡" if y_te[i] == 1 else "存活"}')

# ---------- 6. 算 SHAP 并直接切片取"死亡类"（官方推荐写法）----------
explainer = shap.TreeExplainer(rf)
shap_values = explainer(X_te)               # 返回 Explanation 对象，形状 (n, n_feat, 2)
sv_i = shap_values[i, :, 1]                 # 第 i 个患者、全部特征、死亡类 ← 官方写法

# 用原始（未标准化）特征值做显示，避免 y 轴出现 z 分数，
# 临床期刊要求图上展示可直接读懂的原始值（bpm、mmHg 等）。
sv_i.display_data = X[te_idx[i]]
sv_i.feature_names = feat_names   # 46 个临床特征名（含 mean/min/max），否则 y 轴会显示 Feature N 占位名

base = float(shap_values.base_values[i, 1] if np.ndim(shap_values.base_values) == 2
             else shap_values.base_values[i])
f_x = float(base + sv_i.values.sum())
print(f'基线 E[f(X)] = {base:.4f} → 该患者 f(x) = {f_x:.4f}')

# ---------- 7. 画瀑布图 ----------
shap.plots.waterfall(sv_i, max_display=MAX_DISPLAY, show=False)
plt.gcf().set_size_inches(10, 7)
plt.savefig('figures/shap_waterfall.png', dpi=300, bbox_inches='tight')
print('已保存: shap_waterfall.png')