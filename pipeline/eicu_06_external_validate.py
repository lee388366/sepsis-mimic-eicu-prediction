import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from split_utils import grouped_tts

# =====================================================================
# eicu_06_external_validate.py —— eICU 外部验证
#
# 目的：用 MIMIC 已训模型（6 基线 + LSTM）在 eICU 外验集上评估。
#
# 关键纪律：
#   1. 标准化参数**全部来自 MIMIC 训练集**（绝不用 eICU fit）
#      - 6 基线: models/scaler.joblib（46 维）
#      - LSTM: 重建 sc_seq/sc_sv（在 MIMIC 训练集上 fit）
#   2. 模型权重来自 models/*.joblib + models/model_LSTM.pt（不重训）
#   3. 输出: 内验 vs 外验 AUC 对比 + 校准 + 模型间对比
# =====================================================================

SEED = 42
np.random.seed(SEED)
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# ---------- 1. 加载 MIMIC 训练数据（重建 LSTM 的 scaler）----------
d_tr = np.load('sepsis_training_data.npz')
X_hcv_tr = d_tr['X_hcv'].astype(np.float32)
X_sv_tr  = d_tr['X_sv'].astype(np.float32)
y_tr_all = d_tr['y'].astype(np.int64)
N_tr = len(y_tr_all)

# 复刻 06b 的划分（MIMIC 训练集，受试者级分组——与 06b 完全一致）
idx_tr = np.arange(N_tr)
tr_tr, va_tr, _ = grouped_tts(d_tr['subject_id'].astype(np.int64), y_tr_all)

# LSTM 的 scaler：在 MIMIC 训练子集上 fit
X_seq_tr = X_hcv_tr.transpose(0, 2, 1)   # [n, 48, 14]
sc_seq = StandardScaler().fit(X_seq_tr[tr_tr].reshape(-1, 14))
sc_sv  = StandardScaler().fit(X_sv_tr[tr_tr])
print('LSTM scaler 已从 MIMIC 训练集重建')

# ---------- 2. 加载 eICU 外验数据（已插补）----------
de = np.load('eicu_training_data_imputed.npz')
X_hcv_e = de['X_hcv'].astype(np.float32)   # [n, 14, 48]
X_sv_e  = de['X_sv'].astype(np.float32)
y_e     = de['y'].astype(np.int64)
n_e = len(y_e)
print(f'eICU 外验集: {n_e} 例 | 死亡率 {100*y_e.mean():.1f}%')

# ---------- 3. 6 基线模型预测（46 维聚合特征 + MIMIC scaler）----------
scaler46 = joblib.load('models/scaler.joblib')
CLINICAL = ['heart_rate','sbp','dbp','mbp','resp_rate','spo2',
            'temperature','glucose','fio2','ph','gcs','gcs_eyes','gcs_motor','gcs_verbal']
feat_parts = []
for j in range(14):
    s = X_hcv_e[:, j, :]
    feat_parts += [s.mean(1), s.min(1), s.max(1)]
X_e46 = np.column_stack(feat_parts + [X_sv_e[:,0], X_sv_e[:,1], X_sv_e[:,2], X_sv_e[:,3]]).astype(np.float32)
X_e46_std = scaler46.transform(X_e46)

BASELINES = ['LogisticRegression', 'DecisionTree', 'RandomForest', 'KNN', 'SVM', 'GaussianNB']
baseline_probs = {}
for name in BASELINES:
    clf = joblib.load(f'models/model_{name}.joblib')
    if hasattr(clf, 'predict_proba'):
        baseline_probs[name] = clf.predict_proba(X_e46_std)[:, 1]
    else:
        baseline_probs[name] = clf.decision_function(X_e46_std)
    print(f'  {name:20s} 外验 AUC = {roc_auc_score(y_e, baseline_probs[name]):.4f}')

# ---------- 4. LSTM 预测（重建模型 + MIMIC scaler）----------
X_seq_e = X_hcv_e.transpose(0, 2, 1)                     # [n, 48, 14]
X_seq_e_std = sc_seq.transform(X_seq_e.reshape(-1, 14)).reshape(n_e, 48, 14).astype(np.float32)
X_sv_e_std  = sc_sv.transform(X_sv_e).astype(np.float32)

class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.W = nn.Linear(dim, 1)
    def forward(self, h):
        att = torch.softmax(self.W(h).squeeze(-1), dim=1)
        return torch.bmm(att.unsqueeze(1), h).squeeze(1)

class SepsisLSTM(nn.Module):
    def __init__(self, n_seq=14, n_static=4, hidden=64, drop=0.35):
        super().__init__()
        self.lstm = nn.LSTM(n_seq, hidden, batch_first=True, bidirectional=True)
        self.attn = AttentionPool(2*hidden); self.drop = nn.Dropout(drop)
        self.static = nn.Sequential(nn.Linear(n_static, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2*hidden+16, 64), nn.ReLU(),
                                  nn.Dropout(drop), nn.Linear(64, 1))
    def forward(self, xs, xsv):
        out, _ = self.lstm(xs)
        return self.head(torch.cat([self.drop(self.attn(out)), self.static(xsv)], 1)).squeeze(1)

model = SepsisLSTM().to(DEVICE)
model.load_state_dict(torch.load('models/model_LSTM.pt', map_location=DEVICE))
model.eval()

lstm_probs = np.zeros(n_e)
with torch.no_grad():
    for i in range(0, n_e, 256):
        xs = torch.tensor(X_seq_e_std[i:i+256], device=DEVICE)
        xsv = torch.tensor(X_sv_e_std[i:i+256], device=DEVICE)
        lstm_probs[i:i+256] = torch.sigmoid(model(xs, xsv)).cpu().numpy()
print(f'  LSTM              外验 AUC = {roc_auc_score(y_e, lstm_probs):.4f}')

# ---------- 5. 汇总外验表 ----------
from sklearn.metrics import precision_score, recall_score, f1_score

def fbeta(p, r, beta=3.0):
    return (1+beta**2)*p*r/(beta**2*p+r) if (beta**2*p+r) > 0 else 0.0

rows = []
for name in ['LSTM'] + BASELINES:
    prob = lstm_probs if name == 'LSTM' else baseline_probs[name]
    rows.append({
        'Model': name,
        'AUC_ext': roc_auc_score(y_e, prob),
        'AUPRC_ext': average_precision_score(y_e, prob),
        'Brier_ext': brier_score_loss(y_e, prob),
    })

df = pd.DataFrame(rows).sort_values('AUC_ext', ascending=False).reset_index(drop=True)
print('\n===== eICU 外验结果（全部模型）=====')
print(df.to_string(index=False))
df.to_csv('tables/table9_external_validation.csv', index=False)
print('\n已保存: tables/table9_external_validation.csv')

# ---------- 6. SOFA/qSOFA 外验对比 ----------
sofa_ext = de['sofa_max_48h'].astype(float)
qsofa6_ext = de['qsofa6'].astype(float)
print('\n===== 评分外验（对照）=====')
print(f'  SOFA 外验 AUC = {roc_auc_score(y_e, sofa_ext):.4f}')
print(f'  qSOFA-6h 外验 AUC = {roc_auc_score(y_e, qsofa6_ext):.4f}')

# ---------- 7. 内验 vs 外验对比（MIMIC 测试集数字从控制文档取）----------
print('\n===== 内验 vs 外验 AUC 对比（论文核心）=====')
internal = {'LSTM': 0.791, 'RandomForest': 0.786, 'SVM': 0.785, 'LogisticRegression': 0.752}
for name in internal:
    ext = df.loc[df.Model == name, 'AUC_ext'].iloc[0]
    print(f'  {name:20s}: 内验 {internal[name]:.3f} → 外验 {ext:.3f} | 衰减 {ext-internal[name]:+.3f}')

# ---------- 8. 保存外验预测 ----------
np.savez('eicu_predictions.npz',
         LSTM=lstm_probs,
         **{n: baseline_probs[n] for n in BASELINES},
         y_true=y_e,
         sofa=sofa_ext,
         qsofa6=qsofa6_ext,
         stay_id=de['stay_id'])
print('\n已保存: eicu_predictions.npz')