import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score)
from split_utils import grouped_tts

# =====================================================================
# 06c_ablate_timepoint.py —— 时间窗消融实验（量化"时序信息"的贡献）
#
# 目的：回答"LSTM 是只靠最后时刻的信息，还是真的用了整个轨迹？"
#      以及"时间序列相对单点预测到底带来多少增量？"
#
# 四个 RF 对照（同一 test set、seed=42 划分、同 RF 超参）：
#   A. 单点-末时刻 : 只用第 48h 的 14 变量 → 单点预测
#   B. 单点-首时刻 : 只用第 1h 的 14 变量 → 对照"末 vs 首"
#   C. 末 24h 窗口 : 第 25-48h，聚合 mean/min/max → 42 维
#   D. 全 48h 窗口 : 完整 48h，聚合 mean/min/max → 42 维
# 另加 E. LSTM 完整序列（用 06b 已训模型，不重训）作为参考
#
# 输出: tables/table6_timepoint_ablation.csv + 终端增量解读
# =====================================================================

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# ---------- 1. 加载 ----------
d = np.load('sepsis_training_data.npz')
X_hcv = d['X_hcv'].astype(np.float32)   # [n, 14, 48]
X_sv  = d['X_sv'].astype(np.float32)    # [n, 4]
y     = d['y'].astype(np.int64)
N = len(y)
print(f'数据: {N} 例 | 死亡率 {100*y.mean():.1f}%')

# ---------- 2. 划分（受试者级分组，与 06/06b 一致）----------
subject_id = d['subject_id'].astype(np.int64)
tr_idx, va_idx, te_idx = grouped_tts(subject_id, y)
print(f'train {len(tr_idx)} | val {len(va_idx)} | test {len(te_idx)}')
y_te = y[te_idx]

# ---------- 3. 特征构造（四种窗口，均聚合 mean/min/max + 4 静态）----------
def aggregate(series):
    parts = []
    for j in range(14):
        s = series[:, j, :]
        parts += [s.mean(1), s.min(1), s.max(1)]
    return np.column_stack(parts + [X_sv[:, 0], X_sv[:, 1], X_sv[:, 2], X_sv[:, 3]]).astype(np.float32)

windows = {
    'A_single_t48': X_hcv[:, :, 47:48],   # 仅第 48h（单点）
    'B_single_t1':  X_hcv[:, :, 0:1],     # 仅第 1h（单点）
    'C_last24h':    X_hcv[:, :, 24:48],   # 第 25-48h
    'D_full48h':    X_hcv,                # 完整 48h
}
X_all = {name: aggregate(series) for name, series in windows.items()}

# ---------- 4. 每个窗口训 RF ----------
def fbeta(p, r, beta=3.0):
    return (1 + beta**2) * p * r / (beta**2 * p + r) if (beta**2 * p + r) > 0 else 0.0

results = []
for name, X in X_all.items():
    sc = StandardScaler().fit(X[tr_idx])
    X_tr = sc.transform(X[tr_idx]); X_va = sc.transform(X[va_idx]); X_te = sc.transform(X[te_idx])
    y_tr, y_va = y[tr_idx], y[va_idx]

    rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                class_weight='balanced', n_jobs=-1, random_state=SEED)
    rf.fit(X_tr, y_tr)
    va_prob = rf.predict_proba(X_va)[:, 1]
    te_prob = rf.predict_proba(X_te)[:, 1]

    # 验证集 F3 最优阈值（与 06 同口径）
    best_f3, best_th = 0.0, 0.5
    for th in np.arange(0.10, 0.86, 0.01):
        yp = (va_prob >= th).astype(int)
        p = precision_score(y_va, yp, zero_division=0)
        r = recall_score(y_va, yp, zero_division=0)
        f3 = fbeta(p, r)
        if f3 > best_f3:
            best_f3, best_th = f3, th

    bin_pred = (te_prob >= best_th).astype(int)
    p = precision_score(y_te, bin_pred, zero_division=0)
    r = recall_score(y_te, bin_pred, zero_division=0)
    results.append({
        'Window': name, 'Input': f'{X.shape[1]}D',
        'AUC': roc_auc_score(y_te, te_prob),
        'AUPRC': average_precision_score(y_te, te_prob),
        'best_th': round(best_th, 2),
        'Precision': p, 'Recall': r,
        'F1': f1_score(y_te, bin_pred, zero_division=0),
        'F3': fbeta(p, r),
    })
    print(f'{name:15s} | AUC={results[-1]["AUC"]:.4f} | AUPRC={results[-1]["AUPRC"]:.4f} | F3={results[-1]["F3"]:.4f}')

# ---------- 5. LSTM 参考（用 06b 已训模型 + sequence_results 同源操作点）----------
# 操作点指标从 sequence_results.csv 读取（与 table2 同源、同阈值，2026-08-27 后为 0.25），不再留空。
seq_res = pd.read_csv('sequence_results.csv').set_index('model')
lstm_row = seq_res.loc['LSTM']
X_seq = X_hcv.transpose(0, 2, 1)
sc_seq = StandardScaler().fit(X_seq[tr_idx].reshape(-1, 14))
sc_sv  = StandardScaler().fit(X_sv[tr_idx])
X_seq = sc_seq.transform(X_seq.reshape(-1, 14)).reshape(N, 48, 14).astype(np.float32)
X_sv_s = sc_sv.transform(X_sv).astype(np.float32)

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
with torch.no_grad():
    xs = torch.tensor(X_seq[te_idx], device=DEVICE)
    xsv = torch.tensor(X_sv_s[te_idx], device=DEVICE)
    lstm_prob = torch.sigmoid(model(xs, xsv)).cpu().numpy()

results.append({
    'Window': 'E_LSTM_full', 'Input': '48×14 seq',
    'AUC': roc_auc_score(y_te, lstm_prob),
    'AUPRC': average_precision_score(y_te, lstm_prob),
    'best_th': lstm_row['best_threshold'],
    'Precision': lstm_row['Precision'], 'Recall': lstm_row['Recall'],
    'F1': lstm_row['F1'], 'F3': lstm_row['F3'],
})
print(f'E_LSTM_full     | AUC={results[-1]["AUC"]:.4f} | AUPRC={results[-1]["AUPRC"]:.4f}')

# ---------- 6. 汇总输出 ----------
df = pd.DataFrame(results)
print('\n===== 时间窗消融（同一 test set 3742 例）=====')
print(df.to_string(index=False))
df.to_csv('tables/table6_timepoint_ablation.csv', index=False)
print('\n已保存: tables/table6_timepoint_ablation.csv')

auc_a = df.loc[df.Window == 'A_single_t48', 'AUC'].iloc[0]
auc_d = df.loc[df.Window == 'D_full48h', 'AUC'].iloc[0]
auc_e = df.loc[df.Window == 'E_LSTM_full', 'AUC'].iloc[0]
print('\n===== 增量解读（供论文引用）=====')
print(f'单点(48h) RF     : {auc_a:.4f}')
print(f'全48h聚合 RF     : {auc_d:.4f}  (窗口信息增量 = {auc_d-auc_a:+.4f})')
print(f'LSTM 完整序列    : {auc_e:.4f}  (相对单点增量 = {auc_e-auc_a:+.4f})')