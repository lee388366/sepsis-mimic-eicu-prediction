import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score, accuracy_score)
from split_utils import grouped_tts

# =====================================================================
# 06b_train_sequence.py —— v2：双向 LSTM + 注意力池化（调优版）
#
# 与 06_train_baseline.py 同源：同划分(70/15/15, seed=42)、同验证集 F3 阈值扫描
# v2 改动：双向 LSTM(hidden=64/方向) + 时间注意力池化 + LR=5e-4 + patience=10
#
# 输出（覆盖 B 轨自身产物，不碰 A 轨）：
#   models/model_LSTM.pt / sequence_results.csv / sequence_predictions.npz
# =====================================================================

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'设备: {DEVICE}')

# ---------- 1. 加载 ----------
d = np.load('sepsis_training_data.npz')
X_hcv = d['X_hcv'].astype(np.float32)
X_sv  = d['X_sv'].astype(np.float32)
y     = d['y'].astype(np.int64)
subject_id = d['subject_id'].astype(np.int64)
N = len(y)
X_seq = X_hcv.transpose(0, 2, 1)                # [n, 48, 14]
print(f'数据: {N} 例 | X_seq {X_seq.shape} | 死亡率 {100*y.mean():.1f}%')

# ---------- 2. 划分（受试者级分组，同 06）----------
tr_idx, va_idx, te_idx = grouped_tts(subject_id, y)
print(f'train {len(tr_idx)} | val {len(va_idx)} | test {len(te_idx)}')

# ---------- 3. 标准化（只 fit 训练集）----------
sc_seq = StandardScaler().fit(X_seq[tr_idx].reshape(-1, 14))
sc_sv  = StandardScaler().fit(X_sv[tr_idx])
X_seq = sc_seq.transform(X_seq.reshape(-1, 14)).reshape(N, 48, 14).astype(np.float32)
X_sv  = sc_sv.transform(X_sv).astype(np.float32)

n_pos = int((y[tr_idx] == 1).sum()); n_neg = int((y[tr_idx] == 0).sum())
pos_w = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=DEVICE)
print(f'pos_weight = {pos_w.item():.2f}')

# ---------- 4. 模型：双向 LSTM + 注意力池化 ----------
class AttentionPool(nn.Module):
    """时间步上的可学习注意力：加权求和 48 个时间步的隐状态"""
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, 1)
    def forward(self, h):                          # h: [n, 48, dim]
        att = torch.softmax(self.W(h).squeeze(-1), dim=1)   # [n, 48]
        return torch.bmm(att.unsqueeze(1), h).squeeze(1)    # [n, dim]

class SepsisLSTM(nn.Module):
    def __init__(self, n_seq=14, n_static=4, hidden=64, drop=0.35):
        super().__init__()
        self.lstm = nn.LSTM(n_seq, hidden, batch_first=True, bidirectional=True)
        self.attn = AttentionPool(2 * hidden)
        self.drop = nn.Dropout(drop)
        self.static = nn.Sequential(nn.Linear(n_static, 16), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + 16, 64), nn.ReLU(),
            nn.Dropout(drop), nn.Linear(64, 1))

    def forward(self, xs, xsv):
        out, _ = self.lstm(xs)                     # [n, 48, 2*hidden]
        ctx = self.attn(out)                       # 注意力池化
        return self.head(torch.cat([self.drop(ctx), self.static(xsv)], 1)).squeeze(1)

model = SepsisLSTM().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

def tensors(i_arr):
    return (torch.tensor(X_seq[i_arr], device=DEVICE),
            torch.tensor(X_sv[i_arr], device=DEVICE),
            torch.tensor(y[i_arr], dtype=torch.float32, device=DEVICE))

# ---------- 5. 训练（patience=10，监控验证集 loss）----------
best_val, patience, epochs = 1e9, 10, 80
BATCH = 256
for ep in range(epochs):
    model.train()
    perm = np.random.permutation(len(tr_idx))
    for b in range(0, len(tr_idx), BATCH):
        xs, xsv, yy = tensors(tr_idx[perm[b:b+BATCH]])
        opt.zero_grad()
        loss = loss_fn(model(xs, xsv), yy)
        loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        xs, xsv, yy = tensors(va_idx)
        val_loss = loss_fn(model(xs, xsv), yy).item()
    if val_loss < best_val - 1e-4:
        best_val = val_loss; patience = 10
        torch.save(model.state_dict(), 'models/model_LSTM.pt')
    else:
        patience -= 1
        if patience == 0:
            print(f'  早停于 epoch {ep+1}'); break
    if (ep + 1) % 10 == 0:
        print(f'  epoch {ep+1:3d} | val loss = {val_loss:.4f}')
print(f'训练结束 | best val loss = {best_val:.4f}')

# ---------- 6. 验证集 F3 阈值扫描（同 06）----------
def fbeta(p, r, beta=3.0):
    return (1 + beta**2) * p * r / (beta**2 * p + r) if (beta**2 * p + r) > 0 else 0.0

model.load_state_dict(torch.load('models/model_LSTM.pt')); model.eval()
with torch.no_grad():
    xs, xsv, _ = tensors(va_idx)
    va_prob = torch.sigmoid(model(xs, xsv)).cpu().numpy()
    xs, xsv, _ = tensors(te_idx)
    te_prob = torch.sigmoid(model(xs, xsv)).cpu().numpy()
y_va, y_te = y[va_idx], y[te_idx]

best_f3, best_th = 0.0, 0.5
for th in np.arange(0.10, 0.86, 0.01):
    yp = (va_prob >= th).astype(int)
    p = precision_score(y_va, yp, zero_division=0)
    r = recall_score(y_va, yp, zero_division=0)
    f3 = fbeta(p, r)
    if f3 > best_f3:
        best_f3, best_th = f3, th
print(f'验证集最优阈值: {best_th:.2f} (val F3={best_f3:.4f})')

# ---------- 7. 测试集评估 ----------
bin_pred = (te_prob >= best_th).astype(int)
p = precision_score(y_te, bin_pred, zero_division=0)
r = recall_score(y_te, bin_pred, zero_division=0)

row = {
    'model': 'LSTM',
    'AUC': roc_auc_score(y_te, te_prob),
    'AUPRC': average_precision_score(y_te, te_prob),
    'best_threshold': round(best_th, 2),
    'Accuracy': accuracy_score(y_te, bin_pred),
    'Precision': p,
    'Recall': r,
    'F1': f1_score(y_te, bin_pred, zero_division=0),
    'F3': fbeta(p, r),
}
df = pd.DataFrame([row])
print('\n===== LSTM v2 测试集结果 =====')
print(df.to_string(index=False))
df.to_csv('sequence_results.csv', index=False)
np.savez('sequence_predictions.npz', LSTM=te_prob, y_true=y_te,
         stay_id=d['stay_id'][te_idx])
print('\n已保存: models/model_LSTM.pt + sequence_results.csv + sequence_predictions.npz')