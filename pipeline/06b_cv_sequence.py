import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from split_utils import grouped_kfold

# =====================================================================
# 06b_cv_sequence.py —— LSTM 5 折分层交叉验证（稳定性验证）
#
# 与 06b_train_sequence.py v2 同架构：双向 LSTM + 注意力池化
# 与 06_cross_validation.py（RF 10 折）同源同口径，但用 5 折（LSTM 训练贵）
#
# 关键纪律：
#   - 每折内部重新 StandardScaler fit（只 fit 训练折），防泄漏
#   - 每折内部早停（监控验证折 loss），用验证折选最优 epoch
#   - pos_weight 每折按该折训练分布计算
#
# 输出: tables/table3_cv_lstm_5fold.csv（每折 AUC/AP + mean/SD 汇总行）
# =====================================================================

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'设备: {DEVICE}')

# ---------- 1. 加载（同 06b）----------
d = np.load('sepsis_training_data.npz')
X_hcv = d['X_hcv'].astype(np.float32)   # [n, 14, 48]
X_sv  = d['X_sv'].astype(np.float32)    # [n, 4]
y     = d['y'].astype(np.int64)
subject_id = d['subject_id'].astype(np.int64)
X_seq = X_hcv.transpose(0, 2, 1)        # [n, 48, 14]
N = len(y)
print(f'数据: {N} 例 | X_seq {X_seq.shape} | 死亡率 {100*y.mean():.1f}%')

# ---------- 2. 模型（同 06b v2）----------
class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, 1)
    def forward(self, h):                               # h: [n, 48, dim]
        att = torch.softmax(self.W(h).squeeze(-1), dim=1)
        return torch.bmm(att.unsqueeze(1), h).squeeze(1)

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
        out, _ = self.lstm(xs)
        return self.head(torch.cat([self.drop(self.attn(out)), self.static(xsv)], 1)).squeeze(1)

# ---------- 3. 5 折受试者级分层交叉验证 ----------
print('\n=== 5 折受试者级分层交叉验证 ===')
auc_list, ap_list = [], []

for fold, (tr_idx, va_idx) in enumerate(grouped_kfold(X_seq, y, subject_id, n_splits=5), 1):
    # 每折内重新标准化（防泄漏）
    sc_seq = StandardScaler().fit(X_seq[tr_idx].reshape(-1, 14))
    sc_sv  = StandardScaler().fit(X_sv[tr_idx])
    Xtr = sc_seq.transform(X_seq[tr_idx].reshape(-1, 14)).reshape(len(tr_idx), 48, 14).astype(np.float32)
    Xva = sc_seq.transform(X_seq[va_idx].reshape(-1, 14)).reshape(len(va_idx), 48, 14).astype(np.float32)
    SVtr, SVva = sc_sv.transform(X_sv[tr_idx]).astype(np.float32), sc_sv.transform(X_sv[va_idx]).astype(np.float32)
    ytr, yva = y[tr_idx], y[va_idx]

    pos_w = torch.tensor([(ytr == 0).sum() / (ytr == 1).sum()], dtype=torch.float32, device=DEVICE)
    model = SepsisLSTM().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    def tens(Xs, Xv, yy):
        return (torch.tensor(Xs, device=DEVICE),
                torch.tensor(Xv, device=DEVICE),
                torch.tensor(yy, dtype=torch.float32, device=DEVICE))

    best_val, patience = 1e9, 10
    for ep in range(60):
        model.train()
        perm = np.random.permutation(len(tr_idx))
        for b in range(0, len(tr_idx), 256):
            xs, xsv, yy = tens(Xtr[perm[b:b+256]], SVtr[perm[b:b+256]], ytr[perm[b:b+256]])
            opt.zero_grad()
            loss = loss_fn(model(xs, xsv), yy)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            xs, xsv, yy = tens(Xva, SVva, yva)
            vl = loss_fn(model(xs, xsv), yy).item()
        if vl < best_val - 1e-4:
            best_val, patience = vl, 10
        else:
            patience -= 1
            if patience == 0:
                break

    model.eval()
    with torch.no_grad():
        xs, xsv, _ = tens(Xva, SVva, yva)
        prob = torch.sigmoid(model(xs, xsv)).cpu().numpy()
    a = roc_auc_score(yva, prob)
    p = average_precision_score(yva, prob)
    auc_list.append(a); ap_list.append(p)
    print(f'  Fold {fold}: AUC={a:.4f} | AP={p:.4f}')

auc_arr, ap_arr = np.array(auc_list), np.array(ap_list)
print('\n===== LSTM 5 折汇总（供正文引用）=====')
print(f'AUC: {auc_arr.mean():.4f} ± {auc_arr.std(ddof=1):.4f} (范围 {auc_arr.min():.4f}–{auc_arr.max():.4f})')
print(f'AP : {ap_arr.mean():.4f} ± {ap_arr.std(ddof=1):.4f} (范围 {ap_arr.min():.4f}–{ap_arr.max():.4f})')

df = pd.DataFrame({
    'Model': 'LSTM',
    'fold': [f'Fold{i}' for i in range(1, 6)],
    'AUC': auc_arr, 'AP': ap_arr,
})
df.loc[len(df)] = ['LSTM', 'mean', auc_arr.mean(), ap_arr.mean()]
df.loc[len(df)] = ['LSTM', 'SD',   auc_arr.std(ddof=1), ap_arr.std(ddof=1)]
df.to_csv('tables/table3_cv_lstm_5fold.csv', index=False)
print('已保存: tables/table3_cv_lstm_5fold.csv')