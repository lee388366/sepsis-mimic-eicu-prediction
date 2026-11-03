import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from split_utils import grouped_tts

# =====================================================================
# 07_learning_curve_lstm.py —— LSTM 学习曲线（诊断过拟合 / 样本量效应）
#
# 与 06b_train_sequence.py 完全同构：同一模型（双向 LSTM + 注意力池化）、
# 同一特征聚合、同一受试者级划分、同一训练配置（Adam lr=5e-4, pos_weight,
# 基于验证集 loss 的早停）。
#
# 做法: 训练样本量从 2000 逐步增至全量（6 档），每档重复 N_REPEAT 次取均值，
#       画 训练集 AUC 与 验证集 AUC，检验两者是否收敛。
#
# 输出: figures/learning_curve_lstm.png
# =====================================================================

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
# 注意：MPS 上 LSTM 训练结果与 CPU 不一致（验证 AUC 虚高），
# 学习曲线必须用 CPU 以获得与 model_LSTM.pt 可比的数值。
DEVICE = 'cpu'
torch.set_num_threads(4)
print(f'设备: {DEVICE}')

d = np.load('sepsis_training_data.npz')
X_hcv = d['X_hcv'].astype(np.float32)
X_sv  = d['X_sv'].astype(np.float32)
y     = d['y'].astype(np.int64)
subject_id = d['subject_id'].astype(np.int64)
N = len(y)
X_seq = X_hcv.transpose(0, 2, 1)                # [n, 48, 14]

tr_idx, va_idx, te_idx = grouped_tts(subject_id, y)
print(f'train {len(tr_idx)} | val {len(va_idx)} | test {len(te_idx)}')

# ---------- 标准化（只 fit 训练集）----------
sc_seq = StandardScaler().fit(X_seq[tr_idx].reshape(-1, 14))
sc_sv  = StandardScaler().fit(X_sv[tr_idx])
X_seq = sc_seq.transform(X_seq.reshape(-1, 14)).reshape(N, 48, 14).astype(np.float32)
X_sv  = sc_sv.transform(X_sv).astype(np.float32)

n_pos = int((y[tr_idx] == 1).sum()); n_neg = int((y[tr_idx] == 0).sum())
pos_w = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=DEVICE)


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, 1)
    def forward(self, h):
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
        ctx = self.attn(out)
        return self.head(torch.cat([self.drop(ctx), self.static(xsv)], 1)).squeeze(1)


def tensors(i_arr):
    return (torch.tensor(X_seq[i_arr], device=DEVICE),
            torch.tensor(X_sv[i_arr], device=DEVICE),
            torch.tensor(y[i_arr], dtype=torch.float32, device=DEVICE))


def train_one(n_train, seed):
    """在训练集随机抽 n_train 例训练 LSTM，返回 (训练集AUC, 验证集AUC)。
    早停全程跟踪验证集 loss 最低的权重，评估时恢复该权重（与 06b 一致）。"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tr_idx))
    sub = perm[:n_train]

    model = SepsisLSTM().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    BATCH = 256
    MAX_EPOCHS = 80
    PATIENCE = 10
    best_val, patience = 1e9, PATIENCE
    best_state = None

    xs_va, xsv_va, y_va_t = tensors(va_idx)
    for ep in range(MAX_EPOCHS):
        model.train()
        p2 = np.random.permutation(len(sub))
        for b in range(0, len(sub), BATCH):
            i = sub[p2[b:b+BATCH]]
            xs, xsv, yy = tensors(i)
            opt.zero_grad()
            loss = loss_fn(model(xs, xsv), yy)
            loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xs_va, xsv_va), y_va_t).item()
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            patience = PATIENCE
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience == 0:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        xs, xsv, _ = tensors(sub)
        tr_prob = torch.sigmoid(model(xs, xsv)).cpu().numpy()
        va_prob = torch.sigmoid(model(xs_va, xsv_va)).cpu().numpy()
    tr_auc = roc_auc_score(y[sub], tr_prob)
    va_auc = roc_auc_score(y[va_idx], va_prob)
    return tr_auc, va_auc


TRAIN_SIZES = [2000, 4000, 7000, 10000, 14000, len(tr_idx)]
N_REPEAT = 2

train_aucs = np.zeros((len(TRAIN_SIZES), N_REPEAT))
val_aucs = np.zeros((len(TRAIN_SIZES), N_REPEAT))

print(f'训练集: {len(tr_idx)} 例 | 验证集: {len(va_idx)} 例')
for si, n_train in enumerate(TRAIN_SIZES):
    for rep in range(N_REPEAT):
        seed = SEED * 100 + si * 10 + rep
        tr_auc, va_auc = train_one(n_train, seed)
        train_aucs[si, rep], val_aucs[si, rep] = tr_auc, va_auc
        print(f'  n={n_train:>6} rep{rep}: 训练AUC={tr_auc:.4f}  验证AUC={va_auc:.4f}')
    print(f'  n={n_train:>6}: 训练AUC={train_aucs[si].mean():.4f}±{train_aucs[si].std():.4f}  '
          f'验证AUC={val_aucs[si].mean():.4f}±{val_aucs[si].std():.4f}')

plt.figure(figsize=(8, 6))
tr_mean, tr_std = train_aucs.mean(1), train_aucs.std(1)
va_mean, va_std = val_aucs.mean(1), val_aucs.std(1)

plt.fill_between(TRAIN_SIZES, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color='#D62728')
plt.fill_between(TRAIN_SIZES, va_mean - va_std, va_mean + va_std, alpha=0.15, color='#1F77B4')
plt.plot(TRAIN_SIZES, tr_mean, 'o-', color='#D62728', lw=2.2, label='Training set')
plt.plot(TRAIN_SIZES, va_mean, 's-', color='#1F77B4', lw=2.2, label='Validation set')

plt.xlabel('Number of training samples', fontsize=13)
plt.ylabel('AUC', fontsize=13)
plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/learning_curve_lstm.png', dpi=300, bbox_inches='tight')
print('已保存: figures/learning_curve_lstm.png')
