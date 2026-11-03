import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from split_utils import grouped_tts

# =====================================================================
# 08b_lstm_attention.py —— LSTM 注意力权重可视化（时序可解释性）
#
# 原理：LSTM v2 的注意力池化层会为每个时间步（小时）学习一个权重，
#       权重越高 = 该小时对预测的贡献越大。此脚本提取这些权重，
#       展示模型在 48 小时内"什么时候最关注"。
#
# 输出: figures/lstm_attention.png（平均注意力 + 典型病例注意力）
# =====================================================================

SEED = 42
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# ---------- 1. 复现 06b 的预处理（受试者级分组划分 + 标准化，seed 固定可复现）----------
d = np.load('sepsis_training_data.npz')
X_hcv = d['X_hcv'].astype(np.float32)
X_sv  = d['X_sv'].astype(np.float32)
y     = d['y'].astype(np.int64)
N = len(y)
X_seq = X_hcv.transpose(0, 2, 1)

tr_idx, va_idx, te_idx = grouped_tts(d['subject_id'].astype(np.int64), y)

sc_seq = StandardScaler().fit(X_seq[tr_idx].reshape(-1, 14))
sc_sv  = StandardScaler().fit(X_sv[tr_idx])
X_seq = sc_seq.transform(X_seq.reshape(-1, 14)).reshape(N, 48, 14).astype(np.float32)
X_sv  = sc_sv.transform(X_sv).astype(np.float32)

# ---------- 2. 重建模型（与 06b 完全一致）并加载权重 ----------
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
        return self.head(torch.cat([self.drop(self.attn(out)), self.static(xsv)], 1)).squeeze(1)

model = SepsisLSTM().to(DEVICE)
model.load_state_dict(torch.load('models/model_LSTM.pt', map_location=DEVICE))
model.eval()

# ---------- 3. 提取测试集所有样本的注意力权重 ----------
xs = torch.tensor(X_seq[te_idx], device=DEVICE)
with torch.no_grad():
    out, _ = model.lstm(xs)
    att_logit = model.attn.W(out).squeeze(-1)          # [n_test, 48]
    att = torch.softmax(att_logit, dim=1).cpu().numpy() # [n_test, 48]

y_te = y[te_idx]
hours = np.arange(48)
mean_att = att.mean(axis=0)                            # 平均注意力

# ---------- 4. 画图：左=平均注意力，右=典型死亡病例 ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(hours, mean_att, color='#1F77B4', alpha=0.85)
axes[0].axvline(24, color='k', ls='--', lw=1, label='Hour 24')
axes[0].set_xlabel('ICU hour', fontsize=12)
axes[0].set_ylabel('Mean attention weight', fontsize=12)
axes[0].set_title('Average attention across test patients', fontsize=13)
axes[0].legend()

# 典型病例：实际死亡且模型预测概率最高的
model.eval()
with torch.no_grad():
    xsv_te = torch.tensor(X_sv[te_idx], device=DEVICE)
    prob = torch.sigmoid(model(xs, xsv_te)).cpu().numpy()
deceased = np.where(y_te == 1)[0]
i = deceased[np.argmax(prob[deceased])]
axes[1].bar(hours, att[i], color='#D62728', alpha=0.85)
axes[1].set_xlabel('ICU hour', fontsize=12)
axes[1].set_ylabel('Attention weight', fontsize=12)
axes[1].set_title(f'Representative deceased patient (pred prob={prob[i]:.2f})', fontsize=13)

plt.tight_layout()
plt.savefig('figures/lstm_attention.png', dpi=300, bbox_inches='tight')
print('已保存: figures/lstm_attention.png')
print(f'平均注意力最高的 5 个小时: {np.argsort(mean_att)[-5:]}')