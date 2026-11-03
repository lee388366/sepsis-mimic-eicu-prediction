import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from split_utils import grouped_tts

# =====================================================================
# 08c_attention_faithfulness.py —— 注意力遮挡/扰动忠实性检验（纯推理，不重训）
#
# 目的：验证 LSTM 注意力权重是否真的对齐"对预测重要的时间步"。
#   方法（occlusion faithfulness）：把注意力权重最高/最低的 k 个小时
#   整段遮住（14 特征全部置 0 = 训练集均值，即"最无信息"），重新前向，
#   看 AUC/AUPRC 掉多少。若注意力忠实，则遮"高权重小时"应比遮"低权重
#   小时"、比"随机小时"损伤更大。
#
# 与 06c（时间窗消融）的区别：06c 消融的是"窗口跨度"，这里消融的是
#   "注意力挑出来的具体小时"，直接检验注意力本身的可信度。
#
# 口径（与 06b/06c/08b 完全一致）：
#   - 同一受试者级分组划分（grouped_tts, seed=42），同一测试集 n=3742
#   - 同一预处理（StandardScaler 只 fit 训练集）
#   - 用已训好的 models/model_LSTM.pt（不重训）
#   - 遮挡值=0（训练集均值）；报告 AUC/AUPRC（无阈值依赖，遮挡会破坏校准）
#   - 随机遮挡 NSEED 个种子，报告均值±SD，并算 z = (top降 - 随机降)/SD
#
# 输出:
#   tables/table8c_attention_faithfulness.csv（AUC/AUPRC 明细表）
#   figures/attention_faithfulness.png（两面板：AUC 降幅 + AUPRC 降幅）
#   终端打印论文可引用数字
# =====================================================================

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
NSEED = 30          # 随机遮挡种子数（z 值稳定性）
KS = [1, 3, 6, 12, 24]

# ---------- 1. 加载与预处理（复现 06b）----------
d = np.load('sepsis_training_data.npz')
X_hcv = d['X_hcv'].astype(np.float32)
X_sv  = d['X_sv'].astype(np.float32)
y     = d['y'].astype(np.int64)
N = len(y)
X_seq = X_hcv.transpose(0, 2, 1)                 # [n, 48, 14]

tr_idx, va_idx, te_idx = grouped_tts(d['subject_id'].astype(np.int64), y)
print(f'train {len(tr_idx)} | val {len(va_idx)} | test {len(te_idx)}')

sc_seq = StandardScaler().fit(X_seq[tr_idx].reshape(-1, 14))
sc_sv  = StandardScaler().fit(X_sv[tr_idx])
X_seq = sc_seq.transform(X_seq.reshape(-1, 14)).reshape(N, 48, 14).astype(np.float32)
X_sv  = sc_sv.transform(X_sv).astype(np.float32)
y_te  = y[te_idx]

# ---------- 2. 重建模型并加载权重（与 06b 一致）----------
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
        self.attn = AttentionPool(2 * hidden); self.drop = nn.Dropout(drop)
        self.static = nn.Sequential(nn.Linear(n_static, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * hidden + 16, 64), nn.ReLU(),
                                  nn.Dropout(drop), nn.Linear(64, 1))
    def forward(self, xs, xsv):
        out, _ = self.lstm(xs)
        return self.head(torch.cat([self.drop(self.attn(out)), self.static(xsv)], 1)).squeeze(1)

model = SepsisLSTM().to(DEVICE)
model.load_state_dict(torch.load('models/model_LSTM.pt', map_location=DEVICE))
model.eval()

xs  = torch.tensor(X_seq[te_idx], device=DEVICE)
xsv = torch.tensor(X_sv[te_idx],  device=DEVICE)
with torch.no_grad():
    out, _ = model.lstm(xs)
    att = torch.softmax(model.attn.W(out).squeeze(-1), dim=1).cpu().numpy()   # [n, 48]
    base_prob = torch.sigmoid(model(xs, xsv)).cpu().numpy()

ba  = roc_auc_score(y_te, base_prob)
bp  = average_precision_score(y_te, base_prob)
print(f'基线 AUC={ba:.4f}  AUPRC={bp:.4f}  (n_test={len(te_idx)})')

# ---------- 3. 遮挡工具 ----------
def masked_prob(keep_hours):
    """keep_hours: [n, m] 每样本保留的小时（其余置 0=训练集均值）。返回新概率。"""
    mask = np.zeros((len(te_idx), 48), dtype=np.float32)
    mask[np.arange(len(te_idx))[:, None], keep_hours] = 1.0
    xm = xs * torch.tensor(mask[..., None], device=DEVICE)
    with torch.no_grad():
        return torch.sigmoid(model(xm, xsv)).cpu().numpy()

def run(p):
    return roc_auc_score(y_te, p), average_precision_score(y_te, p)

all_hours = np.arange(48)
mean_att = att.mean(axis=0)

rows = []
for k in KS:
    keep_all  = np.tile(all_hours[~np.isin(all_hours, np.argsort(mean_att)[-k:])], (len(te_idx), 1))
    keep_glob = np.tile(all_hours[~np.isin(all_hours, np.argsort(mean_att)[:k])],  (len(te_idx), 1))
    per_top = np.argsort(att, 1)[:, ::-1][:, :k]
    per_bot = np.argsort(att, 1)[:, :k]
    keep_pt = np.array([all_hours[~np.isin(all_hours, per_top[i])] for i in range(len(te_idx))])
    keep_pb = np.array([all_hours[~np.isin(all_hours, per_bot[i])] for i in range(len(te_idx))])

    a_all, p_all = run(masked_prob(keep_all))    # 全局 top-k（平均注意力最高）
    a_glo, p_glo = run(masked_prob(keep_glob))   # 全局 bottom-k（平均注意力最低）
    a_pt,  p_pt  = run(masked_prob(keep_pt))     # 逐样本 top-k
    a_pb,  p_pb  = run(masked_prob(keep_pb))     # 逐样本 bottom-k

    # 随机遮挡：每样本随机 k 小时（无重复），NSEED 个种子
    rand_auc, rand_prc = [], []
    for s in range(NSEED):
        rng = np.random.default_rng(SEED + s)
        keep_rnd = np.stack([np.delete(all_hours, rng.choice(48, k, replace=False))
                             for _ in range(len(te_idx))])
        a_r, p_r = run(masked_prob(keep_rnd))
        rand_auc.append(a_r); rand_prc.append(p_r)
    rand_auc = np.array(rand_auc); rand_prc = np.array(rand_prc)

    z = (ba - a_pt - (ba - rand_auc.mean())) / rand_auc.std()   # top降 vs 随机降 的标准化差

    rows.append({
        'k_hours': k,
        'AUC_global_top':    a_all, 'AUC_global_bottom': a_glo,
        'AUC_per_sample_top': a_pt, 'AUC_per_sample_bottom': a_pb,
        'AUC_random_mean': rand_auc.mean(), 'AUC_random_sd': rand_auc.std(),
        'AUPRC_global_top': p_all, 'AUPRC_global_bottom': p_glo,
        'AUPRC_per_sample_top': p_pt, 'AUPRC_per_sample_bottom': p_pb,
        'AUPRC_random_mean': rand_prc.mean(), 'AUPRC_random_sd': rand_prc.std(),
        'z_top_vs_random': z,
    })
    print(f'k={k:>2} | 逐样本top ΔAUC={ba-a_pt:+.4f} | 全局top ΔAUC={ba-a_all:+.4f} '
          f'| 逐样本bottom ΔAUC={ba-a_pb:+.4f} | 随机 ΔAUC={ba-rand_auc.mean():+.4f}±{rand_auc.std():.4f} | z={z:+.2f}')

df = pd.DataFrame(rows)
df.to_csv('tables/table8c_attention_faithfulness.csv', index=False)
print('\n已保存: tables/table8c_attention_faithfulness.csv')

# ---------- 4. 画图（AUC 降幅 + AUPRC 降幅）----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ks = np.array(KS)
drop_all = ba  - df['AUC_global_top'].values
drop_glo = ba  - df['AUC_global_bottom'].values
drop_pt  = ba  - df['AUC_per_sample_top'].values
drop_pb  = ba  - df['AUC_per_sample_bottom'].values
drop_rnd = ba  - df['AUC_random_mean'].values
sd_rnd   = df['AUC_random_sd'].values

drop_all_p = bp - df['AUPRC_global_top'].values
drop_pt_p  = bp - df['AUPRC_per_sample_top'].values
drop_rnd_p = bp - df['AUPRC_random_mean'].values
sd_rnd_p   = df['AUPRC_random_sd'].values

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

panels = [
    dict(ax=axes[0], d_top=drop_all, d_bot=drop_glo, d_per=drop_pt, d_perbot=drop_pb,
         dr=drop_rnd, sd=sd_rnd, title='AUC drop'),
    dict(ax=axes[1], d_top=drop_all_p, d_bot=None, d_per=drop_pt_p, d_perbot=None,
         dr=drop_rnd_p, sd=sd_rnd_p, title='AUPRC drop'),
]
for p in panels:
    ax = p['ax']
    ax.fill_between(ks, p['dr'] - p['sd'], p['dr'] + p['sd'], color='gray', alpha=0.18, label='Random occlusion (mean±SD)')
    ax.plot(ks, p['dr'], 'o-', color='gray', lw=1.8)
    ax.plot(ks, p['d_per'], 'o-', color='#D62728', lw=2.2, label='Per-sample top-k attention')
    ax.plot(ks, p['d_top'], 's--', color='#FF7F0E', lw=1.8, label='Global top-k attention')
    if p['d_bot'] is not None:
        ax.plot(ks, p['d_perbot'], '^--', color='#1F77B4', lw=1.8, label='Per-sample bottom-k attention')
        ax.plot(ks, p['d_bot'], 'v--', color='#2CA02C', lw=1.8, label='Global bottom-k attention')
    ax.set_xlabel('Number of masked hours (k)', fontsize=12)
    ax.set_ylabel('Performance drop from baseline', fontsize=12)
    ax.set_title(p['title'], fontsize=13)
    ax.set_xticks(ks)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/attention_faithfulness.png', dpi=300, bbox_inches='tight')
print('已保存: figures/attention_faithfulness.png')

# ---------- 5. 论文可引用数字 ----------
print('\n===== 论文可引用数字 =====')
print(f'基线 AUC={ba:.4f} (与 table2 官方 0.791 一致)')
for _, r in df.iterrows():
    k = int(r['k_hours'])
    print(f'遮 {k:>2}h: 逐样本top降ΔAUC={ba-r["AUC_per_sample_top"]:+.4f}, '
          f'全局top降ΔAUC={ba-r["AUC_global_top"]:+.4f}, '
          f'逐样本bottom降ΔAUC={ba-r["AUC_per_sample_bottom"]:+.4f}, '
          f'随机降ΔAUC={ba-r["AUC_random_mean"]:+.4f}±{r["AUC_random_sd"]:.4f}, '
          f'z={r["z_top_vs_random"]:+.2f}')

# 平均注意力集中度（供方法段引用）
top3_share = np.sort(att, 1)[:, ::-1][:, :3].sum(1).mean()
print(f'\n注意力集中度: 每样本 top-3 权重和均值 = {top3_share:.3f} (均匀时={3/48:.3f})')
print(f'平均注意力 top5 小时: {np.argsort(mean_att)[-5:][::-1].tolist()}')
