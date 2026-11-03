#!/usr/bin/env python3
"""s08b_occlusion_lowarm.py —— 遮挡忠实性分析补做「低注意力时段」对照臂

背景
----
旧稿曾写道"遮挡 24 小时时，移除低注意力时段的损伤更大"。
新的 s08_explain.py 只重算了 全局top / 逐样本top / 随机 三个臂，
**没有**低注意力臂，因此旧句无法由新产物支持。

本脚本用**新训练的 LSTM 权重**（models_trainonly/<V>/model_LSTM.pt）
补做两个臂：
  * per_sample_low：每个样本移除其注意力最低的 k 个小时
  * global_low    ：移除全样本平均注意力最低的 k 个小时
随机臂沿用 s08 的 20 个种子（SEED + s），保证与 table8c 可比。

输出
----
  tables/<V>/table8c_attention_occlusion_lowarm.csv
  qa/s08b_occlusion_lowarm_<V>.json

纪律：不加载任何旧权重；不修改 s08 已生成的表；只追加对照臂。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent.parent
SEED = 42
KS = [1, 3, 6, 12, 24]
NSEED = 20
HCV = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'spo2', 'temperature',
       'glucose', 'fio2', 'ph', 'gcs', 'gcs_eyes', 'gcs_motor', 'gcs_verbal']


class AttentionPool(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = torch.nn.Linear(dim, 1)

    def forward(self, h):
        att = torch.softmax(self.W(h).squeeze(-1), dim=1)
        return torch.bmm(att.unsqueeze(1), h).squeeze(1), att


class SepsisLSTM(torch.nn.Module):
    def __init__(self, n_seq=14, n_static=4, hidden=64, drop=0.35):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_seq, hidden, batch_first=True, bidirectional=True)
        self.attn = AttentionPool(2 * hidden)
        self.drop = torch.nn.Dropout(drop)
        self.static = torch.nn.Sequential(torch.nn.Linear(n_static, 16), torch.nn.ReLU())
        self.head = torch.nn.Sequential(torch.nn.Linear(2 * hidden + 16, 64), torch.nn.ReLU(),
                                        torch.nn.Dropout(drop), torch.nn.Linear(64, 1))

    def forward(self, xs, xsv, return_attn=False):
        out, _ = self.lstm(xs)
        ctx, att = self.attn(out)
        logit = self.head(torch.cat([self.drop(ctx), self.static(xsv)], 1)).squeeze(1)
        return (logit, att) if return_attn else logit


def load_variant(V):
    d = np.load(HERE / 'data' / 'trainonly_mimic.npz', allow_pickle=False)
    X_hcv, X_sv = d['X_hcv'], d['X_sv']
    y, split = d['y'].astype(np.int64), d['split'].astype(str)
    tinf = d['t_inf_h'].astype(float)
    mask = np.ones(len(y), bool) if V == 'A' else (tinf <= 48)
    return X_hcv[mask], X_sv[mask], y[mask], split[mask]


def main() -> int:
    V = (sys.argv[1] if len(sys.argv) > 1 else 'A').upper()
    TAB = HERE / 'tables' / V
    QA = HERE / 'qa'
    MD = HERE / 'models_trainonly' / V
    TAB.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    X_hcv, X_sv, y, split = load_variant(V)
    te = np.flatnonzero(split == 'test')

    sc_seq = joblib.load(MD / 'scaler_seq.joblib')
    sc_sv = joblib.load(MD / 'scaler_sv.joblib')
    net = SepsisLSTM()
    net.load_state_dict(torch.load(MD / 'model_LSTM.pt', map_location='cpu'))
    net.eval()

    seq = X_hcv[te].transpose(0, 2, 1)
    A_te = sc_seq.transform(seq.reshape(-1, len(HCV))).reshape(
        len(te), 48, len(HCV)).astype(np.float32)
    B_te = sc_sv.transform(X_sv[te]).astype(np.float32)
    with torch.no_grad():
        logit, att = net(torch.tensor(A_te), torch.tensor(B_te), return_attn=True)
    p_te = torch.sigmoid(logit).numpy()
    att = att.numpy()
    y_te = y[te]
    ba = roc_auc_score(y_te, p_te)
    bp = average_precision_score(y_te, p_te)
    mean_att = att.mean(axis=0)
    print(f'[队列 {V}] LSTM 新权重 baseline AUC={ba:.4f} AUPRC={bp:.4f}')

    all_h = np.arange(48)
    n_te = len(te)
    B = 512

    def masked_prob(keep):
        out = np.empty(n_te)
        for i in range(0, n_te, B):
            n = min(B, n_te - i)
            m = np.zeros((n, 48), dtype=np.float32)
            for r in range(n):
                m[r, keep[i + r]] = 1.0
            with torch.no_grad():
                o, _ = net(torch.tensor(A_te[i:i + B]) * torch.tensor(m)[..., None],
                           torch.tensor(B_te[i:i + B]), return_attn=True)
                out[i:i + n] = torch.sigmoid(o).numpy()
        return out

    def run(p):
        return roc_auc_score(y_te, p), average_precision_score(y_te, p)

    rows = []
    for k in KS:
        per_top = np.argsort(att, 1)[:, ::-1][:, :k]                  # 注意力最高 k 小时
        per_low = np.argsort(att, 1)[:, :k]                           # 注意力最低 k 小时
        keep_pt = np.array([all_h[~np.isin(all_h, per_top[i])] for i in range(n_te)])
        keep_pl = np.array([all_h[~np.isin(all_h, per_low[i])] for i in range(n_te)])
        keep_gl = np.tile(all_h[~np.isin(all_h, np.argsort(mean_att)[-k:])], (n_te, 1))
        keep_gl_low = np.tile(all_h[~np.isin(all_h, np.argsort(mean_att)[:k])], (n_te, 1))

        a_pt, p_pt = run(masked_prob(keep_pt))
        a_pl, p_pl = run(masked_prob(keep_pl))
        a_glo, p_glo = run(masked_prob(keep_gl))
        a_glo_low, p_glo_low = run(masked_prob(keep_gl_low))
        ra, rp = [], []
        for s in range(NSEED):
            rng = np.random.default_rng(SEED + s)
            keep_r = np.stack([np.delete(all_h, rng.choice(48, k, replace=False))
                               for _ in range(n_te)])
            a_r, p_r = run(masked_prob(keep_r))
            ra.append(a_r); rp.append(p_r)
        ra, rp = np.array(ra), np.array(rp)

        rows.append({
            'k_hours': k,
            'dAUC_global_top': ba - a_glo,
            'dAUC_per_sample_top': ba - a_pt,
            'dAUC_per_sample_low': ba - a_pl,
            'dAUC_global_low': ba - a_glo_low,
            'dAUC_random_mean': ba - ra.mean(),
            'dAUC_random_sd': ra.std(),
            'AUPRC_per_sample_low': p_pl,
            'AUPRC_global_low': p_glo_low,
            'z_per_sample_low_vs_random':
                ((ba - a_pl) - (ba - ra.mean())) / ra.std() if ra.std() > 0 else np.nan,
        })
        r = rows[-1]
        print(f"  k={k:>2} | ΔAUC 逐样本top={r['dAUC_per_sample_top']:+.4f} "
              f"逐样本low={r['dAUC_per_sample_low']:+.4f} "
              f"全局top={r['dAUC_global_top']:+.4f} 全局low={r['dAUC_global_low']:+.4f} "
              f"随机={r['dAUC_random_mean']:+.4f}±{r['dAUC_random_sd']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(TAB / 'table8c_attention_occlusion_lowarm.csv', index=False)
    (QA / f's08b_occlusion_lowarm_{V}.json').write_text(json.dumps({
        'variant': V, 'seed': SEED, 'n_random_seeds': NSEED, 'ks': KS,
        'baseline_auc': ba, 'baseline_auprc': bp,
        'mean_att_top5_hours': [int(h) for h in np.argsort(mean_att)[-5:][::-1]],
        'mean_att_bottom5_hours': [int(h) for h in np.argsort(mean_att)[:5]],
        'rows': rows,
        'caveat': ('遮挡值=0（训练集均值），遮挡会破坏概率校准，故只报 AUC/AUPRC；'
                   '注意力权重只反映隐藏表示的加权，不能定位因果生理阶段'),
        'source_weights': f'models_trainonly/{V}/model_LSTM.pt',
    }, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print(f'-> {TAB}/table8c_attention_occlusion_lowarm.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
