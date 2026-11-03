#!/usr/bin/env python3
"""s06_lstm_seed_stability.py —— LSTM 重训随机性对结论稳定性的影响

动机（2026-09-13 发现）：
  s03 的 LSTM 是**真正重新训练**的。与旧稿 LSTM 相比，新 LSTM 在测试集上的
  预测相关系数 r=0.968、平均绝对差 0.043，远大于 6 个经典模型（RF r=0.9991）。
  直接后果是亚组交互检验 p 值从旧的 0.301 变为 0.903（同一数据、同一代码、
  仅 LSTM 权重不同）。因此必须量化"LSTM 重训随机性"这一不确定性来源，
  否则任何基于单次 LSTM 的交互/配对推断都不可靠。

本脚本做什么：
  在**完全相同的**训练集独立数据、划分（SEED=42）、超参下，只用不同随机种子
  重新初始化并重训 attention-based BiLSTM，重复 N_SEEDS 次，报告：
    - 每次的 test AUC / AUPRC / 验证集 F3 阈值
    - 每次的 ΔAUC(LSTM−RF) 与 DeLong p（RF 固定为 s03 的 A 变体预测）
    - 每次的 3 个亚组交互检验 p（Age / Initial SOFA / Cardiovascular SOFA）
  从而给出 LSTM 性能与推断的**种子间离散度**，而不是单点估计。

注意：这不改变主分析。主分析仍用 SEED=42 的 LSTM（与旧稿同种子）。
     本脚本只是给出不确定性范围，供正文如实说明。

输出:
  tables/A/lstm_seed_stability.csv        每次种子的明细
  qa/s06_lstm_seed_stability.json         汇总（均值/标准差/极差/结论稳定性）
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_eval as C  # noqa: E402
from s03_train_models import LSTM_CFG, HCV_VARS, SepsisLSTM, select_best_threshold  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent.parent.parent
DATA, TAB, QA = HERE / 'data', HERE / 'tables' / 'A', HERE / 'qa'
for d in (DATA, TAB, QA):
    d.mkdir(parents=True, exist_ok=True)

V = 'A'
SEEDS = [42, 43, 44, 45, 46, 47]
DEVICE = 'cpu'


def fit_lstm(Xs, Xv, y, tr, va, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_pos, n_neg = int((y[tr] == 1).sum()), int((y[tr] == 0).sum())
    pos_w = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=DEVICE)
    model = SepsisLSTM(hidden=LSTM_CFG['hidden'], drop=LSTM_CFG['drop']).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LSTM_CFG['lr'])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    def T(i):
        return (torch.tensor(Xs[i], device=DEVICE),
                torch.tensor(Xv[i], device=DEVICE),
                torch.tensor(y[i], dtype=torch.float32, device=DEVICE))

    best_val, patience, ep_done, best_state = float('inf'), LSTM_CFG['patience'], 0, None
    for ep in range(LSTM_CFG['epochs']):
        model.train()
        perm = np.random.permutation(len(tr))
        for b in range(0, len(tr), LSTM_CFG['batch']):
            xs, xv, yy = T(tr[perm[b:b + LSTM_CFG['batch']]])
            opt.zero_grad()
            loss_fn(model(xs, xv), yy).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            xs, xv, yy = T(va)
            val_loss = loss_fn(model(xs, xv), yy).item()
        ep_done = ep + 1
        if val_loss < best_val - 1e-4:
            best_val, patience = val_loss, LSTM_CFG['patience']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience == 0:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(Xs, device=DEVICE),
                                torch.tensor(Xv, device=DEVICE))).cpu().numpy()
    return p, best_val, ep_done


def interaction_p(y, lp, g):
    X = np.column_stack([np.ones(len(y)), lp, g, lp * g])
    m = sm.Logit(y, X).fit(disp=0)
    return float(2 * (1 - scipy.stats.norm.cdf(abs(m.params[3] / m.bse[3])))), float(m.params[3])


def main() -> int:
    d = np.load(DATA / 'trainonly_mimic.npz', allow_pickle=False)
    pred = np.load(DATA / f'preds_{V}.npz', allow_pickle=False)
    X_hcv = d['X_hcv'].astype(np.float32)
    X_sv = d['X_sv'].astype(np.float32)
    y = d['y'].astype(np.int64)
    stay = d['stay_id'].astype(np.int64)
    subj = d['subject_id'].astype(np.int64)
    split = d['split'].astype(str)

    mask = np.ones(len(y), bool)   # 变体 A
    X_hcv, X_sv, y = X_hcv[mask], X_sv[mask], y[mask]
    stay, subj, split = stay[mask], subj[mask], split[mask]
    tr = np.flatnonzero(split == 'train')
    va = np.flatnonzero(split == 'val')
    te = np.flatnonzero(split == 'test')

    X_seq = X_hcv.transpose(0, 2, 1)
    sc_seq = StandardScaler().fit(X_seq[tr].reshape(-1, len(HCV_VARS)))
    sc_sv = StandardScaler().fit(X_sv[tr])
    Xs = sc_seq.transform(X_seq.reshape(-1, len(HCV_VARS))).reshape(
        len(y), 48, len(HCV_VARS)).astype(np.float32)
    Xv = sc_sv.transform(X_sv).astype(np.float32)
    print(f'队列 {V}: {len(y)} 条 | train {len(tr)} val {len(va)} test {len(te)}')
    print(f'种子集合: {SEEDS}（每次从零重训 LSTM；scaler 与划分完全固定）')

    # 交互检验所需的亚组变量（固定不变）
    se = pd.read_csv(ROOT / 'sofa_extract.csv')
    se['stay_id'] = se['stay_id'].astype(np.int64)
    sofa = np.load(ROOT / 'sofa_predictions.npz')
    aux = pd.DataFrame({'stay_id': stay, 'age': X_sv[:, 0]})
    aux = aux.merge(pd.DataFrame({'stay_id': sofa['stay_id'].astype(np.int64),
                                  'SOFA': sofa['sofa'].astype(float)}), on='stay_id', how='left')
    aux = aux.merge(se[['stay_id', 'cardiovascular']], on='stay_id', how='left')
    age_g = (aux.age.values >= 65).astype(float)              # 0=age<65, 1=age>=65
    sofa_g = (aux.SOFA.values >= 8).astype(float)             # 0=SOFA<8, 1=SOFA>=8
    shock_g = (aux.cardiovascular.values >= 2).astype(float)  # 0=nonshock, 1=shock_proxy
    y_te = y[te]

    rf_te = pred['prob_RandomForest'].astype(float)[te]
    rf_auc = roc_auc_score(y_te, rf_te)

    rows = []
    for s in SEEDS:
        t0 = time.time()
        p_all, bvl, eps = fit_lstm(Xs, Xv, y, tr, va, s)
        p_te = p_all[te]
        th, _ = select_best_threshold(y[va], p_all[va])
        auc = roc_auc_score(y_te, p_te)
        ap = average_precision_score(y_te, p_te)
        dl = C.delong_test(y_te, p_te, rf_te)
        lp = C._logit(p_te)
        p_age, b_age = interaction_p(y_te, lp, age_g[te])
        p_sofa, b_sofa = interaction_p(y_te, lp, sofa_g[te])
        p_shock, b_shock = interaction_p(y_te, lp, shock_g[te])
        rows.append({
            'seed': s, 'epochs': eps, 'best_val_loss': round(bvl, 4),
            'F3_threshold': th, 'test_AUC': round(auc, 4), 'test_AUPRC': round(ap, 4),
            'RF_test_AUC_fixed': round(rf_auc, 4),
            'dAUC_LSTM_minus_RF': round(auc - rf_auc, 4),
            'DeLong_p_vs_RF': round(dl['p'], 4),
            'LSTM_higher_than_RF': bool(auc > rf_auc),
            'interaction_p_Age': round(p_age, 4), 'beta_Age': round(b_age, 4),
            'interaction_p_SOFA': round(p_sofa, 4), 'beta_SOFA': round(b_sofa, 4),
            'interaction_p_CVSOFA': round(p_shock, 4), 'beta_CVSOFA': round(b_shock, 4),
            'seconds': round(time.time() - t0, 1),
        })
        print(f'  seed={s} | AUC={auc:.4f} AUPRC={ap:.4f} th={th:.2f} '
              f'ΔAUC_vs_RF={auc-rf_auc:+.4f} p={dl["p"]:.4f} | '
              f'interaction p: Age={p_age:.3f} SOFA={p_sofa:.3f} CV={p_shock:.3f} '
              f'| {rows[-1]["seconds"]}s')

    res = pd.DataFrame(rows)
    res.to_csv(TAB / 'lstm_seed_stability.csv', index=False)

    summ = {
        'seeds': SEEDS, 'n_seeds': len(SEEDS), 'variant': V,
        'note': ('同一数据/划分/超参，仅随机种子不同（重新初始化+重训）。'
                 'RF 预测固定为 s03 变体 A 的产物，用于配对比较。'),
        'test_AUC_mean': round(float(res.test_AUC.mean()), 4),
        'test_AUC_sd': round(float(res.test_AUC.std(ddof=1)), 4),
        'test_AUC_min': round(float(res.test_AUC.min()), 4),
        'test_AUC_max': round(float(res.test_AUC.max()), 4),
        'test_AUC_range': round(float(res.test_AUC.max() - res.test_AUC.min()), 4),
        'test_AUPRC_mean': round(float(res.test_AUPRC.mean()), 4),
        'test_AUPRC_sd': round(float(res.test_AUPRC.std(ddof=1)), 4),
        'dAUC_vs_RF_mean': round(float(res.dAUC_LSTM_minus_RF.mean()), 4),
        'dAUC_vs_RF_min': round(float(res.dAUC_LSTM_minus_RF.min()), 4),
        'dAUC_vs_RF_max': round(float(res.dAUC_LSTM_minus_RF.max()), 4),
        'n_seeds_LSTM_above_RF': int(res.LSTM_higher_than_RF.sum()),
        'n_seeds_DeLong_p_lt_0.05': int((res.DeLong_p_vs_RF < 0.05).sum()),
        'F3_threshold_values': sorted(res.F3_threshold.unique().tolist()),
        'interaction_p_ranges': {
            'Age': [round(float(res.interaction_p_Age.min()), 4),
                    round(float(res.interaction_p_Age.max()), 4)],
            'SOFA': [round(float(res.interaction_p_SOFA.min()), 4),
                     round(float(res.interaction_p_SOFA.max()), 4)],
            'CVSOFA': [round(float(res.interaction_p_CVSOFA.min()), 4),
                       round(float(res.interaction_p_CVSOFA.max()), 4)],
        },
        'any_interaction_significant_at_0.05': bool(
            (res[['interaction_p_Age', 'interaction_p_SOFA', 'interaction_p_CVSOFA']] < 0.05)
            .any().any()),
        'torch_version': torch.__version__, 'device': DEVICE,
    }
    (QA / 's06_lstm_seed_stability.json').write_text(
        json.dumps(summ, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n===== 汇总 =====')
    print(json.dumps(summ, ensure_ascii=False, indent=2))
    print(f'\n-> {TAB}/lstm_seed_stability.csv | {QA}/s06_lstm_seed_stability.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
