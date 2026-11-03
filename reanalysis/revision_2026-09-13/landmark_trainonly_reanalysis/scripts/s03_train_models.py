#!/usr/bin/env python3
# =====================================================================
# s03_train_models.py —— 阶段4：在训练集独立数据上**真正重新训练**全部模型
#
# 用法：python s03_train_models.py A       # 队列A：数据库定义回顾性地标队列（24,894）
#       python s03_train_models.py B       # 队列B：t_inf≤48h 限制性地标队列（22,778）
#
# 与旧流水线（06_train_baseline.py / 06b_train_sequence.py）的差异：
#   1. 输入为训练集独立插补产物（data/trainonly_mimic.npz），不是旧 v2 插补文件；
#   2. 全部模型重新训练（旧 joblib / .pt 权重一律不加载）；
#   3. 标准化参数用**本变体的训练子集**重新拟合（训练集独立）；
#   4. 队列 B 额外把 t_inf>48h 的记录同时从训练/验证/测试中剔除，
#      使其成为真正"地标时点可判定脓毒症表型"的人群；
#   5. 预测覆盖 train/val/test 全部记录并保存 subject_id/split，
#      供后续患者级再校准与聚类 bootstrap 使用（旧 npz 只有 test）。
#
# 保持不变（用户要求"保持旧稿已确定的超参数、划分原则和随机种子"）：
#   · 超参数与 06/06b 完全一致；· 受试者级分组划分与 SEED=42；
#   · 验证集 F3 阈值扫描区间与步长（0.10–0.85，0.01）。
#
# 输出：
#   models_trainonly/<variant>/model_*.joblib, model_LSTM.pt, scalers
#   data/preds_<variant>.npz
#   tables/results_<variant>.csv
# =====================================================================
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent.parent.parent
DATA, TAB, QA = HERE / 'data', HERE / 'tables', HERE / 'qa'
for d in (DATA, TAB, QA):
    d.mkdir(exist_ok=True)

SEED = 42
HCV_VARS = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'spo2',
            'temperature', 'glucose', 'fio2', 'ph',
            'gcs', 'gcs_eyes', 'gcs_motor', 'gcs_verbal']
MODELS_CLS = ['LogisticRegression', 'DecisionTree', 'RandomForest', 'KNN', 'SVM',
              'GaussianNB']

# ---- 与旧脚本逐字一致的超参数 ----
BASELINE_KW = {
    'LogisticRegression': dict(max_iter=2000, class_weight='balanced', random_state=SEED),
    'DecisionTree': dict(max_depth=10, class_weight='balanced', random_state=SEED),
    'RandomForest': dict(n_estimators=200, max_depth=15, class_weight='balanced',
                         n_jobs=-1, random_state=SEED),
    'KNN': dict(n_neighbors=7, weights='distance'),
    'SVM': None,                     # CalibratedClassifierCV(SVC(rbf,C=1,class_weight=balanced))
    'GaussianNB': dict(),
}
LSTM_CFG = dict(hidden=64, drop=0.35, lr=5e-4, epochs=80, batch=256, patience=10)


def fbeta(p, r, beta=3.0):
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r) if (beta ** 2 * p + r) > 0 else 0.0


def select_best_threshold(y_val, prob_val):
    """与 06_train_baseline.py 完全一致的验证集 F3 扫描。"""
    best_f3, best_th = 0.0, 0.5
    for th in np.arange(0.10, 0.86, 0.01):
        yp = (prob_val >= th).astype(int)
        p = precision_score(y_val, yp, zero_division=0)
        r = recall_score(y_val, yp, zero_division=0)
        f3 = fbeta(p, r)
        if f3 > best_f3:
            best_f3, best_th = f3, th
    return best_th, best_f3


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, 1)

    def forward(self, h):
        att = torch.softmax(self.W(h).squeeze(-1), dim=1)
        return torch.bmm(att.unsqueeze(1), h).squeeze(1), att


class SepsisLSTM(nn.Module):
    """与 06b_train_sequence.py 结构逐行一致；forward 额外返回注意力权重。"""

    def __init__(self, n_seq=14, n_static=4, hidden=64, drop=0.35):
        super().__init__()
        self.lstm = nn.LSTM(n_seq, hidden, batch_first=True, bidirectional=True)
        self.attn = AttentionPool(2 * hidden)
        self.drop = nn.Dropout(drop)
        self.static = nn.Sequential(nn.Linear(n_static, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * hidden + 16, 64), nn.ReLU(),
                                  nn.Dropout(drop), nn.Linear(64, 1))

    def forward(self, xs, xsv, return_attn=False):
        out, _ = self.lstm(xs)
        ctx, att = self.attn(out)
        logit = self.head(torch.cat([self.drop(ctx), self.static(xsv)], 1)).squeeze(1)
        return (logit, att) if return_attn else logit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('variant', choices=['A', 'B'])
    args = ap.parse_args()
    V = args.variant

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    DEVICE = 'cpu'                       # 本机 MPS 不可用；固定 CPU 以保证可复现
    print(f'=== 变体 {V} | 设备 {DEVICE} | torch {torch.__version__} ===')

    # ---------- 1. 加载训练集独立数据 ----------
    d = np.load(DATA / 'trainonly_mimic.npz', allow_pickle=False)
    X_hcv_all = d['X_hcv'].astype(np.float32)
    X_sv_all = d['X_sv'].astype(np.float32)
    X_flat_all = d['X_flat'].astype(np.float32)
    y_all = d['y'].astype(np.int64)
    stay_all = d['stay_id'].astype(np.int64)
    subj_all = d['subject_id'].astype(np.int64)
    split_all = d['split'].astype(str)
    tinf_all = d['t_inf_h'].astype(np.float64)

    mask = np.ones(len(y_all), bool) if V == 'A' else (tinf_all <= 48)
    X_hcv, X_sv, X_flat = X_hcv_all[mask], X_sv_all[mask], X_flat_all[mask]
    y, stay, subj, split, tinf = (y_all[mask], stay_all[mask], subj_all[mask],
                                  split_all[mask], tinf_all[mask])
    tr = np.flatnonzero(split == 'train')
    va = np.flatnonzero(split == 'val')
    te = np.flatnonzero(split == 'test')
    print(f'队列 {V}: {len(y)} 条记录 | {len(np.unique(subj))} 受试者 | '
          f'死亡 {y.sum()} ({100*y.mean():.2f}%)')
    print(f'  train {len(tr)} | val {len(va)} | test {len(te)} | '
          f'死亡 train {y[tr].sum()} val {y[va].sum()} test {y[te].sum()}')
    # 受试者级互斥断言
    assert not (set(subj[tr]) & set(subj[va])) and not (set(subj[tr]) & set(subj[te])) \
        and not (set(subj[va]) & set(subj[te])), '受试者跨集泄漏'

    outdir = HERE / 'models_trainonly' / V
    outdir.mkdir(parents=True, exist_ok=True)
    timings = {}

    # ---------- 2. 经典模型 ----------
    sc_flat = StandardScaler().fit(X_flat[tr])
    F_tr, F_va, F_te = sc_flat.transform(X_flat[tr]), sc_flat.transform(X_flat[va]), \
        sc_flat.transform(X_flat[te])
    print(f'\n[scaler] X_flat 训练集拟合: mean[:3]={np.round(sc_flat.mean_[:3], 4)}')

    probs = {}          # model -> [n] 全队列预测
    thresholds = {}
    rows = []
    for name in MODELS_CLS:
        t0 = time.time()
        if name == 'SVM':
            clf = CalibratedClassifierCV(
                SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=SEED),
                ensemble=False)
        else:
            clf = {'LogisticRegression': LogisticRegression,
                   'DecisionTree': DecisionTreeClassifier,
                   'RandomForest': RandomForestClassifier,
                   'KNN': KNeighborsClassifier,
                   'GaussianNB': GaussianNB}[name](**BASELINE_KW[name])
        clf.fit(F_tr, y[tr])
        p_tr = clf.predict_proba(F_tr)[:, 1]
        p_va = clf.predict_proba(F_va)[:, 1]
        p_te = clf.predict_proba(F_te)[:, 1]
        best_th, best_f3 = select_best_threshold(y[va], p_va)
        thresholds[name] = float(best_th)
        full = np.empty(len(y), float)
        full[tr], full[va], full[te] = p_tr, p_va, p_te
        probs[name] = full
        for tag, idx, p in [('train', tr, p_tr), ('val', va, p_va), ('test', te, p_te)]:
            b = (p >= best_th).astype(int)
            rows.append(dict(model=name, split=tag, n=len(idx), deaths=int(y[idx].sum()),
                             AUC=roc_auc_score(y[idx], p),
                             AUPRC=average_precision_score(y[idx], p),
                             threshold=best_th, Accuracy=accuracy_score(y[idx], b),
                             Precision=precision_score(y[idx], b, zero_division=0),
                             Recall=recall_score(y[idx], b, zero_division=0),
                             F1=f1_score(y[idx], b, zero_division=0),
                             F3=fbeta(precision_score(y[idx], b, zero_division=0),
                                      recall_score(y[idx], b, zero_division=0)),
                             val_F3=best_f3))
        joblib.dump(clf, outdir / f'model_{name}.joblib')
        timings[name] = round(time.time() - t0, 2)
        print(f'  {name:20s} val AUC={roc_auc_score(y[va], p_va):.4f} '
              f'th={best_th:.2f} test AUC={roc_auc_score(y[te], p_te):.4f} '
              f'({timings[name]}s)')
    joblib.dump(sc_flat, outdir / 'scaler_flat.joblib')

    # ---------- 3. LSTM（真正重新训练）----------
    print('\n[LSTM] 重新训练…')
    t0 = time.time()
    X_seq = X_hcv.transpose(0, 2, 1)                     # [n,48,14]
    sc_seq = StandardScaler().fit(X_seq[tr].reshape(-1, len(HCV_VARS)))
    sc_sv = StandardScaler().fit(X_sv[tr])
    Xs = sc_seq.transform(X_seq.reshape(-1, len(HCV_VARS))).reshape(
        len(y), 48, len(HCV_VARS)).astype(np.float32)
    Xv = sc_sv.transform(X_sv).astype(np.float32)
    joblib.dump(sc_seq, outdir / 'scaler_seq.joblib')
    joblib.dump(sc_sv, outdir / 'scaler_sv.joblib')

    n_pos, n_neg = int((y[tr] == 1).sum()), int((y[tr] == 0).sum())
    pos_w = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=DEVICE)
    print(f'  pos_weight={pos_w.item():.4f} | scaler 仅用训练集拟合')

    torch.manual_seed(SEED)
    model = SepsisLSTM(hidden=LSTM_CFG['hidden'], drop=LSTM_CFG['drop']).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LSTM_CFG['lr'])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    def T(i_arr):
        return (torch.tensor(Xs[i_arr], device=DEVICE),
                torch.tensor(Xv[i_arr], device=DEVICE),
                torch.tensor(y[i_arr], dtype=torch.float32, device=DEVICE))

    best_val, patience, ep_done = float('inf'), LSTM_CFG['patience'], 0
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
            torch.save(model.state_dict(), outdir / 'model_LSTM.pt')
        else:
            patience -= 1
            if patience == 0:
                print(f'  早停于 epoch {ep_done}')
                break
        if ep_done % 10 == 0:
            print(f'  epoch {ep_done:3d} | val loss={val_loss:.4f}')
    timings['LSTM'] = round(time.time() - t0, 2)
    print(f'  best val loss={best_val:.4f} | 训练了 {ep_done} epochs | {timings["LSTM"]}s')

    model.load_state_dict(torch.load(outdir / 'model_LSTM.pt'))
    model.eval()
    with torch.no_grad():
        p_all = torch.sigmoid(model(torch.tensor(Xs, device=DEVICE),
                                    torch.tensor(Xv, device=DEVICE))).cpu().numpy()
    p_tr, p_va, p_te = p_all[tr], p_all[va], p_all[te]
    best_th, best_f3 = select_best_threshold(y[va], p_va)
    thresholds['LSTM'] = float(best_th)
    probs['LSTM'] = p_all
    rows.append(dict(model='LSTM', split='train', n=len(tr), deaths=int(y[tr].sum()),
                     AUC=roc_auc_score(y[tr], p_tr), AUPRC=average_precision_score(y[tr], p_tr),
                     threshold=best_th, Accuracy=accuracy_score(y[tr], (p_tr >= best_th)),
                     Precision=precision_score(y[tr], (p_tr >= best_th), zero_division=0),
                     Recall=recall_score(y[tr], (p_tr >= best_th), zero_division=0),
                     F1=f1_score(y[tr], (p_tr >= best_th), zero_division=0),
                     F3=best_f3, val_F3=best_f3))
    for tag, idx, p in [('val', va, p_va), ('test', te, p_te)]:
        b = (p >= best_th).astype(int)
        pr = precision_score(y[idx], b, zero_division=0)
        rc = recall_score(y[idx], b, zero_division=0)
        rows.append(dict(model='LSTM', split=tag, n=len(idx), deaths=int(y[idx].sum()),
                         AUC=roc_auc_score(y[idx], p), AUPRC=average_precision_score(y[idx], p),
                         threshold=best_th, Accuracy=accuracy_score(y[idx], b),
                         Precision=pr, Recall=rc, F1=f1_score(y[idx], b, zero_division=0),
                         F3=fbeta(pr, rc), val_F3=best_f3))
    print(f'  LSTM test AUC={roc_auc_score(y[te], p_te):.4f} th={best_th:.2f}')

    # ---------- 4. 保存 ----------
    np.savez(DATA / f'preds_{V}.npz',
             **{f'prob_{k}': v.astype(np.float32) for k, v in probs.items()},
             y_true=y, stay_id=stay, subject_id=subj, split=split.astype('U8'),
             t_inf_h=tinf.astype(np.float32),
             cohorts=np.array([V] * len(y), dtype='U1'),
             **{f'threshold_{k}': np.float64(v) for k, v in thresholds.items()})

    res = pd.DataFrame(rows)
    res.to_csv(TAB / f'results_{V}.csv', index=False)
    print(f'\n=== 测试集结果（变体 {V}）===')
    print(res[res.split == 'test'].drop(columns=['val_F3']).round(4).to_string(index=False))

    meta = {'variant': V, 'n_records': int(len(y)),
            'n_subjects': int(len(np.unique(subj))),
            'n_train': int(len(tr)), 'n_val': int(len(va)), 'n_test': int(len(te)),
            'deaths_total': int(y.sum()), 'deaths_train': int(y[tr].sum()),
            'deaths_val': int(y[va].sum()), 'deaths_test': int(y[te].sum()),
            'train_only_fit': 'static mean + all scalers fit on this variant train split only',
            'old_weights_loaded': False, 'old_predictions_loaded': False,
            'lstm_retrained': True, 'epochs_run': int(ep_done),
            'thresholds_from': 'validation set F3 scan 0.10-0.85 step 0.01',
            'thresholds': thresholds, 'timings_sec': timings,
            'torch_version': torch.__version__, 'device': DEVICE,
            'lstm_cfg': LSTM_CFG, 'seed': SEED}
    (QA / f'train_meta_{V}.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n-> data/preds_{V}.npz | tables/results_{V}.csv | qa/train_meta_{V}.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
