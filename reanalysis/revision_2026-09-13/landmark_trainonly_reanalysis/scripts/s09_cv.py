#!/usr/bin/env python3
# =====================================================================
# s09_cv.py —— 交叉验证重算（在"训练集独立插补"的新数据上）
#
# 背景：
#   一位外部审阅者指出原分析的静态变量插补在 train/val/test 划分之前
#   用整个开发队列的均值完成，构成预处理泄漏。landmark 重分析已产出
#   训练集独立插补的新数据 data/trainonly_mimic.npz（s02），并在此数据上
#   真正重训了全部模型（s03）。因此旧稿 Table 3 的交叉验证结果也是在
#   有泄漏的数据上算出的，必须用新数据重算。
#
# 本脚本只做一件事：重算
#   (1) RandomForest 10 折受试者级分层交叉验证  → tables/A/table3_cv_10fold.csv
#   (2) attention-based BiLSTM 5 折受试者级分层交叉验证
#                                               → tables/A/table3_cv_lstm_5fold.csv
#
# 纪律（逐条对应任务要求）：
#   · 数据只来自 data/trainonly_mimic.npz；只评估 cohort A（全队列 24,894 条）
#   · 受试者级分组：StratifiedGroupKFold(groups=subject_id, seed=42, shuffle=True)
#   · RF n_splits=10，LSTM n_splits=5
#   · 每一折内部 StandardScaler().fit(该折训练部分) 后 transform 验证部分
#   · 每一折模型重新初始化、重新训练（不加载 models_trainonly/ 权重）
#   · 特征构造与 s03 逐行一致：14 时变变量 mean/min/max + 4 静态 = 46 维；
#     LSTM 序列 = X_hcv.transpose(0,2,1) → [n,48,14]，静态 = X_sv
#   · RF 超参与 LSTM 架构/超参全部照抄 s03（不改动）
#   · 设备 cpu（与 s03 一致）
#
# 输出：
#   tables/A/table3_cv_10fold.csv      （Model, fold, AUC, AP + mean/SD 行）
#   tables/A/table3_cv_lstm_5fold.csv  （同结构）
#   qa/s09_cv.json                     （数据哈希、折划分、泄漏检查、
#                                       每折 scaler 证据、逐折指标、版本等）
# =====================================================================
from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# 路径（全部写入 landmark_trainonly_reanalysis/，不触碰任何旧目录）
# ---------------------------------------------------------------------
HERE = Path(__file__).resolve().parent.parent          # .../landmark_trainonly_reanalysis
DATA = HERE / 'data'
VARIANT = (sys.argv[1].upper() if len(sys.argv) > 1 else 'A')
if VARIANT not in ('A', 'B'):
    raise SystemExit(f'用法: python s09_cv.py A|B（收到 {VARIANT!r}）')
TAB_A = HERE / 'tables' / VARIANT                      # 兼容旧名，实际按变体落盘
QA = HERE / 'qa'
LOGS = HERE / 'logs'
for d in (DATA, TAB_A, QA, LOGS):
    d.mkdir(parents=True, exist_ok=True)

NPZ_PATH = DATA / 'trainonly_mimic.npz'

SEED = 42
DEVICE = 'cpu'          # 与 s03 一致（s03 固定 cpu 以保证可复现）

HCV_VARS = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'spo2',
            'temperature', 'glucose', 'fio2', 'ph',
            'gcs', 'gcs_eyes', 'gcs_motor', 'gcs_verbal']          # 14 个时变变量

# ---- 与旧稿 / s03 逐字一致的超参数（不得改动）----
RF_KW = dict(n_estimators=200, max_depth=15, class_weight='balanced',
             n_jobs=-1, random_state=SEED)
LSTM_CFG = dict(hidden=64, drop=0.35, lr=5e-4, epochs=80, batch=256, patience=10)


# =====================================================================
# 模型定义：逐字复制自 scripts/s03_train_models.py（不得修改）
# =====================================================================
class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, 1)

    def forward(self, h):
        att = torch.softmax(self.W(h).squeeze(-1), dim=1)
        return torch.bmm(att.unsqueeze(1), h).squeeze(1), att


class SepsisLSTM(nn.Module):
    """与 06b_train_sequence.py / s03_train_models.py 结构逐行一致。"""

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


# =====================================================================
# 工具
# =====================================================================
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def py(o):
    """把 numpy / torch 标量转成 JSON 可序列化类型（bool 必须先于 int 判断）。"""
    if isinstance(o, dict):
        return {k: py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [py(v) for v in o]
    if isinstance(o, bool):
        return bool(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def make_folds(X, y, groups, n_splits):
    """受试者级分层 K 折；返回 (splitter, folds[(tr_idx, va_idx)...])。"""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = [(np.asarray(tr, dtype=int), np.asarray(va, dtype=int))
             for tr, va in sgkf.split(X, y, groups=groups)]
    return sgkf, folds


def leak_check(folds, groups):
    """跨折 / 折内训练-验证 的 subject 集合互斥检查。"""
    val_sets = [set(groups[va].tolist()) for _, va in folds]
    pairwise = []
    ok = True
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            inter = val_sets[i] & val_sets[j]
            if inter:
                ok = False
            pairwise.append({'fold_i': i + 1, 'fold_j': j + 1,
                             'n_subject_intersection': len(inter)})
    train_val_within = []
    within_ok = True
    for k, (tr, va) in enumerate(folds):
        inter = set(groups[tr].tolist()) & set(groups[va].tolist())
        if inter:
            within_ok = False
        train_val_within.append({'fold': k + 1, 'n_subject_intersection': len(inter)})
    union = set()
    for s in val_sets:
        union |= s
    coverage = {
        'n_subjects_total': int(len(np.unique(groups))),
        'n_subjects_in_val_union': int(len(union)),
        'union_equals_all_subjects': bool(len(union) == len(np.unique(groups))),
        'sum_of_val_subject_counts': int(sum(len(s) for s in val_sets)),
        'each_subject_in_exactly_one_val_fold':
            bool(sum(len(s) for s in val_sets) == len(union) == len(np.unique(groups))),
    }
    return bool(ok and within_ok), {
        'subject_disjoint_across_val_folds': bool(ok),
        'subject_disjoint_train_vs_val_within_each_fold': bool(within_ok),
        'pairwise_val_intersections': pairwise,
        'train_val_within_fold_intersections': train_val_within,
        'val_fold_subject_partition': coverage,
    }


def to_tensors(Xs, Xv, yy):
    return (torch.tensor(Xs, device=DEVICE),
            torch.tensor(Xv, device=DEVICE),
            torch.tensor(yy, dtype=torch.float32, device=DEVICE))


# =====================================================================
# 主流程
# =====================================================================
def main() -> int:
    t_start = time.time()
    print(f'=== s09_cv | seed={SEED} | 设备={DEVICE} | '
          f'torch={torch.__version__} | sklearn={__import__("sklearn").__version__} ===')

    # ---------- 0. 数据来源与字段 ----------
    npz_sha = sha256_of(NPZ_PATH)
    d = np.load(NPZ_PATH, allow_pickle=False)
    fields = [{'name': k, 'shape': list(d[k].shape), 'dtype': str(d[k].dtype)}
              for k in d.files]

    X_hcv = d['X_hcv'].astype(np.float32)        # [n, 14, 48] 患者内局部插补后
    X_sv = d['X_sv'].astype(np.float32)          # [n, 4] 静态（训练集均值插补后）
    X_flat = d['X_flat'].astype(np.float32)      # [n, 46] 聚合特征
    y = d['y'].astype(np.int64)
    subject = d['subject_id'].astype(np.int64)
    stay = d['stay_id'].astype(np.int64)
    t_inf = d['t_inf_h'].astype(np.float64)

    # ---- 变体 B：只保留疑似感染时间 ≤ intime+48h 的限制性地标队列 ----
    # 权威依据是 preds_B.npz（由 s03 变体 B 写出）；此处用 t_inf_h ≤ 48 重建掩码
    # 并逐字段核对 stay_id 的顺序与集合完全一致，避免"以为在算 B 其实在算 A"。
    b_mask = None
    if VARIANT == 'B':
        b_mask = t_inf <= 48.0
        ref = np.load(DATA / 'preds_B.npz')
        same_set = set(stay[b_mask].tolist()) == set(ref['stay_id'].tolist())
        same_order = bool(np.array_equal(stay[b_mask], ref['stay_id']))
        print(f'[变体B掩码自检] n={int(b_mask.sum())} | 与 preds_B.npz stay_id '
              f'集合一致={same_set} | 顺序一致={same_order}')
        if not (same_set and same_order):
            raise SystemExit('变体 B 掩码与 preds_B.npz 不一致，停止以避免算错队列')
        X_hcv, X_sv, X_flat = X_hcv[b_mask], X_sv[b_mask], X_flat[b_mask]
        y, subject, stay, t_inf = y[b_mask], subject[b_mask], stay[b_mask], t_inf[b_mask]

    n = len(y)
    print(f'数据源: {NPZ_PATH}')
    print(f'  sha256: {npz_sha}')
    print(f'  cohort {VARIANT}: n={n} | 受试者={len(np.unique(subject))} | '
          f'死亡={int(y.sum())} ({100 * y.mean():.2f}%)')

    # ---- 特征构造自检：npz 的 X_flat == s03/s02 的 46 维构造 ----
    # s02 在 float64 的 V_imp 上算 mean/min/max 再 .astype(float32) 存入 npz；
    # 因此这里同样用 float64 重建（npz 只保存了 float32，残余差异仅来自
    # 输入张量的 float32 存储舍入，量级 ~1.5e-5）。
    feat_parts = []
    for j in range(len(HCV_VARS)):
        s = X_hcv[:, j, :].astype(np.float64)
        feat_parts += [s.mean(1), s.min(1), s.max(1)]
    X_flat_rebuilt = np.column_stack(
        feat_parts + [X_sv[:, 0].astype(np.float64), X_sv[:, 1].astype(np.float64),
                      X_sv[:, 2].astype(np.float64), X_sv[:, 3].astype(np.float64)]
    ).astype(np.float32)
    flat_max_abs_diff = float(np.abs(X_flat_rebuilt - X_flat).max())
    flat_rebuild_ok = bool(np.allclose(X_flat_rebuilt, X_flat, rtol=0, atol=1e-4))
    print(f'[自检] X_flat(42 mean/min/max + 4 static) float64 重建 vs npz: '
          f'max|diff|={flat_max_abs_diff:.3e} | 一致(atol=1e-4)={flat_rebuild_ok}')

    # ---- LSTM 输入（与 s03 完全相同的转置方式）----
    X_seq = X_hcv.transpose(0, 2, 1)             # [n, 48, 14]
    print(f'[LSTM 输入] X_seq={X_seq.shape} (X_hcv.transpose(0,2,1)) | X_sv={X_sv.shape}')

    # ---------- 1. 折划分 ----------
    X_for_split = X_flat
    sgkf_rf, folds_rf = make_folds(X_for_split, y, subject, n_splits=10)
    sgkf_lstm, folds_lstm = make_folds(X_for_split, y, subject, n_splits=5)
    splitter_params_rf = {'n_splits': 10, 'shuffle': True, 'random_state': SEED,
                          'groups': 'subject_id'}
    splitter_params_lstm = {'n_splits': 5, 'shuffle': True, 'random_state': SEED,
                            'groups': 'subject_id'}

    # ---- 泄漏检查：用同 seed 的 StratifiedGroupKFold 原样重跑一遍 ----
    _, folds_rf_recheck = make_folds(X_for_split, y, subject, n_splits=10)
    _, folds_lstm_recheck = make_folds(X_for_split, y, subject, n_splits=5)
    folds_reproduced = {
        'rf_identical': all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                            for a, b in zip(folds_rf, folds_rf_recheck)),
        'lstm_identical': all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                              for a, b in zip(folds_lstm, folds_lstm_recheck)),
    }
    leak_rf_ok, leak_rf = leak_check(folds_rf, subject)
    leak_lstm_ok, leak_lstm = leak_check(folds_lstm, subject)
    print(f'[泄漏检查] RF 10 折 subject 跨折不相交={leak_rf_ok} | '
          f'LSTM 5 折 subject 跨折不相交={leak_lstm_ok} | '
          f'重跑折一致={folds_reproduced}')

    # ---------- 2. RandomForest 10 折 ----------
    print('\n=== RandomForest 10 折受试者级分层 CV ===')
    rf_rows, rf_scaler_evidence, rf_details = [], [], []
    for k, (tr, va) in enumerate(folds_rf, 1):
        t0 = time.time()
        scaler = StandardScaler().fit(X_flat[tr])          # ← 只 fit 该折训练部分
        Xtr = scaler.transform(X_flat[tr])
        Xva = scaler.transform(X_flat[va])
        rf = RandomForestClassifier(**RF_KW)               # ← 每折重新初始化
        rf.fit(Xtr, y[tr])
        p_va = rf.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(y[va], p_va)
        ap = average_precision_score(y[va], p_va)
        rf_rows.append(dict(Model='RandomForest', fold=k, AUC=float(auc), AP=float(ap)))
        rf_scaler_evidence.append({'fold': k, 'n_train': int(len(tr)), 'n_val': int(len(va)),
                                   'mean_first3': [float(v) for v in scaler.mean_[:3]],
                                   'scale_first3': [float(v) for v in scaler.scale_[:3]]})
        rf_details.append({'fold': k, 'n_train': int(len(tr)), 'n_val': int(len(va)),
                           'deaths_val': int(y[va].sum()), 'AUC': float(auc), 'AP': float(ap),
                           'seconds': round(time.time() - t0, 2)})
        print(f'  Fold {k:>2}: AUC={auc:.4f} | AP={ap:.4f} '
              f'| n_train={len(tr)} ({rf_details[-1]["seconds"]}s)')
    rf_auc = np.array([r['AUC'] for r in rf_rows])
    rf_ap = np.array([r['AP'] for r in rf_rows])
    print(f'  AUC: {rf_auc.mean():.4f} ± {rf_auc.std(ddof=1):.4f} '
          f'(范围 {rf_auc.min():.4f}–{rf_auc.max():.4f})')
    print(f'  AP : {rf_ap.mean():.4f} ± {rf_ap.std(ddof=1):.4f} '
          f'(范围 {rf_ap.min():.4f}–{rf_ap.max():.4f})')

    # ---------- 3. LSTM 5 折 ----------
    print('\n=== BiLSTM 5 折受试者级分层 CV ===')
    lstm_rows, lstm_seq_scaler_evidence, lstm_sv_scaler_evidence, lstm_details = [], [], [], []
    for k, (tr, va) in enumerate(folds_lstm, 1):
        t0 = time.time()
        # 每折独立随机种子（保证逐折可复现；同一 seed 起点，折间数据不同）
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        # ---- 每折 scaler 只 fit 训练部分 ----
        sc_seq = StandardScaler().fit(X_seq[tr].reshape(-1, len(HCV_VARS)))
        sc_sv = StandardScaler().fit(X_sv[tr])
        Xtr = sc_seq.transform(X_seq[tr].reshape(-1, len(HCV_VARS))).reshape(
            len(tr), 48, len(HCV_VARS)).astype(np.float32)
        SVtr = sc_sv.transform(X_sv[tr]).astype(np.float32)
        Xva = sc_seq.transform(X_seq[va].reshape(-1, len(HCV_VARS))).reshape(
            len(va), 48, len(HCV_VARS)).astype(np.float32)
        SVva = sc_sv.transform(X_sv[va]).astype(np.float32)
        ytr, yva = y[tr], y[va]

        # ---- pos_weight 按该折训练分布计算 ----
        n_pos, n_neg = int((ytr == 1).sum()), int((ytr == 0).sum())
        pos_w = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=DEVICE)

        # ---- 每折模型重新初始化 ----
        model = SepsisLSTM(hidden=LSTM_CFG['hidden'], drop=LSTM_CFG['drop']).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LSTM_CFG['lr'])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        # ---- s03 的训练循环：早停监控该折验证 loss，恢复最优权重 ----
        best_val, patience, ep_done = float('inf'), LSTM_CFG['patience'], 0
        best_state = None
        for ep in range(LSTM_CFG['epochs']):
            model.train()
            perm = np.random.permutation(len(tr))
            for b in range(0, len(tr), LSTM_CFG['batch']):
                sl = perm[b:b + LSTM_CFG['batch']]
                xs, xv, yy = to_tensors(Xtr[sl], SVtr[sl], ytr[sl])
                opt.zero_grad()
                loss_fn(model(xs, xv), yy).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                xs, xv, yy = to_tensors(Xva, SVva, yva)
                val_loss = loss_fn(model(xs, xv), yy).item()
            ep_done = ep + 1
            if val_loss < best_val - 1e-4:
                best_val, patience = val_loss, LSTM_CFG['patience']
                best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
            else:
                patience -= 1
                if patience == 0:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            xs, xv, _ = to_tensors(Xva, SVva, yva)
            p_va = torch.sigmoid(model(xs, xv)).cpu().numpy()
        auc = roc_auc_score(yva, p_va)
        ap = average_precision_score(yva, p_va)

        lstm_rows.append(dict(Model='LSTM', fold=f'Fold{k}', AUC=float(auc), AP=float(ap)))
        lstm_seq_scaler_evidence.append(
            {'fold': k, 'n_train': int(len(tr)), 'n_val': int(len(va)),
             'mean_first3': [float(v) for v in sc_seq.mean_[:3]],
             'scale_first3': [float(v) for v in sc_seq.scale_[:3]]})
        lstm_sv_scaler_evidence.append(
            {'fold': k, 'n_train': int(len(tr)), 'n_val': int(len(va)),
             'mean': [float(v) for v in sc_sv.mean_],
             'scale': [float(v) for v in sc_sv.scale_]})
        lstm_details.append({'fold': k, 'n_train': int(len(tr)), 'n_val': int(len(va)),
                             'deaths_train': int(ytr.sum()), 'deaths_val': int(yva.sum()),
                             'pos_weight': float(pos_w.item()), 'epochs_run': int(ep_done),
                             'best_val_loss': float(best_val),
                             'AUC': float(auc), 'AP': float(ap),
                             'seconds': round(time.time() - t0, 2)})
        print(f'  Fold {k}: AUC={auc:.4f} | AP={ap:.4f} '
              f'| epochs={ep_done} best_val_loss={best_val:.4f} '
              f'| pos_w={pos_w.item():.3f} ({lstm_details[-1]["seconds"]}s)')
    lstm_auc = np.array([r['AUC'] for r in lstm_rows])
    lstm_ap = np.array([r['AP'] for r in lstm_rows])
    print(f'  AUC: {lstm_auc.mean():.4f} ± {lstm_auc.std(ddof=1):.4f} '
          f'(范围 {lstm_auc.min():.4f}–{lstm_auc.max():.4f})')
    print(f'  AP : {lstm_ap.mean():.4f} ± {lstm_ap.std(ddof=1):.4f} '
          f'(范围 {lstm_ap.min():.4f}–{lstm_ap.max():.4f})')

    # ---------- 4. 保存 CSV ----------
    rf_out = rf_rows + [
        dict(Model='RandomForest', fold='mean', AUC=float(rf_auc.mean()), AP=float(rf_ap.mean())),
        dict(Model='RandomForest', fold='SD', AUC=float(rf_auc.std(ddof=1)), AP=float(rf_ap.std(ddof=1))),
    ]
    lstm_out = lstm_rows + [
        dict(Model='LSTM', fold='mean', AUC=float(lstm_auc.mean()), AP=float(lstm_ap.mean())),
        dict(Model='LSTM', fold='SD', AUC=float(lstm_auc.std(ddof=1)), AP=float(lstm_ap.std(ddof=1))),
    ]
    rf_csv = TAB_A / 'table3_cv_10fold.csv'
    lstm_csv = TAB_A / 'table3_cv_lstm_5fold.csv'
    pd.DataFrame(rf_out, columns=['Model', 'fold', 'AUC', 'AP']).to_csv(rf_csv, index=False)
    pd.DataFrame(lstm_out, columns=['Model', 'fold', 'AUC', 'AP']).to_csv(lstm_csv, index=False)
    print(f'\n-> {rf_csv}')
    print(f'-> {lstm_csv}')

    # ---------- 5. QA json ----------
    qa = {
        'script': str(Path(__file__).resolve()),
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': '在新数据（训练集独立插补）上重算 Table 3 交叉验证',
        'command': (f'cd "…/revision_2026-09-13/landmark_trainonly_reanalysis" && '
                    f'"…/.venv/bin/python" -u scripts/s09_cv.py {VARIANT} 2>&1 | '
                    f'tee logs/s09_cv{"" if VARIANT == "A" else "_" + VARIANT}.log'),
        'data_source': {
            'path': str(NPZ_PATH.resolve()),
            'sha256': npz_sha,
            'n_records': int(n),
            'n_subjects': int(len(np.unique(subject))),
            'deaths': int(y.sum()),
            'cohort_evaluated': ('A (all 24,894 records，数据库定义回顾性地标队列)'
                                 if VARIANT == 'A' else
                                 'B (22,778 records，疑似感染时间 ≤ intime+48h 的限制性地标队列)'),
            'cohort_B_evaluated': VARIANT == 'B',
            'variant_mask_rule': ('无（使用全部记录）' if b_mask is None else
                                  't_inf_h <= 48h，并与 preds_B.npz 的 stay_id 集合与顺序逐条核对一致'),
        },
        'npz_fields': fields,
        'fields_used': {
            'random_forest': ['X_flat', 'y', 'subject_id'],
            'lstm': ['X_hcv', 'X_sv', 'y', 'subject_id'],
            'group_variable': 'subject_id',
        },
        'feature_construction': {
            'X_flat': '14 时变变量 × {mean,min,max}（42）+ 4 静态变量（age,gender_m,weight,height）= 46 维',
            'X_flat_rebuild_vs_npz_max_abs_diff': flat_max_abs_diff,
            'X_flat_rebuild_consistent': flat_rebuild_ok,
            'X_flat_rebuild_note': ('重建在 float64 下进行（与 s02 一致），差异仅来自 npz 中 '
                                    'X_hcv 的 float32 存储舍入；RF 实际直接使用 npz 的 X_flat，'
                                    '与 s03 完全一致'),
        },
        'lstm_arrays_used': {
            'sequence_array': 'X_hcv', 'X_hcv_shape': list(X_hcv.shape),
            'X_hcv_semantics': '14 时变变量 × 48 小时，患者内局部插补后（无 NaN）',
            'sequence_transposed': 'X_hcv.transpose(0,2,1)',
            'sequence_shape': list(X_seq.shape),
            'static_array': 'X_sv', 'X_sv_shape': list(X_sv.shape),
            'static_semantics': 'age, gender_m, weight, height（训练集均值插补后）',
        },
        'random_forest': {
            'n_splits': 10, 'seed': SEED, 'splitter': 'StratifiedGroupKFold',
            'splitter_params': splitter_params_rf,
            'scaler': "StandardScaler().fit(X_flat[train_fold]) only",
            'hyperparams': RF_KW,
            'model_reinitialized_each_fold': True,
            'old_weights_loaded': False,
            'per_fold': rf_details,
            'scaler_fit_evidence': rf_scaler_evidence,
        },
        'lstm': {
            'n_splits': 5, 'seed': SEED, 'splitter': 'StratifiedGroupKFold',
            'splitter_params': splitter_params_lstm,
            'scaler_seq': 'StandardScaler().fit(X_seq[train_fold].reshape(-1,14)) only',
            'scaler_sv': 'StandardScaler().fit(X_sv[train_fold]) only',
            'hyperparams': LSTM_CFG, 'architecture': 'SepsisLSTM (s03 copy, unmodified)',
            'device': DEVICE, 'mps_available': bool(torch.backends.mps.is_available()),
            'pos_weight': 'recomputed per fold from that fold train distribution',
            'early_stopping': 'monitor that fold val BCEWithLogitsLoss; patience=10; restore best state',
            'model_reinitialized_each_fold': True,
            'old_weights_loaded': False,
            'reseed_per_fold': 'random/np/torch seeded to 42 at each fold start',
            'per_fold': lstm_details,
            'scaler_seq_fit_evidence': lstm_seq_scaler_evidence,
            'scaler_sv_fit_evidence': lstm_sv_scaler_evidence,
        },
        'leakage_checks': {
            'splitter_rerun_same_seed_folds_identical': folds_reproduced,
            'rf_10fold': leak_rf,
            'lstm_5fold': leak_lstm,
            'subject_disjoint_across_folds': bool(leak_rf_ok and leak_lstm_ok),
        },
        'summary': {
            'RandomForest': {
                'n_splits': 10, 'AUC_mean': float(rf_auc.mean()),
                'AUC_SD_ddof1': float(rf_auc.std(ddof=1)),
                'AUC_min': float(rf_auc.min()), 'AUC_max': float(rf_auc.max()),
                'AP_mean': float(rf_ap.mean()), 'AP_SD_ddof1': float(rf_ap.std(ddof=1)),
                'AP_min': float(rf_ap.min()), 'AP_max': float(rf_ap.max()),
            },
            'LSTM': {
                'n_splits': 5, 'AUC_mean': float(lstm_auc.mean()),
                'AUC_SD_ddof1': float(lstm_auc.std(ddof=1)),
                'AUC_min': float(lstm_auc.min()), 'AUC_max': float(lstm_auc.max()),
                'AP_mean': float(lstm_ap.mean()), 'AP_SD_ddof1': float(lstm_ap.std(ddof=1)),
                'AP_min': float(lstm_ap.min()), 'AP_max': float(lstm_ap.max()),
            },
        },
        'outputs': {'rf_csv': str(rf_csv.resolve()), 'lstm_csv': str(lstm_csv.resolve())},
        'environment': {
            'python': sys.version, 'platform': platform.platform(),
            'torch': torch.__version__, 'sklearn': __import__('sklearn').__version__,
            'numpy': np.__version__, 'pandas': pd.__version__,
        },
        'elapsed_sec': round(time.time() - t_start, 2),
    }
    qa_path = QA / f's09_cv{"" if VARIANT == "A" else "_" + VARIANT}.json'
    qa_path.write_text(json.dumps(py(qa), ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'-> {qa_path}')
    print(f'\n完成，用时 {qa["elapsed_sec"]}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
