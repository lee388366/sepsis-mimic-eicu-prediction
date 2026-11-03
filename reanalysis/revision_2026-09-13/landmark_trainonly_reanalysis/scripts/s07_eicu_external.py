#!/usr/bin/env python3
"""s07_eicu_external.py —— eICU 跨库探索性评价 + uniquepid 患者级再校准

用法: python s07_eicu_external.py A|B
      A = MIMIC 数据库定义回顾性地标队列训练的模型
      B = MIMIC t_inf≤48h 限制性地标队列训练的模型

关键纪律（回应审阅意见）:
  1. 外部 CI 的抽样单位是 **uniquepid（患者）**，不是 patientunitstayid。
     本队列 6,812 条记录来自 6,488 位患者，其中 284 位患者贡献 608 条记录，
     即 608−284=324 条"额外记录"。记录级 bootstrap 会低估不确定性。
  2. 再校准按 uniquepid 做 50/50 划分：同一患者的全部记录整体落入
     拟合半或评价半，两半患者集合交集必须为 0。
  3. 记录级划半结果只作敏感性对照，不作为主要证据。
  4. eICU 的脓毒症识别来自诊断字符串（24h 窗），与 MIMIC 的 sepsis3 定义
     不同 —— 因此只能称 **跨数据库探索性评价（exploratory cross-database
     evaluation）**，不得称 external validation。
  5. 再校准后的概率与 raw 概率的阈值不可混用；两者分别报告。

输出:
  tables/<V>/table9_external_validation.csv           外部判别性能（uniquepid 聚类 CI）
  tables/<V>/table2_ci_eicu_record_vs_patient.csv      记录级 vs 患者级 CI 对照
  tables/<V>/table10_calibration_external.csv          外部校准（raw）
  tables/<V>/table10_recalibration_eicu_patient_split.csv  患者级再校准（主）
  tables/<V>/table10_recalibration_eicu_record_split.csv   记录级再校准（敏感性）
  tables/<V>/table9b_subgroup_external.csv             外部亚组
  tables/<V>/dca_eicu.csv                              外部 DCA 曲线
  tables/<V>/threshold_operating_characteristics_external.csv
  calibration_primary_patient_clustered_external_<V>.csv
  qa/s07_eicu_<V>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_eval as C  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent.parent.parent

SEED = 42
N_BOOT = 1000
HCV_VARS = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'spo2',
            'temperature', 'glucose', 'fio2', 'ph',
            'gcs', 'gcs_eyes', 'gcs_motor', 'gcs_verbal']
BASELINES = ['LogisticRegression', 'DecisionTree', 'RandomForest', 'KNN', 'SVM', 'GaussianNB']
MODELS = ['LSTM'] + BASELINES
THRESH_GRID = np.round(np.arange(0.01, 0.99, 0.01), 4)


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


def fbeta(p, r, beta):
    b2 = beta ** 2
    return (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) > 0 else 0.0


def main() -> int:
    V = (sys.argv[1] if len(sys.argv) > 1 else 'A').upper()
    TAB = HERE / 'tables' / V
    QA = HERE / 'qa'
    TAB.mkdir(parents=True, exist_ok=True)
    MD = HERE / 'models_trainonly' / V

    # ---------------- 1. eICU 数据 ----------------
    e = np.load(HERE / 'data' / 'trainonly_eicu.npz', allow_pickle=False)
    X_hcv = e['X_hcv'].astype(np.float32)          # [n,14,48]（已插补）
    X_sv = e['X_sv'].astype(np.float32)            # [n,4]（缺失用 MIMIC 训练集均值）
    y = e['y'].astype(int)
    stay = e['stay_id'].astype(np.int64)
    upid = e['uniquepid'].astype(str)
    n = len(y)
    n_pat = len(np.unique(upid))
    vc = pd.Series(upid).value_counts()
    print(f'=== 变体 {V} | eICU {n} 条记录 / {n_pat} 位患者 | 死亡 {int(y.sum())} '
          f'({y.mean():.3%}) ===')
    print(f'  多次入住患者 {int((vc>1).sum())} 人，贡献 {int(vc[vc>1].sum())} 条记录 '
          f'(额外 {int(vc[vc>1].sum())-int((vc>1).sum())} 条)')

    # 46 维聚合（与 s02/s03 完全同构造）
    parts = []
    for j in range(len(HCV_VARS)):
        s = X_hcv[:, j, :]
        parts += [s.mean(1), s.min(1), s.max(1)]
    X_flat = np.column_stack(parts + [X_sv[:, 0], X_sv[:, 1], X_sv[:, 2],
                                      X_sv[:, 3]]).astype(np.float32)
    assert X_flat.shape[1] == 46, f'聚合特征应为 46 维，实际 {X_flat.shape[1]}'

    # ---------------- 2. 用 MIMIC 训练集模型预测 ----------------
    sc_flat = joblib.load(MD / 'scaler_flat.joblib')
    F = sc_flat.transform(X_flat)
    probs = {}
    for name in BASELINES:
        clf = joblib.load(MD / f'model_{name}.joblib')
        probs[name] = clf.predict_proba(F)[:, 1].astype(float)

    sc_seq = joblib.load(MD / 'scaler_seq.joblib')
    sc_sv = joblib.load(MD / 'scaler_sv.joblib')
    X_seq = X_hcv.transpose(0, 2, 1)
    Xs = sc_seq.transform(X_seq.reshape(-1, len(HCV_VARS))).reshape(
        n, 48, len(HCV_VARS)).astype(np.float32)
    Xv = sc_sv.transform(X_sv).astype(np.float32)
    net = SepsisLSTM()
    net.load_state_dict(torch.load(MD / 'model_LSTM.pt', map_location='cpu'))
    net.eval()
    with torch.no_grad():
        pl = torch.sigmoid(net(torch.tensor(Xs), torch.tensor(Xv))).numpy()
    probs['LSTM'] = pl.astype(float)
    for name in MODELS:
        print(f'  {name:20s} 外部 AUC = {roc_auc_score(y, probs[name]):.4f}')

    thr_raw = {m: float(np.load(HERE / 'data' / f'preds_{V}.npz')[f'threshold_{m}'])
               for m in MODELS}

    # ---------------- 3. 外部判别性能 + 患者聚类 CI ----------------
    b_pat = C.cluster_batches(upid, N_BOOT, SEED)
    rng = np.random.default_rng(1)
    b_rec = [rng.integers(0, n, n) for _ in range(N_BOOT)]

    rows, ci_rows = [], []
    for m in MODELS:
        p = probs[m]
        auc = roc_auc_score(y, p)
        ap = average_precision_score(y, p)
        a_pat, ap_pat = C.auc_ap_batches(y, p, b_pat)
        a_rec, ap_rec = C.auc_ap_batches(y, p, b_rec)
        rows.append({
            'Model': m, 'n_records': n, 'n_patients': n_pat, 'deaths': int(y.sum()),
            'AUC_ext': round(auc, 4),
            'AUC_CI_low_patient': round(float(np.percentile(a_pat, 2.5)), 4),
            'AUC_CI_high_patient': round(float(np.percentile(a_pat, 97.5)), 4),
            'AUPRC_ext': round(ap, 4),
            'AP_CI_low_patient': round(float(np.percentile(ap_pat, 2.5)), 4),
            'AP_CI_high_patient': round(float(np.percentile(ap_pat, 97.5)), 4),
            'Brier_ext': round(brier_score_loss(y, p), 4),
        })
        ci_rows.append({'Model': m,
                        'AUC_rec_low': round(float(np.percentile(a_rec, 2.5)), 4),
                        'AUC_rec_high': round(float(np.percentile(a_rec, 97.5)), 4),
                        'AUC_pat_low': round(float(np.percentile(a_pat, 2.5)), 4),
                        'AUC_pat_high': round(float(np.percentile(a_pat, 97.5)), 4),
                        'AUC_rec_width': round(float(np.percentile(a_rec, 97.5)
                                                     - np.percentile(a_rec, 2.5)), 4),
                        'AUC_pat_width': round(float(np.percentile(a_pat, 97.5)
                                                     - np.percentile(a_pat, 2.5)), 4)})
    ext = pd.DataFrame(rows).sort_values('AUC_ext', ascending=False).reset_index(drop=True)
    sofa_ext = e['sofa_max_48h'].astype(float)
    qsofa_ext = e['qsofa6'].astype(float)
    ext = pd.concat([ext, pd.DataFrame([
        {'Model': 'SOFA (rule-based, 48h max)', 'n_records': n, 'n_patients': n_pat,
         'deaths': int(y.sum()), 'AUC_ext': round(roc_auc_score(y, sofa_ext), 4),
         'AUC_CI_low_patient': round(float(np.percentile(
             [roc_auc_score(y[i], sofa_ext[i]) for i in b_pat], 2.5)), 4),
         'AUC_CI_high_patient': round(float(np.percentile(
             [roc_auc_score(y[i], sofa_ext[i]) for i in b_pat], 97.5)), 4),
         'AUPRC_ext': round(average_precision_score(y, sofa_ext), 4),
         'Brier_ext': np.nan},
        {'Model': 'qSOFA-6h (rule-based)', 'n_records': n, 'n_patients': n_pat,
         'deaths': int(y.sum()), 'AUC_ext': round(roc_auc_score(y, qsofa_ext), 4),
         'AUC_CI_low_patient': round(float(np.percentile(
             [roc_auc_score(y[i], qsofa_ext[i]) for i in b_pat], 2.5)), 4),
         'AUC_CI_high_patient': round(float(np.percentile(
             [roc_auc_score(y[i], qsofa_ext[i]) for i in b_pat], 97.5)), 4),
         'AUPRC_ext': round(average_precision_score(y, qsofa_ext), 4),
         'Brier_ext': np.nan},
    ])], ignore_index=True)
    ext.to_csv(TAB / 'table9_external_validation.csv', index=False)
    pd.DataFrame(ci_rows).to_csv(TAB / 'table2_ci_eicu_record_vs_patient.csv', index=False)
    print('\n===== 外部判别性能（uniquepid 聚类 95%CI）=====')
    print(ext[['Model', 'AUC_ext', 'AUC_CI_low_patient', 'AUC_CI_high_patient',
               'AUPRC_ext', 'Brier_ext']].to_string(index=False))
    print('\n===== 记录级 vs 患者级 CI 宽度 =====')
    print(pd.DataFrame(ci_rows)[['Model', 'AUC_rec_width', 'AUC_pat_width']].to_string(index=False))

    # ---------------- 4. 外部校准（raw）+ 患者聚类 CI ----------------
    cal_rows = []
    for m in MODELS:
        p = probs[m]
        st = C.calibration_stats(y, p)
        sl, ic, ct, br = [], [], [], []
        for idx in b_pat:
            s2 = C.calibration_stats(y[idx], p[idx])
            sl.append(s2['slope']); ic.append(s2['intercept']); ct.append(s2['citl'])
            br.append(brier_score_loss(y[idx], p[idx]))
        cal_rows.append({
            'Model': m, 'n_records': n, 'n_patients': n_pat,
            'brier_raw': round(st['brier'], 4),
            'brier_raw_CI_low_patient': round(float(np.nanpercentile(br, 2.5)), 4),
            'brier_raw_CI_high_patient': round(float(np.nanpercentile(br, 97.5)), 4),
            'slope': round(st['slope'], 4),
            'slope_CI_low_asymptotic': round(st['slope_ci_low'], 4),
            'slope_CI_high_asymptotic': round(st['slope_ci_high'], 4),
            'slope_CI_low_patient': round(float(np.nanpercentile(sl, 2.5)), 4),
            'slope_CI_high_patient': round(float(np.nanpercentile(sl, 97.5)), 4),
            'intercept': round(st['intercept'], 4),
            'intercept_CI_low_patient': round(float(np.nanpercentile(ic, 2.5)), 4),
            'intercept_CI_high_patient': round(float(np.nanpercentile(ic, 97.5)), 4),
            'CITL': round(st['citl'], 4),
            'CITL_CI_low_asymptotic': round(st['citl_ci_low'], 4),
            'CITL_CI_high_asymptotic': round(st['citl_ci_high'], 4),
            'CITL_CI_low_patient': round(float(np.nanpercentile(ct, 2.5)), 4),
            'CITL_CI_high_patient': round(float(np.nanpercentile(ct, 97.5)), 4),
        })
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(TAB / 'table10_calibration_external.csv', index=False)
    cal.to_csv(HERE / f'calibration_primary_patient_clustered_external_{V}.csv', index=False)
    print('\n===== 外部校准（raw 概率，uniquepid 聚类 CI）=====')
    print(cal[['Model', 'brier_raw', 'slope', 'slope_CI_low_patient', 'slope_CI_high_patient',
               'CITL', 'CITL_CI_low_patient', 'CITL_CI_high_patient']].to_string(index=False))

    # ---------------- 5. 患者级再校准（主）/ 记录级（敏感性）----------------
    rep = pd.DataFrame({'pid': upid, 'y': y}).groupby('pid').agg(
        rep_y=('y', 'first')).reset_index()
    tr_p, va_p = train_test_split(rep.pid.values, test_size=0.5,
                                  stratify=rep.rep_y.values, random_state=SEED)
    tr_pid, va_pid = set(tr_p), set(va_p)
    tr_mask = np.isin(upid, list(tr_pid))
    va_mask = np.isin(upid, list(va_pid))
    overlap = len(tr_pid & va_pid)
    assert overlap == 0, '患者级划半出现重叠'
    print(f'\n患者级划半: 拟合半 {tr_mask.sum()} 条/{len(tr_pid)} 人（死亡 '
          f'{int(y[tr_mask].sum())}）| 评价半 {va_mask.sum()} 条/{len(va_pid)} 人（死亡 '
          f'{int(y[va_mask].sum())}）| 患者交集 {overlap}')

    idx = np.arange(n)
    tr_i, va_i = train_test_split(idx, test_size=0.5, stratify=y, random_state=SEED)
    rec_overlap = int(len(set(upid[tr_i]) & set(upid[va_i])))
    print(f'记录级划半(敏感性): 拟合半 {len(tr_i)} 条 | 评价半 {len(va_i)} 条 | '
          f'其中患者重叠 {rec_overlap} 人')

    def recal_block(tag, trm, vam):
        out = []
        for m in MODELS:
            p = probs[m]
            a_, b_ = C.fit_logistic_recal(y[trm], p[trm])
            p_log = C.invlogit(a_ + b_ * C._logit(p[vam]))
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(p[trm], y[trm])
            p_iso = iso.predict(p[vam])
            yy, pp = y[vam], p[vam]
            out.append({
                'Model': m, 'split_unit': tag,
                'n_fit_records': int(np.count_nonzero(trm)) if trm.dtype == bool else int(trm.size),
                'n_eval_records': int(np.count_nonzero(vam)) if vam.dtype == bool else int(vam.size),
                'recal_a': round(a_, 4), 'recal_b': round(b_, 4),
                'Brier_raw': round(brier_score_loss(yy, pp), 4),
                'Brier_logistic': round(brier_score_loss(yy, p_log), 4),
                'Brier_isotonic': round(brier_score_loss(yy, p_iso), 4),
                'slope_raw': round(C.calibration_stats(yy, pp)['slope'], 4),
                'intercept_raw': round(C.calibration_stats(yy, pp)['intercept'], 4),
                'CITL_raw': round(C.calibration_stats(yy, pp)['citl'], 4),
                'slope_logistic': round(C.calibration_stats(yy, p_log)['slope'], 4),
                'intercept_logistic': round(C.calibration_stats(yy, p_log)['intercept'], 4),
                'CITL_logistic': round(C.calibration_stats(yy, p_log)['citl'], 4),
                'slope_isotonic': round(C.calibration_stats(yy, p_iso)['slope'], 4),
                'intercept_isotonic': round(C.calibration_stats(yy, p_iso)['intercept'], 4),
                'CITL_isotonic': round(C.calibration_stats(yy, p_iso)['citl'], 4),
            })
        return pd.DataFrame(out)

    recal_pat = recal_block('uniquepid (primary)', tr_mask, va_mask)
    recal_pat.to_csv(TAB / 'table10_recalibration_eicu_patient_split.csv', index=False)
    recal_rec = recal_block('record (sensitivity)', tr_i, va_i)
    recal_rec.to_csv(TAB / 'table10_recalibration_eicu_record_split.csv', index=False)
    print('\n===== 患者级再校准（主）=====')
    print(recal_pat[['Model', 'recal_a', 'recal_b', 'Brier_raw', 'Brier_logistic',
                     'Brier_isotonic']].to_string(index=False))
    print('\n===== 记录级再校准（敏感性对照）=====')
    print(recal_rec[['Model', 'Brier_raw', 'Brier_logistic', 'Brier_isotonic']].to_string(index=False))

    # 再校准后 Brier 的患者级聚类 CI（评价半之内按患者重抽）
    eval_pid = upid[va_mask]
    b_eval = C.cluster_batches(eval_pid, N_BOOT, SEED)
    yv = y[va_mask]
    recal_ci_rows = []
    for m in MODELS:
        p = probs[m]
        a_, b_ = C.fit_logistic_recal(y[tr_mask], p[tr_mask])
        iso = IsotonicRegression(out_of_bounds='clip').fit(p[tr_mask], y[tr_mask])
        pv = p[va_mask]
        pv_log = C.invlogit(a_ + b_ * C._logit(pv))
        pv_iso = iso.predict(pv)
        # (i) 固定方程：只重抽评价集
        br_fix_log = [brier_score_loss(yv[i], pv_log[i]) for i in b_eval]
        br_fix_iso = [brier_score_loss(yv[i], pv_iso[i]) for i in b_eval]
        br_fix_raw = [brier_score_loss(yv[i], pv[i]) for i in b_eval]
        # (ii) 每次重抽内重新拟合方程（总不确定性）：拟合半重抽 + 方程重估
        b_fit = C.cluster_batches(upid[tr_mask], 200, SEED)
        yf, pf = y[tr_mask], p[tr_mask]
        br_all = []
        for j, i_eval in enumerate(b_eval[:200]):
            i_fit = b_fit[j]
            if len(np.unique(yf[i_fit])) < 2:
                br_all.append(np.nan); continue
            a2, b2 = C.fit_logistic_recal(yf[i_fit], pf[i_fit])
            br_all.append(brier_score_loss(yv[i_eval],
                                           C.invlogit(a2 + b2 * C._logit(pv[i_eval]))))
        recal_ci_rows.append({
            'Model': m,
            'Brier_logistic_fixed_CI_low_patient': round(float(np.nanpercentile(br_fix_log, 2.5)), 4),
            'Brier_logistic_fixed_CI_high_patient': round(float(np.nanpercentile(br_fix_log, 97.5)), 4),
            'Brier_isotonic_fixed_CI_low_patient': round(float(np.nanpercentile(br_fix_iso, 2.5)), 4),
            'Brier_isotonic_fixed_CI_high_patient': round(float(np.nanpercentile(br_fix_iso, 97.5)), 4),
            'Brier_raw_fixed_CI_low_patient': round(float(np.nanpercentile(br_fix_raw, 2.5)), 4),
            'Brier_raw_fixed_CI_high_patient': round(float(np.nanpercentile(br_fix_raw, 97.5)), 4),
            'Brier_logistic_refit_CI_low_patient': round(float(np.nanpercentile(br_all, 2.5)), 4),
            'Brier_logistic_refit_CI_high_patient': round(float(np.nanpercentile(br_all, 97.5)), 4),
            'n_boot_refit': int(np.sum(np.isfinite(br_all))),
        })
    recal_ci = pd.DataFrame(recal_ci_rows)
    recal_ci.to_csv(TAB / 'table10_recalibration_eicu_patient_ci.csv', index=False)
    print('\n===== 再校准 Brier 的患者级聚类区间 =====')
    print(recal_ci.to_string(index=False))

    # ---------------- 6. 外部 DCA + 临床影响曲线 ----------------
    dca_rows = []
    for m in MODELS:
        p = probs[m]
        nb = C.dca_net_benefit(y, p, THRESH_GRID)
        fl, hits = C.clinical_impact(y, p, THRESH_GRID)
        for t, v, f, h in zip(THRESH_GRID, nb, fl, hits):
            dca_rows.append({'Model': m, 'prob_type': 'raw', 'threshold_prob': t,
                             'net_benefit': round(v, 5), 'flagged_n': f,
                             'deaths_among_flagged': h})
    for t, v in zip(THRESH_GRID, C.dca_treat_all(y, THRESH_GRID)):
        dca_rows.append({'Model': 'TreatAll', 'prob_type': 'raw', 'threshold_prob': t,
                         'net_benefit': round(v, 5), 'flagged_n': n,
                         'deaths_among_flagged': int(y.sum())})
    for t in THRESH_GRID:
        dca_rows.append({'Model': 'TreatNone', 'prob_type': 'raw', 'threshold_prob': t,
                         'net_benefit': 0.0, 'flagged_n': 0, 'deaths_among_flagged': 0})
    # 再校准概率的 DCA（用评价半）
    recal_prob = {}
    for m in MODELS:
        p = probs[m]
        a_, b_ = C.fit_logistic_recal(y[tr_mask], p[tr_mask])
        recal_prob[m] = C.invlogit(a_ + b_ * C._logit(p))
        nb = C.dca_net_benefit(y[va_mask], recal_prob[m][va_mask], THRESH_GRID)
        for t, v in zip(THRESH_GRID, nb):
            dca_rows.append({'Model': m, 'prob_type': 'logistic_recalibrated_full_cohort',
                             'threshold_prob': t, 'net_benefit': round(v, 5),
                             'flagged_n': np.nan, 'deaths_among_flagged': np.nan})
    pd.DataFrame(dca_rows).to_csv(TAB / 'dca_eicu.csv', index=False)
    key_t = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    print('\n===== 外部 DCA 净获益（raw 概率）=====')
    nbt = pd.DataFrame({m: C.dca_net_benefit(y, probs[m], key_t) for m in MODELS},
                       index=[f'{t:.2f}' for t in key_t])
    nbt['TreatAll'] = C.dca_treat_all(y, key_t)
    print(nbt.round(4).to_string())

    # ---------------- 7. 阈值操作特征（外部）----------------
    # raw 概率：在全部 eICU 记录上套用 MIMIC 验证集阈值（仅描述性）
    # 再校准概率：方程由"拟合半"估计，只在"患者不重叠的评价半"上评估，
    #             分母也必须是评价半的记录数（否则分子分母口径不一致）。
    n_eval = int(np.count_nonzero(va_mask))
    th_rows = []
    for m in MODELS:
        p = probs[m]
        for tag, yy, pv, denom, note in [
                ('raw_prob_same_MIMIC_threshold', y, p, n,
                 'MIMIC 验证集 F3 操作点直接套用于 eICU raw 概率（概率尺度不可迁移，仅作描述）'),
                ('logistic_recalibrated_prob_eval_half', y[va_mask], recal_prob[m][va_mask],
                 n_eval,
                 ('logistic 再校准方程由患者不重叠的拟合半估计，'
                  '仅在评价半以固定风险阈值 0.15 评估（患者交集=0）'))]:
            thr = thr_raw[m] if tag.startswith('raw') else 0.15
            pred = (pv >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(yy, pred, labels=[0, 1]).ravel()
            th_rows.append({
                'Model': m, 'prob_type': tag, 'threshold': round(float(thr), 4),
                'threshold_origin': note,
                'n_evaluated': int(denom),
                'flagged_n': int(tp + fp), 'flagged_percent': round(100 * (tp + fp) / denom, 2),
                'deaths_captured': int(tp), 'sensitivity': round(tp / (tp + fn), 4),
                'specificity': round(tn / (tn + fp), 4),
                'PPV': round(tp / (tp + fp) if (tp + fp) else np.nan, 4),
                'NPV': round(tn / (tn + fn) if (tn + fn) else np.nan, 4),
                'false_positive_n': int(fp),
                'false_positive_percent_of_all': round(100 * fp / denom, 2),
                'one_minus_PPV': round(1 - tp / (tp + fp) if (tp + fp) else np.nan, 4),
                'net_benefit_at_threshold': round(float(C.dca_net_benefit(
                    yy, pv, np.array([thr]))[0]), 5),
                'net_benefit_treat_all_at_threshold': round(float(
                    C.dca_treat_all(yy, np.array([thr]))[0]), 5),
            })
    th_df = pd.DataFrame(th_rows)
    th_df.to_csv(TAB / 'threshold_operating_characteristics_external.csv', index=False)
    print('\n===== 外部阈值操作特征 =====')
    print(th_df[['Model', 'prob_type', 'threshold', 'flagged_percent', 'sensitivity',
                 'PPV', 'one_minus_PPV', 'net_benefit_at_threshold',
                 'net_benefit_treat_all_at_threshold']].to_string(index=False))

    # ---------------- 8. 外部亚组 ----------------
    demo = pd.read_csv(ROOT / 'eicu_demographics.csv')
    demo['patientunitstayid'] = demo['patientunitstayid'].astype(np.int64)
    dm = dict(zip(demo.patientunitstayid.tolist(), zip(demo.gender.tolist(),
                                                       demo.race_group.tolist())))
    gender = np.array([dm.get(int(s), ('Unknown', 'Unknown'))[0] for s in stay])
    race = np.array([dm.get(int(s), ('Unknown', 'Unknown'))[1] for s in stay])
    sg_rows = []
    for dim, arr in [('Gender', gender), ('Race', race)]:
        for g in sorted(set(arr.tolist())):
            sel = arr == g
            if sel.sum() < 100:
                continue
            row = {'Dimension': dim, 'Subgroup': g, 'n': int(sel.sum()),
                   'n_patients': int(len(np.unique(upid[sel]))),
                   'Death_rate': round(float(y[sel].mean()), 4)}
            for m in ['LSTM', 'RandomForest', 'SOFA', 'qsofa6']:
                pv = probs[m][sel] if m in probs else (sofa_ext[sel] if m == 'SOFA' else qsofa_ext[sel])
                row[f'{m}_AUC'] = round(float(roc_auc_score(y[sel], pv)), 4)
                sub_b = C.cluster_batches(upid[sel], 300, SEED)
                vals = [roc_auc_score(y[sel][i], pv[i]) for i in sub_b
                        if len(np.unique(y[sel][i])) > 1]
                row[f'{m}_CI_low_patient'] = round(float(np.percentile(vals, 2.5)), 4)
                row[f'{m}_CI_high_patient'] = round(float(np.percentile(vals, 97.5)), 4)
            sg_rows.append(row)
    pd.DataFrame(sg_rows).to_csv(TAB / 'table9b_subgroup_external.csv', index=False)

    # ---------------- 9. QA ----------------
    qa = {
        'variant': V, 'unit': 'eICU CRD',
        'definition_caveat': ('eICU 脓毒症识别基于诊断字符串（24h 窗），与 MIMIC sepsis3 '
                             '定义不同 → 只能称跨数据库探索性评价，不得称 external validation'),
        'n_records': n, 'n_patients_uniquepid': n_pat,
        'n_patients_multi_record': int((vc > 1).sum()),
        'n_records_from_multi': int(vc[vc > 1].sum()),
        'n_extra_records_beyond_one_per_patient': int(vc[vc > 1].sum() - (vc > 1).sum()),
        'external_bootstrap_unit': 'uniquepid (主结果)',
        'external_bootstrap_unit_sensitivity': 'record (对照)',
        'recalibration_primary_split': 'uniquepid 50/50 (stratified on patient representative label)',
        'recalibration_fit_records': int(tr_mask.sum()),
        'recalibration_fit_patients': int(len(tr_pid)),
        'recalibration_fit_deaths': int(y[tr_mask].sum()),
        'recalibration_eval_records': int(va_mask.sum()),
        'recalibration_eval_patients': int(len(va_pid)),
        'recalibration_eval_deaths': int(y[va_mask].sum()),
        'recalibration_patient_intersection': int(overlap),
        'recalibration_record_split_patient_overlap_sensitivity': int(rec_overlap),
        'calibration_ci_definition': (
            '评价半内按 uniquepid 重抽，固定再校准方程（Brier_*_fixed_*）与'
            '每次重抽内重新拟合方程（Brier_logistic_refit_*）分别报告'),
        'threshold_note': ('raw 概率阈值来自 MIMIC 验证集，不能与再校准后概率阈值混用；'
                           '再校准概率只在患者不重叠的评价半上、以固定风险阈值 0.15 报告'
                           '（flagged_percent 的分母为评价半记录数）'),
        'recalibration_eval_half_n_records': int(n_eval),
        'threshold_operating_characteristics_note': (
            'prob_type=raw_prob_same_MIMIC_threshold 在全部 eICU 记录上计算；'
            'prob_type=logistic_recalibrated_prob_eval_half 只在评价半计算'),
        'models_from': f'models_trainonly/{V}（MIMIC 训练集拟合参数，未在 eICU 重新估计）',
        'eicu_static_missing_handling': ('age/weight/height 的缺失用 MIMIC 训练集均值；'
                                         'gender 缺失被填为 0（Female）—— 沿用旧装配规则，'
                                         '属映射限制，已在局限中说明'),
        'eicu_sofa_qsofa_from': 'eicu_sofa.csv / trainonly_eicu.npz（0–48h 窗，与地标一致）',
    }
    (QA / f's07_eicu_{V}.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2),
                                           encoding='utf-8')
    np.savez(HERE / 'data' / f'eicu_probs_{V}.npz',
             **{f'prob_{m}': probs[m].astype(np.float32) for m in MODELS},
             **{f'prob_logrecal_{m}': recal_prob[m].astype(np.float32) for m in MODELS},
             y_true=y, stay_id=stay, uniquepid=upid, sofa=sofa_ext, qsofa6=qsofa_ext,
             recal_fit_mask=tr_mask, recal_eval_mask=va_mask,
             record_fit_mask=np.isin(np.arange(n), tr_i),
             record_eval_mask=np.isin(np.arange(n), va_i))
    print(f'\n-> tables/{V}/ 与 qa/s07_eicu_{V}.json 已写出')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
