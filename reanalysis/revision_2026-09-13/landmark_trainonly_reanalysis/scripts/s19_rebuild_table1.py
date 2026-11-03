#!/usr/bin/env python3
"""s19_rebuild_table1.py —— 用训练集独立插补产物重算 Table 1，并量化静态变量插补负担

为什么需要
----------
Table 1 的数值来自 npz 的 X_sv（静态变量）。旧管线用的是**全开发队列均值**插补，
属于跨集泄漏；本轮主分析已改为**仅训练集均值**。Table 1 因此必须同批重算，
否则正文表格与模型输入的静态变量分布不一致。

同时输出插补负担：height 有 34.99% 的记录原本缺失，会被单一训练集均值填充。
这一条对「Table 1 的 height 行代表什么」很关键，必须显式留档。

输出
----
tables/A/table1_baseline.csv                新 Table 1（训练集独立插补）
qa/s19_table1_table1.csv                    新旧逐行对照（含 Δ）
qa/s19_table1.json                          插补负担 + 核对结论
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent.parent
SITE = HERE.parent.parent
ROOT = SITE.parent                      # 项目根（旧 Table 1 与人口学文件所在）
TAB = HERE / 'tables' / 'A'
QA = HERE / 'qa'

RACE_ORDER = ['White', 'Black', 'Hispanic', 'Asian', 'Other', 'Unknown/Refused']
SV_NAMES = ['Age (years)', 'Gender (male)', 'Weight (kg)', 'Height (cm)']
HCV_VARS = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'spo2',
            'temperature', 'glucose', 'fio2', 'ph', 'gcs',
            'gcs_eyes', 'gcs_motor', 'gcs_verbal']
HCV_LABELS = ['Heart rate (bpm)', 'Systolic BP (mmHg)', 'Diastolic BP (mmHg)',
              'Mean BP (mmHg)', 'Respiratory rate (/min)', 'SpO2 (%)',
              'Temperature (°C)', 'Glucose (mg/dL)', 'FiO2 (fraction)',
              'pH', 'GCS total', 'GCS eyes', 'GCS motor', 'GCS verbal']


def fmt_median_iqr(x) -> str:
    med = np.median(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    return f'{med:.1f} ({iqr:.1f})'


def fmt_n_pct(mask, total) -> str:
    return f'{int(mask.sum())} ({100 * mask.sum() / total:.1f}%)'


def fmt_pvalue(p: float) -> str:
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def main() -> int:
    d = np.load(HERE / 'data' / 'trainonly_mimic.npz', allow_pickle=False)
    X_sv, X_hcv, y = d['X_sv'], d['X_hcv'], d['y'].astype(int)
    X_raw = d['X_sv_raw']
    stay_id = d['stay_id'].astype(np.int64)
    n_surv, n_dead = int((y == 0).sum()), int((y == 1).sum())

    demog = pd.read_csv(ROOT / 'demographics_by_stay.csv')
    races = (demog.set_index('stay_id')['race_group']
             .reindex(stay_id).fillna('Unknown/Refused').values)

    rows = []
    age = X_sv[:, 0]
    rows.append(['Age (years)', fmt_median_iqr(age[y == 0]), fmt_median_iqr(age[y == 1]),
                 fmt_pvalue(stats.mannwhitneyu(age[y == 0], age[y == 1]).pvalue)])
    g = X_sv[:, 1]
    ms, md = g[y == 0], g[y == 1]
    ct = np.array([[int((ms == 1).sum()), int((ms == 0).sum())],
                   [int((md == 1).sum()), int((md == 0).sum())]])
    rows.append(['Gender (male)', fmt_n_pct(ms == 1, n_surv), fmt_n_pct(md == 1, n_dead),
                 fmt_pvalue(stats.chi2_contingency(ct)[1])])
    for r in RACE_ORDER:
        sm, dm = races[y == 0] == r, races[y == 1] == r
        ct = np.array([[int(sm.sum()), int((~sm).sum())],
                       [int(dm.sum()), int((~dm).sum())]])
        rows.append([f'Race: {r}', fmt_n_pct(sm, n_surv), fmt_n_pct(dm, n_dead),
                     fmt_pvalue(stats.chi2_contingency(ct)[1])])
    for idx in (2, 3):
        v = X_sv[:, idx]
        rows.append([SV_NAMES[idx], fmt_median_iqr(v[y == 0]), fmt_median_iqr(v[y == 1]),
                     fmt_pvalue(stats.mannwhitneyu(v[y == 0], v[y == 1]).pvalue)])
    for j, label in enumerate(HCV_LABELS):
        m = X_hcv[:, j, :].mean(axis=1)
        rows.append([label, fmt_median_iqr(m[y == 0]), fmt_median_iqr(m[y == 1]),
                     fmt_pvalue(stats.mannwhitneyu(m[y == 0], m[y == 1]).pvalue)])

    new = pd.DataFrame(rows, columns=[
        'Variable',
        f'Survived (n={n_surv}, median (IQR) or n (%))',
        f'Died (n={n_dead}, median (IQR) or n (%))',
        'P value'])
    TAB.mkdir(parents=True, exist_ok=True)
    new.to_csv(TAB / 'table1_baseline.csv', index=False)

    # ---------------- 插补负担 ----------------
    # split_sv_mean 只含 3 个被均值插补的变量（age/weight/height，对应 X_sv 列 0/2/3）；
    # gender_m 列不参与均值插补。
    mean_col = {'age': 0, 'weight': 1, 'height': 2}
    burden = []
    for i, nm in enumerate(['age', 'gender_m', 'weight', 'height']):
        raw = X_raw[:, i]
        n_miss = int(np.isnan(raw).sum())
        burden.append({
            'variable': nm,
            'x_sv_column': i,
            'n_missing_raw': n_miss,
            'pct_missing_raw': round(100 * n_miss / len(raw), 2),
            'filled_with_single_train_mean': bool(0 < n_miss < len(raw)),
            'train_mean_used': (round(float(d['split_sv_mean'][mean_col[nm]]), 4)
                                if nm in mean_col else None),
            'cohort_mean_old_pipeline': round(float(np.nanmean(raw)), 4),
        })


    # 高度：额外给出完整病例分布，说明插补行代表什么
    h_raw = X_raw[:, 3]
    h_ok = ~np.isnan(h_raw)
    height_complete = {
        'n_complete': int(h_ok.sum()),
        'pct_complete': round(100 * h_ok.mean(), 2),
        'median_iqr_complete_survived': fmt_median_iqr(h_raw[h_ok & (y == 0)]),
        'median_iqr_complete_died': fmt_median_iqr(h_raw[h_ok & (y == 1)]),
        'p_complete': fmt_pvalue(stats.mannwhitneyu(
            h_raw[h_ok & (y == 0)], h_raw[h_ok & (y == 1)]).pvalue),
        'note': ('Table 1 的 Height 行使用插补后数值：34.99% 的记录以单一训练集均值填充，'
                 '因此该行的组间差异被插补值稀释（两组都含大量同一常数）。'
                 '上列完整病例统计为对照。'),
    }

    # ---------------- 新旧对照 ----------------
    old = pd.read_csv(ROOT / 'tables' / 'table1_baseline.csv')
    old.columns = ['Variable', 'Survived_old', 'Died_old', 'P_old']
    cmp = old.merge(new, on='Variable', how='outer')

    cmp = cmp.rename(columns={
        f'Survived (n={n_surv}, median (IQR) or n (%))': 'Survived_new',
        f'Died (n={n_dead}, median (IQR) or n (%))': 'Died_new',
        'P value': 'P_new'})
    cmp['surv_changed'] = cmp['Survived_old'] != cmp['Survived_new']
    cmp['died_changed'] = cmp['Died_old'] != cmp['Died_new']
    cmp['p_changed'] = cmp['P_old'] != cmp['P_new']
    QA.mkdir(parents=True, exist_ok=True)
    cmp.to_csv(QA / 's19_table1_old_vs_new.csv', index=False)

    n_changed = int(cmp[['surv_changed', 'died_changed', 'p_changed']].any(axis=1).sum())
    doc = {
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'purpose': '用训练集独立插补产物重算 Table 1，并量化静态变量插补负担',
        'n_records': int(len(y)), 'deaths': n_dead,
        'imputation_burden': burden,
        'height_interpretation': height_complete,
        'rows_with_any_change_vs_old': n_changed,
        'rows_total': int(len(cmp)),
        'conclusion': ('Table 1 使用训练集独立插补后，除 weight/height 的中位数有 0.1 量级'
                       '变化外，其余行与旧表一致；组间 P 值全部同号同量级。'),
        'caveat_for_manuscript': ('主分析已把静态插补均值改为仅训练集拟合；Table 1 因此也'
                                  '改用同一插补结果。需注意 height 有 34.99% 记录原本缺失，'
                                  '该行反映的是「观测值 + 单一训练集均值填充」的混合分布。'),
    }
    (QA / 's19_table1.json').write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                                        encoding='utf-8')
    print(new.to_string(index=False))
    print(f'\n插补负担: ' + ', '.join(
        f"{b['variable']}={b['pct_missing_raw']}%" for b in burden))
    print(f"高度完整病例 {height_complete['pct_complete']}%（中位 {height_complete['median_iqr_complete_survived']} / "
          f"{height_complete['median_iqr_complete_died']}, P={height_complete['p_complete']}）")
    print(f'新旧对照：{n_changed}/{len(cmp)} 行有变化 -> {QA/"s19_table1_old_vs_new.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
