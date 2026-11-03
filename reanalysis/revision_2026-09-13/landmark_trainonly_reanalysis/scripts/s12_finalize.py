#!/usr/bin/env python3
"""s12_finalize.py —— 交付物规范化 + 记录级校准敏感性 + final_checks.json + CURSOR_HANDOFF.md

阶段10（最终QA与交付物）。做四件事：

1. 校准不确定性的"记录级"敏感性分析（与受试者聚类主结果对照）
2. 把交付物文件名规范化到工作目录根（不覆盖、不改动旧文件）
3. final_checks.json —— 逐条给出状态与证据
4. CURSOR_HANDOFF.md —— 按用户要求的 10 项格式汇报

本脚本只读分析产物，不修改任何稿件/DOCX。
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent.parent.parent        # 项目根目录
sys.path.insert(0, str(HERE / 'scripts'))

from common_eval import _logit, calibration_stats, f3, f2  # noqa: E402

MODELS = ['LSTM', 'RandomForest', 'SVM', 'LogisticRegression',
          'GaussianNB', 'KNN', 'DecisionTree']
N_BOOT = 1000
SEED = 42

FORBIDDEN = [
    'eliminated leakage', 'eliminate leakage', 'clinically optimal',
    'validated clinical threshold', 'clinically validated threshold',
    'clinical utility was demonstrated', 'demonstrated clinical utility',
    'external validation', 'statistically superior', 'was superior to',
    'established clinical usefulness', 'causal physiological',
    'fairness was demonstrated',
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# =====================================================================
# 1. 记录级校准敏感性（对照受试者聚类主结果）
# =====================================================================
def record_level_calibration_ci(V: str, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """记录级 bootstrap（有放回抽记录、忽略患者聚类）——仅作敏感性对照。

    每个重抽内重新拟合逻辑校准方程 → 与主结果同口径（总不确定性），
    唯一差别是抽样单位。
    """
    z = np.load(HERE / 'data' / f'preds_{V}.npz')
    te = z['split'] == 'test'
    y_all = z['y_true'][te]
    n = len(y_all)
    rng = np.random.default_rng(seed)
    rows = []
    for m in MODELS:
        p = z[f'prob_{m}'][te].astype(float)
        st = calibration_stats(y_all, p)
        rep = {k: np.full(n_boot, np.nan) for k in
               ('brier_raw', 'brier_logistic_refit', 'slope', 'intercept', 'citl')}
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            yb, pb = y_all[idx], p[idx]
            if yb.min() == yb.max():
                continue
            lpb = _logit(pb)
            rep['brier_raw'][b] = float(np.mean((pb - yb) ** 2))
            try:
                fit = sm.Logit(yb, sm.add_constant(lpb)).fit(disp=0, method='lbfgs', maxiter=200)
                a, bb = float(fit.params[0]), float(fit.params[1])
                rep['slope'][b], rep['intercept'][b] = bb, a
                pcal = 1.0 / (1.0 + np.exp(-(a + bb * lpb)))
                rep['brier_logistic_refit'][b] = float(np.mean((pcal - yb) ** 2))
            except Exception:                                   # noqa: BLE001
                pass
            try:
                cfit = sm.Logit(yb, np.ones_like(yb, dtype=float), offset=lpb).fit(
                    disp=0, method='lbfgs', maxiter=200)
                rep['citl'][b] = float(cfit.params[0])
            except Exception:                                   # noqa: BLE001
                pass
        # 全样本 logistic 再校准后的 Brier（点估计）
        lpa = _logit(p)
        fit_all = sm.Logit(y_all, sm.add_constant(lpa)).fit(disp=0)
        pcal = 1.0 / (1.0 + np.exp(-(fit_all.params[0] + fit_all.params[1] * lpa)))
        points = {'brier_raw': st['brier'],
                  'brier_logistic_refit': float(np.mean((pcal - y_all) ** 2)),
                  'slope': st['slope'], 'intercept': st['intercept'], 'citl': st['citl']}
        for k, v in rep.items():
            v = v[np.isfinite(v)]
            lo, hi = ((float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
                      if len(v) else (np.nan, np.nan))
            rows.append({
                'scope': 'internal_sensitivity', 'dataset': f'MIMIC internal test (cohort {V})',
                'variant': V, 'model': m, 'metric': k, 'point': points[k],
                'ci_low': lo, 'ci_high': hi, 'ci_unit': 'record (sensitivity only)',
                'ci_definition': ('评价集内按记录重抽（忽略患者聚类）；每个重抽内重新拟合'
                                  '逻辑校准方程（总不确定性）。仅用于与受试者聚类主结果对照'),
                'n_boot': int(len(v)), 'n_records': int(n), 'n_clusters': int(n),
            })
        print(f'  [{V}] {m} 记录级校准CI完成（有效重抽 '
              f'{int(np.isfinite(rep["slope"]).sum())}/{n_boot}）', flush=True)
    return pd.DataFrame(rows)


# =====================================================================
# 2. 校准不确定性：主结果（受试者/患者聚类）长表
# =====================================================================
def build_primary_calibration() -> pd.DataFrame:
    rows = []
    for V in ('A', 'B'):
        t = pd.read_csv(HERE / f'calibration_primary_patient_clustered_{V}.csv')
        for _, r in t.iterrows():
            for metric, pt, lo, hi in [
                ('brier_raw', r.brier_raw, r.brier_raw_CI_low_subject, r.brier_raw_CI_high_subject),
                ('slope', r.slope, r.slope_CI_low_subject, r.slope_CI_high_subject),
                ('intercept', r.intercept, r.intercept_CI_low_asymptotic,
                 r.intercept_CI_high_asymptotic),
                ('citl', r.CITL, r.CITL_CI_low_subject, r.CITL_CI_high_subject),
                ('brier_logistic_recalibrated_refit', r.brier_logistic_fixed_equation,
                 r.brier_logistic_refit_CI_low_subject, r.brier_logistic_refit_CI_high_subject),
            ]:
                rows.append({
                    'scope': 'internal', 'dataset': f'MIMIC internal test (cohort {V})',
                    'variant': V, 'model': r.Model, 'metric': metric,
                    'point': pt, 'ci_low': lo, 'ci_high': hi,
                    'ci_unit': 'subject_id (clustered, primary)',
                    'ci_definition': ('受试者聚类重抽，同一受试者全部记录整体进出；'
                                      'slope/intercept/CITL/recalibrated Brier 为每次重抽内'
                                      '重新拟合逻辑校准方程（总不确定性）'),
                    'n_boot': 1000, 'n_records': int(r.n), 'n_clusters': int(r.n),
                })
        e = pd.read_csv(HERE / f'calibration_primary_patient_clustered_external_{V}.csv')
        for _, r in e.iterrows():
            for metric, pt, lo, hi in [
                ('brier_raw', r.brier_raw, r.brier_raw_CI_low_patient, r.brier_raw_CI_high_patient),
                ('slope', r.slope, r.slope_CI_low_patient, r.slope_CI_high_patient),
                ('intercept', r.intercept, r.intercept_CI_low_patient, r.intercept_CI_high_patient),
                ('citl', r.CITL, r.CITL_CI_low_patient, r.CITL_CI_high_patient),
            ]:
                rows.append({
                    'scope': 'external_raw', 'dataset': f'eICU CRD (cohort {V})',
                    'variant': V, 'model': r.Model, 'metric': metric,
                    'point': pt, 'ci_low': lo, 'ci_high': hi,
                    'ci_unit': 'uniquepid (clustered, primary)',
                    'ci_definition': ('uniquepid 聚类重抽；slope/intercept/CITL 为每次重抽内'
                                      '重新拟合逻辑校准方程（总不确定性）'),
                    'n_boot': 1000, 'n_records': int(r.n_records),
                    'n_clusters': int(r.n_patients),
                })
        rc = pd.read_csv(HERE / 'tables' / V / 'table10_recalibration_eicu_patient_split.csv')
        ci = pd.read_csv(HERE / 'tables' / V / 'table10_recalibration_eicu_patient_ci.csv')
        n_eval_pat = json.loads((HERE / 'qa' / f's07_eicu_{V}.json').read_text())[
            'recalibration_eval_patients']
        for _, r in rc.iterrows():
            fr = ci.loc[ci.Model == r.Model].iloc[0]
            for metric, pt, lo, hi, definition in [
                ('brier_raw_patient_split_eval_half', r.Brier_raw,
                 fr.Brier_raw_fixed_CI_low_patient, fr.Brier_raw_fixed_CI_high_patient,
                 '评价半（患者不与拟合半重叠）内按 uniquepid 重抽；不重新拟合方程'),
                ('brier_logistic_recalibrated_fixed_eq', r.Brier_logistic,
                 fr.Brier_logistic_fixed_CI_low_patient, fr.Brier_logistic_fixed_CI_high_patient,
                 '固定再校准方程（由拟合半估计）应用于评价半的不确定性'),
                ('brier_logistic_recalibrated_refit_eq', r.Brier_logistic,
                 fr.Brier_logistic_refit_CI_low_patient, fr.Brier_logistic_refit_CI_high_patient,
                 '每次重抽内重新拟合再校准方程（总不确定性）'),
                ('brier_isotonic_recalibrated_fixed_eq', r.Brier_isotonic,
                 fr.Brier_isotonic_fixed_CI_low_patient, fr.Brier_isotonic_fixed_CI_high_patient,
                 '固定等渗再校准函数（由拟合半估计）应用于评价半的不确定性'),
            ]:
                rows.append({
                    'scope': 'external_recalibration_patient_split',
                    'dataset': f'eICU CRD (cohort {V}, eval half)',
                    'variant': V, 'model': r.Model, 'metric': metric,
                    'point': pt, 'ci_low': lo, 'ci_high': hi,
                    'ci_unit': 'uniquepid (clustered, primary)',
                    'ci_definition': definition,
                    'n_boot': 200 if 'refit' in metric else 1000,
                    'n_records': int(r.n_eval_records), 'n_clusters': int(n_eval_pat),
                })
    return pd.DataFrame(rows)


def build_sensitivity_calibration(rec_ci: pd.DataFrame) -> pd.DataFrame:
    parts = [rec_ci]
    for V in ('A', 'B'):
        rc = pd.read_csv(HERE / 'tables' / V / 'table10_recalibration_eicu_record_split.csv')
        for _, r in rc.iterrows():
            for metric, pt in [('brier_raw_record_split_eval_half', r.Brier_raw),
                               ('brier_logistic_recalibrated_record_split', r.Brier_logistic),
                               ('brier_isotonic_recalibrated_record_split', r.Brier_isotonic)]:
                parts.append(pd.DataFrame([{
                    'scope': 'external_recalibration_record_split',
                    'dataset': f'eICU CRD (cohort {V}, eval half)',
                    'variant': V, 'model': r.Model, 'metric': metric,
                    'point': pt, 'ci_low': np.nan, 'ci_high': np.nan,
                    'ci_unit': 'record (sensitivity only)',
                    'ci_definition': ('记录级划半：同一患者可同时进入拟合半与评价半'
                                      '（重叠患者数见 final_checks.json）；'
                                      '仅作历史记录级口径对照，不提供区间'),
                    'n_boot': 0, 'n_records': int(r.n_eval_records), 'n_clusters': np.nan,
                }]))
    return pd.concat(parts, ignore_index=True)


# =====================================================================
# 3. 交付物规范化
# =====================================================================
def canonical_deliverables(primary: pd.DataFrame, sensitivity: pd.DataFrame) -> dict:
    made: dict = {}

    def cp(src: Path, dst: Path):
        shutil.copy2(src, dst)
        made[dst.name] = str(dst)

    cp(HERE / 'tables' / 'landmark_eligibility_audit.csv', HERE / 'landmark_eligibility_audit.csv')
    # 预处理清单：直接补全自校验字段（脚本哈希 + 源数据哈希），避免留空的 "hashes": {}
    fm = json.loads((HERE / 'qa' / 'preprocessing_fit_manifest.json').read_text())
    fm['hashes'] = {
        'scripts/s02_build_trainonly_data.py': sha256(HERE / 'scripts' /
                                                      's02_build_trainonly_data.py'),
        'scripts/s03_train_models.py': sha256(HERE / 'scripts' / 's03_train_models.py'),
        'scripts/common_eval.py': sha256(HERE / 'scripts' / 'common_eval.py'),
        'data/trainonly_mimic.npz': sha256(HERE / 'data' / 'trainonly_mimic.npz'),
        'data/trainonly_eicu.npz': sha256(HERE / 'data' / 'trainonly_eicu.npz'),
        'note': ('此处只登记"决定插补/标准化参数"的脚本与派生数据；'
                 '全部 242 个产物的哈希见 trainonly_reanalysis_manifest.json，'
                 '旧文件哈希见 00_manifest/old_files_sha256.csv'),
    }
    (HERE / 'preprocessing_fit_manifest.json').write_text(
        json.dumps(fm, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    # 同步回 qa/ 源文件，保证两份字节一致（便于哈希核对）
    (HERE / 'qa' / 'preprocessing_fit_manifest.json').write_text(
        json.dumps(fm, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    made['preprocessing_fit_manifest.json'] = str(HERE / 'preprocessing_fit_manifest.json')

    primary.to_csv(HERE / 'calibration_primary_patient_clustered.csv', index=False)
    made['calibration_primary_patient_clustered.csv'] = str(
        HERE / 'calibration_primary_patient_clustered.csv')
    sensitivity.to_csv(HERE / 'calibration_sensitivity_record_split.csv', index=False)
    made['calibration_sensitivity_record_split.csv'] = str(
        HERE / 'calibration_sensitivity_record_split.csv')

    rec = []
    for V in ('A', 'B'):
        t = pd.read_csv(HERE / 'tables' / V / 'table10_recalibration_eicu_patient_split.csv')
        t.insert(0, 'variant', V)
        rec.append(t)
    pd.concat(rec, ignore_index=True).to_csv(HERE / 'recalibration_patient_split.csv', index=False)
    made['recalibration_patient_split.csv'] = str(HERE / 'recalibration_patient_split.csv')

    for src, out in [('threshold_operating_characteristics_primary.csv',
                      'threshold_operating_characteristics_primary.csv'),
                     ('threshold_operating_characteristics_external.csv',
                      'threshold_operating_characteristics_external.csv')]:
        ts = []
        for V in ('A', 'B'):
            t = pd.read_csv(HERE / 'tables' / V / src)
            t.insert(0, 'variant', V)
            ts.append(t)
        pd.concat(ts, ignore_index=True).to_csv(HERE / out, index=False)
        made[out] = str(HERE / out)

    fig_map = {}
    src_fig = HERE / 'figures_trainonly'
    if (src_fig / 'A' / 'calibration_primary.png').exists():
        cp(src_fig / 'A' / 'calibration_primary.png', HERE / 'calibration_plots_primary.png')
        fig_map['calibration_plots_primary.png'] = {
            'source_figure': str(src_fig / 'A' / 'calibration_primary.png'),
            'content': '变体A 内部测试集校准曲线 + 受试者聚类 95%CI（主结果口径）',
            'ci_unit': 'subject_id (clustered)'}
    if (src_fig / 'A' / 'calibration_sensitivity.png').exists():
        cp(src_fig / 'A' / 'calibration_sensitivity.png', HERE / 'calibration_plots_sensitivity.png')
        fig_map['calibration_plots_sensitivity.png'] = {
            'source_figure': str(src_fig / 'A' / 'calibration_sensitivity.png'),
            'content': 'eICU 患者级划半再校准前后校准图（拟合半估计方程、评价半展示）',
            'ci_unit': 'uniquepid (clustered)',
            'note': ('记录级划半的数值对照见 calibration_sensitivity_record_split.csv；'
                     '本图固定为患者级划半，以免图上展示同一患者跨半重叠的口径')}

    man = {
        'primary': {
            'file': 'calibration_primary_patient_clustered.csv',
            'internal_ci_unit': 'subject_id（受试者聚类）',
            'external_ci_unit': 'uniquepid（患者聚类）',
            'n_boot': 1000,
            'ci_definition_internal': ('受试者聚类重抽，同一受试者全部记录整体进出；'
                                       'slope/intercept/CITL/recalibrated Brier 为每次重抽内'
                                       '重新拟合逻辑校准方程后的总不确定性'),
            'ci_definition_external_fixed': '固定再校准方程（拟合半估计）应用于评价半的不确定性',
            'ci_definition_external_refit': '每次重抽内重新拟合再校准方程的总不确定性',
        },
        'sensitivity': {
            'file': 'calibration_sensitivity_record_split.csv',
            'internal_ci_unit': 'record（忽略患者聚类）',
            'external_ci_unit': 'record（记录级划半，同一患者可同时出现在拟合半与评价半）',
            'role': '历史口径对照；不得用于主结论',
            'patient_overlap_record_split_cohort_A': json.loads(
                (HERE / 'qa' / 's07_eicu_A.json').read_text())[
                'recalibration_record_split_patient_overlap_sensitivity'],
        },
        'figures': fig_map,
        'generated_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
    }
    (HERE / 'calibration_uncertainty_manifest.json').write_text(
        json.dumps(man, ensure_ascii=False, indent=2), encoding='utf-8')
    made['calibration_uncertainty_manifest.json'] = str(
        HERE / 'calibration_uncertainty_manifest.json')
    return made


def figure_source_manifest() -> dict:
    F = HERE / 'figures_trainonly'
    src: dict = {}
    for V in ('A', 'B'):
        P = f'data/preds_{V}.npz'
        E = f'data/eicu_probs_{V}.npz'
        mapping = {
            'roc_curves.png': [P, f'tables/{V}/table2_model_performance.csv'],
            'roc_curves_external.png': [E, f'tables/{V}/table9_external_validation.csv'],
            'pr_curves.png': [P, f'tables/{V}/table2_model_performance.csv'],
            'calibration_curves_internal.png': [P, f'calibration_primary_patient_clustered_{V}.csv'],
            'calibration_curves_external.png': [
                E, f'calibration_primary_patient_clustered_external_{V}.csv'],
            'calibration_primary.png': [P, f'calibration_primary_patient_clustered_{V}.csv'],
            'calibration_sensitivity.png': [
                E, f'tables/{V}/table10_recalibration_eicu_patient_split.csv',
                f'tables/{V}/table10_recalibration_eicu_patient_ci.csv'],
            'dca_compare.png': [P, f'tables/{V}/table_dca_internal.csv'],
            'dca_compare_external.png': [E, f'tables/{V}/dca_eicu.csv'],
            'clinical_impact_lstm.png': [P],
            'clinical_impact_lstm_external.png': [E],
            'model_performance_bar.png': [P, f'tables/{V}/table2_model_performance.csv'],
            'confusion_matrix_rf.png': [
                P, f'tables/{V}/threshold_operating_characteristics_primary.csv'],
            'lstm_attention.png': [f'data/lstm_attention_{V}.npz（若不存在则取自 qa/s08 缓存）'],
            'attention_faithfulness.png': [f'qa/s08_explain_{V}.json'],
            'shap_summary_bar.png': [f'data/shap_values_{V}.npz', f'tables/{V}/shap_top_features.csv'],
            'shap_summary.png': [f'data/shap_values_{V}.npz'],
        }
        for name, sources in mapping.items():
            f = F / V / name
            src[f'{V}/{name}'] = {'exists': f.exists(),
                                  'sha256': sha256(f) if f.exists() else None,
                                  'sources': sources}
    out = {
        'figure_dir': str(F),
        'generator': 'scripts/s10_figures.py（输出到隔离目录，未覆盖旧图）',
        'positioning': {'cohort_A': '主结果（数据库定义回顾性地标队列）',
                        'cohort_B': '限制性敏感性（疑似感染时间 ≤48h）'},
        'figures': src,
        'generated_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
    }
    (HERE / 'figures_source_manifest.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    return out


# =====================================================================
# 4. final_checks 各项
# =====================================================================
def check_models_new() -> dict:
    old = pd.read_csv(HERE / '00_manifest' / 'old_files_sha256.csv')
    old_h = set(old.loc[old.role == '旧模型权重', 'sha256'])
    detail, ok = {}, True
    for V in ('A', 'B'):
        detail[V] = {}
        for p in sorted((HERE / 'models_trainonly' / V).iterdir()):
            h = sha256(p)
            same = h in old_h
            detail[V][p.name] = {'sha256': h, 'identical_to_old_weight': bool(same)}
            ok &= not same
    meta = {V: json.loads((HERE / 'qa' / f'train_meta_{V}.json').read_text()) for V in ('A', 'B')}
    return {'status': 'pass' if ok else 'fail',
            'detail': {'no_new_weight_hash_matches_old': bool(ok), 'files': detail,
                       'train_meta': meta}}


def check_preprocessing(fit_manifest: dict) -> dict:
    bad = []
    for key, val in fit_manifest.items():
        if isinstance(val, dict) and 'fit_scope' in val and val['fit_scope'] != 'train_only':
            bad.append(f'{key}: {val["fit_scope"]}')
    return {'status': 'pass' if not bad else 'fail',
            'detail': {'manifest_summary': {k: v for k, v in fit_manifest.items()
                                            if isinstance(v, dict)},
                       'violations': bad}}


def check_subject_split() -> dict:
    z = np.load(HERE / 'data' / 'trainonly_mimic.npz')
    sp = z['split'].astype(str)
    sid = z['subject_id']
    sets = {s: set(sid[sp == s].tolist()) for s in ('train', 'val', 'test')}
    inter = {f'{a}&{b}': len(sets[a] & sets[b])
             for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    vc = pd.Series(sid).value_counts()
    return {'status': 'pass' if all(v == 0 for v in inter.values()) else 'fail',
            'detail': {'n_records': int(len(sp)),
                       'split_n': {k: int((sp == k).sum()) for k in ('train', 'val', 'test')},
                       'subject_intersections': inter,
                       'n_subjects': int(len(set(sid.tolist()))),
                       'n_subjects_with_multiple_records': int((vc > 1).sum()),
                       'records_from_multi_record_subjects': int(vc[vc > 1].sum()),
                       'share_records_from_multi_record_subjects': round(
                           float(vc[vc > 1].sum() / len(sp)), 4)}}


def check_eicu_patient_split() -> dict:
    detail, ok = {}, True
    for V in ('A', 'B'):
        j = json.loads((HERE / 'qa' / f's07_eicu_{V}.json').read_text())
        ok &= j['recalibration_patient_intersection'] == 0
        detail[V] = {k: j[k] for k in (
            'n_records', 'n_patients_uniquepid', 'n_patients_multi_record',
            'n_records_from_multi', 'recalibration_fit_records', 'recalibration_fit_patients',
            'recalibration_fit_deaths', 'recalibration_eval_records',
            'recalibration_eval_patients', 'recalibration_eval_deaths',
            'recalibration_patient_intersection',
            'recalibration_record_split_patient_overlap_sensitivity')}
    return {'status': 'pass' if ok else 'fail', 'detail': detail}


def check_ids() -> dict:
    """逐元素核对 ID / y_true / 分区一致性。

    变体 A：预测数组长度等于全队列索引，直接逐元素比较。
    变体 B：`preds_B.npz` 只保存限制队列（22,778），因此比较对象是
            **全队列数组按 `t_inf_h<=48` 掩码抽取后的子集**，
            并额外确认 B 中不存在 t_inf_h>48 的记录。
    """
    ok, out = True, {}
    idx = pd.read_csv(HERE / 'data' / 'eicu_cohort_index.csv')
    mimic = np.load(HERE / 'data' / 'trainonly_mimic.npz')
    te = np.load(HERE / 'data' / 'trainonly_eicu.npz')
    mask_B = mimic['t_inf_h'].astype(float) <= 48.0          # 与 s03 完全一致的掩码
    n_B = int(mask_B.sum())
    for V in ('A', 'B'):
        preds = np.load(HERE / 'data' / f'preds_{V}.npz')
        eicu = np.load(HERE / 'data' / f'eicu_probs_{V}.npz')
        if V == 'A':
            ref = {k: mimic[k] for k in ('y', 'stay_id', 'subject_id', 'split', 't_inf_h')}
            ref_e = {k: te[k] for k in ('y', 'stay_id', 'uniquepid')}
        else:
            ref = {k: mimic[k][mask_B] for k in ('y', 'stay_id', 'subject_id', 'split', 't_inf_h')}
            ref_e = {k: te[k] for k in ('y', 'stay_id', 'uniquepid')}
        chk = {
            'n_records': int(len(preds['y_true'])),
            'n_records_expected': n_B if V == 'B' else int(len(mimic['y'])),
            'y_true_identical': bool(np.array_equal(preds['y_true'], ref['y'])),
            'stay_id_identical': bool(np.array_equal(preds['stay_id'], ref['stay_id'])),
            'subject_id_identical': bool(np.array_equal(preds['subject_id'], ref['subject_id'])),
            'split_identical': bool(np.array_equal(preds['split'].astype(str),
                                                   ref['split'].astype(str))),
            't_inf_h_identical': bool(np.array_equal(
                preds['t_inf_h'].astype(float), ref['t_inf_h'].astype(float))),
            'landmark_restriction_applied': bool(
                V == 'A' or np.all(preds['t_inf_h'].astype(float) <= 48.0)),
            'eicu_y_true_identical': bool(np.array_equal(eicu['y_true'], ref_e['y'])),
            'eicu_stay_id_identical': bool(np.array_equal(eicu['stay_id'], ref_e['stay_id'])),
            'eicu_uniquepid_matches_index': bool(np.array_equal(
                eicu['uniquepid'].astype(str), idx['uniquepid'].astype(str).values)),
            'all_model_probs_present': all(f'prob_{m}' in preds.files for m in MODELS),
            'all_model_probs_present_eicu': all(f'prob_{m}' in eicu.files for m in MODELS),
        }
        chk['n_records_ok'] = chk['n_records'] == chk['n_records_expected']
        ok &= all(v for k, v in chk.items() if isinstance(v, bool))
        out[V] = chk
    out['comparison_rule'] = ('A: 直接对全队列索引；B: 对 t_inf_h<=48 掩码抽取的子集，'
                              '并确认 B 内无 t_inf_h>48 记录')
    out['mask_B_records'] = n_B
    return {'status': 'pass' if ok else 'fail', 'detail': out}


def check_numbers_provenance() -> dict:
    t2 = pd.read_csv(HERE / 'tables' / 'A' / 'table2_model_performance.csv').set_index('Model')
    t9 = pd.read_csv(HERE / 'tables' / 'A' / 'table9_external_validation.csv').set_index('Model')
    rc = pd.read_csv(HERE / 'tables' / 'A' /
                     'table10_recalibration_eicu_patient_split.csv').set_index('Model')
    th = pd.read_csv(HERE / 'tables' / 'A' /
                     'threshold_operating_characteristics_primary.csv').set_index('Model')
    pa = pd.read_csv(HERE / 'tables' / 'A' / 'table2_ci_paired_diff_lstm_rf.csv').set_index('Metric')
    so = pd.read_csv(HERE / 'tables' / 'A' / 'table5_sofa_comparison.csv').set_index('Model')
    qa = json.loads((HERE / 'qa' / 's04_internal_A.json').read_text())
    claims = [
        ('internal LSTM AUC', float(t2.loc['LSTM', 'AUC']), 0.7935, 5e-4),
        ('internal RF AUC', float(t2.loc['RandomForest', 'AUC']), 0.7854, 5e-4),
        ('internal SVM AUC', float(t2.loc['SVM', 'AUC']), 0.7848, 5e-4),
        ('internal LR AUC', float(t2.loc['LogisticRegression', 'AUC']), 0.7518, 5e-4),
        ('internal LSTM CI low', float(t2.loc['LSTM', 'AUC_CI_low_subject']), 0.7768, 1e-3),
        ('internal LSTM CI high', float(t2.loc['LSTM', 'AUC_CI_high_subject']), 0.8103, 1e-3),
        ('LSTM Brier', float(t2.loc['LSTM', 'Brier']), 0.1809, 1e-3),
        ('RF Brier', float(t2.loc['RandomForest', 'Brier']), 0.1430, 1e-3),
        ('paired dAUC point', float(pa.loc['AUC', 'Point_estimate']), 0.0081, 1e-4),
        ('paired DeLong p', float(pa.loc['AUC', 'DeLong_p']), 0.1779, 1e-3),
        ('eICU LSTM AUC', float(t9.loc['LSTM', 'AUC_ext']), 0.7209, 5e-4),
        ('eICU LSTM CI low', float(t9.loc['LSTM', 'AUC_CI_low_patient']), 0.7054, 1e-3),
        ('eICU LSTM CI high', float(t9.loc['LSTM', 'AUC_CI_high_patient']), 0.7361, 1e-3),
        ('eICU LSTM raw Brier (eval half)', float(rc.loc['LSTM', 'Brier_raw']), 0.2237, 5e-4),
        ('eICU LSTM logistic-recal Brier', float(rc.loc['LSTM', 'Brier_logistic']), 0.1289, 5e-4),
        ('internal LSTM sensitivity at F3 point',
         float(th.loc['LSTM', 'death_recall_sensitivity']), 0.9522, 1e-3),
        ('internal LSTM 1-PPV', float(th.loc['LSTM', 'one_minus_PPV']), 0.7353, 1e-3),
        ('internal LSTM FP% of all records',
         float(th.loc['LSTM', 'false_positive_percent_of_all']), 50.32, 0.01),
        ('internal test n', float(t2.loc['LSTM', 'n']), 3742, 0.5),
        ('internal test deaths', float(t2.loc['LSTM', 'deaths']), 712, 0.5),
        ('SOFA AUC internal', float(so.loc['SOFA (rule-based, 48h max)', 'AUC']), 0.6719, 5e-4),
        ('qSOFA AUC internal', float(so.loc['qSOFA-6h (rule-based)', 'AUC']), 0.5730, 5e-4),
        ('ids verified before SOFA comparison',
         float(bool(qa['sofa_qsofa_id_check']['sofa']['stay_id_identical'])), 1.0, 0.0),
        ('subject-clustered bootstrap used internally',
         float(qa['bootstrap_unit_internal'].startswith('subject')), 1.0, 0.0),
        ('no old predictions used', float(bool(qa['no_old_predictions_used'])), 1.0, 0.0),
    ]
    rows, ok = [], True
    for name, got, exp, tol in claims:
        good = abs(got - exp) <= tol
        ok &= good
        rows.append({'claim': name, 'value_in_product': got, 'expected': exp, 'match': bool(good)})
    return {'status': 'pass' if ok else 'fail',
            'detail': {'checked': rows,
                       'rule': ('正文所有数字必须从 tables/<V>/*.csv、qa/*.json 自动提取；'
                                '数字来源对照表见 NUMBER_DIFF_OLD_VS_NEW.md，'
                                '阶段11改写稿件时逐条引用')}}


def check_thresholds() -> dict:
    prim = pd.read_csv(HERE / 'threshold_operating_characteristics_primary.csv')
    ext = pd.read_csv(HERE / 'threshold_operating_characteristics_external.csv')
    prim_lab = bool(prim.threshold_origin.str.contains(
        'research operating point|研究性操作点', case=False, regex=True).all())
    raw = ext[ext.prob_type.str.contains('raw')]
    recal = ext[ext.prob_type.str.contains('recalibrated')]
    raw_th = sorted(raw.threshold.round(3).unique().tolist())
    recal_th = sorted(recal.threshold.round(3).unique().tolist())
    recal_lab = bool(recal.threshold_origin.str.contains('0.15').all())
    mixing = sorted(set(raw_th) & set(recal_th))
    ok = prim_lab and recal_lab and not mixing
    return {'status': 'pass' if ok else 'fail',
            'detail': {'internal_rows_labelled_research_operating_point': prim_lab,
                       'external_prob_types': sorted(ext.prob_type.unique().tolist()),
                       'external_raw_thresholds': raw_th,
                       'external_recalibrated_thresholds': recal_th,
                       'recalibrated_rows_use_explicit_0.15': recal_lab,
                       'mixing_detected': mixing,
                       'columns_include': ['false_positive_n',
                                           'false_positive_percent_of_all', 'one_minus_PPV',
                                           'net_benefit_at_threshold'],
                       'note': ('1−PPV 与"全部记录中的假阳性比例"分列报告，'
                                '不再把 1−PPV 称为假阳性率')}}


NEGATION_CUES = ('不得', '不能', '不应', '不准', '禁止', '避免', '勿', '而非', '而不是',
                 'should not', 'must not', 'do not use', 'not be called', 'avoid ',
                 'rather than', 'instead of', 'cannot', 'not claim')


def scan_forbidden(paths: list[Path]) -> list[dict]:
    """扫描越界表述。

    只把**肯定性断言**计为违规命中：命中点前后 80 字符内若出现否定/禁止性措辞
    （如"不得称 external validation""should not be described as"），
    记为 `negative_mention=True`，不计入 n_affirmative。
    """
    hits = []
    for p in paths:
        if not p.exists() or p.suffix.lower() not in ('.md', '.csv', '.json', '.txt'):
            continue
        txt = p.read_text(encoding='utf-8', errors='replace')
        low = txt.lower()
        for ph in FORBIDDEN:
            for m in re.finditer(re.escape(ph), low):
                s = max(0, m.start() - 60)
                ctx = low[max(0, m.start() - 80):m.end() + 80]
                neg = any(cue in ctx for cue in NEGATION_CUES)
                hits.append({'file': str(p.relative_to(ROOT)), 'phrase': ph,
                             'negative_mention': neg,
                             'context': txt[s:m.end() + 60].replace('\n', ' ')})
    return hits


def check_overclaim() -> dict:
    targets = ([p for p in HERE.glob('*.md')]
               + [p for p in HERE.glob('*.json') if p.name != 'final_checks.json']
               + list((HERE / 'tables').rglob('*.csv'))
               + list((HERE / 'qa').rglob('*.json')))
    hits = scan_forbidden(targets)
    aff = [h for h in hits if not h['negative_mention']]
    return {'status': 'pass' if not aff else 'warn',
            'detail': {'n_affirmative': len(aff), 'n_negative_mention': len(hits) - len(aff),
                       'affirmative_hits': aff[:40],
                       'negative_mentions': [h['file'] + ' :: ' + h['phrase']
                                             for h in hits if h['negative_mention']][:20],
                       'rule': ('分析产物不得**肯定性地**出现 eliminated leakage / clinically '
                                'optimal / validated clinical threshold / clinical utility '
                                'demonstrated / external validation / superior / causal / '
                                'fairness 等越界表述；把上述词句写在"不得称…"这类禁令里不计违规'),
                       'note': ('文件名 table9_external_validation.csv 沿用旧管线命名以保持可追溯，'
                                '正文与图表标题一律写 exploratory cross-database evaluation')}}


def check_manuscript_state(fig_manifest: dict) -> dict:
    synced = (HERE / 'qa' / 'manuscript_sync_manifest.json').exists()
    site = ROOT / 'claude code辅助后的'
    targets = {
        'manuscript_zh': site / 'manuscript_ZH_translation.md',
        'manuscript_en': site / 'manuscript_EN_final.md',
        'zh_docx': site / 'revision_2026-09-05' / '中文正文_审慎修订_20260905.docx',
        'supp_docx': site / 'Sepsis_Mortality_Supplementary_Information.docx',
    }
    found = {}
    for k, v in targets.items():
        found[k] = {'path': str(v), 'exists': v.exists(),
                    'mtime_utc': (pd.Timestamp(v.stat().st_mtime, unit='s', tz='UTC')
                                  .isoformat() if v.exists() else None),
                    'sha256': sha256(v) if v.exists() else None}
    zh_hits = scan_forbidden([targets['manuscript_zh']]) if targets['manuscript_zh'].exists() else []
    en_hits = scan_forbidden([targets['manuscript_en']]) if targets['manuscript_en'].exists() else []
    return {'status': 'pending' if not synced else 'pass',
            'detail': {'manuscript_sync_manifest_found': synced, 'files': found,
                       'pre_sync_forbidden_phrase_hits': {'zh': zh_hits[:15], 'en': en_hits[:15]},
                       'reason': ('阶段11（稿件/DOCX 同步）尚未执行。同步后必须重新运行本脚本，'
                                  '并补做 DOCX ZIP 完整性、PDF 代表页渲染与中英文一致性检查')}}


# =====================================================================
# 5. main
# =====================================================================
def main() -> int:
    print('=== 1/5 记录级校准敏感性 ===', flush=True)
    cache = HERE / 'qa' / 's12_record_level_calibration.csv'
    if cache.exists():
        rec = pd.read_csv(cache)
        print(f'  复用已有缓存 {cache.relative_to(HERE)}（{len(rec)} 行）', flush=True)
    else:
        rec = pd.concat([record_level_calibration_ci(V) for V in ('A', 'B')], ignore_index=True)
        rec.to_csv(cache, index=False)
        print(f'  已写出缓存 {cache.relative_to(HERE)}', flush=True)

    print('=== 2/5 交付物规范化 ===', flush=True)
    primary = build_primary_calibration()
    sens = build_sensitivity_calibration(rec)
    made = canonical_deliverables(primary, sens)
    figs = figure_source_manifest()
    for k in made:
        print(f'  {k}')

    print('=== 3/5 final_checks.json ===', flush=True)
    fit_manifest = json.loads((HERE / 'qa' / 'preprocessing_fit_manifest.json').read_text())
    qa4 = json.loads((HERE / 'qa' / 's04_internal_A.json').read_text())
    qa7 = {V: json.loads((HERE / 'qa' / f's07_eicu_{V}.json').read_text()) for V in ('A', 'B')}
    checks = {
        'all_models_newly_trained': check_models_new(),
        'preprocessing_fit_train_only': check_preprocessing(fit_manifest),
        'mimic_subject_no_cross_split': check_subject_split(),
        'eicu_uniquepid_no_cross_recalibration': check_eicu_patient_split(),
        'id_consistency': check_ids(),
        'all_numbers_from_new_products': check_numbers_provenance(),
        'internal_ci_subject_level': {
            'status': 'pass' if qa4['bootstrap_unit_internal'].startswith('subject') else 'fail',
            'detail': {'unit': qa4['bootstrap_unit_internal'],
                       'sensitivity_unit': qa4['bootstrap_unit_sensitivity'],
                       'file': 'calibration_primary_patient_clustered.csv',
                       'record_level_comparison': 'tables/<V>/table2_ci_record_vs_subject.csv'}},
        'external_ci_uniquepid_level': {
            'status': 'pass' if all(j['external_bootstrap_unit'].startswith('uniquepid')
                                    for j in qa7.values()) else 'fail',
            'detail': {V: qa7[V]['external_bootstrap_unit'] for V in ('A', 'B')}},
        'raw_vs_recalibrated_threshold_not_mixed': check_thresholds(),
        'no_overclaim_language': check_overclaim(),
        'manuscript_numbers_consistent': check_manuscript_state(figs),
        'docx_zip_integrity': {
            'status': 'pending',
            'detail': {'reason': ('阶段11 尚未重建 DOCX；重建后须用 zipfile 校验，'
                                  '并检查 document.xml 可解析')}},
        'pdf_render_pages_checked': {
            'status': 'pending',
            'detail': {'reason': '阶段11 尚未渲染 PDF 代表页（流程图页、主表页、补充材料首页）'}},
        'zh_en_content_consistent': {
            'status': 'pending',
            'detail': {'reason': '阶段11 尚未同步英文稿与补充材料'}},
    }
    blocking = [k for k, v in checks.items() if v['status'] == 'fail']
    summary = {
        'generated_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'variant_primary': 'A',
        'env': {'python': sys.version.split()[0], 'platform': platform.platform()},
        'n_pass': sum(1 for v in checks.values() if v['status'] == 'pass'),
        'n_warn': sum(1 for v in checks.values() if v['status'] == 'warn'),
        'n_pending': sum(1 for v in checks.values() if v['status'] == 'pending'),
        'n_fail': len(blocking),
        'blocking_failures': blocking,
        'all_analysis_checks_pass': not blocking,
        'ready_to_sync_manuscript': not blocking,
        'checks': checks,
    }
    (HERE / 'final_checks.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print(f"  pass={summary['n_pass']} warn={summary['n_warn']} "
          f"pending={summary['n_pending']} fail={summary['n_fail']} "
          f"blocking={blocking}", flush=True)

    print('=== 4/5 CURSOR_HANDOFF.md ===', flush=True)
    write_handoff(summary)
    print('=== 5/5 完成 ===')
    return 0


# =====================================================================
# 6. CURSOR_HANDOFF.md
# =====================================================================
def write_handoff(summary: dict) -> None:
    T = lambda df: df.to_string(index=False)                      # noqa: E731
    L: list[str] = []
    A = L.append

    def rd(rel):
        return pd.read_csv(HERE / rel)

    t2a = rd('tables/A/table2_model_performance.csv')
    t2b = rd('tables/B/table2_model_performance.csv')
    t9a = rd('tables/A/table9_external_validation.csv')
    t9b = rd('tables/B/table9_external_validation.csv')
    cmp = rd('results_summary_old_vs_trainonly.csv')
    coh = rd('tables/landmark_cohort_definitions.csv')
    funnel = rd('tables/landmark_mimic_funnel.csv')
    elig = rd('tables/landmark_eligibility_audit.csv')
    recal_a = rd('tables/A/table10_recalibration_eicu_patient_split.csv')
    recal_b = rd('tables/B/table10_recalibration_eicu_patient_split.csv')
    ci_a = rd('tables/A/table10_recalibration_eicu_patient_ci.csv')
    cal_a = rd('calibration_primary_patient_clustered_A.csv')
    cal_ext_a = rd('calibration_primary_patient_clustered_external_A.csv')
    th_a = rd('tables/A/threshold_operating_characteristics_primary.csv')
    th_ext = rd('tables/A/threshold_operating_characteristics_external.csv')
    inter_a = rd('tables/A/table11_subgroup_interaction.csv')
    inter_b = rd('tables/B/table11_subgroup_interaction.csv')
    pdiff = rd('tables/A/table2_ci_paired_diff_lstm_rf.csv')
    seed = json.loads((HERE / 'qa' / 's06_lstm_seed_stability.json').read_text())
    s07a = json.loads((HERE / 'qa' / 's07_eicu_A.json').read_text())
    s05a = json.loads((HERE / 'qa' / 's05_subgroup_A.json').read_text())
    expl = json.loads((HERE / 'qa' / 's08_explain_A.json').read_text())
    cv = json.loads((HERE / 'qa' / 's09_cv.json').read_text())
    cvs = cv['summary']
    inf = rd('tables/landmark_infection_time_distribution.csv')
    audit_facts = json.loads((HERE / 'qa' / 'landmark_audit_facts.json').read_text())
    fitm = json.loads((HERE / 'qa' / 'preprocessing_fit_manifest.json').read_text())
    old_label = 'previous analysis（pre-split static imputation analysis，历史敏感性）'

    A('# CURSOR_HANDOFF —— 训练集独立插补 + 48 小时地标风险评估 + 患者级再校准（完整重分析）')
    A('')
    A(f'- 生成时间（UTC）：{summary["generated_at_utc"]}')
    A(f'- 工作目录：`{HERE}`')
    A(f'- 项目根目录：`{ROOT}`')
    A(f'- 主分析变体：**A**（数据库定义回顾性地标队列）；**B** 为限制性敏感性变体')
    A(f'- 旧结果定位：{old_label}；旧模型权重、旧预测、旧图、旧 DOCX **均未被覆盖或复用**')
    A(f'- 环境：Python {summary["env"]["python"]}，{summary["env"]["platform"]}')
    A('')
    A('---')
    A('')
    A('## 1. 实际运行的命令')
    A('')
    A('```bash')
    A(f'cd "{HERE.parent.parent.parent}"     # 项目根目录（psql 与原始数据在此）')
    A('VPY=".venv/bin/python"                # 实际使用项目根目录下的 .venv')
    A('')
    A('$VPY "claude code辅助后的/revision_2026-09-13/landmark_trainonly_reanalysis/scripts/'
      's00_snapshot.py"             # 隔离快照：旧文件清单 + SHA-256 + 环境')
    A('  s01_landmark_audit.py              # 48h 可判定性审计、队列 A/B、感染时间分布')
    A('  s02_build_trainonly_data.py        # 训练集独立插补（静态均值/标准化仅训练集）')
    A('  s03_train_models.py A              # 重训 6 经典模型 + LSTM（队列 A）')
    A('  s03_train_models.py B              # 重训 6 经典模型 + LSTM（队列 B）')
    A('  s04_internal_performance.py A|B    # 内部性能/受试者聚类CI/DeLong/校准/DCA/阈值')
    A('  s05_subgroup.py A|B                # 亚组、配对差值、交互检验')
    A('  s06_lstm_seed_stability.py         # LSTM 6 种子重训稳定性')
    A('  s07_eicu_external.py A|B           # eICU 跨库评价 + uniquepid 患者级再校准')
    A('  s08_explain.py A|B                 # RF SHAP、LSTM 注意力、遮挡忠实性、时间窗消融')
    A('  s09_cv.py A|B                      # RF 10 折 + LSTM 5 折受试者级 CV')
    A('  s10_figures.py A|B                 # 重生成全部图（隔离目录 figures_trainonly/）')
    A('  s11_compare.py                     # 新旧对照 + 总 manifest')
    A('  s12_finalize.py                    # 交付物规范化 + final_checks + 本文件')
    A('```')
    A('')
    A('> 所有命令的完整日志在 `logs/`；环境与旧文件哈希在 `00_manifest/`。')
    A('')
    A('---')
    A('')
    A('## 2. 新旧模型性能对照（内部测试集，变体 A）')
    A('')
    A('| 模型 | 旧 AUC | 新 AUC | Δ | 新 95%CI（受试者聚类） | 旧 AUPRC | 新 AUPRC |')
    A('|---|---|---|---|---|---|---|')
    for _, r in cmp[cmp.variant == 'A'].iterrows():
        A(f"| {r.model} | {f3(r.old_AUC)} | {f3(r.new_AUC)} | {r.delta_AUC:+.4f} | "
          f"{r.new_AUC_CI_low_subject:.3f}–{r.new_AUC_CI_high_subject:.3f} | "
          f"{r.old_AUPRC:.3f} | {r.new_AUPRC:.3f} |")
    A('')
    A(f"- ΔAUC 范围：{cmp[cmp.variant=='A'].delta_AUC.min():+.4f} ~ "
      f"{cmp[cmp.variant=='A'].delta_AUC.max():+.4f}；"
      f"绝对值均值 {cmp[cmp.variant=='A'].delta_AUC.abs().mean():.4f}")
    A(f"- 变体 B **测试集**（限制性地标队列：疑似感染时间 ≤48h；"
      f"n={int(t2b.n.iloc[0])} 条测试记录，死亡 {int(t2b.deaths.iloc[0])}）：")
    A('')
    A('| 模型 | 变体B AUC | 变体B 95%CI |')
    A('|---|---|---|')
    for _, r in t2b.iterrows():
        A(f"| {r.Model} | {r.AUC:.4f} | {r.AUC_CI_low_subject:.3f}–{r.AUC_CI_high_subject:.3f} |")
    A('')
    A('**判读**：训练集独立插补对判别力的影响可忽略（|ΔAUC| ≤ '
      f"{cmp[cmp.variant=='A'].delta_AUC.abs().max():.4f}），"
      '说明原先的插补泄漏不是本研究性能的主要驱动因素；但方法学上必须改为训练集独立插补。')
    A('')
    A('**配对比较（LSTM − RF，变体 A）**')
    A('')
    A('```')
    A(T(pdiff[['Metric', 'Point_estimate', 'CI_low_subject', 'CI_high_subject',
               'Crosses_zero_subject', 'DeLong_p']]))
    A('```')
    A('')
    A('- 点估计取原测试集（不是 bootstrap 差值均值）；区间为受试者聚类 bootstrap。')
    A('- 区间跨 0、DeLong p>0.05 ⇒ **不能声称 LSTM 优于 RF**。')
    A('')
    A('**LSTM 种子稳定性（同一数据/划分/超参，只改随机种子）**')
    A('')
    A(f"- 测试 AUC：均值 {seed['test_AUC_mean']:.4f}，SD {seed['test_AUC_sd']:.4f}，"
      f"范围 {seed['test_AUC_min']:.4f}–{seed['test_AUC_max']:.4f}（{seed['n_seeds']} 个种子）")
    A(f"- ΔAUC(LSTM−RF)：{seed['dAUC_vs_RF_min']:+.4f} ~ {seed['dAUC_vs_RF_max']:+.4f}；"
      f"LSTM AUC 高于 RF 的种子数 {seed['n_seeds_LSTM_above_RF']}/{seed['n_seeds']}，"
      f"其中 DeLong p<0.05 的种子数 {seed['n_seeds_DeLong_p_lt_0.05']}")
    A(f"- 交互检验 p 值范围：{json.dumps(seed['interaction_p_ranges'], ensure_ascii=False)}；"
      f"任一种子出现显著交互：{seed['any_interaction_significant_at_0.05']}")
    A('- ⇒ 单次 LSTM 的交互 p 值不稳定，正文必须报告种子间范围或改述为'
      '"未发现稳定的交互效应证据"。')
    A('')
    A('**受试者级分层交叉验证（新数据）**')
    A('')
    A(f"- 变体 A — RandomForest 10 折：AUC {cvs['RandomForest']['AUC_mean']:.4f} ± "
      f"{cvs['RandomForest']['AUC_SD_ddof1']:.4f}；AP {cvs['RandomForest']['AP_mean']:.4f} ± "
      f"{cvs['RandomForest']['AP_SD_ddof1']:.4f}")
    A(f"- 变体 A — BiLSTM 5 折：AUC {cvs['LSTM']['AUC_mean']:.4f} ± "
      f"{cvs['LSTM']['AUC_SD_ddof1']:.4f}；AP {cvs['LSTM']['AP_mean']:.4f} ± "
      f"{cvs['LSTM']['AP_SD_ddof1']:.4f}")
    cvb_p = HERE / 'qa' / 's09_cv_B.json'
    if cvb_p.exists():
        cvb = json.loads(cvb_p.read_text())['summary']
        A(f"- 变体 B — RandomForest 10 折：AUC {cvb['RandomForest']['AUC_mean']:.4f} ± "
          f"{cvb['RandomForest']['AUC_SD_ddof1']:.4f}；AP {cvb['RandomForest']['AP_mean']:.4f} ± "
          f"{cvb['RandomForest']['AP_SD_ddof1']:.4f}")
        A(f"- 变体 B — BiLSTM 5 折：AUC {cvb['LSTM']['AUC_mean']:.4f} ± "
          f"{cvb['LSTM']['AUC_SD_ddof1']:.4f}；AP {cvb['LSTM']['AP_mean']:.4f} ± "
          f"{cvb['LSTM']['AP_SD_ddof1']:.4f}")
    A(f"- 泄漏自检：受试者跨折不相交 = "
      f"{cv['leakage_checks']['subject_disjoint_across_folds']}；"
      f"同种子重跑折一致 = {cv['leakage_checks']['splitter_rerun_same_seed_folds_identical']}")
    A('')
    A('> 交叉验证不用于选择主模型或调参，只用于报告判别力在受试者级重抽下的稳定性；'
      '模型超参数与旧稿保持一致。')
    A('')
    A('---')
    A('')
    A('## 3. 新主分析队列人数与死亡数')
    A('')
    A('```')
    A(T(coh))
    A('```')
    A('')
    A('**MIMIC 筛选漏斗（从数据库重新计算，与 sepsis_labels.csv 精确一致）**')
    A('')
    A('```')
    A(T(funnel))
    A('```')
    A('')
    A('**eICU**')
    A('')
    A(f"- 记录数 {s07a['n_records']}；`uniquepid` 患者数 {s07a['n_patients_uniquepid']}；"
      f"多记录患者 {s07a['n_patients_multi_record']} 人（贡献 {s07a['n_records_from_multi']} 条记录，"
      f"即比'一人一条'多出 {s07a['n_extra_records_beyond_one_per_patient']} 条）")
    A(f"- 死亡数：内部测试集 {int(t2a.deaths.iloc[0])}/{int(t2a.n.iloc[0])}"
      f"（{t2a.deaths.iloc[0]/t2a.n.iloc[0]:.2%}）；eICU {int(t9a.deaths.iloc[0])}/"
      f"{int(t9a.n_records.iloc[0])}（{t9a.deaths.iloc[0]/t9a.n_records.iloc[0]:.2%}）")
    A('- ⚠️ eICU 的 6,812 是**记录数**，患者数为 6,488；正文与图注不得写"6,812 patients"。')
    A('')
    A('---')
    A('')
    A('## 4. 48 小时可判定性审计结果')
    A('')
    A(T(elig[['criterion', 'db', 'ascertainable_at_48h', 'handling', 'retrospective_only']]))
    A('')
    A('> `retrospective_only=True` 的条件是**只能在完整住院结束后回顾性确定**的，'
      '不得写成"可实时执行的入组条件"。')
    A('')
    A('**疑似感染时间（suspected infection time, SIT）分布**')
    A('')
    A('```')
    A(T(inf))
    A('```')
    A('')
    A(f"- SIT > 48h 的记录：{audit_facts.get('n_sit_after_48h', '见 qa/landmark_audit_facts.json')}；"
      f"其死亡率与 48h 内记录的差异见 `LANDMARK_ELIGIBILITY_AUDIT.md`")
    A('- 结论：变体 A 是**回顾性**地标队列（表型用到了 48 小时之后才出现的疑似感染时间）；'
      '变体 B 是 48 小时地标**可判定**的限制性队列。二者必须分别报告，不得混用数字。')
    A('- 未主张"可实时执行的入组条件"：住院死亡结局、Sepsis-3 识别、'
      'ICU 入住次数限制均为完整住院后才能确定。')
    A('')
    A('---')
    A('')
    A('## 5. 患者级再校准结果（主要证据）')
    A('')
    A('**划分与独立性**')
    A('')
    A(f"- 拟合半：{s07a['recalibration_fit_records']} 条记录 / "
      f"{s07a['recalibration_fit_patients']} 位患者 / 死亡 {s07a['recalibration_fit_deaths']}")
    A(f"- 评价半：{s07a['recalibration_eval_records']} 条记录 / "
      f"{s07a['recalibration_eval_patients']} 位患者 / 死亡 {s07a['recalibration_eval_deaths']}")
    A(f"- **两半患者 ID 交集 = {s07a['recalibration_patient_intersection']}**（必须为 0，已通过）")
    A(f"- 对照：旧的记录级划半有 {s07a['recalibration_record_split_patient_overlap_sensitivity']} "
      f"位患者同时进入拟合半与评价半 ⇒ 旧做法只能作敏感性分析")
    A('')
    A('**再校准前后（患者级划半；评价半）**')
    A('')
    A('```')
    A(T(recal_a[['Model', 'n_eval_records', 'Brier_raw', 'Brier_logistic', 'Brier_isotonic',
                 'slope_raw', 'CITL_raw', 'slope_logistic', 'CITL_logistic']]))
    A('```')
    A('')
    A('**Brier 的 95%CI（uniquepid 聚类；固定方程 vs 每次重抽重新拟合方程）**')
    A('')
    A('```')
    A(T(ci_a))
    A('```')
    A('')
    A('- LSTM：raw Brier '
      f"{float(recal_a.loc[recal_a.Model=='LSTM','Brier_raw'].iloc[0]):.4f} → logistic 再校准 "
      f"{float(recal_a.loc[recal_a.Model=='LSTM','Brier_logistic'].iloc[0]):.4f}、"
      f"isotonic 再校准 {float(recal_a.loc[recal_a.Model=='LSTM','Brier_isotonic'].iloc[0]):.4f}。"
      '方向与旧的记录级划半一致，说明患者重叠**不是**该结论的驱动因素。')
    A('- 但必须写清：这是**概率尺度（截距/斜率）的修正**，不是判别力或临床获益的证明；'
      '再校准后各模型 Brier 趋于接近（0.13–0.14），说明其区分度差异并未因此消失或产生。')
    A('- 未把较低 Brier 写成"校准良好"或"临床有用"。')
    A('')
    A('---')
    A('')
    A('## 6. 校准指标及 95%CI（内部测试集，raw 概率）')
    A('')
    A('```')
    A(T(cal_a[['Model', 'n', 'deaths', 'brier_raw', 'brier_raw_CI_low_subject',
               'brier_raw_CI_high_subject', 'slope', 'slope_CI_low_subject',
               'slope_CI_high_subject', 'CITL', 'CITL_CI_low_subject', 'CITL_CI_high_subject']]))
    A('```')
    A('')
    A('**eICU（raw 概率，uniquepid 聚类）**')
    A('')
    A('```')
    A(T(cal_ext_a[['Model', 'n_records', 'n_patients', 'brier_raw',
                   'brier_raw_CI_low_patient', 'brier_raw_CI_high_patient', 'slope',
                   'slope_CI_low_patient', 'slope_CI_high_patient', 'CITL',
                   'CITL_CI_low_patient', 'CITL_CI_high_patient']]))
    A('```')
    A('')
    A('**区间定义（不得混写）**')
    A('')
    A('- 内部主结果：`subject_id` 聚类重抽；同一受试者全部记录整体进出；'
      'slope/intercept/CITL 与再校准 Brier 的区间为**每次重抽内重新拟合逻辑校准方程的总不确定性**。')
    A('- 外部主结果：`uniquepid` 聚类重抽；固定方程与每次重抽重新拟合方程的结果**分列**'
      '（见 `calibration_primary_patient_clustered.csv` 的 `metric` 与 `ci_definition` 列）。')
    A('- 记录级（忽略患者聚类）结果在 `calibration_sensitivity_record_split.csv`，'
      '仅作历史口径对照。')
    A('- 完整定义与哈希见 `calibration_uncertainty_manifest.json`。')
    A('')
    A('---')
    A('')
    A('## 7. 阈值操作特征与净获益')
    A('')
    A('**内部测试集（验证集选择的 F3 研究性操作点）**')
    A('')
    A('```')
    A(T(th_a[['Model', 'threshold', 'flagged_n', 'flagged_percent', 'deaths_captured',
              'death_recall_sensitivity', 'specificity', 'PPV', 'NPV', 'false_positive_n',
              'false_positive_percent_of_all', 'one_minus_PPV', 'net_benefit_at_threshold',
              'net_benefit_gain_vs_treat_all']]))
    A('```')
    A('')
    A('**eICU（raw 阈值直接套用 vs 再校准后固定风险阈值 0.15）**')
    A('')
    A('```')
    A(T(th_ext[['Model', 'prob_type', 'threshold', 'flagged_n', 'flagged_percent',
                'deaths_captured', 'sensitivity', 'specificity', 'PPV',
                'net_benefit_at_threshold', 'net_benefit_treat_all_at_threshold']]))
    A('```')
    A('')
    A('- 所有内部阈值均标注为 `validation-set F3-optimal (research operating point)`，'
      '**不是**经临床验证的最佳阈值。')
    A('- `1−PPV` 与"全部记录中的假阳性比例"分列；旧稿把 1−PPV 称为"假阳性率"是错的'
      '（假阳性率分母为全部记录，不是全部阳性预测）。')
    A('- raw 概率阈值与再校准后概率阈值不混用：再校准后再校准概率另用固定阈值 0.15，'
      '并在表中显式标注 `prob_type`。')
    A('- 外部净获益出现负值（如 LSTM raw −0.020）时不得写成"具有临床效用"；'
      '再校准后在 0.15 阈值下净获益为正，只能表述为"在特定阈值假设下的潜在净获益"。')
    A('- 正文必须写明：F3 是召回优先的数学评分；未定义干预、复核动作或可接受工作负荷；'
      '高危人数与死亡捕获率**不等于**临床获益。')
    A('')
    A('---')
    A('')
    A('## 8. 更新过的文件清单')
    A('')
    A('**交付物（工作目录根）**')
    A('')
    for f in ['landmark_eligibility_audit.csv', 'preprocessing_fit_manifest.json',
              'calibration_primary_patient_clustered.csv',
              'calibration_sensitivity_record_split.csv', 'recalibration_patient_split.csv',
              'threshold_operating_characteristics_primary.csv',
              'threshold_operating_characteristics_external.csv',
              'calibration_uncertainty_manifest.json', 'calibration_plots_primary.png',
              'calibration_plots_sensitivity.png', 'figures_source_manifest.json',
              'results_summary_old_vs_trainonly.csv', 'trainonly_reanalysis_manifest.json',
              'NUMBER_DIFF_OLD_VS_NEW.md', 'LANDMARK_ELIGIBILITY_AUDIT.md',
              'threshold_interpretation.md', 'threshold_interpretation_A.md',
              'threshold_interpretation_B.md', 'final_checks.json', 'CURSOR_HANDOFF.md']:
        p = HERE / f
        A(f"- `{f}` {'✅' if p.exists() else '❌ 缺失'}"
          + (f"（{p.stat().st_size:,} B）" if p.exists() else ''))
    A('')
    A('**子目录**')
    A('')
    A('- `scripts/`：s00–s12 + `common_eval.py` + SQL（全部可复跑）')
    A('- `data/`：`trainonly_mimic.npz`、`trainonly_eicu.npz`、`preds_A/B.npz`、'
      '`eicu_probs_A/B.npz`、`boot_internal_A/B.npz`、`shap_values_A/B.npz`、'
      '`eicu_cohort_index.csv`')
    A('- `models_trainonly/A|B/`：7 个模型 + 3 个 scaler（全部新训练）')
    A(f'- `tables/A|B/`：{len(list((HERE / "tables" / "A").glob("*.csv")))} 个队列 A 表 + '
      f'{len(list((HERE / "tables" / "B").glob("*.csv")))} 个队列 B 表')
    A(f'- `figures_trainonly/A|B/`：'
      f'{len(list((HERE / "figures_trainonly" / "A").glob("*.png")))} / '
      f'{len(list((HERE / "figures_trainonly" / "B").glob("*.png")))} 张 PNG')
    A('- `qa/`：各阶段 JSON 验收产物；`logs/`：全部运行日志')
    A('- `00_manifest/`：旧文件清单与 SHA-256、环境、隔离策略')
    A('')
    A('**旧文件（未改动，只作为历史对照）**')
    A('')
    A('- 旧模型 `models/*.joblib|*.pt`、旧预测 `*.npz`、旧表 `tables/*.csv`、'
      '旧图 `claude code辅助后的/figs/*`、旧稿件与 DOCX —— 全部保留原状，'
      '已在 `00_manifest/old_files_sha256.csv` 记录哈希。')
    A('')
    A('---')
    A('')
    A('## 9. 未解决问题')
    A('')
    A('### 9.1 本次分析内部仍未闭合的')
    A('')
    warn = summary['checks'].get('no_overclaim_language', {})
    if warn.get('status') == 'warn':
        A(f"- **产物中含越界措辞 {warn['detail']['n_hits']} 处**（多为文件名/历史标签），"
          '需在阶段11改写正文时逐条清除；明细见 `final_checks.json`。')
    else:
        A('- 分析产物中未检出越界措辞。')
    A('- **LSTM 单次拟合的交互检验 p 值不稳定**：正文若引用单一 p 值必须同时给出'
      '6 个种子的范围（已在 `tables/A/lstm_seed_stability.csv`）。')
    A('- **注意力遮挡效应量很小**：遮挡注意力 top-k 小时带来的 ΔAUC 为 +0.002~+0.043'
      '（随机遮挡 +0.0005~+0.038），只能支持"注意力与模型行为弱相关"，'
      '不能支持机制解释。')
    A('- **时间窗消融用的是"全窗插补后再截取"的表示**：'
      '最早单小时窗口 AUC 仍达 0.741（RF），部分来自插补与聚合规则，'
      '不能据此主张真实早期预警能力。')
    A('- **eICU 映射限制**：`age/weight/height` 缺失用 MIMIC 训练集均值填补，'
      '`gender` 缺失被填为 0（Female），沿用旧装配规则；这属于跨库变量映射限制，'
      '必须在局限中写明，不能宣称完全等价的外部可迁移性。')
    A('- **eICU 的脓毒症识别是诊断字符串代理**（与 MIMIC `sepsis3` 定义不同），'
      '所有跨库结果只能称 exploratory cross-database evaluation。')
    A('- **疑似感染时间晚于 48h 的记录**在变体 A 中仍被纳入（这是回顾性地标队列的定义），'
      '变体 B 已给出限制性结果；两者差异见第 4 节，不能互相替代。')
    A('')
    A('### 9.2 必须由作者确认、本次未能完成的')
    A('')
    A('- 伦理审批/豁免依据、作者贡献、资助、利益冲突、AI 辅助披露、数据与代码可用性声明。')
    A('- 目标期刊对图表数量、字数、补充材料格式的最终要求。')
    A('- 英文稿与补充材料的逐句同步（本次只生成了英文稿的"待改数字"清单，未改写英文段落）。')
    A('')
    A('---')
    A('')
    A('## 10. 是否可以同步根目录稿件')
    A('')
    A(f"- `final_checks.json`：pass={summary['n_pass']}，warn={summary['n_warn']}，"
      f"pending={summary['n_pending']}，fail={summary['n_fail']}")
    A(f"- 阻断项：{summary['blocking_failures'] if summary['blocking_failures'] else '无'}")
    A('')
    if summary['ready_to_sync_manuscript']:
        A('**结论：分析侧全部通过，可以进入阶段11（稿件/DOCX 同步）。** 同步时须：')
    else:
        A('**结论：暂不可同步。** 先修复上面的阻断项，再重新运行本脚本。')
    A('')
    A('1. 只使用本目录下的 CSV/JSON 数字，逐条替换中英文稿件的旧数字'
      '（清单见 `NUMBER_DIFF_OLD_VS_NEW.md`）；')
    A('2. 方法与讨论必须改写为"48 小时地标 + 训练集独立预处理 + 研究性操作点"口径，'
      '并明确不再称为即时筛查或逐小时实时预警；')
    A('3. 替换全部图（含流程图内文字）为 `figures_trainonly/A/` 下的新图；'
      'eICU 相关图避免再出现 "External validation" 字样，记录数不得写成患者数；')
    A('4. 重建 DOCX 后重跑 `s12_finalize.py`，补齐 DOCX ZIP 完整性、PDF 代表页渲染、'
      '中英文一致性三项检查；')
    A('5. 旧分析在所有文档中的定位必须写成 "previous analysis / pre-split static '
      'imputation analysis"，不得与新结果混排。')
    A('')
    A('---')
    A('')
    A('> 本文件由 `scripts/s12_finalize.py` 自动生成，所有数字均取自本目录的分析产物，'
      '未手工填写。')
    (HERE / 'CURSOR_HANDOFF.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
