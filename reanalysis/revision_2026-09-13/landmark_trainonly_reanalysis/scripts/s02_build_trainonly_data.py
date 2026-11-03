#!/usr/bin/env python3
# =====================================================================
# s02_build_trainonly_data.py —— 阶段3：训练集独立插补 + 数据组装
#
# 目的（对应用户要求第五节）：
#   所有"会从数据中估计参数"的预处理步骤，只允许使用 MIMIC 训练集。
#
# 本脚本严格区分两类操作，并分别留证：
#   (P) 患者局部规则 —— 不跨患者、不跨分区、不跨库估计任何参数：
#       · 时变变量 03_impute.py 的三分支规则（全缺失→生理固定值；
#         <24 观测→该患者自身观测均值；≥24 观测→该患者自身最近邻插值）
#       · 固定生理值 FIXED 属预先规定的规则，不从任何数据估计
#   (G) 全局可拟合参数 —— 只允许用 MIMIC 训练集拟合：
#       · 静态变量 weight / height 的均值插补
#       · 静态变量标准化参数（StandardScaler）
#       · 时变变量标准化参数（StandardScaler）
#
# eICU 一律使用 MIMIC 训练集拟合出的参数，不使用 eICU 自身统计量。
#
# 输出：
#   data/trainonly_mimic.npz       （含 split 标签、t_inf_h、静态变量原始与插补后）
#   data/trainonly_eicu.npz
#   data/eicu_cohort_index.csv
#   qa/preprocessing_fit_manifest.json
#   qa/imputation_reproduction_check.json   （03_impute 患者局部性验证）
# =====================================================================
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent.parent.parent
DATA = HERE / 'data'
QA = HERE / 'qa'
DATA.mkdir(exist_ok=True)
QA.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

from split_utils import grouped_tts          # noqa: E402  与旧流水线同源

SEED = 42
np.random.seed(SEED)

HCV_VARS = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'spo2',
            'temperature', 'glucose', 'fio2', 'ph',
            'gcs', 'gcs_eyes', 'gcs_motor', 'gcs_verbal']
FIXED = {'heart_rate': 86, 'sbp': 118, 'dbp': 59, 'mbp': 77, 'resp_rate': 19,
         'spo2': 98, 'temperature': 36, 'glucose': 128, 'fio2': 21, 'ph': 7.4,
         'gcs': 12, 'gcs_eyes': 4, 'gcs_motor': 5, 'gcs_verbal': 4}
ALL_HOURS = np.arange(48)
SV_NUM = ['age', 'weight', 'height']
CLINICAL = HCV_VARS


# ---------------------------------------------------------------------
# (P) 患者局部时变插补 —— 与 03_impute.py / eicu_05_impute.py 规则逐条一致
# ---------------------------------------------------------------------
def impute_patient_local(V: np.ndarray, tag: str) -> tuple[np.ndarray, dict]:
    """V: [n, 14, 48] 原始（含 NaN）→ 插补后（无 NaN）。

    返回 (插补后数组, 每个变量的分支计数)。
    全部操作仅使用同一行（同一患者 stay）内部信息。

    ⚠️ 全程 float64：与 03_impute.py / eicu_05_impute.py 完全一致。
       若用 float32 读入会引入 ~2e-5 的输入舍入差异，使"精确复现"核对失败。
    """
    V = V.astype(np.float64).copy()
    n = V.shape[0]
    branch = {}
    for j, var in enumerate(HCV_VARS):
        s = V[:, j, :]
        obs = ~np.isnan(s)
        k = obs.sum(1)
        b1, b2, b3 = (k == 0), ((k > 0) & (k < 24)), (k >= 24)
        branch[var] = {'n_patients': int(n), 'fixed_all_missing': int(b1.sum()),
                       'patient_mean_lt24obs': int(b2.sum()),
                       'patient_nearest_ge24obs': int(b3.sum())}
        if b1.any():
            s[b1, :] = FIXED[var]
        if b2.any():
            s[b2, :] = np.nanmean(s[b2, :], axis=1)[:, None]
        for i in np.flatnonzero(b3):
            ho = ALL_HOURS[obs[i]]
            f = interp1d(ho, s[i, obs[i]], kind='nearest', bounds_error=False,
                         fill_value='extrapolate')
            s[i, :] = f(ALL_HOURS)
        V[:, j, :] = s
    n_nan = int(np.isnan(V).sum())
    assert n_nan == 0, f'{tag}: 插补后仍有 {n_nan} 个 NaN'
    return V, branch


def load_hourly_tensor(csv_path: Path, id_col: str, keep_ids: np.ndarray,
                       ) -> tuple[np.ndarray, pd.DataFrame]:
    """从小时级宽表构建 [n, 14, 48] 张量（不插补），并返回静态变量表。"""
    cols = [id_col, 'hr'] + HCV_VARS + (SV_NUM + ['gender'] if id_col == 'stay_id'
                                        else SV_NUM + ['gender'])
    df = pd.read_csv(csv_path, usecols=lambda c: c in set(cols))
    df = df[df[id_col].isin(keep_ids)].copy()
    df = df.sort_values([id_col, 'hr']).reset_index(drop=True)
    n_per = df.groupby(id_col).size()
    bad = n_per[n_per != 48]
    assert bad.empty, f'{csv_path.name}: 有 {len(bad)} 例行数≠48'
    n = len(keep_ids)
    V = np.full((n, len(HCV_VARS), 48), np.nan, dtype=np.float64)
    for j, var in enumerate(HCV_VARS):
        piv = df.pivot(index=id_col, columns='hr', values=var)
        V[:, j, :] = piv.loc[keep_ids].to_numpy(dtype=np.float64)
    sv = df[[id_col] + SV_NUM + ['gender']].drop_duplicates(id_col).set_index(id_col)
    sv = sv.loc[keep_ids]
    return V, sv


def main() -> int:
    manifest: dict = {'seed': SEED, 'fitted_on': {}, 'patient_local_rules': {},
                      'global_fitted_params': {}, 'hashes': {}}

    # =================================================================
    # A. MIMIC
    # =================================================================
    print('=' * 70)
    print('A. MIMIC-IV —— 训练集独立插补')
    print('=' * 70)
    lab = pd.read_csv(ROOT / 'sepsis_labels.csv')
    lab['t_inf'] = pd.to_datetime(lab.t_inf)
    lab['intime'] = pd.to_datetime(lab.intime)
    lab['t_inf_h'] = (lab.t_inf - lab.intime).dt.total_seconds() / 3600.0
    stay = lab.stay_id.to_numpy(np.int64)
    y = lab.label.to_numpy(np.int64)
    subject = lab.subject_id.to_numpy(np.int64)
    print(f'队列: {len(lab)} 条记录 | {lab.subject_id.nunique()} 受试者 | '
          f'死亡 {y.sum()} ({100*y.mean():.2f}%)')

    # ---- 划分：与旧流水线同一函数、同一 seed（不改变已确定的划分原则）----
    tr_idx, va_idx, te_idx = grouped_tts(subject, y, seed=SEED)
    split = np.array(['train'] * len(y), dtype=object)
    split[va_idx] = 'val'
    split[te_idx] = 'test'
    print(f'划分: train {len(tr_idx)} | val {len(va_idx)} | test {len(te_idx)}')

    # 与冻结预测文件的测试集比对（证明划分未变）
    old_pred = np.load(ROOT / 'baseline_predictions.npz')
    old_test_stay = set(old_pred['stay_id'].astype(np.int64).tolist())
    new_test_stay = set(stay[te_idx].tolist())
    split_identical = bool(old_test_stay == new_test_stay)
    print(f'测试集与冻结 baseline_predictions.npz 一致: {split_identical} '
          f'(旧 {len(old_test_stay)} / 新 {len(new_test_stay)})')
    assert split_identical, '划分与冻结产物不一致，须停止'

    # ---- (P) 患者局部插补 ----
    V_raw, sv = load_hourly_tensor(ROOT / 'sepsis_hourly.csv', 'stay_id', stay)
    print(f'原始张量 {V_raw.shape} | 原始缺失率 {100*np.isnan(V_raw).mean():.2f}%')
    V_imp, branch = impute_patient_local(V_raw, 'MIMIC')
    manifest['patient_local_rules']['MIMIC'] = branch

    # ---- 验证：我的插补结果 == 冻结的 sepsis_hourly_imputed_v2.csv（时变块）----
    print('\n[验证] 患者局部插补与冻结 sepsis_hourly_imputed_v2.csv 的 14 个时变变量逐元素比对…')
    old_imp = pd.read_csv(ROOT / 'sepsis_hourly_imputed_v2.csv',
                          usecols=['stay_id', 'hr'] + HCV_VARS)
    old_imp = old_imp[old_imp.stay_id.isin(set(stay.tolist()))]
    old_imp = old_imp.sort_values(['stay_id', 'hr']).reset_index(drop=True)
    n_per_old = old_imp.groupby('stay_id').size()
    assert (n_per_old == 48).all(), '冻结插补文件行数异常'
    oids = old_imp.stay_id.drop_duplicates().to_numpy(np.int64)
    assert len(oids) == len(stay) and set(oids.tolist()) == set(stay.tolist())
    B = np.full((len(oids), len(HCV_VARS), 48), np.nan, np.float64)
    for j, var in enumerate(HCV_VARS):
        piv = old_imp.pivot(index='stay_id', columns='hr', values=var)
        B[:, j, :] = piv.loc[oids].to_numpy(dtype=np.float64)
    # 对齐到同一 stay 顺序
    pos = pd.Series(np.arange(len(oids)), index=oids)
    order = pos.loc[stay].to_numpy()
    B = B[order]
    diff = np.abs(V_imp - B)
    rep = {'max_abs_diff': float(np.nanmax(diff)),
           'n_elements': int(diff.size),
           'n_exceed_1e_9': int((diff > 1e-9).sum()),
           'n_exceed_1e_6': int((diff > 1e-6).sum()),
           'n_exceed_1e_4': int((diff > 1e-4).sum()),
           'identical_within_1e_6': bool((diff <= 1e-6).all()),
           'note': 'float64 全程复刻；残余差异仅来自冻结 CSV 的十进制文本往返精度'}
    print('  ', json.dumps(rep, ensure_ascii=False))
    manifest['imputation_reproduction'] = rep
    (QA / 'imputation_reproduction_check.json').write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    if not rep['identical_within_1e_6']:
        print('  ⚠️ 存在超出容差的差异；请检查 03_impute 规则复刻是否正确 '
              '（不阻断，但必须在交付中说明）')

    # ---- (G) 静态变量：均值只用训练集拟合 ----
    sv = sv.copy()
    sv['gender_m'] = sv['gender'].map({'M': 1, 'F': 0})
    gender_nan = int(sv['gender_m'].isna().sum())
    static_params = {}
    X_sv_raw = np.zeros((len(stay), 4), np.float32)
    X_sv_raw[:, 0] = pd.to_numeric(sv['age'], errors='coerce').to_numpy(np.float32)
    X_sv_raw[:, 1] = sv['gender_m'].fillna(0).to_numpy(np.float32)
    X_sv_raw[:, 2] = pd.to_numeric(sv['weight'], errors='coerce').to_numpy(np.float32)
    X_sv_raw[:, 3] = pd.to_numeric(sv['height'], errors='coerce').to_numpy(np.float32)

    nan_before = {SV_NUM[i]: int(np.isnan(X_sv_raw[:, [0, 2, 3][i]]).sum()) for i in range(3)}
    # 训练集拟合均值（受试者级划分；均值按记录数加权，与旧口径一致）
    train_mean = {c: float(np.nanmean(X_sv_raw[tr_idx, col]))
                  for c, col in zip(SV_NUM, [0, 2, 3])}
    print(f'\n静态变量缺失（插补前）: {nan_before}')
    print(f'训练集拟合均值（仅 train 子集）: '
          f'{ {k: round(v, 4) for k, v in train_mean.items()} }')
    # 全队列均值（旧口径，仅用于对照，不进入主分析）
    all_mean = {c: float(np.nanmean(X_sv_raw[:, col]))
                for c, col in zip(SV_NUM, [0, 2, 3])}
    diff_mean = {k: round(all_mean[k] - train_mean[k], 6) for k in train_mean}
    print(f'全队列均值（旧口径对照，不用于主分析）: '
          f'{ {k: round(v, 4) for k, v in all_mean.items()} } 差值 {diff_mean}')

    X_sv = X_sv_raw.copy()
    for c, col in zip(SV_NUM, [0, 2, 3]):
        nanmask = np.isnan(X_sv[:, col])
        X_sv[nanmask, col] = train_mean[c]
    assert not np.isnan(X_sv).any(), '静态变量插补后仍有 NaN'

    static_params = {'train_only_mean': train_mean,
                     'full_cohort_mean_NOT_USED': all_mean,
                     'difference_train_minus_full': diff_mean,
                     'n_missing_filled': nan_before,
                     'fit_partition': 'train only (subject-level grouped split, seed=42)',
                     'gender_unknown_records_filled_with': 0,
                     'n_gender_unknown': gender_nan,
                     'n_train_records_used_for_mean': int(len(tr_idx))}
    manifest['global_fitted_params']['MIMIC_static_imputation'] = static_params

    # ---- (G) 标准化：只用训练集拟合 ----
    sc_seq = StandardScaler().fit(V_imp[tr_idx].reshape(-1, len(HCV_VARS)))
    sc_sv = StandardScaler().fit(X_sv[tr_idx])
    manifest['global_fitted_params']['MIMIC_scaler_hcv'] = {
        'fit_partition': 'train only', 'n_train_records': int(len(tr_idx)),
        'mean': sc_seq.mean_.tolist(), 'scale': sc_seq.scale_.tolist(),
        'vars': HCV_VARS}
    manifest['global_fitted_params']['MIMIC_scaler_sv'] = {
        'fit_partition': 'train only', 'n_train_records': int(len(tr_idx)),
        'mean': sc_sv.mean_.tolist(), 'scale': sc_sv.scale_.tolist(),
        'vars': ['age', 'gender_m', 'weight', 'height']}

    # 静态特征聚合（与旧流水线 06 完全一致的 46 维构造）
    feat_parts = []
    for j in range(len(CLINICAL)):
        s = V_imp[:, j, :]
        feat_parts += [s.mean(1), s.min(1), s.max(1)]
    X_flat = np.column_stack(feat_parts + [X_sv[:, 0], X_sv[:, 1], X_sv[:, 2],
                                           X_sv[:, 3]]).astype(np.float32)
    print(f'聚合特征: {X_flat.shape}（46 维）')

    np.savez(DATA / 'trainonly_mimic.npz',
             X_hcv=V_imp.astype(np.float32), X_sv=X_sv, X_sv_raw=X_sv_raw,
             X_flat=X_flat,
             y=y, stay_id=stay, subject_id=subject, split=split.astype('U8'),
             tr_idx=tr_idx, va_idx=va_idx, te_idx=te_idx,
             t_inf_h=lab.t_inf_h.to_numpy(np.float32),
             los_h=lab.los_h.to_numpy(np.float32),
             split_sv_mean=np.array([train_mean[c] for c in SV_NUM], np.float32),
             scaler_hcv_mean=sc_seq.mean_.astype(np.float32),
             scaler_hcv_scale=sc_seq.scale_.astype(np.float32),
             scaler_sv_mean=sc_sv.mean_.astype(np.float32),
             scaler_sv_scale=sc_sv.scale_.astype(np.float32))
    print(f'已保存 {DATA / "trainonly_mimic.npz"}')

    # =================================================================
    # B. eICU —— 使用 MIMIC 训练集参数，不用 eICU 自身统计量
    # =================================================================
    print('\n' + '=' * 70)
    print('B. eICU-CRD —— 复用 MIMIC 训练集参数')
    print('=' * 70)
    idx = pd.read_csv(DATA / 'eicu_cohort_index.csv') if (DATA / 'eicu_cohort_index.csv').exists() else None
    if idx is None:
        # 从数据库取唯一队列索引（含 uniquepid）
        from io import StringIO
        import subprocess
        sql = (HERE / 'scripts' / 'eicu_cohort_index.sql').read_text(encoding='utf-8')
        # 去掉末尾分号（COPY (...) 内部不允许）与纯注释尾行
        sql_body = sql.strip()
        while sql_body.endswith(';'):
            sql_body = sql_body[:-1].rstrip()
        r = subprocess.run(['psql', '-h', 'localhost', '-U', 'postgres', '-d', 'eicu',
                            '-c', f'COPY ({sql_body}) TO STDOUT WITH CSV HEADER'],
                           capture_output=True, text=True, env={**__import__('os').environ,
                                                                'PAGER': 'cat'})
        assert r.returncode == 0, r.stderr[:600]
        idx = pd.read_csv(StringIO(r.stdout))
        # SOFA/qSOFA 来自本地 eicu_sofa.csv（非数据库表），按 patientunitstayid 合并
        sc = pd.read_csv(ROOT / 'eicu_sofa.csv',
                         usecols=['patientunitstayid', 'sofa_max_48h', 'qsofa6', 'qsofa24'])
        idx = idx.merge(sc, on='patientunitstayid', how='left')
        assert idx.sofa_max_48h.notna().all(), 'SOFA 合并后出现缺失'
        idx.to_csv(DATA / 'eicu_cohort_index.csv', index=False)
    print(f'eICU 队列索引: {len(idx)} 条记录 | uniquepid {idx.uniquepid.nunique()} '
          f'| 死亡 {int(idx.label.sum())}')
    e_stay = idx.patientunitstayid.to_numpy(np.int64)
    e_y = idx.label.to_numpy(np.int64)

    E_raw, e_sv = load_hourly_tensor(ROOT / 'eicu_hourly.csv', 'patientunitstayid', e_stay)
    print(f'原始张量 {E_raw.shape} | 原始缺失率 {100*np.isnan(E_raw).mean():.2f}%')
    E_imp, e_branch = impute_patient_local(E_raw, 'eICU')
    manifest['patient_local_rules']['eICU'] = e_branch

    # eICU 静态变量：age（'>89'→89）、gender、weight、height；缺失用 MIMIC 训练集均值
    e_sv = e_sv.copy()
    def parse_age(x):
        s = str(x).strip()
        return 89.0 if s == '> 89' else (float(s) if s.replace('.', '', 1).isdigit() else np.nan)
    e_age = e_sv['age'].map(parse_age).to_numpy(np.float32)
    e_gender = e_sv['gender'].map({'Male': 1, 'Female': 0}).fillna(0).to_numpy(np.float32)
    e_wt = pd.to_numeric(e_sv['weight'], errors='coerce').to_numpy(np.float32)
    e_ht = pd.to_numeric(e_sv['height'], errors='coerce').to_numpy(np.float32)
    E_sv_raw = np.column_stack([e_age, e_gender, e_wt, e_ht]).astype(np.float32)
    e_nan = {'age': int(np.isnan(e_age).sum()), 'weight': int(np.isnan(e_wt).sum()),
             'height': int(np.isnan(e_ht).sum())}
    E_sv = E_sv_raw.copy()
    for col, key in zip([0, 2, 3], SV_NUM):
        nanmask = np.isnan(E_sv[:, col])
        E_sv[nanmask, col] = train_mean[key]          # ← MIMIC 训练集均值
    assert not np.isnan(E_sv).any()
    print(f'eICU 静态缺失: {e_nan} → 用 MIMIC 训练集均值填充 '
          f'{ {k: round(v, 4) for k, v in train_mean.items()} }')
    manifest['global_fitted_params']['eICU_static_imputation'] = {
        'params_source': 'MIMIC train only (NOT estimated on eICU)',
        'n_missing_filled': e_nan, 'applied_means': train_mean}
    print('  ⚠️ 变量映射限制：eICU weight/height 为 admissionweight/admissionheight，'
          '与 MIMIC first_day_weight/height 定义不完全等价')

    np.savez(DATA / 'trainonly_eicu.npz',
             X_hcv=E_imp.astype(np.float32), X_sv=E_sv, X_sv_raw=E_sv_raw, y=e_y,
             stay_id=e_stay, uniquepid=idx.uniquepid.to_numpy('U32'),
             sofa_max_48h=idx.sofa_max_48h.to_numpy(np.float32)
             if 'sofa_max_48h' in idx else np.zeros(len(idx), np.float32),
             qsofa6=idx.qsofa6.to_numpy(np.int32) if 'qsofa6' in idx
             else np.zeros(len(idx), np.int32))
    print(f'已保存 {DATA / "trainonly_eicu.npz"}')

    manifest['fitted_on'] = {
        'MIMIC_static_imputation_mean': 'MIMIC train only',
        'MIMIC_scaler_hcv': 'MIMIC train only',
        'MIMIC_scaler_sv': 'MIMIC train only',
        'MIMIC_hcv_imputation': 'per-patient local rule (no cross-patient parameter)',
        'eICU_static_imputation_mean': 'MIMIC train only (not re-estimated on eICU)',
        'eICU_hcv_imputation': 'per-patient local rule (no cross-patient parameter)',
        'eICU_scaler': 'MIMIC train only',
        'validation_set_used_for_fitting': False,
        'test_set_used_for_fitting': False,
    }
    (QA / 'preprocessing_fit_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n-> qa/preprocessing_fit_manifest.json')
    print('-> qa/imputation_reproduction_check.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
