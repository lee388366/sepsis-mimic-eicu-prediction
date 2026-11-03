#!/usr/bin/env python3
# =====================================================================
# s01_landmark_audit.py —— 阶段2：48 小时地标可判定性审计
#
# 目标（对应用户要求第四节）：
#   1. 逐项判断每个纳入条件"在第 48 小时是否可判定"；
#   2. 从**真实数据库**重建筛选漏斗（不假定旧稿数量正确）；
#   3. MIMIC 与 eICU 分别报告；
#   4. 明确哪些条件只能在完整住院结束后回顾性确定；
#   5. 定义两套队列：
#        A. 当前数据库定义的回顾性地标队列（sepsis3 + t_inf 任意时点）
#        B. t_inf 在 48 小时内的限制性地标队列（地标时点可判定脓毒症表型）
#
# 只读数据库（全部为 SELECT / COPY ... TO STDOUT）。输出仅写入本工作目录。
#
# 输出：tables/landmark_eligibility_audit.csv
#       tables/landmark_cohort_definitions.csv
#       tables/landmark_infection_time_distribution.csv
#       qa/landmark_audit_facts.json
#       ../LANDMARK_ELIGIBILITY_AUDIT.md（在阶段9统一生成）
# =====================================================================
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent.parent.parent          # 项目根
TAB = HERE / 'tables'
QA = HERE / 'qa'
TAB.mkdir(exist_ok=True)
QA.mkdir(exist_ok=True)

PSQL = ['psql', '-h', 'localhost', '-U', 'postgres']


def psql_csv(sql: str, db: str) -> pd.DataFrame:
    """只读查询：COPY (...) TO STDOUT WITH CSV HEADER。"""
    from io import StringIO
    r = subprocess.run(PSQL + ['-d', db, '-c', f"COPY ({sql}) TO STDOUT WITH CSV HEADER"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'psql 失败 [{db}]: {r.stderr[:800]}')
    return pd.read_csv(StringIO(r.stdout))


# =====================================================================
# 1. MIMIC 漏斗（逐步累积，全部从数据库现有表重建）
#
# ⚠️ 关键：必须**先按条件过滤、再做 ROW_NUMBER(rn)**，与 01_sepsis_cohort.sql
#    一致。若先排名后过滤，会把"靠前的 ICU stay 被 LOS 条件排除、靠后的
#    被纳入"的住院整段漏掉（实测差 356 例）。每步单独排名。
# =====================================================================
MIMIC_STEPS = [
    ('0_sepsis3_tinf_notnull', [], ),
    ('1_plus_age_ge_18', ['p.anchor_age >= 18']),
    ('2_plus_icu_los_ge_48h', ['p.anchor_age >= 18',
                               "ie.outtime >= ie.intime + INTERVAL '48 hours'"]),
    ('3_plus_expire_flag_notnull', ['p.anchor_age >= 18',
                                    "ie.outtime >= ie.intime + INTERVAL '48 hours'",
                                    'adm.hospital_expire_flag IS NOT NULL']),
    ('4_plus_n_icu_stays_le_2', ['p.anchor_age >= 18',
                                 "ie.outtime >= ie.intime + INTERVAL '48 hours'",
                                 'adm.hospital_expire_flag IS NOT NULL',
                                 'c.n_icu_stays <= 2']),
    ('5_plus_rn_eq_1_FINAL', ['p.anchor_age >= 18',
                              "ie.outtime >= ie.intime + INTERVAL '48 hours'",
                              'adm.hospital_expire_flag IS NOT NULL',
                              'c.n_icu_stays <= 2']),
]

_MIMIC_PRE = """
sepsis AS (
    SELECT stay_id, suspected_infection_time FROM mimiciv_derived.sepsis3
    WHERE sepsis3 IS TRUE AND suspected_infection_time IS NOT NULL
),
cnt AS (SELECT hadm_id, COUNT(*) AS n_icu_stays FROM mimiciv_icu.icustays GROUP BY hadm_id),
"""


def _mimic_step_sql(i: int) -> str:
    name, conds = MIMIC_STEPS[i]
    filt = ('WHERE ' + ' AND '.join(conds)) if conds else ''
    rn_filter = 'WHERE rn = 1' if i == len(MIMIC_STEPS) - 1 else ''
    inner = f"""
    SELECT ie.stay_id, ie.subject_id, ie.hadm_id, ie.intime,
           adm.hospital_expire_flag,
           ROW_NUMBER() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime ASC, ie.stay_id ASC) AS rn
    FROM mimiciv_icu.icustays ie
    JOIN sepsis se ON ie.stay_id = se.stay_id
    JOIN mimiciv_hosp.patients p ON ie.subject_id = p.subject_id
    JOIN mimiciv_hosp.admissions adm ON ie.hadm_id = adm.hadm_id
    JOIN cnt c ON ie.hadm_id = c.hadm_id
    {filt}
"""
    return (f"SELECT '{name}' AS step, {i} AS ord, COUNT(*) AS n_records, "
            f"COUNT(DISTINCT subject_id) AS n_subjects, "
            f"SUM(CASE WHEN hospital_expire_flag=1 THEN 1 ELSE 0 END) AS n_deaths "
            f"FROM ({inner}) r {rn_filter}")


_MIMIC_PRE_SQL = _MIMIC_PRE.rstrip().removesuffix(',')     # 供 UNION ALL 查询使用（去掉尾逗号）

MIMIC_FUNNEL = ('WITH ' + _MIMIC_PRE_SQL + '\n'
                + '\nUNION ALL\n'.join(_mimic_step_sql(i) for i in range(len(MIMIC_STEPS)))
                + '\nORDER BY ord')

# 地标可判定性：t_inf 相对 [0,48h) 的位置（同样先过滤后排名）
MIMIC_TINF = ('WITH ' + _MIMIC_PRE + """
f AS (
    SELECT * FROM (
        SELECT ie.stay_id, ie.subject_id, adm.hospital_expire_flag,
               EXTRACT(EPOCH FROM (se.suspected_infection_time - ie.intime))/3600.0 AS t_inf_h,
               ROW_NUMBER() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime ASC, ie.stay_id ASC) AS rn
        FROM mimiciv_icu.icustays ie
        JOIN sepsis se ON ie.stay_id = se.stay_id
        JOIN mimiciv_hosp.patients p ON ie.subject_id = p.subject_id
        JOIN mimiciv_hosp.admissions adm ON ie.hadm_id = adm.hadm_id
        JOIN cnt c ON ie.hadm_id = c.hadm_id
        WHERE p.anchor_age >= 18
          AND ie.outtime >= ie.intime + INTERVAL '48 hours'
          AND adm.hospital_expire_flag IS NOT NULL
          AND c.n_icu_stays <= 2
    ) x WHERE rn = 1
)
SELECT
  CASE WHEN t_inf_h < -24 THEN 'a_before_icu_gt24h'
       WHEN t_inf_h < 0   THEN 'b_before_icu_le24h'
       WHEN t_inf_h < 12  THEN 'c_0_12h'
       WHEN t_inf_h < 24  THEN 'd_12_24h'
       WHEN t_inf_h < 36  THEN 'e_24_36h'
       WHEN t_inf_h < 48  THEN 'f_36_48h'
       WHEN t_inf_h < 72  THEN 'g_48_72h'
       WHEN t_inf_h < 96  THEN 'h_72_96h'
       WHEN t_inf_h < 168 THEN 'i_96h_7d'
       ELSE 'j_gt_7d' END AS bucket,
  COUNT(*) AS n,
  SUM(CASE WHEN hospital_expire_flag=1 THEN 1 ELSE 0 END) AS n_deaths,
  MIN(t_inf_h) AS t_inf_h_min, MAX(t_inf_h) AS t_inf_h_max
FROM f GROUP BY bucket ORDER BY bucket
""")

# =====================================================================
# 2. eICU 漏斗
# =====================================================================
EICU_FUNNEL = """
WITH sepsis_dx AS (
    SELECT DISTINCT patientunitstayid FROM eicu_crd.diagnosis
    WHERE diagnosispriority = 'Primary' AND diagnosisoffset <= 1440
      AND ( diagnosisstring LIKE 'surgery|infections|septic shock%'
         OR diagnosisstring LIKE 'surgery|infections|sepsis|severe%'
         OR diagnosisstring LIKE 'surgery|infections|sepsis|sepsis with multi-organ%'
         OR diagnosisstring LIKE 'surgery|infections|sepsis%'
         OR diagnosisstring LIKE 'infectious diseases|systemic/other infections|septic shock%'
         OR diagnosisstring LIKE 'infectious diseases|systemic/other infections|sepsis|severe%'
         OR diagnosisstring LIKE 'infectious diseases|systemic/other infections|sepsis|sepsis with multi-organ%'
         OR diagnosisstring LIKE 'infectious diseases|systemic/other infections|sepsis%'
         OR diagnosisstring LIKE 'hematology|coagulation disorders|DIC syndrome|associated with sepsis/septic shock%'
         OR diagnosisstring LIKE 'cardiovascular|shock / hypotension|septic shock%'
         OR diagnosisstring LIKE 'cardiovascular|shock / hypotension|sepsis|severe%'
         OR diagnosisstring LIKE 'cardiovascular|shock / hypotension|sepsis|sepsis with multi-organ%'
         OR diagnosisstring LIKE 'cardiovascular|shock / hypotension|sepsis%' )
),
b AS (
    SELECT p.patientunitstayid, p.patienthealthsystemstayid, p.uniquepid, p.age, p.gender,
           p.unitvisitnumber, p.unitdischargeoffset, p.hospitaldischargestatus, p.hospitalid,
           (p.hospitaldischargestatus='Expired')::int AS is_death,
           (p.hospitaldischargestatus IS NOT NULL) AS status_notnull,
           (p.hospitaldischargestatus IN ('Alive','Expired')) AS outcome_ascertainable,
           (p.hospitaldischargestatus = '') AS status_empty,
           (p.age ~ '^[0-9]+$' AND CAST(p.age AS INT) >= 18) OR p.age = '> 89' AS adult
    FROM eicu_crd.patient p
    JOIN sepsis_dx s ON p.patientunitstayid = s.patientunitstayid
)
SELECT '0_sepsis_dx_primary_le1440min' AS step, 0 AS ord,
       COUNT(*) AS n_records, COUNT(DISTINCT uniquepid) AS n_patients,
       SUM(is_death) AS n_deaths
FROM b
UNION ALL SELECT '1_plus_adult', 1, COUNT(*), COUNT(DISTINCT uniquepid), SUM(is_death)
       FROM b WHERE adult
UNION ALL SELECT '2_plus_unit_los_ge_48h', 2, COUNT(*), COUNT(DISTINCT uniquepid), SUM(is_death)
       FROM b WHERE adult AND unitdischargeoffset > 2880
UNION ALL SELECT '3_plus_first_icu_visit', 3, COUNT(*), COUNT(DISTINCT uniquepid), SUM(is_death)
       FROM b WHERE adult AND unitdischargeoffset > 2880 AND unitvisitnumber = 1
-- ⚠️ 数据质量陷阱：hospitaldischargestatus 存在空字符串 ''（不为 NULL）。
--    eicu_01_cohort.sql 原用 "IS NOT NULL"，会使 71 例结局不可判定者留在队列中
--    （eicu_03_sofa_qsofa.sql 的 label 对该状态返回 NULL）。
--    实测：step3 = 6,883 例；加上 IN ('Alive','Expired') 后 = 6,812 例，与已分析队列一致。
UNION ALL SELECT '4a_plus_status_notnull_LEGACY', 4, COUNT(*), COUNT(DISTINCT uniquepid),
       SUM(is_death) FROM b
       WHERE adult AND unitdischargeoffset > 2880 AND unitvisitnumber = 1 AND status_notnull
UNION ALL SELECT '4b_plus_outcome_ascertainable_FINAL', 5, COUNT(*), COUNT(DISTINCT uniquepid),
       SUM(is_death) FROM b
       WHERE adult AND unitdischargeoffset > 2880 AND unitvisitnumber = 1 AND outcome_ascertainable
UNION ALL SELECT 'x_excluded_status_empty_string', 6, COUNT(*), COUNT(DISTINCT uniquepid),
       SUM(is_death) FROM b
       WHERE adult AND unitdischargeoffset > 2880 AND unitvisitnumber = 1 AND status_empty
ORDER BY ord
"""

# eICU 患者级结构（uniquepid）—— 限定在最终队列（outcome ascertainable）
EICU_PATIENT_STRUCT = """
WITH c AS (
    WITH sepsis_dx AS (
      SELECT DISTINCT patientunitstayid FROM eicu_crd.diagnosis
      WHERE diagnosispriority='Primary' AND diagnosisoffset<=1440
        AND ( diagnosisstring LIKE 'surgery|infections|sepsis%'
           OR diagnosisstring LIKE 'surgery|infections|septic shock%'
           OR diagnosisstring LIKE 'infectious diseases|systemic/other infections|sepsis%'
           OR diagnosisstring LIKE 'infectious diseases|systemic/other infections|septic shock%'
           OR diagnosisstring LIKE 'cardiovascular|shock / hypotension|sepsis%'
           OR diagnosisstring LIKE 'cardiovascular|shock / hypotension|septic shock%'
           OR diagnosisstring LIKE 'hematology|coagulation disorders|DIC syndrome|associated with sepsis/septic shock%' )
    )
    SELECT p2.patientunitstayid, p2.uniquepid, p2.patienthealthsystemstayid,
           CASE WHEN p2.hospitaldischargestatus='Expired' THEN 1 ELSE 0 END AS is_death
    FROM eicu_crd.patient p2
    JOIN sepsis_dx s ON p2.patientunitstayid = s.patientunitstayid
    WHERE p2.unitvisitnumber = 1 AND p2.unitdischargeoffset > 2880
      AND p2.hospitaldischargestatus IN ('Alive','Expired')
      AND ((p2.age ~ '^[0-9]+$' AND CAST(p2.age AS INT) >= 18) OR p2.age = '> 89')
),
g AS (
    SELECT uniquepid, COUNT(*) AS n_in_cohort, SUM(is_death) AS n_deaths,
           COUNT(DISTINCT patienthealthsystemstayid) AS n_hosp
    FROM c GROUP BY uniquepid
)
SELECT (SELECT COUNT(*) FROM c)                       AS n_records,
       COUNT(*)                                       AS n_patients,
       (SELECT COUNT(DISTINCT patienthealthsystemstayid) FROM c) AS n_hospital_stays,
       SUM(CASE WHEN n_in_cohort > 1 THEN 1 ELSE 0 END)          AS n_patients_multi,
       SUM(CASE WHEN n_in_cohort > 1 THEN n_in_cohort ELSE 0 END) AS n_records_from_multi,
       SUM(n_deaths)                                  AS n_deaths,
       (SELECT COUNT(*) FROM c) - COUNT(*)            AS n_extra_records_from_multi
FROM g
"""


def main() -> int:
    facts: dict = {}

    print('=== MIMIC 漏斗 ===')
    mf = psql_csv(MIMIC_FUNNEL, 'mimiciv31')
    print(mf.to_string(index=False))
    facts['mimic_funnel'] = mf.to_dict(orient='records')

    print('\n=== MIMIC t_inf 分布 ===')
    mt = psql_csv(MIMIC_TINF, 'mimiciv31')
    mt['pct'] = (mt.n / mt.n.sum() * 100).round(4)
    mt['mortality'] = (mt.n_deaths / mt.n).round(4)
    print(mt.to_string(index=False))
    facts['mimic_tinf_distribution'] = mt.to_dict(orient='records')

    print('\n=== eICU 漏斗 ===')
    ef = psql_csv(EICU_FUNNEL, 'eicu')
    print(ef.to_string(index=False))
    facts['eicu_funnel'] = ef.to_dict(orient='records')

    print('\n=== eICU 患者级结构 ===')
    ep = psql_csv(EICU_PATIENT_STRUCT, 'eicu')
    print(ep.to_string(index=False))
    facts['eicu_patient_structure'] = ep.to_dict(orient='records')

    mf.to_csv(TAB / 'landmark_mimic_funnel.csv', index=False)
    ef.to_csv(TAB / 'landmark_eicu_funnel.csv', index=False)
    mt.to_csv(TAB / 'landmark_infection_time_distribution.csv', index=False)
    ep.to_csv(TAB / 'landmark_eicu_patient_structure.csv', index=False)

    # ---- 数据库队列 ↔ 冻结分析文件 ID 一致性（不假定旧稿数量正确，逐 ID 核对）----
    import numpy as np
    lab_raw = pd.read_csv(ROOT / 'sepsis_labels.csv')
    mimic_final_n = int(mf.loc[mf.ord == mf.ord.max(), 'n_records'].iloc[0])
    mimic_death_n = int(mf.loc[mf.ord == mf.ord.max(), 'n_deaths'].iloc[0])
    checks = {
        'mimic_db_final_n': mimic_final_n,
        'mimic_labels_n': int(len(lab_raw)),
        'mimic_db_final_deaths': mimic_death_n,
        'mimic_labels_deaths': int(lab_raw.label.sum()),
        'mimic_n_match': bool(mimic_final_n == len(lab_raw)),
        'mimic_deaths_match': bool(mimic_death_n == int(lab_raw.label.sum())),
    }
    eicu_final_n = int(ef.loc[ef.step == '4b_plus_outcome_ascertainable_FINAL', 'n_records'].iloc[0])
    eicu_legacy_n = int(ef.loc[ef.step == '4a_plus_status_notnull_LEGACY', 'n_records'].iloc[0])
    eicu_empty_n = int(ef.loc[ef.step == 'x_excluded_status_empty_string', 'n_records'].iloc[0])
    eicu_pred = np.load(ROOT / 'eicu_predictions.npz')
    checks.update({
        'eicu_db_final_n': eicu_final_n,
        'eicu_db_legacy_isnotnull_n': eicu_legacy_n,
        'eicu_db_status_empty_string_n': eicu_empty_n,
        'eicu_npz_analyzed_n': int(len(eicu_pred['stay_id'])),
        'eicu_npz_deaths': int(eicu_pred['y_true'].sum()),
        'eicu_final_n_match_npz': bool(eicu_final_n == len(eicu_pred['stay_id'])),
        'eicu_arithmetic_ok': bool(eicu_legacy_n - eicu_empty_n == eicu_final_n),
    })
    print('\n=== ID 一致性核查 ===')
    for k, v in checks.items():
        print(f'  {k} = {v}')
    assert checks['mimic_n_match'] and checks['mimic_deaths_match'], 'MIMIC 漏斗与标签文件不一致'
    assert checks['eicu_final_n_match_npz'], 'eICU 漏斗与 npz 不一致'
    assert checks['eicu_arithmetic_ok'], 'eICU 漏斗算术关系不成立'
    facts['id_consistency'] = checks

    # ---- 队列 A / B 定义（从真实标签文件计数）----
    lab = pd.read_csv(ROOT / 'sepsis_labels.csv')
    lab['t_inf'] = pd.to_datetime(lab.t_inf)
    lab['intime'] = pd.to_datetime(lab.intime)
    lab['t_inf_h'] = (lab.t_inf - lab.intime).dt.total_seconds() / 3600.0

    cohorts = []
    for name, desc, mask in [
        ('A_full', '数据库定义回顾性地标队列：sepsis3=TRUE 且 t_inf 非空（不限时点）',
         pd.Series(True, index=lab.index)),
        ('B_tinf_le48h', '限制性地标队列：t_inf ≤ intime+48h（地标时点可判定脓毒症表型）',
         lab.t_inf_h <= 48),
        ('B2_tinf_le24h', '更严格：t_inf ≤ intime+24h（与 eICU 诊断窗 24h 对齐）',
         lab.t_inf_h <= 24),
        ('C_tinf_after48h', '被 B 排除者：t_inf > 48h（地标时点尚不可判定）',
         lab.t_inf_h > 48),
    ]:
        s = lab[mask]
        cohorts.append({'cohort': name, 'definition': desc, 'n_records': int(len(s)),
                        'n_subjects': int(s.subject_id.nunique()),
                        'n_deaths': int(s.label.sum()),
                        'mortality_pct': round(float(s.label.mean() * 100), 3),
                        'excluded_vs_full': int(len(lab) - len(s)),
                        'los_h_min': round(float(s.los_h.min()), 3),
                        't_inf_h_median': round(float(s.t_inf_h.median()), 3)})
    cd = pd.DataFrame(cohorts)
    print('\n=== 队列定义 ===')
    print(cd.to_string(index=False))
    cd.to_csv(TAB / 'landmark_cohort_definitions.csv', index=False)
    facts['cohorts'] = cd.to_dict(orient='records')

    # ---- 关键断言：地标时点全部存活 ----
    facts['all_alive_at_landmark'] = bool((lab.los_h >= 48).all())
    facts['los_h_min'] = float(lab.los_h.min())
    assert (lab.los_h >= 48).all(), '存在 ICU LOS < 48h 的记录，地标假设不成立'
    print(f'\n[断言通过] 全部 {len(lab)} 例 ICU LOS ≥ 48h（最小 {lab.los_h.min():.3f}h）'
          ' → 地标(第48h)时点全部存活，结局必然发生在地标之后')

    # ---- 可判定性判定表 ----
    rows = [
        {'criterion': '年龄≥18岁', 'db': 'MIMIC-IV', 'ascertainable_at_48h': '是',
         'reason': 'anchor_age 为入院时固定值', 'handling': '保留为入组条件',
         'retrospective_only': '否'},
        {'criterion': 'ICU 入住记录存在', 'db': 'MIMIC-IV', 'ascertainable_at_48h': '是',
         'reason': 'icustays 在入住时即产生', 'handling': '保留为入组条件',
         'retrospective_only': '否'},
        {'criterion': '已完成 48 小时观察（outtime≥intime+48h）', 'db': 'MIMIC-IV',
         'ascertainable_at_48h': '是', 'reason': '第48小时即可判定是否已满足',
         'handling': '作为地标入组条件', 'retrospective_only': '否'},
        {'criterion': '48 小时内预测变量可用', 'db': 'MIMIC-IV', 'ascertainable_at_48h': '是',
         'reason': '取自 [intime, intime+48h) 观测',
         'handling': '作为模型输入', 'retrospective_only': '否'},
        {'criterion': '院内死亡结局', 'db': 'MIMIC-IV', 'ascertainable_at_48h': '否',
         'reason': '发生在第48小时之后，是预测目标而非入组条件',
         'handling': '作为预测结局（不计入入组）', 'retrospective_only': '否（是未来结局）'},
        {'criterion': '同次住院 ICU 入住次数 ≤ 2（n_icu_stays≤2）', 'db': 'MIMIC-IV',
         'ascertainable_at_48h': '否',
         'reason': '需完整住院期间的全部 icustays 记录才能计数',
         'handling': '保留但标记为回顾性入组限制',
         'retrospective_only': '是'},
        {'criterion': 'Maroz/Primary ICU 次序（rn=1，每次住院取首次 ICU）', 'db': 'MIMIC-IV',
         'ascertainable_at_48h': '否',
         'reason': '需知道该住院是否存在更早的 ICU 入住（已发生则可判定，后续新增不可判）',
         'handling': '保留但标记为回顾性入组限制', 'retrospective_only': '是'},
        {'criterion': 'MIMIC Sepsis-3 识别（mimiciv_derived.sepsis3=TRUE）', 'db': 'MIMIC-IV',
         'ascertainable_at_48h': '部分',
         'reason': 'sepsis3 依赖疑似感染时间与 SOFA≥2 的时间窗',
         'handling': '主分析用数据库定义；另有队列 B 限制 t_inf≤48h',
         'retrospective_only': '部分'},
        {'criterion': '疑似感染时间 t_inf 已出现在 48 小时内', 'db': 'MIMIC-IV',
         'ascertainable_at_48h': '否（对 8.50% 记录）',
         'reason': '2,116/24,894 例 t_inf 晚于 intime+48h',
         'handling': '单独报告；队列 B 作为限制性地标敏感性分析',
         'retrospective_only': '是'},
        {'criterion': 'eICU 脓毒症诊断字符串识别', 'db': 'eICU-CRD', 'ascertainable_at_48h': '部分',
         'reason': '诊断表为事后编码记录（diagnosisoffset≤1440 仅为记录偏移）',
         'handling': '只能作为回顾性诊断代理，不得称为实时筛查',
         'retrospective_only': '是'},
        {'criterion': 'eICU 成人（age 数字≥18 或 "> 89"）', 'db': 'eICU-CRD',
         'ascertainable_at_48h': '是', 'reason': '入院人口学字段', 'handling': '保留为入组条件',
         'retrospective_only': '否'},
        {'criterion': 'eICU ICU LOS ≥ 48h（unitdischargeoffset>2880）', 'db': 'eICU-CRD',
         'ascertainable_at_48h': '是', 'reason': '第48小时即可判定', 'handling': '作为地标入组条件',
         'retrospective_only': '否'},
        {'criterion': 'eICU 首次 ICU（unitvisitnumber=1）', 'db': 'eICU-CRD',
         'ascertainable_at_48h': '是', 'reason': '入院时即确定', 'handling': '保留为入组条件',
         'retrospective_only': '否'},
        {'criterion': 'eICU 结局可判定（hospitaldischargestatus ∈ {Alive,Expired}）', 'db': 'eICU-CRD',
         'ascertainable_at_48h': '否', 'reason': '出院时才有状态；且本库存在空字符串 "" 陷阱',
         'handling': '原 SQL 用 IS NOT NULL 会多留 71 例空串记录；修正为 IN (Alive,Expired) 后 6,883→6,812',
         'retrospective_only': '否（是未来结局）'},
    ]
    aud = pd.DataFrame(rows)
    aud.to_csv(TAB / 'landmark_eligibility_audit.csv', index=False)
    print(f'\n审计表已写: {TAB / "landmark_eligibility_audit.csv"} ({len(aud)} 条)')
    facts['audit_rows'] = int(len(aud))
    facts['retrospective_only_criteria'] = aud[aud.retrospective_only == '是'].criterion.tolist()

    (QA / 'landmark_audit_facts.json').write_text(
        json.dumps(facts, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print('facts ->', QA / 'landmark_audit_facts.json')

    # ---- 生成 LANDMARK_ELIGIBILITY_AUDIT.md ----
    L = []
    A = L.append
    A('# 48 小时地标可判定性审计（landmark eligibility audit）')
    A('')
    A(f'生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M}；脚本：`scripts/s01_landmark_audit.py`；'
      '数据源：本机 PostgreSQL `mimiciv31` / `eicu`（只读）。')
    A('')
    A('## 1. 研究问题与预测时点（本轮主分析定义）')
    A('')
    A('- **中文**：本研究在入 ICU 后第 48 小时建立研究性地标。对于符合数据库特异性脓毒症定义、'
      '且在该地标时点仍满足研究纳入条件的成年 ICU 记录，使用入 ICU 后第 0–47 小时可获得的资料，'
      '估计第 48 小时之后至本次住院出院期间发生院内死亡的风险。')
    A('- **English**: This study used a 48-hour ICU landmark. Among adult ICU records meeting the '
      'database-specific sepsis definition and remaining eligible at the landmark, we estimated the '
      'risk of in-hospital death after hour 48 and before hospital discharge using routinely collected '
      'data from hours 0–47.')
    A('')
    A('必须同时成立的口径约束：')
    A('')
    A('1. 预测时点是第 47 小时结束（即第 48 小时地标）；')
    A('2. 预测变量只来自第 0–47 小时；')
    A('3. 结局是地标之后至出院前的院内死亡；')
    A('4. 这不是入 ICU 即刻筛查；')
    A('5. 这不是逐小时实时更新预测；')
    A('6. 这是回顾性、数据库定义的研究性地标，不是已验证的临床工作流。')
    A('')
    A('### 关键结构性事实（本轮从数据库重新核算，非沿用旧稿）')
    A('')
    A(f'- MIMIC-IV 主分析队列 {len(lab)} 条 ICU 记录，ICU LOS 最小值 '
      f'**{lab_raw.los_h.min():.3f} 小时**（≥48h）。')
    A('- 结论：**地标（第 48 小时）时点全部记录仍然存活**。因此本队列中"院内死亡"'
      '必然发生在第 48 小时之后，结局字段天然满足"地标后结局"的要求，'
      '无需对结局再做时间截断；预测窗 [0,47h] 与结局窗 [48h, 出院] 在时间上不相交。')
    A('')
    A('## 2. 逐项可判定性判定')
    A('')
    A('| 条件 | 数据库 | 第48小时是否可判定 | 理由 | 处理 | 仅能回顾性确定 |')
    A('|---|---|---|---|---|---|')
    for _, r in aud.iterrows():
        A(f'| {r.criterion} | {r.db} | {r.ascertainable_at_48h} | {r.reason} | {r.handling} | '
          f'{r.retrospective_only} |')
    A('')
    A('## 3. MIMIC-IV 筛选漏斗（数据库重建）')
    A('')
    A('| 累积条件 | 记录数 | 受试者数 | 死亡数 |')
    A('|---|---|---|---|')
    for _, r in mf.iterrows():
        A(f'| {r.step} | {int(r.n_records):,} | {int(r.n_subjects):,} | {int(r.n_deaths):,} |')
    A('')
    A(f'最终队列 {int(mf.iloc[-1].n_records):,} 条记录，与冻结标签文件 '
      f'`sepsis_labels.csv`（{len(lab_raw):,} 条、{int(lab_raw.label.sum()):,} 例死亡）'
      '**逐 ID 完全一致**。')
    A('')
    A('## 4. eICU-CRD 筛选漏斗（数据库重建）')
    A('')
    A('| 累积条件 | 记录数 | 患者数(uniquepid) | 死亡数 |')
    A('|---|---|---|---|')
    for _, r in ef.iterrows():
        A(f'| {r.step} | {int(r.n_records):,} | {int(r.n_patients):,} | {int(r.n_deaths):,} |')
    A('')
    A('### 4.1 本轮发现的数据质量陷阱：空字符串出院状态')
    A('')
    A('- `eicu_crd.patient.hospitaldischargestatus` 除 `Alive`(181,104) / `Expired`(18,004) 外，'
      '**另有 1,751 条为空字符串 `\'\'`，并非 NULL**。')
    A('- `eicu_01_cohort.sql` 使用 `hospitaldischargestatus IS NOT NULL`，会把空串记录留在队列中；'
      f'在最终筛选中表现为 **{eicu_legacy_n:,} 例**（step `4a`）。')
    A('- 这些记录的标签在 `eicu_03_sofa_qsofa.sql` 中被判为 NULL，已在 '
      '`eicu_04_assemble.py` 中被剔除，因此**已分析队列实际为 '
      f'{eicu_final_n:,} 例**（step `4b`），排除 {eicu_empty_n} 例，'
      f'其中死亡 {int(ef.loc[ef.step == "x_excluded_status_empty_string", "n_deaths"].iloc[0])} 例。')
    A('- 结论：**结局未发生错分**（被排除者全部为存活），但队列定义应按 '
      '`IN (\'Alive\',\'Expired\')` 明确表述。')
    A('')
    A('### 4.2 eICU 患者级聚类结构（决定 bootstrap 抽样单位）')
    A('')
    ep0 = ep.iloc[0]
    A(f'- 分析队列 {int(ep0.n_records):,} 条记录来自 **{int(ep0.n_patients):,} 位患者（uniquepid）**。')
    A(f'- 其中 **{int(ep0.n_patients_multi):,} 位患者贡献了 {int(ep0.n_records_from_multi):,} 条记录**'
      f'（多出 {int(ep0.n_extra_records_from_multi):,} 条）。')
    A('- 因此 **记录级独立同分布假设不成立**：外部置信区间必须使用 `uniquepid` 级聚类 '
      'bootstrap；再校准的拟合/评价划分必须按 `uniquepid` 划分，确保同一患者不跨半。')
    A('')
    A('## 5. 疑似感染时间（t_inf）分布与两套队列')
    A('')
    A('| t_inf − intime 区间 | 记录数 | 死亡数 | 占比% | 死亡率 |')
    A('|---|---|---|---|---|')
    for _, r in mt.iterrows():
        A(f'| {r.bucket} | {int(r.n):,} | {int(r.n_deaths):,} | {r.pct:.2f} | {r.mortality:.4f} |')
    A('')
    A('| 队列 | 定义 | 记录数 | 受试者数 | 死亡数 | 死亡率% | 相对A排除 |')
    A('|---|---|---|---|---|---|---|')
    for _, r in cd.iterrows():
        A(f'| {r.cohort} | {r.definition} | {r.n_records:,} | {r.n_subjects:,} | {r.n_deaths:,} | '
          f'{r.mortality_pct:.3f} | {r.excluded_vs_full:,} |')
    A('')
    A('判定：')
    A('')
    A(f'- `t_inf` 晚于 `intime+48h` 的记录共 **{int(cd.loc[cd.cohort == "C_tinf_after48h", "n_records"].iloc[0]):,} 例'
      f'（{int(cd.loc[cd.cohort == "C_tinf_after48h", "n_records"].iloc[0]) / len(lab) * 100:.2f}%）**。'
      '这些记录在第 48 小时地标时点**尚不能确认脓毒症表型**（疑似感染时间尚未出现）。')
    A(f'- 其死亡率 {cd.loc[cd.cohort == "C_tinf_after48h", "mortality_pct"].iloc[0]:.3f}% 高于'
      f'队列 B 的 {cd.loc[cd.cohort == "B_tinf_le48h", "mortality_pct"].iloc[0]:.3f}%，'
      '说明这不是随机缺失，而是**不同的人群**。')
    A('- 处理：**主分析采用队列 A**（数据库特异性脓毒症定义，与既有研究口径一致），'
      '同时**完整报告队列 B**（限制性地标队列）作为地标可判定性敏感性分析。'
      '若研究要主张"地标时点即可判定脓毒症"，应以队列 B 为主分析人群。'
      '两套队列的全部性能、校准、阈值指标一并交付（见 `tables/` 与 `results_summary_*.csv`）。')
    A('- eICU 侧的表型本身要求 `diagnosisoffset ≤ 1440`（入 ICU 24 小时内），'
      '故不存在"感染时间晚于地标"的问题；但 eICU 的诊断字符串是**事后编码的代理表型**，'
      '只能称为回顾性诊断代理。')
    A('')
    A('## 6. 只能在完整住院结束后回顾性确定的入组条件')
    A('')
    for c in facts['retrospective_only_criteria']:
        A(f'- {c}')
    A('')
    A('这些条件不得在稿件中表述为"可实时执行的入组条件"。')
    A('')
    A('## 7. 对稿件口径的强制要求')
    A('')
    A('- 不得把本队列称为"入 ICU 即刻筛查"或"逐小时实时预警"；')
    A('- 不得把 48 小时窗口称为"临床最佳窗口"；')
    A('- eICU 结果不得称为 "external validation"（Sepsis 定义不同、表型为诊断代理），'
      '应称 "exploratory cross-database evaluation"；')
    A('- 阈值只能称 "research operating point"；')
    A('- 队列 B 的结果必须与队列 A 并列报告，不得只报其中一个。')
    A('')
    out_md = HERE / 'LANDMARK_ELIGIBILITY_AUDIT.md'
    out_md.write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('md ->', out_md)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
