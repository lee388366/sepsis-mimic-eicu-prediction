#!/usr/bin/env python3
# =====================================================================
# qa_build.py —— DAG-GCS 数据准备：队列核验、覆盖率与质量核查
#
# 只读既有产出（output/*.csv）与本项目历史文件（../*.csv、*.npz），
# 不做数据库连接、不训练模型、不修改任何既有分析文件。
# 所有输出写入 qa/ 目录，全部为汇总级（无患者级明细导出）。
#
# 运行：
#   python3 qa_build.py
# =====================================================================
import json, hashlib, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'output'
QA = HERE / 'qa'
PROJ = HERE.parents[2]          # 项目根目录（含 sepsis_labels.csv 等）
QA.mkdir(exist_ok=True)
sys.dont_write_bytecode = True

MIMIC_N = 24894
EICU_N = 6812
RESULTS = {}


def sha256(p, blocksize=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(blocksize), b''):
            h.update(b)
    return h.hexdigest()


def rd(name, **kw):
    """读 output/ 下的抽取结果。"""
    kw.setdefault('low_memory', False)
    return pd.read_csv(OUT / f'{name}.csv', **kw)


def pct(a, b):
    return round(100.0 * a / b, 4) if b else None


def jdump(obj, name):
    (QA / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                      default=str), encoding='utf-8')


# =====================================================================
# 1. 队列 ID 匹配核验
# =====================================================================
print('[1/7] 队列 ID 匹配核验', flush=True)

mim = rd('mimic_00_cohort_index')
eic = rd('eicu_00_cohort_index')

labels = pd.read_csv(PROJ / 'sepsis_labels.csv')
tr = np.load(PROJ / 'sepsis_training_data.npz', allow_pickle=False)
cache = np.load(PROJ / 'all_predictions.npz', allow_pickle=False)
etr = np.load(PROJ / 'eicu_training_data.npz', allow_pickle=True)
epred = np.load(PROJ / 'eicu_predictions.npz', allow_pickle=True)

def ids(a):
    return set(int(x) for x in np.asarray(a).tolist())

mim_db, mim_lab, mim_npz = ids(mim.stay_id), ids(labels.stay_id), ids(tr['stay_id'])
eic_db, eic_npz, eic_pred = ids(eic.patientunitstayid), ids(etr['stay_id']), ids(epred['stay_id'])

link = {
    'mimic': {
        'db_rows': int(len(mim)),
        'db_unique_stay_id': len(mim_db),
        'local_labels_rows': int(len(labels)),
        'local_npz_rows': int(len(tr['stay_id'])),
        'db_vs_labels_only_in_db': len(mim_db - mim_lab),
        'db_vs_labels_only_in_labels': len(mim_lab - mim_db),
        'db_vs_npz_only_in_db': len(mim_db - mim_npz),
        'db_vs_npz_only_in_npz': len(mim_npz - mim_db),
        'deaths_db': int(mim.label.sum()),
        'deaths_labels': int(labels.label.sum()),
        'deaths_npz': int(tr['y'].sum()),
        'unique_subjects': int(mim.subject_id.nunique()),
        'unique_hadm': int(mim.hadm_id.nunique()),
        'rn_values': sorted(int(x) for x in mim.rn.unique()),
        'n_icu_stays_distribution': {str(k): int(v) for k, v in
                                     mim.n_icu_stays.value_counts().sort_index().items()},
        't_inf_within_48h': int(mim.infection_within_window.sum()),
        't_inf_within_48h_pct': pct(int(mim.infection_within_window.sum()), len(mim)),
        'stay_id_md5_sorted': hashlib.md5(
            ','.join(str(s) for s in sorted(mim_db)).encode()).hexdigest(),
    },
    'eicu': {
        'db_rows': int(len(eic)),
        'db_unique_patientunitstayid': len(eic_db),
        'local_npz_rows': int(len(etr['stay_id'])),
        'db_vs_npz_only_in_db': len(eic_db - eic_npz),
        'db_vs_npz_only_in_npz': len(eic_npz - eic_db),
        'deaths_db': int(eic.label.sum()),
        'deaths_npz': int(etr['y'].sum()),
        'unique_patienthealthsystemstayid': int(eic.patienthealthsystemstayid.nunique()),
        'unique_uniquepid': int(eic.uniquepid.nunique()),
        'unique_hospitalid': int(eic.hospitalid.nunique()),
        'predictions_rows': int(len(epred['stay_id'])),
        'predictions_subset_of_cohort': len(eic_pred - eic_db) == 0,
        'repeat_patient_records': int(
            len(eic) - eic.uniquepid.nunique()),
        'stay_id_md5_sorted': hashlib.md5(
            ','.join(str(s) for s in sorted(eic_db)).encode()).hexdigest(),
    },
}
# eICU 重复患者明细（uniquepid 贡献 >1 条记录者）
vc = eic.uniquepid.value_counts()
link['eicu']['patients_with_multiple_stays'] = int((vc > 1).sum())
link['eicu']['records_from_repeat_patients'] = int(vc[vc > 1].sum())
link['eicu']['pct_records_from_repeat_patients'] = pct(int(vc[vc > 1].sum()), len(eic))

# MIMIC 重复受试者
svc = mim.subject_id.value_counts()
link['mimic']['subjects_with_multiple_records'] = int((svc > 1).sum())
link['mimic']['records_from_repeat_subjects'] = int(svc[svc > 1].sum())
link['mimic']['pct_records_from_repeat_subjects'] = pct(int(svc[svc > 1].sum()), len(mim))

link['conclusion'] = {
    'mimic_exact_id_match': (mim_db == mim_lab == mim_npz),
    'eicu_exact_id_match': (eic_db == eic_npz),
    'note': '两库正式队列 ID 集合与既有缓存逐元素一致；md5 为排序后拼接的 ID 串哈希。',
}
RESULTS['cohort_linkage'] = link
jdump(link, 'cohort_linkage.json')
print('    MIMIC 精确一致:', link['conclusion']['mimic_exact_id_match'],
      ' eICU 精确一致:', link['conclusion']['eicu_exact_id_match'], flush=True)


# =====================================================================
# 2. 覆盖率矩阵
# =====================================================================
print('[2/7] 覆盖率矩阵', flush=True)

rows = []


def cover(db, category, detail, df, idcol, denom):
    n = df[idcol].nunique() if len(df) else 0
    rows.append({'db': db, 'category': category, 'detail': detail,
                 'n_records': int(n), 'denom_records': denom,
                 'pct_records': pct(n, denom), 'n_events': int(len(df))})


M_FILES = {
    'gcs_event_raw': ('mimic_01_gcs_event_raw', 'stay_id', '原始床旁 GCS 事件'),
    'gcs_derived': ('mimic_02_gcs_derived', 'stay_id', '派生 GCS 层'),
    'drug_inputevents': ('mimic_03_drug_inputevents', 'stay_id', 'ICU 实际给药(inputevents)'),
    'drug_emar': ('mimic_04_drug_emar', 'stay_id', 'eMAR 给药执行'),
    'drug_emar_detail': ('mimic_04b_drug_emar_detail', 'stay_id', 'eMAR 定量明细'),
    'drug_prescriptions': ('mimic_05_drug_prescriptions', 'stay_id', '医嘱层'),
    'ventilation_derived': ('mimic_06_ventilation_derived', 'stay_id', '派生呼吸支持时段'),
    'airway_procedures': ('mimic_07_airway_procedures', 'stay_id', '气道操作(procedureevents)'),
    'airway_chartevents': ('mimic_08_airway_chartevents', 'stay_id', '气道/通气护理记录'),
    'rass_cam_icu': ('mimic_09_rass_cam_icu', 'stay_id', 'RASS/CAM-ICU/谵妄'),
    'neuro_diagnoses': ('mimic_10_neuro_diagnoses', 'stay_id', '神经诊断(ICD)'),
    'icd_procedures': ('mimic_11_icd_procedures', 'stay_id', '呼吸支持操作(ICD)'),
}
for key, (fn, idc, desc) in M_FILES.items():
    df = rd(fn)
    cover('MIMIC-IV', key, desc, df, idc, MIMIC_N)

E_FILES = {
    'gcs_nursecharting': ('eicu_01_gcs_nursecharting', 'patientunitstayid', 'GCS(nursecharting)'),
    'rass_delirium_sedation': ('eicu_02_rass_delirium_sedation', 'patientunitstayid', 'RASS/谵妄/疼痛'),
    'sedation_infusiondrug': ('eicu_03_sedation_infusiondrug', 'patientunitstayid', '镇静输注记录'),
    'sedation_medication': ('eicu_04_sedation_medication', 'patientunitstayid', '镇静医嘱/给药'),
    'airway_respiratorycare': ('eicu_05_airway_respiratorycare', 'patientunitstayid', '气道类型与通气时段'),
    'airway_respiratorycharting': ('eicu_06_airway_respiratorycharting', 'patientunitstayid', '呼吸治疗记录'),
    'airway_treatment': ('eicu_07_airway_treatment', 'patientunitstayid', '治疗记录(通气/气道)'),
    'neuro_diagnoses': ('eicu_08_neuro_diagnoses', 'patientunitstayid', '神经诊断'),
    'pasthistory': ('eicu_09_pasthistory', 'patientunitstayid', '既往史(神经)'),
}
for key, (fn, idc, desc) in E_FILES.items():
    df = rd(fn)
    cover('eICU', key, desc, df, idc, EICU_N)

# 药物类别细分
for fn, lab, idc in [('mimic_03_drug_inputevents', 'inputevents', 'stay_id'),
                     ('mimic_04_drug_emar', 'emar', 'stay_id'),
                     ('mimic_05_drug_prescriptions', 'prescriptions', 'stay_id')]:
    df = rd(fn, usecols=[idc, 'drug_class'])
    for cls, g in df.groupby('drug_class'):
        cover('MIMIC-IV', 'drug_by_class', f'{lab}:{cls}', g, idc, MIMIC_N)
df = rd('mimic_04_drug_emar', usecols=['stay_id', 'event_category'])
for cat, g in df.groupby('event_category'):
    cover('MIMIC-IV', 'emar_by_event', cat, g, 'stay_id', MIMIC_N)

for fn, lab in [('eicu_03_sedation_infusiondrug', 'infusiondrug'),
                ('eicu_04_sedation_medication', 'medication')]:
    df = rd(fn, usecols=['patientunitstayid', 'drug_class'])
    for cls, g in df.groupby('drug_class'):
        cover('eICU', 'drug_by_class', f'{lab}:{cls}', g, 'patientunitstayid', EICU_N)

# 呼吸支持细分
df = rd('mimic_06_ventilation_derived', usecols=['stay_id', 'ventilation_status'])
for st, g in df.groupby('ventilation_status'):
    cover('MIMIC-IV', 'ventilation_by_status', st, g, 'stay_id', MIMIC_N)
df = rd('mimic_07_airway_procedures', usecols=['stay_id', 'airway_event_type'])
for st, g in df.groupby('airway_event_type'):
    cover('MIMIC-IV', 'airway_event_by_type', st, g, 'stay_id', MIMIC_N)
df = rd('eicu_05_airway_respiratorycare', usecols=['patientunitstayid', 'airway_state'])
for st, g in df.groupby('airway_state'):
    cover('eICU', 'airway_state', st, g, 'patientunitstayid', EICU_N)
df = rd('eicu_07_airway_treatment', usecols=['patientunitstayid', 'airway_treatment_group'])
for st, g in df.groupby('airway_treatment_group'):
    cover('eICU', 'airway_treatment_group', st, g, 'patientunitstayid', EICU_N)

cov = pd.DataFrame(rows)
cov.to_csv(QA / 'coverage_matrix.csv', index=False)
print('    覆盖率矩阵行数:', len(cov), flush=True)


# =====================================================================
# 3. GCS 质量核查
# =====================================================================
print('[3/7] GCS 质量核查', flush=True)

g = rd('mimic_01_gcs_event_raw')
gd = rd('mimic_02_gcs_derived')

gq = {'source': 'mimic_01_gcs_event_raw + mimic_02_gcs_derived'}
gq['rows'] = int(len(g))
gq['records_with_events'] = int(g.stay_id.nunique())
gq['records_without_events'] = MIMIC_N - gq['records_with_events']
gq['events_by_component'] = {k: int(v) for k, v in g.component.value_counts().items()}
gq['duplicate_stay_charttime_item'] = int(
    g.duplicated(['stay_id', 'charttime', 'itemid']).sum())
gq['duplicate_stay_charttime_item_examples'] = \
    g[g.duplicated(['stay_id', 'charttime', 'itemid'], keep=False)] \
    .groupby(['stay_id', 'charttime', 'itemid']).size().head(5).to_dict()
gq['hours_from_intime_min'] = float(g.hours_from_intime.min())
gq['hours_from_intime_max'] = float(g.hours_from_intime.max())
gq['events_before_minus6h'] = int((g.hours_from_intime < -6).sum())
gq['events_after_48h'] = int((g.hours_from_intime > 48).sum())
gq['events_in_0_48h'] = int(g.hours_from_intime.between(0, 48).sum())
gq['negative_chart_to_store_lag'] = int((g.chart_to_store_lag_h < 0).sum())
gq['zero_chart_to_store_lag'] = int((g.chart_to_store_lag_h == 0).sum())
gq['warning_nonzero'] = int((g.warning.fillna(0) != 0).sum())
gq['value_raw_missing'] = int(g.value_raw.isna().sum())
gq['valuenum_missing'] = int(g.valuenum.isna().sum())
gq['score_mapped_null'] = int(g.score_mapped.isna().sum())
gq['score_mapped_null_values'] = \
    g[g.score_mapped.isna()].value_raw.value_counts().head(10).to_dict()
gq['ett_text_events'] = int(g.any_ett_text_flag.sum())
gq['true_no_response_events'] = int(g.true_no_response_flag.sum())
gq['verbal_value_raw_distribution'] = {
    str(k): int(v) for k, v in
    g[g.component == 'verbal'].value_raw.value_counts().items()}
# 原始 score 与 valuenum 一致性（verbal ETT 的分歧是重点）
gv = g[g.component == 'verbal'].copy()
gv['consistent'] = (gv.score_mapped == gv.valuenum) | \
                   (gv.value_raw == 'No Response-ETT')
gq['verbal_score_vs_valuenum_inconsistent'] = int((~gv.consistent).sum())
gq['verbal_score_vs_valuenum_inconsistent_values'] = {
    str(k): int(v) for k, v in
    gv[~gv.consistent].value_raw.value_counts().items()}

# 派生层
gq['derived_rows'] = int(len(gd))
gq['derived_records'] = int(gd.stay_id.nunique())
gq['derived_records_without_raw_events'] = int(
    len(set(gd.stay_id.unique()) - set(g.stay_id.unique())))
gq['derived_total_minus_sum_nonzero'] = int((gd.total_minus_sum.fillna(0) != 0).sum())
gq['derived_total15_with_sum_below15'] = int(gd.total15_with_sum_below15.sum())
gq['derived_records_with_total15_discordance'] = int(
    gd[gd.total15_with_sum_below15 == 1].stay_id.nunique())
gq['derived_unable_flag_rows'] = int((gd.unable_flag == 1).sum())
gq['derived_verbal_zero_rows'] = int(gd.verbal_zero_flag.sum())
gq['derived_verbal_zero_without_unable'] = int(
    ((gd.verbal_zero_flag == 1) & (gd.unable_flag == 0)).sum())
gq['derived_total_range'] = [float(gd.gcs_total.min()), float(gd.gcs_total.max())]
gq['derived_component_null_counts'] = {
    c: int(gd[c].isna().sum()) for c in
    ['gcs_total', 'gcs_eyes', 'gcs_verbal', 'gcs_motor']}

# 逐事件对照：同一 (stay_id, charttime) 上原始 ETT 与派生 unable
key = ['stay_id', 'charttime']
raw_ett = g[g.component == 'verbal'].groupby(key).agg(
    has_ett=('ett_flag', 'max'), has_true_no_resp=('true_no_response_flag', 'max')).reset_index()
mrg = gd.merge(raw_ett, on=key, how='left')
matched = mrg[mrg.has_ett.notna()]
gq['derived_rows_matching_raw_verbal_events'] = int(len(matched))
gq['ett_matched_derived_unable_eq1'] = int(
    ((matched.has_ett == 1) & (matched.unable_flag == 1)).sum())
gq['ett_matched_derived_unable_eq0'] = int(
    ((matched.has_ett == 1) & (matched.unable_flag == 0)).sum())
gq['derived_rows_without_raw_verbal_at_same_time'] = int(mrg.has_ett.isna().sum())
gq['derived_verbal_zero_no_raw_match'] = int(
    ((mrg.verbal_zero_flag == 1) & (mrg.has_ett.isna())).sum())

jdump(gq, 'gcs_quality_mimic.json')

# ---- eICU GCS ----
eg = rd('eicu_01_gcs_nursecharting')
eq = {'source': 'eicu_01_gcs_nursecharting'}
eq['rows'] = int(len(eg))
eq['records_with_gcs'] = int(eg.patientunitstayid.nunique())
eq['records_without_gcs'] = EICU_N - eq['records_with_gcs']
eq['rows_by_field'] = {f'{a} | {b}': int(v) for (a, b), v in
                       eg.groupby(['field_label', 'field_name']).size().items()}
eq['duplicate_pid_offset_field'] = int(
    eg.duplicated(['patientunitstayid', 'nursingchartoffset', 'field_label', 'field_name']).sum())
eq['offset_min'] = int(eg.nursingchartoffset.min())
eq['offset_max'] = int(eg.nursingchartoffset.max())
eq['entry_minus_chart_min_min'] = int(eg.entry_minus_chart_min.min())
eq['entry_minus_chart_min_max'] = int(eg.entry_minus_chart_min.max())
eq['negative_entry_minus_chart'] = int((eg.entry_minus_chart_min < 0).sum())
eq['value_state_distribution'] = {k: int(v) for k, v in eg.value_state.value_counts().items()}
eq['gcs_value_class_distribution'] = {k: int(v) for k, v in
                                      eg.gcs_value_class.value_counts().items()}
eq['unable_due_to_medication_rows'] = int(
    (eg.gcs_value_class == 'unable_due_to_medication').sum())
eq['unable_due_to_medication_records'] = int(
    eg[eg.gcs_value_class == 'unable_due_to_medication'].patientunitstayid.nunique())
eq['blank_rows'] = int((eg.gcs_value_class == 'blank').sum())
eq['blank_by_field'] = {f'{a} | {b}': int(v) for (a, b), v in
                        eg[eg.gcs_value_class == 'blank']
                        .groupby(['field_label', 'field_name']).size().items()}
eq['numeric_range_by_field'] = {
    f'{a} | {b}': [float(gg.value_numeric.min()), float(gg.value_numeric.max())]
    for (a, b), gg in eg[eg.value_numeric.notna()]
    .groupby(['field_label', 'field_name'])}

# 同一时刻三分项是否齐全；总分是否等于三分项之和
comp = eg[eg.field_label == 'Glasgow coma score'].copy()
# 注意：不要用 pivot_table 的行数当"时间点数"。pivot_table 在缺失组合下会产生
# 笛卡尔膨胀（曾观测到 1,460 万这一不可能值）。直接用去重后的键计数。
eq['timestamps_total'] = int(
    comp[['patientunitstayid', 'nursingchartoffset']].drop_duplicates().shape[0])
piv = comp.pivot_table(index=['patientunitstayid', 'nursingchartoffset'],
                       columns='field_name', values='value_numeric',
                       aggfunc='first', dropna=False)
eq['timestamps_all_three_components'] = int(
    piv[['Eyes', 'Motor', 'Verbal']].notna().all(axis=1).sum())
eq['timestamps_have_gcs_total'] = int(piv['GCS Total'].notna().sum())
both = piv[piv[['Eyes', 'Motor', 'Verbal', 'GCS Total']].notna().all(axis=1)].copy()
both['sum3'] = both[['Eyes', 'Motor', 'Verbal']].sum(axis=1)
both['diff'] = both['GCS Total'] - both['sum3']
eq['timestamps_total_and_all_components'] = int(len(both))
eq['timestamps_total_ne_sum3'] = int((both['diff'].abs() > 1e-9).sum())
eq['timestamps_total_ne_sum3_pct'] = pct(int((both['diff'].abs() > 1e-9).sum()), len(both))
eq['diff_distribution'] = {str(k): int(v) for k, v in
                           both['diff'].value_counts().head(10).items()}

jdump(eq, 'gcs_quality_eicu.json')
print('    MIMIC GCS 行数:', gq['rows'], ' eICU GCS 行数:', eq['rows'], flush=True)


# =====================================================================
# 4. 药物层质量核查
# =====================================================================
print('[4/7] 药物层质量核查', flush=True)

dq = {}
# --- inputevents ---
ie = rd('mimic_03_drug_inputevents')
dq['inputevents'] = {
    'rows': int(len(ie)), 'records': int(ie.stay_id.nunique()),
    'duplicate_stay_orderid_item_starttime': int(
        ie.duplicated(['stay_id', 'orderid', 'itemid', 'starttime']).sum()),
    'endtime_before_starttime': int((ie.endtime < ie.starttime).sum()),
    'rateuom_distribution': {str(k): int(v) for k, v in ie.rateuom.value_counts(dropna=False).head(20).items()},
    'amountuom_distribution': {str(k): int(v) for k, v in ie.amountuom.value_counts(dropna=False).head(20).items()},
    'administration_mode': {str(k): int(v) for k, v in ie.administration_mode.value_counts(dropna=False).items()},
    'statusdescription': {str(k): int(v) for k, v in ie.statusdescription.value_counts(dropna=False).head(20).items()},
    'amount_null': int(ie.amount.isna().sum()),
    'rate_null': int(ie.rate.isna().sum()),
    'both_amount_and_rate_null': int((ie.amount.isna() & ie.rate.isna()).sum()),
    'amount_zero_or_negative': int((ie.amount.fillna(1) <= 0).sum()),
    'rate_zero_or_negative': int((ie.rate.fillna(1) <= 0).sum()),
    'totalamount_null': int(ie.totalamount.isna().sum()),
    'isopenbag_distribution': {str(k): int(v) for k, v in ie.isopenbag.value_counts(dropna=False).items()},
    'started_before_window': int(ie.started_before_window.sum()),
    'continuous_spanning_window_boundary': int(
        ((ie.started_before_window == 1) & (ie.overlaps_window == 1)).sum()),
}
# --- emar ---
em = rd('mimic_04_drug_emar')
cre = em.event_category.value_counts()
dq['emar'] = {
    'rows': int(len(em)), 'records': int(em.stay_id.nunique()),
    'event_category': {str(k): int(v) for k, v in cre.items()},
    'not_given_by_drug_class': {
        f'{a}|{b}': int(v) for (a, b), v in
        em[em.event_category == 'not_given'].groupby(['drug_class', 'drug_name']).size().items()},
    'given_by_drug_class': {
        f'{a}|{b}': int(v) for (a, b), v in
        em[em.event_category == 'given'].groupby(['drug_class', 'drug_name']).size().items()},
    'null_event_txt': int(em.event_txt.isna().sum()),
    'duplicate_emar_id_seq': int(em.duplicated(['emar_id', 'emar_seq']).sum()),
    'capitalisation_variants_detected': {
        str(n): int(em[em.drug_name == n].medication_clean.nunique())
        for n in em.drug_name.dropna().unique() if
        em[em.drug_name == n].medication_clean.nunique() > 1},
}
# --- prescriptions ---
pr = rd('mimic_05_drug_prescriptions')
dq['prescriptions'] = {
    'rows': int(len(pr)), 'records': int(pr.stay_id.nunique()),
    'drug_type': {str(k): int(v) for k, v in pr.drug_type.value_counts(dropna=False).items()},
    'window_relation': {str(k): int(v) for k, v in pr.window_relation.value_counts(dropna=False).items()},
    'stoptime_null': int(pr.stoptime.isna().sum()),
    'stoptime_before_starttime': int((pr.stoptime < pr.starttime).sum()),
    'dose_unit_rx_top': {str(k): int(v) for k, v in pr.dose_unit_rx.value_counts(dropna=False).head(15).items()},
    'route_top': {str(k): int(v) for k, v in pr.route.value_counts(dropna=False).head(15).items()},
    'dose_val_rx_null': int(pr.dose_val_rx.isna().sum()),
    'dose_val_rx_non_numeric': int(
        pd.to_numeric(pr.dose_val_rx.astype(str).str.replace(',', ''), errors='coerce')
        .isna().sum() - pr.dose_val_rx.isna().sum()),
}
# --- eicu infusiondrug ---
ei = rd('eicu_03_sedation_infusiondrug')
ei['drugrate_numeric'] = pd.to_numeric(ei.drugrate, errors='coerce')
dq['eicu_infusiondrug'] = {
    'rows': int(len(ei)), 'records': int(ei.patientunitstayid.nunique()),
    'duplicate_pid_offset_drugname': int(
        ei.duplicated(['patientunitstayid', 'infusionoffset', 'drugname']).sum()),
    'rate_unit_in_name_distribution': {str(k): int(v) for k, v in
                                       ei.rate_unit_in_name.value_counts(dropna=False).head(20).items()},
    'drugname_with_empty_unit_parens': int(ei.drugname.str.contains(r'\(\s*\)$', regex=True).sum()),
    'drugrate_null': int(ei.drugrate.isna().sum()),
    'drugrate_nonnumeric': int(ei.drugrate_numeric.isna().sum() - ei.drugrate.isna().sum()),
    'drugrate_zero_or_negative': int((ei.drugrate_numeric.fillna(1) <= 0).sum()),
    'drugamount_null': int(ei.drugamount.isna().sum()),
    'patientweight_null': int(ei.patientweight.isna().sum()),
    'has_no_end_time_field': True,
    'note': 'infusiondrug 无结束时间字段，无法直接重建输注区间。',
}
# --- eicu medication ---
emd = rd('eicu_04_sedation_medication')
dq['eicu_medication'] = {
    'rows': int(len(emd)), 'records': int(emd.patientunitstayid.nunique()),
    'lifecycle_completeness': {str(k): int(v) for k, v in
                               emd.lifecycle_completeness.value_counts(dropna=False).items()},
    'order_cancelled_true': int(emd.order_cancelled_flag.sum()),
    'drugordercancelled_distribution': {str(k): int(v) for k, v in
                                        emd.drugordercancelled.value_counts(dropna=False).items()},
    'duration_negative': int((emd.duration_h < 0).sum()),
    'duration_null_but_both_offsets': int(
        ((emd.drugstartoffset.notna()) & (emd.drugstopoffset.notna())
         & (emd.duration_h.isna())).sum()),
    'prn_distribution': {str(k): int(v) for k, v in emd.prn.value_counts(dropna=False).head(10).items()},
    'routeadmin_top': {str(k): int(v) for k, v in emd.routeadmin.value_counts(dropna=False).head(15).items()},
    'dosage_null': int(emd.dosage.isna().sum()),
    'dosage_examples': [str(x) for x in emd.dosage.dropna().unique()[:15]],
}
jdump(dq, 'drug_quality.json')


# =====================================================================
# 5. 气道/呼吸支持质量核查
# =====================================================================
print('[5/7] 气道/呼吸支持质量核查', flush=True)

aq = {}
ven = rd('mimic_06_ventilation_derived')
aq['mimic_ventilation_derived'] = {
    'rows': int(len(ven)), 'records': int(ven.stay_id.nunique()),
    'status': {str(k): int(v) for k, v in ven.ventilation_status.value_counts().items()},
    'records_by_status': {str(k): int(v) for k, v in
                          ven.groupby('ventilation_status').stay_id.nunique().items()},
    'endtime_null': int(ven.endtime.isna().sum()),
    'duration_zero_or_negative': int((ven.duration_h <= 0).sum()),
    'duration_median_h': float(ven.duration_h.median()),
    'started_before_window': int(ven.started_before_window.sum()),
    'started_in_0_48h': int(ven.started_in_0_48h.sum()),
    'duplicate_stay_starttime_status': int(
        ven.duplicated(['stay_id', 'starttime', 'ventilation_status']).sum()),
}
proc = rd('mimic_07_airway_procedures')
aq['mimic_airway_procedures'] = {
    'rows': int(len(proc)), 'records': int(proc.stay_id.nunique()),
    'by_event_type': {str(k): int(v) for k, v in proc.airway_event_type.value_counts().items()},
    'records_by_event_type': {str(k): int(v) for k, v in
                              proc.groupby('airway_event_type').stay_id.nunique().items()},
    'endtime_null': int(proc.endtime.isna().sum()),
    'duration_zero_or_negative': int((proc.duration_h <= 0).sum()),
    'statusdescription': {str(k): int(v) for k, v in
                          proc.statusdescription.value_counts(dropna=False).head(10).items()},
}
chv = rd('mimic_08_airway_chartevents', usecols=['stay_id', 'airway_field', 'value_raw', 'itemid'])
aq['mimic_airway_chartevents'] = {
    'rows': int(len(chv)), 'records': int(chv.stay_id.nunique()),
    'by_field': {str(k): int(v) for k, v in chv.airway_field.value_counts().items()},
    'airway_type_values': {str(k): int(v) for k, v in
                           chv[chv.airway_field == 'airway_type'].value_raw.value_counts().head(20).items()},
    'mechanically_ventilated_values': {str(k): int(v) for k, v in
                                       chv[chv.airway_field == 'mechanically_ventilated_flag']
                                       .value_raw.value_counts(dropna=False).items()},
    'o2_device_values': {str(k): int(v) for k, v in
                         chv[chv.airway_field == 'o2_delivery_device'].value_raw
                         .value_counts().head(20).items()},
}
icd = rd('mimic_11_icd_procedures')
aq['mimic_icd_procedures'] = {
    'rows': int(len(icd)), 'records': int(icd.stay_id.nunique()),
    'by_group': {str(k): int(v) for k, v in icd.airway_procedure_group.value_counts().items()},
    'records_by_group': {str(k): int(v) for k, v in
                         icd.groupby('airway_procedure_group').stay_id.nunique().items()},
    'icd_version': {str(k): int(v) for k, v in icd.icd_version.value_counts().items()},
}
# 四来源交叉：有创通气
inv_derived = set(ven[ven.ventilation_status == 'InvasiveVent'].stay_id)
inv_proc = set(proc[proc.airway_event_type == 'invasive_ventilation'].stay_id)
inv_intub = set(proc[proc.airway_event_type == 'intubation'].stay_id)
inv_icd = set(icd[icd.airway_procedure_group == 'mechanical_ventilation'].stay_id)
niv_proc = set(proc[proc.airway_event_type == 'noninvasive_ventilation'].stay_id)
niv_derived = set(ven[ven.ventilation_status == 'NonInvasiveVent'].stay_id)
aq['cross_source_invasive'] = {
    'n_derived': len(inv_derived), 'n_procedure_invasive': len(inv_proc),
    'n_procedure_intubation': len(inv_intub), 'n_icd_mech_vent': len(inv_icd),
    'derived_and_proc_invasive': len(inv_derived & inv_proc),
    'derived_and_icd': len(inv_derived & inv_icd),
    'derived_and_intubation': len(inv_derived & inv_intub),
    'derived_only': len(inv_derived - inv_proc - inv_intub - inv_icd),
    'proc_or_icd_but_not_derived': len((inv_proc | inv_intub | inv_icd) - inv_derived),
    'union_all_four': len(inv_derived | inv_proc | inv_intub | inv_icd),
}
aq['cross_source_noninvasive'] = {
    'n_derived': len(niv_derived), 'n_procedure_niv': len(niv_proc),
    'both': len(niv_derived & niv_proc),
    'derived_only': len(niv_derived - niv_proc), 'proc_only': len(niv_proc - niv_derived),
}
# GCS 语言=0 与有创通气是否一致（检验"verbal=0 不能当通气金标准"）
gdz = gd.groupby('stay_id').agg(any_verbal_zero=('verbal_zero_flag', 'max')).reset_index()
sz = set(gdz[gdz.any_verbal_zero == 1].stay_id)
aq['verbal_zero_vs_invasive'] = {
    'records_any_verbal_zero': len(sz),
    'verbal_zero_and_any_invasive_source': len(sz & (inv_derived | inv_proc | inv_intub | inv_icd)),
    'verbal_zero_without_any_invasive_source': len(sz - (inv_derived | inv_proc | inv_intub | inv_icd)),
    'note': '语言分项为 0（含 ETT 编码）不能单独证明当时存在有创通气。',
}

# eICU
erc = rd('eicu_05_airway_respiratorycare')
aq['eicu_respiratorycare'] = {
    'rows': int(len(erc)), 'records': int(erc.patientunitstayid.nunique()),
    'airway_state': {str(k): int(v) for k, v in erc.airway_state.value_counts(dropna=False).items()},
    'records_by_airway_state': {str(k): int(v) for k, v in
                                erc.groupby('airway_state').patientunitstayid.nunique().items()},
    # 【关键语义】vent* 字段是零填充而非 NULL
    'ventstart_state': {str(k): int(v) for k, v in erc.ventstart_state.value_counts(dropna=False).items()},
    'ventend_state': {str(k): int(v) for k, v in erc.ventend_state.value_counts(dropna=False).items()},
    'vent_interval_valid_records': int(
        erc[erc.vent_interval_valid == 1].patientunitstayid.nunique()),
    'vent_before_unit_admit_records': int(
        erc[erc.vent_before_unit_admit == 1].patientunitstayid.nunique()),
    'vent_started_in_icu_records': int(
        erc[erc.vent_started_in_icu == 1].patientunitstayid.nunique()),
    'has_prior_vent_episode_records': int(
        erc[erc.has_prior_vent_episode == 1].patientunitstayid.nunique()),
    'any_vent_signal_records': int(
        erc[erc.any_vent_signal == 1].patientunitstayid.nunique()),
    'any_vent_signal_pct_of_cohort': pct(
        int(erc[erc.any_vent_signal == 1].patientunitstayid.nunique()), EICU_N),
    'critical_finding': (
        'ventendoffset 在队列内 20,566 行中 100% 为 0（全库 865,381 行中 99.98% 为 0，'
        '仅 157 行为正值）。因此 eICU 并未记录通气结束时间，'
        'ventendoffset=0 意为"无结束时间"而非"在入室时刻结束"。'
        '据此，eICU 无法构造任何有效的通气时段区间（vent_interval_valid 记录数 = 0），'
        '只能得到"入室前已在通气""入室后某刻开始通气"两类时点信号与既往通气史。'),
}
etc = rd('eicu_07_airway_treatment')
aq['eicu_treatment'] = {
    'rows': int(len(etc)), 'records': int(etc.patientunitstayid.nunique()),
    'by_group': {str(k): int(v) for k, v in etc.airway_treatment_group.value_counts().items()},
    'records_by_group': {str(k): int(v) for k, v in
                         etc.groupby('airway_treatment_group').patientunitstayid.nunique().items()},
}
erch = rd('eicu_06_airway_respiratorycharting')
aq['eicu_respiratorycharting'] = {
    'rows': int(len(erch)), 'records': int(erch.patientunitstayid.nunique()),
    'by_field_group': {str(k): int(v) for k, v in erch.field_group.value_counts(dropna=False).items()},
    'et_text_flag_records': int(erch[erch.et_text_flag == 1].patientunitstayid.nunique()),
    'trach_text_flag_records': int(erch[erch.trach_text_flag == 1].patientunitstayid.nunique()),
    'invasive_text_flag_records': int(erch[erch.invasive_text_flag == 1].patientunitstayid.nunique()),
    'oxygen_delivery_method_values': {
        str(k): int(v) for k, v in
        erch[erch.field_label == 'Oxygen Delivery Method'].value_raw.value_counts().head(20).items()},
    'value_state': {str(k): int(v) for k, v in erch.value_state.value_counts().items()},
}
# eICU 交叉：人工气道 / 通气时点信号 / 治疗
e_airway = set(erc[erc.artificial_airway_flag == 1].patientunitstayid)
e_ventint = set(erc[erc.any_vent_signal == 1].patientunitstayid)
e_mech = set(etc[etc.airway_treatment_group == 'invasive_mechanical_ventilation'].patientunitstayid)
e_niv = set(etc[etc.airway_treatment_group == 'noninvasive_ventilation'].patientunitstayid)
e_ettext = set(erch[erch.et_text_flag == 1].patientunitstayid)
aq['eicu_cross_source'] = {
    'n_artificial_airway': len(e_airway),
    'n_any_vent_signal': len(e_ventint),
    'n_treatment_mech': len(e_mech), 'n_treatment_niv': len(e_niv),
    'n_et_text': len(e_ettext),
    'airway_and_vent_signal': len(e_airway & e_ventint),
    'airway_and_treatment_mech': len(e_airway & e_mech),
    'vent_signal_and_treatment_mech': len(e_ventint & e_mech),
    'union_all_sources': len(e_airway | e_ventint | e_mech | e_ettext),
    'union_pct_of_cohort': pct(len(e_airway | e_ventint | e_mech | e_ettext), EICU_N),
    'artificial_airway_only_no_vent_signal': len(e_airway - e_ventint),
    'treatment_mech_without_airway_record': len(e_mech - e_airway),
    'niv_without_mech': len(e_niv - e_mech),
    'note': ('无任何来源可给出完整通气区间；此处 union 表示"至少有通气证据"的住院数，'
             '不等于"某一时点正在通气"。'),
}
jdump(aq, 'airway_quality.json')


# =====================================================================
# 6. "未发生 / 未知 / 未记录" 可区分性 + eICU 医院间差异
# =====================================================================
print('[6/7] 三态可区分性 + 医院间覆盖差异', flush=True)

tri = {
    'mimic': {
        'gcs_ett': {
            'can_distinguish_not_assessable_from_no_response': True,
            'evidence': "raw value 有 'No Response-ETT'（不可评估）与 'No Response'（无反应）两种文本",
            'ett_events': gq['ett_text_events'],
            'true_no_response_events': gq['true_no_response_events'],
        },
        'gcs_carry_forward': {
            'can_distinguish_bedside_from_derived': True,
            'evidence': 'mimic_01 为 chartevents 原始层，mimic_02 为派生层，可按 (stay_id, charttime) 对照',
            'derived_rows_without_raw_verbal_at_same_time': gq['derived_rows_without_raw_verbal_at_same_time'],
        },
        'sedation_not_recorded_vs_not_given': {
            'can_distinguish': True,
            'evidence': "eMAR event_txt 的 'Not Given' 可证否给药；NULL 仅表示未记录",
            'not_given_events': int((em.event_category == 'not_given').sum()),
            'null_event_txt': int(em.event_txt.isna().sum()),
        },
        'ventilation_not_recorded_vs_absent': {
            'can_distinguish': True,
            'evidence': 'mimic_06 给出显式时段；未出现时段 ≠ 未通气（可能发生在窗口外或另一来源）',
            'cross_source_union': aq['cross_source_invasive']['union_all_four'],
        },
        'neuro_baseline': {
            'can_distinguish': False,
            'reason': 'ICD 诊断为住院级汇总，无发生时间，不能区分入院前既有 vs 住院期间新发',
        },
    },
    'eicu': {
        'gcs_ett': {
            'can_distinguish_not_assessable_from_no_response': False,
            'reason': "eICU GCS 无 ETT 语言标记；只有 'Unable to score due to medication' 这一种显式不可评估",
            'unable_due_to_medication_rows': eq['unable_due_to_medication_rows'],
        },
        'gcs_blank_vs_null': {
            'can_distinguish': True,
            'evidence': "value_state 区分 'empty_string' 与 'null'",
            'empty_string_rows': int((eg.value_state == 'empty_string').sum()),
            'null_rows': int((eg.value_state == 'null').sum()),
        },
        'airway_not_recorded_vs_no_airway': {
            'can_distinguish': True,
            'evidence': "respiratorycare.airwaytype 的 ''（未记录）与 'No Artificial Airway'（明确无人道气道）已分别归组",
            'not_recorded_rows': int((erc.airway_state == 'not_recorded').sum()),
            'no_artificial_airway_rows': int((erc.airway_state == 'no_artificial_airway').sum()),
        },
        'sedation_infusion_no_end_time': {
            'can_distinguish': False,
            'reason': 'infusiondrug 无结束时间字段，只能给出记录的首次/末次时刻，不能界定"已停药"',
        },
        'medication_order_vs_given': {
            'can_distinguish': True,
            'evidence': 'medication 同时有 drugorderoffset/drugstartoffset/drugstopoffset/drugordercancelled，可区分医嘱、开始、停止、取消',
            'order_cancelled_true': dq['eicu_medication']['order_cancelled_true'],
            'lifecycle': dq['eicu_medication']['lifecycle_completeness'],
        },
    },
}
jdump(tri, 'three_state_distinguishability.json')

# eICU 医院间覆盖差异
hosp = eic[['patientunitstayid', 'hospitalid', 'region', 'teachingstatus']].copy()
n_hosp = hosp.hospitalid.nunique()
hosp['has_gcs'] = hosp.patientunitstayid.isin(set(eg.patientunitstayid)).astype(int)
hosp['has_gcs_unable_marker'] = hosp.patientunitstayid.isin(
    set(eg[eg.gcs_value_class == 'unable_due_to_medication'].patientunitstayid)).astype(int)
rass_ids = pd.read_csv(OUT / 'eicu_02_rass_delirium_sedation.csv',
                       usecols=['patientunitstayid', 'assessment_field'])
hosp['has_rass'] = hosp.patientunitstayid.isin(
    set(rass_ids[rass_ids.assessment_field == 'rass_actual'].patientunitstayid)).astype(int)
hosp['has_sedation_infusion'] = hosp.patientunitstayid.isin(
    set(ei.patientunitstayid)).astype(int)
hosp['has_sedation_medication'] = hosp.patientunitstayid.isin(
    set(emd.patientunitstayid)).astype(int)
hosp['has_airway_record'] = hosp.patientunitstayid.isin(e_airway).astype(int)
hosp['has_airway_any_state_row'] = hosp.patientunitstayid.isin(set(erc.patientunitstayid)).astype(int)
hosp['has_vent_treatment'] = hosp.patientunitstayid.isin(e_mech | e_niv).astype(int)
hosp['has_respcharting'] = hosp.patientunitstayid.isin(set(erch.patientunitstayid)).astype(int)
hosp['has_neuro_dx'] = hosp.patientunitstayid.isin(
    set(rd('eicu_08_neuro_diagnoses', usecols=['patientunitstayid']).patientunitstayid)).astype(int)
hosp['has_pasthistory'] = hosp.patientunitstayid.isin(
    set(rd('eicu_09_pasthistory', usecols=['patientunitstayid']).patientunitstayid)).astype(int)

agg = hosp.groupby(['hospitalid', 'region', 'teachingstatus']).agg(
    n_records=('patientunitstayid', 'size'),
    pct_gcs=('has_gcs', 'mean'),
    pct_gcs_unable_marker=('has_gcs_unable_marker', 'mean'),
    pct_rass=('has_rass', 'mean'),
    pct_sedation_infusion=('has_sedation_infusion', 'mean'),
    pct_sedation_medication=('has_sedation_medication', 'mean'),
    pct_any_airway_record=('has_airway_record', 'mean'),
    pct_airway_state_row=('has_airway_any_state_row', 'mean'),
    pct_vent_treatment=('has_vent_treatment', 'mean'),
    pct_respcharting=('has_respcharting', 'mean'),
    pct_neuro_dx=('has_neuro_dx', 'mean'),
    pct_pasthistory=('has_pasthistory', 'mean'),
).reset_index()
for c in agg.columns:
    if c.startswith('pct_'):
        agg[c] = (agg[c] * 100).round(2)
agg = agg.sort_values(['n_records'], ascending=False)
agg.to_csv(QA / 'eicu_hospital_coverage.csv', index=False)

hosp_summary = {'n_hospitals': int(n_hosp)}
for c in agg.columns:
    if c.startswith('pct_'):
        hosp_summary[c] = {
            'median': float(agg[c].median()), 'min': float(agg[c].min()),
            'max': float(agg[c].max()),
            'n_hospitals_with_zero': int((agg[c] == 0).sum()),
            'n_hospitals_with_100': int((agg[c] == 100).sum()),
        }
hosp_summary['hospitals_with_no_gcs_at_all'] = int((agg.pct_gcs == 0).sum())
hosp_summary['records_in_hospitals_with_no_gcs'] = int(
    agg[agg.pct_gcs == 0].n_records.sum())
hosp_summary['hospitals_with_no_rass_at_all'] = int((agg.pct_rass == 0).sum())
hosp_summary['records_in_hospitals_with_no_rass'] = int(
    agg[agg.pct_rass == 0].n_records.sum())
hosp_summary['hospitals_with_no_airway_row_at_all'] = int((agg.pct_airway_state_row == 0).sum())
hosp_summary['records_in_hospitals_with_no_airway_row'] = int(
    agg[agg.pct_airway_state_row == 0].n_records.sum())
hosp_summary['note'] = ('若某医院整体贡献 0 条某类记录，应视为该院未提供该表，'
                        '不得推断该院患者未接受治疗。')
jdump(hosp_summary, 'eicu_hospital_coverage_summary.json')
print('    eICU 医院数:', n_hosp, flush=True)


# =====================================================================
# 7. 版本、SHA-256 与数据字典
# =====================================================================
print('[7/7] 版本与 SHA-256', flush=True)

src_files = [
    PROJ / '01_sepsis_cohort.sql', PROJ / '02_extract_hourly.sql',
    PROJ / '03_impute.py', PROJ / '05_assemble_training_data.py',
    PROJ / 'eicu_01_cohort.sql', PROJ / 'eicu_02_extract_hourly.sql',
    PROJ / 'eicu_04_assemble.py', PROJ / 'eicu_05_impute.py',
    PROJ / 'split_utils.py',
    PROJ / 'sepsis_labels.csv', PROJ / 'sepsis_hourly.csv',
    PROJ / 'sepsis_training_data.npz', PROJ / 'all_predictions.npz',
    PROJ / 'eicu_hourly.csv', PROJ / 'eicu_sofa.csv',
    PROJ / 'eicu_training_data.npz', PROJ / 'eicu_predictions.npz',
]
prov = {
    'databases': {
        'mimic': {'db': 'mimiciv31', 'server': 'PostgreSQL 12.18',
                  'access': 'read-only (SELECT/COPY TO STDOUT)',
                  'derived_tables_schema': 'mimiciv_derived'},
        'eicu': {'db': 'eicu', 'server': 'PostgreSQL 12.18',
                 'access': 'read-only (SELECT/COPY TO STDOUT)',
                 'schema': 'eicu_crd'},
    },
    'derived_sql_version': {
        'local_install_sql_saved': False,
        'searched': "项目内无 mimic-code 目录、无 gcs.sql 等派生脚本副本",
        'consequence': ('无法证明本地 mimiciv_derived.gcs 由哪个版本的 gcs.sql 生成；'
                        '因此本交付只报告派生结果的实际行为，不对版本作断言。'),
        'upstream_reference': ('MIT-LCP mimic-code gcs.sql（2026-09-13 访问），'
                               '仅作背景参照，不作为本地执行证据。'),
    },
    'scripts': {},
    'outputs': {},
    'sources': {},
    'environment': {
        'python': platform.python_version(),
        'numpy': np.__version__, 'pandas': pd.__version__,
    },
}
for p in sorted((HERE / 'sql').glob('*.sql')):
    prov['scripts'][p.name] = {'bytes': p.stat().st_size, 'sha256': sha256(p)}
for p in sorted(OUT.glob('*.csv')):
    prov['outputs'][p.name] = {'bytes': p.stat().st_size, 'sha256': sha256(p),
                              'rows': int(sum(1 for _ in open(p, encoding='utf-8')) - 1)}
for p in src_files:
    if p.exists():
        prov['sources'][p.name] = {'bytes': p.stat().st_size, 'sha256': sha256(p),
                                   'mtime': pd.Timestamp(p.stat().st_mtime, unit='s').isoformat()}
    else:
        prov['sources'][p.name] = {'missing': True}
for p in sorted(HERE.glob('*.sh')) + sorted(HERE.glob('*.py')):
    prov['scripts'][p.name] = {'bytes': p.stat().st_size, 'sha256': sha256(p)}
jdump(prov, 'provenance.json')

# 数据字典（从实际 CSV 列推导 + 人工说明）
DICT = [
    ('mimic_00_cohort_index', 'subject_id/hadm_id/stay_id', '患者/住院/ICU 入住标识', 'MIMIC-IV', 'integer'),
    ('mimic_00_cohort_index', 'intime', 't0：ICU 入室时刻', 'MIMIC-IV', 'timestamp'),
    ('mimic_00_cohort_index', 'window_end', '观察窗结束 = intime + 48h', 'MIMIC-IV', 'timestamp'),
    ('mimic_00_cohort_index', 't_inf_minus_intime_h', '疑似感染时间相对 t0 的小时数（可为负）', 'MIMIC-IV', 'float, hour'),
    ('mimic_00_cohort_index', 'infection_within_window', '疑似感染时间是否 <= t0+48h', 'MIMIC-IV', '0/1'),
    ('mimic_00_cohort_index', 'n_icu_stays', '同次住院内 ICU 入住次数（队列要求 <=2）', 'MIMIC-IV', 'integer'),
    ('mimic_00_cohort_index', 'rn', '同一 hadm_id 内按 intime 排序序号；正式队列取 rn=1', 'MIMIC-IV', 'integer'),
    ('mimic_01_gcs_event_raw', 'hours_from_intime', '(charttime - intime) 小时，保留 4 位小数', 'MIMIC-IV', 'float, hour'),
    ('mimic_01_gcs_event_raw', 'component', 'eyes/verbal/motor，由 itemid 映射', 'MIMIC-IV', 'text'),
    ('mimic_01_gcs_event_raw', 'value_raw', '床旁原始文本，未改写', 'MIMIC-IV', 'text'),
    ('mimic_01_gcs_event_raw', 'ett_flag', "verbal 文本为 'No Response-ETT' = 1（不可评估）", 'MIMIC-IV', '0/1'),
    ('mimic_01_gcs_event_raw', 'true_no_response_flag', "verbal 文本为 'No Response' = 1（无反应，非 ETT）", 'MIMIC-IV', '0/1'),
    ('mimic_01_gcs_event_raw', 'score_mapped', '按 GCS 标准文本→分值映射，ETT 记 0；不改原始列', 'MIMIC-IV', 'number'),
    ('mimic_01_gcs_event_raw', 'chart_to_store_lag_h', '(storetime - charttime) 小时，录入延迟', 'MIMIC-IV', 'float, hour'),
    ('mimic_02_gcs_derived', 'gcs_unable', '派生层无法评估标记（=1 与 ETT 文本事件对应）', 'MIMIC-IV', '0/1'),
    ('mimic_02_gcs_derived', 'total_minus_sum', '总分 - 三分项之和', 'MIMIC-IV', 'number'),
    ('mimic_02_gcs_derived', 'total15_with_sum_below15', '总分 15 但三分项之和 <15', 'MIMIC-IV', '0/1'),
    ('mimic_03_drug_inputevents', 'start_hours_from_intime', '给药开始相对 t0 小时', 'MIMIC-IV', 'float, hour'),
    ('mimic_03_drug_inputevents', 'administration_mode', "rateuom 含 min/hour 记为 infusion，否则 other_rate", 'MIMIC-IV', 'text'),
    ('mimic_03_drug_inputevents', 'overlaps_window', '该次给药区间是否与 [-6h,48h] 重叠', 'MIMIC-IV', '0/1'),
    ('mimic_04_drug_emar', 'event_category', 'given/not_given/infusion_event/pump_confirm/assessed_not_dose/unknown_reason', 'MIMIC-IV', 'text'),
    ('mimic_04b_drug_emar_detail', 'dose_given', '实际给予剂量（文本，单位另列）', 'MIMIC-IV', 'text'),
    ('mimic_04b_drug_emar_detail', 'complete_dose_not_given', '整剂未给标记', 'MIMIC-IV', 'text'),
    ('mimic_05_drug_prescriptions', 'window_relation', 'overlaps_window/open_ended/ended_before_window', 'MIMIC-IV', 'text'),
    ('mimic_05_drug_prescriptions', 'stoptime', '医嘱停止时间；常远晚于实际停药，不可当停药时刻', 'MIMIC-IV', 'timestamp'),
    ('mimic_06_ventilation_derived', 'ventilation_status', 'InvasiveVent/NonInvasiveVent/HFNC/SupplementalOxygen/Tracheostomy/None', 'MIMIC-IV', 'text'),
    ('mimic_06_ventilation_derived', 'overlaps_window', '通气时段是否与 [-6h,48h] 重叠', 'MIMIC-IV', '0/1'),
    ('mimic_07_airway_procedures', 'airway_event_type', 'intubation/extubation/tracheostomy/invasive_ventilation/noninvasive_ventilation', 'MIMIC-IV', 'text'),
    ('mimic_08_airway_chartevents', 'airway_field', 'airway_type/ventilator_mode/o2_delivery_device 等', 'MIMIC-IV', 'text'),
    ('mimic_09_rass_cam_icu', 'assessment_field', 'rass_actual/rass_goal/cam_icu_*/riker_sas/delirium_*', 'MIMIC-IV', 'text'),
    ('mimic_09_rass_cam_icu', 'rass_numeric', 'RASS 文本→数值（-5~+4）映射', 'MIMIC-IV', 'integer'),
    ('mimic_10_neuro_diagnoses', 'neuro_group', 'delirium_encephalopathy/cerebrovascular/... 见脚本注释', 'MIMIC-IV', 'text'),
    ('mimic_11_icd_procedures', 'airway_procedure_group', 'mechanical_ventilation/endotracheal_intubation/tracheostomy', 'MIMIC-IV', 'text'),
    ('eicu_00_cohort_index', 'patientunitstayid', 'ICU 入住级标识', 'eICU', 'integer'),
    ('eicu_00_cohort_index', 'patienthealthsystemstayid', '医院住院级标识', 'eICU', 'integer'),
    ('eicu_00_cohort_index', 'uniquepid', '患者级标识（同一患者可多次入住）', 'eICU', 'text'),
    ('eicu_00_cohort_index', 'unitdischargeoffset', 'ICU 出室相对入室的分钟数', 'eICU', 'integer, min'),
    ('eicu_00_cohort_index', 'hospitaldischargestatus', 'Alive/Expired；本队列已排除空字符串与 NULL', 'eICU', 'text'),
    ('eicu_01_gcs_nursecharting', 'nursingchartoffset', '相对 ICU 入室的分钟数（可为负）', 'eICU', 'integer, min'),
    ('eicu_01_gcs_nursecharting', 'value_state', 'numeric/text/empty_string/null 四态', 'eICU', 'text'),
    ('eicu_01_gcs_nursecharting', 'gcs_value_class', 'scored/unable_due_to_medication/blank/other_text', 'eICU', 'text'),
    ('eicu_02_rass_delirium_sedation', 'assessment_field', 'rass_actual/sedation_scale/delirium_*/pain_*', 'eICU', 'text'),
    ('eicu_03_sedation_infusiondrug', 'infusionoffset', '输注记录时刻（分钟）；本表无结束时间', 'eICU', 'integer, min'),
    ('eicu_03_sedation_infusiondrug', 'rate_unit_in_name', '从 drugname 尾部括号解析出的速率单位', 'eICU', 'text'),
    ('eicu_04_sedation_medication', 'drugorderoffset/drugstartoffset/drugstopoffset', '医嘱/开始/停止（分钟）', 'eICU', 'integer, min'),
    ('eicu_04_sedation_medication', 'lifecycle_completeness', 'order_only_no_start_recorded/started_no_stop_recorded/start_and_stop_recorded', 'eICU', 'text'),
    ('eicu_04_sedation_medication', 'drugordercancelled', '医嘱是否取消（可区分"开了未给"）', 'eICU', 'text'),
    ('eicu_05_airway_respiratorycare', 'airwaytype', '原始气道类型文本（含空字符串）', 'eICU', 'text'),
    ('eicu_05_airway_respiratorycare', 'airway_state', "not_recorded/no_artificial_airway/endotracheal_tube/tracheostomy/...", 'eICU', 'text'),
    ('eicu_05_airway_respiratorycare', 'ventstartoffset/ventendoffset', '本次通气时段（分钟）', 'eICU', 'integer, min'),
    ('eicu_06_airway_respiratorycharting', 'field_group', 'oxygen_delivery/endotracheal_tube/ventilator_mode/fio2/peep/...', 'eICU', 'text'),
    ('eicu_07_airway_treatment', 'airway_treatment_group', 'invasive_mechanical_ventilation/noninvasive_ventilation/cpap_peep/tracheostomy/...', 'eICU', 'text'),
    ('eicu_08_neuro_diagnoses', 'neuro_group', '与 MIMIC 同名归组；eICU 仅 ICD-9', 'eICU', 'text'),
    ('eicu_08_neuro_diagnoses', 'coding_source', 'icd9_available/diagnosisstring_only', 'eICU', 'text'),
    ('eicu_09_pasthistory', 'neuro_history_group', '既往史神经归组（文本匹配，可能漏检）', 'eICU', 'text'),
]
pd.DataFrame(DICT, columns=['file', 'field', 'description', 'database', 'unit_or_type']) \
    .to_csv(QA / 'data_dictionary.csv', index=False)

RULES = """# 时间与单位换算规则（DAG-GCS 数据准备）

## MIMIC-IV（库名 mimiciv31）
- 时间基准 t0 = `mimiciv_icu.icustays.intime`（ICU 入室时刻）。
- 观察窗 = `[t0 - 6h, t0 + 48h]`；主窗为 `[t0, t0+48h)`。
  - 向前多取 6h：仅用于观察窗口起点的记录惯性与"窗口开始时是否已在治疗中"。
  - 48h 为硬上界，与模型特征窗一致。
- `hours_from_intime = (event_time - t0) / 3600`（秒→小时），保留 4 位小数。
- 时长换算：`duration_h = (end - start) / 3600`。
- `chart_to_store_lag_h = (storetime - charttime) / 3600`；可为负（录入早于记录时刻，数据异常线索）。
- 结局时间：`dischtime_minus_intime_h = (dischtime - t0)/3600`；
  `dod`（死亡日期）为日期级，不含时刻，不能用于小时级先后判断。
- 长度单位：eICU 用 cm/kg，MIMIC 用 cm/kg，两库一致，未做换算。
- 剂量单位：不做跨库换算。MIMIC inputevents 的 `rateuom` 与 `amountuom` 原样保留；
  emar_detail 的 `dose_given_unit` 原样保留。
  ⚠️ 已知异构：`rateuom` 同时存在 mcg/kg/min、mg/hr、units/hr 等，
  跨药物求和前必须先按单位分池。

## eICU（库名 eicu）
- eICU 的偏移量一律为**分钟**，相对 ICU 入室（unit admit）：
  `nursingchartoffset`、`infusionoffset`、`respcarestatusoffset`、
  `respchartoffset`、`treatmentoffset`、`diagnosisoffset`、
  `pasthistoryoffset`、`drugorderoffset`/`drugstartoffset`/`drugstopoffset`。
- 换算：`hours_from_unit_admit = offset / 60`（保留 4 位小数）。
- 观察窗 = 偏移 `[-360, +2880]` 分钟，与 MIMIC 的 `[-6h, +48h]` 等价。
- ICU 停留时长：`unitdischargeoffset`（分钟）；队列要求 `> 2880`（即 >48h）。
- 医院级时间：`hospitaladmitoffset`、`hospitaldischargeoffset` 同为分钟偏移。
- 结局判定：`hospitaldischargestatus ∈ {'Alive','Expired'}`。
  ⚠️ 该字段存在**空字符串**（非 NULL）。原 `eicu_01_cohort.sql` 用 `IS NOT NULL`
  放行了空字符串记录，导致 SQL 直跑 6,883 条 vs 正式队列 6,812 条（差 71 条，
  均为非死亡）。本次已改为显式枚举并在索引脚本中记录。
- eICU 无年月日绝对时间可用（`*time24` 为字符串、年份仅到 `hospitaldischargeyear`），
  因此两库之间的绝对时间不可直接对齐，只能各自相对 t0 计算。

## 药物名称的单位耦合（eICU 特有）
- `infusiondrug.drugname` 把药物与速率单位写在一起，如
  `Propofol (mcg/kg/min)`、`Fentanyl (ml/hr)`、`Cisatracurium IVF Infused (ml/hr)`。
- 单位由 `rate_unit_in_name` 从名称尾部括号解析；存在 `Propofol ()` 这类空括号。
- 已发现的数据内拼写变体（靠正则兜住，不修改原值）：
  `Dexmetetomidine (Precedex)`（少一个 d）、`Diprivan`（丙泊酚商品名）、
  `Dilaudid`（氢吗啡酮商品名）。

## MIMIC 药物名称的大小写变体
- `prescriptions.drug` 与 `emar.medication` 使用混合大小写，例如
  `HYDROmorphone (Dilaudid)`、`LORazepam`、`OxycoDONE`，且存在**尾部空格**变体。
- 匹配用大小写不敏感正则，并在输出中同时保留 `medication_raw` 与 `medication_clean`（btrim 后）。
- 复合制剂以 `|` 连接（如 `Bupivacaine 0.1%|HYDROmorphone (Dilaudid)`），未拆分。

## 时间窗重叠规则
- 一律使用**重叠**而非包含：`start <= win_end AND end >= win_start`。
  否则跨越窗口边界的持续治疗会被丢弃，导致"评分时点是否在治疗中"无法判断。
- `endtime` 为 NULL 时，按 `COALESCE(endtime, starttime)` 处理窗口重叠，
  但原始 NULL 在输出中保留（NULL 表示未记录结束，不等于瞬时事件）。
"""
(QA / 'time_unit_rules.md').write_text(RULES, encoding='utf-8')

print('全部完成。输出目录:', QA, flush=True)
