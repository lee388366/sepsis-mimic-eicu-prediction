#!/usr/bin/env python3
# =====================================================================
# qa_assessability.py —— GCS 评分时点可判定性链接分析（委托第 4 节·第 7 问）
#
# 回答的问题：
#   现在能否判断"每一次 GCS 评分时点"的
#     ① 镇静暴露、② 呼吸支持状态、③ 该次评分是否可评估？
#   若只能部分判断，范围是什么？
#
# 方法（只做区间/邻近链接，不建模、不推断缺失为阴性）：
#   - 区间重叠用 np.searchsorted "着色"：对每个区间 [start,end) 找出落在其中的
#     评分时点。复杂度 O(n log m)，等价于 SQL 的重叠逻辑，但可在本地复核。
#   - 点事件（eMAR 给药、eICU 输注记录、RASS）用"最近邻时间差"描述，
#     不把"最近一条记录很远"当作"当时未使用"。
#
# 输出：
#   qa/assessability_linkage.json     汇总（无患者级明细）
#   qa/assessability_by_hour.csv      按观察窗小时分层的可判定性覆盖
# =====================================================================
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'output'
QA = HERE / 'qa'
QA.mkdir(exist_ok=True)


def rd(name, **kw):
    kw.setdefault('low_memory', False)
    return pd.read_csv(OUT / f'{name}.csv', **kw)


def paint_interval(query_ids, query_t, iv_ids, iv_start, iv_end, value=True):
    """把区间 [start,end) 的 value 涂到落在其中的评分时点上。

    query_ids/query_t: 评分时点（同一患者可多行）
    iv_*: 区间表
    返回与 query_ids 等长的 object 数组，未命中为 None。
    """
    res = np.array([None] * len(query_t), dtype=object)
    if len(iv_ids) == 0 or len(query_t) == 0:
        return res
    order = np.argsort(query_ids, kind='stable')
    qid_sorted = np.asarray(query_ids)[order]
    qt_sorted = np.asarray(query_t, dtype=float)[order]
    # 每个患者在该排序数组中的切片
    uniq, starts_idx = np.unique(qid_sorted, return_index=True)
    ends_idx = np.append(starts_idx[1:], len(qid_sorted))
    span = {u: (s, e) for u, s, e in zip(uniq, starts_idx, ends_idx)}

    iv = pd.DataFrame({'id': iv_ids, 's': np.asarray(iv_start, dtype=float),
                       'e': np.asarray(iv_end, dtype=float), 'v': value})
    iv = iv.dropna(subset=['s'])
    for pid, g in iv.groupby('id'):
        if pid not in span:
            continue
        a, b = span[pid]
        ts = qt_sorted[a:b]
        for s, e, v in zip(g.s.values, g.e.values, g.v.values):
            lo = np.searchsorted(ts, s, side='left')
            hi = np.searchsorted(ts, e, side='right')
            if hi > lo:
                res[order[a + lo:a + hi]] = v
    return res


def nearest_gap_hours(query_ids, query_t, ev_ids, ev_t):
    """每个评分时点到"最近一次事件"的绝对时间差（小时）；无事件者为 NaN。"""
    out = np.full(len(query_t), np.nan)
    dfq = pd.DataFrame({'id': query_ids, 't': np.asarray(query_t, dtype=float)})
    dfe = pd.DataFrame({'id': ev_ids, 't': np.asarray(ev_t, dtype=float)}).dropna()
    if len(dfe) == 0:
        return out
    for pid, gq in dfq.groupby('id'):
        ge = dfe[dfe.id == pid]
        if len(ge) == 0:
            continue
        te = np.sort(ge.t.values)
        pos = np.searchsorted(te, gq.t.values)
        left = np.where(pos > 0, gq.t.values - te[np.clip(pos - 1, 0, len(te) - 1)], np.inf)
        right = np.where(pos < len(te), te[np.clip(pos, 0, len(te) - 1)] - gq.t.values, np.inf)
        out[gq.index.values] = np.minimum(np.abs(left), np.abs(right)) / 60.0
    return out


def prior_gap_hours(query_ids, query_t, ev_ids, ev_t):
    """每个评分时点到"之前最近一次事件"的时间差（小时）；之前无事件为 NaN。"""
    out = np.full(len(query_t), np.nan)
    dfq = pd.DataFrame({'id': query_ids, 't': np.asarray(query_t, dtype=float)})
    dfe = pd.DataFrame({'id': ev_ids, 't': np.asarray(ev_t, dtype=float)}).dropna()
    if len(dfe) == 0:
        return out
    for pid, gq in dfq.groupby('id'):
        ge = dfe[dfe.id == pid]
        if len(ge) == 0:
            continue
        te = np.sort(ge.t.values)
        pos = np.searchsorted(te, gq.t.values, side='right')
        ok = pos > 0
        vals = np.where(ok, gq.t.values - te[np.clip(pos - 1, 0, len(te) - 1)], np.nan)
        out[gq.index.values] = np.where(ok, vals / 60.0, np.nan)
    return out


R = {}

# =====================================================================
# MIMIC 侧
# =====================================================================
print('MIMIC: 构造评分时点', flush=True)
gd = rd('mimic_02_gcs_derived', usecols=['stay_id', 'hours_from_intime', 'gcs_total',
                                         'gcs_eyes', 'gcs_verbal', 'gcs_motor',
                                         'gcs_unable', 'unable_flag', 'verbal_zero_flag'])
gd = gd[(gd.hours_from_intime >= 0) & (gd.hours_from_intime <= 48)].copy()
qid = gd.stay_id.values
qt = (gd.hours_from_intime * 60.0).values           # 统一用分钟作为内部时间轴

ie = rd('mimic_03_drug_inputevents')
ie['s_min'] = ie.start_hours_from_intime * 60.0
ie['e_min'] = ie.end_hours_from_intime * 60.0
sed = ie[ie.drug_class == 'sedative_hypnotic']
opd = ie[ie.drug_class == 'opioid']
nmb = ie[ie.drug_class == 'neuromuscular_blocker']

sed_active = paint_interval(qid, qt, sed.stay_id.values, sed.s_min.values, sed.e_min.values, 'yes')
opd_active = paint_interval(qid, qt, opd.stay_id.values, opd.s_min.values, opd.e_min.values, 'yes')
nmb_active = paint_interval(qid, qt, nmb.stay_id.values, nmb.s_min.values, nmb.e_min.values, 'yes')

# eMAR 实际给药（点事件）与医嘱（区间）
em = rd('mimic_04_drug_emar')
em_given_sed = em[(em.event_category == 'given') & (em.drug_class == 'sedative_hypnotic')]
em_given_opd = em[(em.event_category == 'given') & (em.drug_class == 'opioid')]
em['s_min'] = em.chart_hours_from_intime * 60.0
em_ord_sed = em[em.drug_class == 'sedative_hypnotic']

ven = rd('mimic_06_ventilation_derived')
ven['s_min'] = ven.start_hours_from_intime * 60.0
ven['e_min'] = ven.end_hours_from_intime * 60.0
vent_status = paint_interval(qid, qt, ven.stay_id.values, ven.s_min.values,
                             ven.e_min.values, ven.ventilation_status.values)

ras = rd('mimic_09_rass_cam_icu', usecols=['stay_id', 'hours_from_intime', 'assessment_field'])
rass = ras[ras.assessment_field == 'rass_actual']
rass_gap_h = nearest_gap_hours(qid, qt, rass.stay_id.values, (rass.hours_from_intime * 60).values)
delir = ras[ras.assessment_field.isin(
    ['delirium_assessment', 'cam_icu_altered_loc', 'cam_icu_inattention',
     'cam_icu_disorganized_thinking', 'cam_icu_ms_change'])]
delir_gap_h = nearest_gap_hours(qid, qt, delir.stay_id.values, (delir.hours_from_intime * 60).values)

n_q = len(gd)
has_sed_data = pd.Series(qid).isin(set(ie[ie.drug_class.isin(
    ['sedative_hypnotic', 'opioid'])].stay_id)).values
has_vent_data = pd.Series(qid).isin(set(ven.stay_id)).values
has_rass_data = pd.Series(qid).isin(set(rass.stay_id)).values
has_delir_data = pd.Series(qid).isin(set(delir.stay_id)).values

m = {
    'n_gcs_scoring_timestamps_0_48h': int(n_q),
    'n_records': int(pd.Series(qid).nunique()),
    'assessability': {
        'unable_flag_eq_1': int((gd.unable_flag == 1).sum()),
        'unable_flag_eq_1_pct': round(100 * (gd.unable_flag == 1).mean(), 4),
        'verbal_zero_flag_eq_1': int((gd.verbal_zero_flag == 1).sum()),
        'verbal_zero_flag_eq_1_pct': round(100 * (gd.verbal_zero_flag == 1).mean(), 4),
        'can_label_assessability': True,
        'evidence': "mimiciv_derived.gcs.gcs_unable 与该次评分同键，直接给出可评估性",
    },
    'sedation_exposure': {
        'structural_data_available_records': int(pd.Series(qid)[has_sed_data].nunique()),
        'structural_data_available_pct_of_records': round(
            100 * pd.Series(qid)[has_sed_data].nunique() / pd.Series(qid).nunique(), 4),
        'timestamps_with_sedative_infusion_active': int((sed_active == 'yes').sum()),
        'timestamps_with_sedative_infusion_active_pct': round(
            100 * (sed_active == 'yes').mean(), 4),
        'timestamps_with_opioid_infusion_active': int((opd_active == 'yes').sum()),
        'timestamps_with_opioid_infusion_active_pct': round(
            100 * (opd_active == 'yes').mean(), 4),
        'timestamps_with_nmb_infusion_active': int((nmb_active == 'yes').sum()),
        'timestamps_with_nmb_infusion_active_pct': round(
            100 * (nmb_active == 'yes').mean(), 4),
        'nearest_emar_given_sedative_gap_h': {
            'median': round(float(np.nanmedian(
                nearest_gap_hours(qid, qt, em_given_sed.stay_id.values,
                                  em_given_sed.chart_hours_from_intime.values * 60))), 4),
            'pct_within_1h': round(100 * np.nanmean(
                nearest_gap_hours(qid, qt, em_given_sed.stay_id.values,
                                  em_given_sed.chart_hours_from_intime.values * 60) <= 1), 4),
        },
        'determinable': 'partial',
        'reason': ('区间法（inputevents 有 start/end）可判定"某时点是否在输注中"；'
                   '但"当时无镇静"无法由无记录推出，故只能判定阳性暴露，不能判定阴性。'),
    },
    'respiratory_support': {
        'structural_data_available_records': int(pd.Series(qid)[has_vent_data].nunique()),
        'structural_data_available_pct_of_records': round(
            100 * pd.Series(qid)[has_vent_data].nunique() / pd.Series(qid).nunique(), 4),
        'status_at_timestamp_counts': {
            str(k): int(v) for k, v in
            pd.Series(vent_status).value_counts(dropna=False).items()},
        'timestamps_covered_by_any_vent_interval_pct': round(
            100 * (pd.Series(vent_status).notna()).mean(), 4),
        'determinable': 'yes_for_positive',
        'reason': ('mimiciv_derived.ventilation 给出显式起止时段，可判定"某时点是否在'
                   '有创/无创/高流量/氧疗中"。未覆盖的时点不能判定为"未通气"，'
                   '因为该派生表可能有遗漏，且窗口外的通气不在表内。'),
    },
    'auxiliary_assessments': {
        'rass_available_records': int(pd.Series(qid)[has_rass_data].nunique()),
        'rass_available_records_pct': round(
            100 * pd.Series(qid)[has_rass_data].nunique() / pd.Series(qid).nunique(), 4),
        'timestamps_with_rass_within_1h_pct': round(100 * np.nanmean(rass_gap_h <= 1), 4),
        'timestamps_with_rass_within_2h_pct': round(100 * np.nanmean(rass_gap_h <= 2), 4),
        'rass_gap_h_median': round(float(np.nanmedian(rass_gap_h)), 4),
        'delirium_assessment_records': int(pd.Series(qid)[has_delir_data].nunique()),
        'timestamps_with_delirium_assessment_within_2h_pct': round(
            100 * np.nanmean(delir_gap_h <= 2), 4),
    },
}

# 交叉表：不可评估 × 镇静中 × 通气中
m['cross_tab_unable_by_sedation_and_vent'] = {
    'unable_and_sedative_active': int(((gd.unable_flag == 1) & (sed_active == 'yes')).sum()),
    'unable_and_opioid_active': int(((gd.unable_flag == 1) & (opd_active == 'yes')).sum()),
    'unable_and_nmb_active': int(((gd.unable_flag == 1) & (nmb_active == 'yes')).sum()),
    'unable_and_invasive_vent': int(((gd.unable_flag == 1) &
                                     (pd.Series(vent_status) == 'InvasiveVent').values).sum()),
    'unable_and_any_vent': int(((gd.unable_flag == 1) &
                                pd.Series(vent_status).notna().values).sum()),
    'unable_and_no_sedative_no_nmb_and_invasive_vent': int(
        ((gd.unable_flag == 1) & (sed_active != 'yes') & (nmb_active != 'yes') &
         (pd.Series(vent_status) == 'InvasiveVent').values).sum()),
    'no_vent_data_at_unable_timestamps': int(
        ((gd.unable_flag == 1) & pd.Series(vent_status).isna().values).sum()),
    'interpretation': ('gcs_unable 在 MIMIC 中只由 ETT 文本触发，因此它与"有创通气"'
                       '高度共现，但不能证明当时正在通气，也不能证明当时有镇静暴露。'),
}
R['mimic'] = m
print('  MIMIC 评分时点:', n_q, flush=True)

# =====================================================================
# eICU 侧
# =====================================================================
print('eICU: 构造评分时点', flush=True)
eg = rd('eicu_01_gcs_nursecharting', usecols=['patientunitstayid', 'nursingchartoffset',
                                              'field_label', 'field_name', 'value_numeric',
                                              'gcs_value_class'])
core = eg[eg.field_label == 'Glasgow coma score'].copy()
core = core[core.nursingchartoffset.between(0, 2880)]
eq_keys = core[['patientunitstayid', 'nursingchartoffset']].drop_duplicates()
eqid = eq_keys.patientunitstayid.values
eqt = eq_keys.nursingchartoffset.values.astype(float)

# 该时点是否为 unable 标记
un = core[core.gcs_value_class == 'unable_due_to_medication'][
    ['patientunitstayid', 'nursingchartoffset']].drop_duplicates()
un_keys = set(map(tuple, un.values))
is_unable = np.array([k in un_keys for k in map(tuple, eq_keys.values)])

# 结构完整性：该时点有几个分项可用
comp_cnt = core[core.value_numeric.notna()].groupby(
    ['patientunitstayid', 'nursingchartoffset']).size()
eq_keys = eq_keys.assign(n_components=[int(comp_cnt.get(tuple(k), 0))
                                       for k in eq_keys.values])
n3 = (eq_keys.n_components >= 3).values

# eICU 镇静：输注记录（无结束时间 → 只能用"之前最近一次记录"的间隔）
ei = rd('eicu_03_sedation_infusiondrug')
ei_sed = ei[ei.drug_class == 'sedative_hypnotic']
ei_opd = ei[ei.drug_class == 'opioid']
ei_nmb = ei[ei.drug_class == 'neuromuscular_blocker']
sed_prior_h = prior_gap_hours(eqid, eqt, ei_sed.patientunitstayid.values,
                              ei_sed.infusionoffset.values)
opd_prior_h = prior_gap_hours(eqid, eqt, ei_opd.patientunitstayid.values,
                              ei_opd.infusionoffset.values)
nmb_prior_h = prior_gap_hours(eqid, eqt, ei_nmb.patientunitstayid.values,
                              ei_nmb.infusionoffset.values)

# eICU 医嘱层可给出区间（有 stop 的）
emd = rd('eicu_04_sedation_medication')
emd_sed = emd[(emd.drug_class == 'sedative_hypnotic') & emd.drugstartoffset.notna()].copy()
# 开放性医嘱（无 stop）单独标记，不擅自假设何时结束
emd_sed_closed = emd_sed[emd_sed.drugstopoffset.notna()]
emd_sed_open = emd_sed[emd_sed.drugstopoffset.isna()]
sed_interval_active = paint_interval(
    eqid, eqt, emd_sed_closed.patientunitstayid.values,
    emd_sed_closed.drugstartoffset.values, emd_sed_closed.drugstopoffset.values, 'yes')
sed_open_started = paint_interval(
    eqid, eqt, emd_sed_open.patientunitstayid.values,
    emd_sed_open.drugstartoffset.values,
    np.full(len(emd_sed_open), 1e9), 'open_ended_started_not_stopped')

# eICU 呼吸支持
# 【关键修正】respiratorycare 的 ventendoffset 是零填充（队列内 100% 为 0），
# 因此**无法**构造通气区间，也就无法判定"某评分时点是否在通气中"。
# 这里只使用两类真实可用的信号，并明确其仅为住院级：
#   ① airway_state（气道类型，瞬时记录）
#   ② ventstartoffset 的符号（<0 = 入室前已在通气；>0 = 入室后某刻开始）
#   ③ treatment 表的机械通气治疗记录时刻（时点，无结束时间）
erc = rd('eicu_05_airway_respiratorycare')
airway_iv = paint_interval(eqid, eqt, erc.patientunitstayid.values,
                           erc.respcarestatusoffset.values,
                           erc.respcarestatusoffset.values + 1e-9,
                           erc.airway_state.values)  # 状态记录为瞬时点
erc_vent_signal = erc[erc.any_vent_signal == 1]
etc = rd('eicu_07_airway_treatment')
etc_mech = etc[etc.airway_treatment_group == 'invasive_mechanical_ventilation']
mech_gap_h = nearest_gap_hours(eqid, eqt, etc_mech.patientunitstayid.values,
                               etc_mech.treatmentoffset.values)

# eICU RASS
eras = rd('eicu_02_rass_delirium_sedation', usecols=['patientunitstayid',
                                                     'nursingchartoffset', 'assessment_field'])
eras_r = eras[eras.assessment_field == 'rass_actual']
rass_gap_h_e = nearest_gap_hours(eqid, eqt, eras_r.patientunitstayid.values,
                                 eras_r.nursingchartoffset.values)

n_qe = len(eq_keys)
has_sed_e = pd.Series(eqid).isin(set(ei[ei.drug_class.isin(
    ['sedative_hypnotic', 'opioid'])].patientunitstayid)).values
has_vent_e = pd.Series(eqid).isin(set(erc.patientunitstayid)).values
has_rass_e = pd.Series(eqid).isin(set(eras_r.patientunitstayid)).values

e = {
    'n_gcs_scoring_timestamps_0_48h': int(n_qe),
    'n_records': int(pd.Series(eqid).nunique()),
    'assessability': {
        'unable_due_to_medication_timestamps': int(is_unable.sum()),
        'unable_due_to_medication_pct': round(100 * is_unable.mean(), 4),
        'timestamps_with_all_three_components': int(n3.sum()),
        'timestamps_with_all_three_components_pct': round(100 * n3.mean(), 4),
        'timestamps_with_zero_components': int((eq_keys.n_components == 0).sum()),
        'can_label_assessability': 'partial',
        'reason': ("eICU 只有一个显式不可评估取值 'Unable to score due to medication'，"
                   "没有 MIMIC 那样的 ETT 语言标记。因此'因插管不能评估'在 eICU 里"
                   "无法与'真实语言反应差'区分；只能识别因药物无法评估。"),
    },
    'sedation_exposure': {
        'structural_data_available_records': int(pd.Series(eqid)[has_sed_e].nunique()),
        'structural_data_available_pct_of_records': round(
            100 * pd.Series(eqid)[has_sed_e].nunique() / pd.Series(eqid).nunique(), 4),
        'record_level_only_no_end_time': True,
        'minutes_since_prior_sedative_infusion_record': {
            'pct_within_60min': round(100 * np.nanmean(sed_prior_h <= 1), 4),
            'pct_within_120min': round(100 * np.nanmean(sed_prior_h <= 2), 4),
            'pct_within_240min': round(100 * np.nanmean(sed_prior_h <= 4), 4),
            'pct_within_480min': round(100 * np.nanmean(sed_prior_h <= 8), 4),
            'median_h_when_exists': round(float(np.nanmedian(sed_prior_h)), 4),
        },
        'minutes_since_prior_opioid_infusion_record_within_120min_pct': round(
            100 * np.nanmean(opd_prior_h <= 2), 4),
        'minutes_since_prior_nmb_infusion_record_within_120min_pct': round(
            100 * np.nanmean(nmb_prior_h <= 2), 4),
        'medication_interval_active_timestamps': int((sed_interval_active == 'yes').sum()),
        'medication_interval_active_pct': round(
            100 * (sed_interval_active == 'yes').mean(), 4),
        'open_ended_order_started_timestamps': int(
            (sed_open_started == 'open_ended_started_not_stopped').sum()),
        'lifecycle_completeness': {str(k): int(v) for k, v in
                                   emd.lifecycle_completeness.value_counts().items()},
        'determinable': 'partial_weak',
        'reason': ('infusiondrug 无结束时间，只能给出"之前最近一次输注记录的间隔"，'
                   '不能界定"当时是否仍在输注"；medication 层有 open/closed 医嘱，'
                   '仅 closed 医嘱可构成确定区间。'),
    },
    'respiratory_support': {
        'structural_data_available_records': int(pd.Series(eqid)[has_vent_e].nunique()),
        'structural_data_available_pct_of_records': round(
            100 * pd.Series(eqid)[has_vent_e].nunique() / pd.Series(eqid).nunique(), 4),
        'records_with_any_vent_signal': int(erc_vent_signal.patientunitstayid.nunique()),
        'records_with_artificial_airway_recorded': int(
            erc[erc.artificial_airway_flag == 1].patientunitstayid.nunique()),
        'timestamps_with_airway_state_pct': round(
            100 * pd.Series(airway_iv).notna().mean(), 4),
        'airway_state_at_timestamp_counts': {
            str(k): int(v) for k, v in
            pd.Series(airway_iv).value_counts(dropna=False).items()},
        'timestamps_with_mech_vent_treatment_within_2h_pct': round(
            100 * np.nanmean(mech_gap_h <= 2), 4),
        'can_determine_vent_status_at_timestamp': False,
        'determinable': 'no_timestamp_level_assessment',
        'reason': ("respiratorycare 的 ventendoffset 为零填充（队列内 20,566 行中 100% 为 0，"
                   "全库 865,381 行中 99.98% 为 0，仅 157 行为正值），"
                   "因此 eICU 不提供通气结束时间，也无法构造任何有效通气区间"
                   "（有效区间记录数 = 0）。可用的仅是住院级信号："
                   "入室前已在通气、入室后某刻开始通气、既往通气史，"
                   "以及 airwaytype 与 treatment 的时点记录。"),
        'note': ('本项与 MIMIC 不可比：MIMIC 的派生 ventilation 表给出完整起止时段，'
                 'eICU 完全没有对应结构。任何"两库都逐时校正了呼吸支持"的说法不成立。'),
    },
    'auxiliary_assessments': {
        'rass_available_records': int(pd.Series(eqid)[has_rass_e].nunique()),
        'rass_available_records_pct': round(
            100 * pd.Series(eqid)[has_rass_e].nunique() / pd.Series(eqid).nunique(), 4),
        'timestamps_with_rass_within_1h_pct': round(100 * np.nanmean(rass_gap_h_e <= 1), 4),
        'timestamps_with_rass_within_2h_pct': round(100 * np.nanmean(rass_gap_h_e <= 2), 4),
        'rass_gap_h_median': round(float(np.nanmedian(rass_gap_h_e)), 4),
    },
}
e['cross_tab_unable_by_sedation_and_vent'] = {
    'unable_and_sedative_record_within_120min': int(
        (is_unable & (sed_prior_h <= 2)).sum()),
    'unable_without_any_prior_sedative_record': int(
        (is_unable & np.isnan(sed_prior_h)).sum()),
    'unable_and_artificial_airway_recorded_at_timestamp': int(
        (is_unable & pd.Series(airway_iv).isin(
            ['endotracheal_tube', 'tracheostomy', 'cricothyrotomy',
             'double_lumen_tube']).values).sum()),
    'interpretation': ("eICU 的 'Unable to score due to medication' 明确指向药物；"
                       "但只有 153 条记录出现该标记，因此它远不能覆盖全部镇静状态下的评分。"),
}
R['eicu'] = e
print('  eICU 评分时点:', n_qe, flush=True)

# =====================================================================
# 第 7 问的直接回答
# =====================================================================
R['question_7_answer'] = {
    'q': '现在能否判断每次 GCS 评分时点的镇静暴露、呼吸支持状态及可评估性？',
    'a': '只能部分判断，且两库范围不同。',
    'mimic': {
        '可评估性': '可直接判定（gcs_unable 与评分同键；ETT 触发的不可评估可识别）。'
                    f"48h 内 {round(100*(gd.unable_flag==1).mean(),2)}% 的评分时点标记为不可评估。",
        '镇静暴露': '只能判定阳性（是否正处于输注区间内）；无记录不能判为未使用。'
                    f"结构性数据覆盖 {round(100*pd.Series(qid)[has_sed_data].nunique()/pd.Series(qid).nunique(),2)}% 的住院。",
        '呼吸支持状态': '可判定阳性（落在派生通气时段内）；'
                        f"覆盖率 {round(100*pd.Series(vent_status).notna().mean(),2)}% 的评分时点。"
                        '未覆盖时点不能判为未通气。',
    },
    'eicu': {
        '可评估性': "仅能识别'因药物无法评估'（占 "
                    f"{round(100*is_unable.mean(),2)}% 的评分时点，涉及 "
                    f"{int(un.patientunitstayid.nunique())} 条记录）；"
                    "无法识别因插管不能评估语言，因此与'真实语言反应差'不可区分。",
        '镇静暴露': '弱于 MIMIC：输注记录无结束时间，只能给出"之前最近一条记录的间隔"；'
                    '仅 closed 医嘱可构成确定区间。',
        '呼吸支持状态': '不可按评分时点判定：无通气结束时间（ventendoffset 100% 为零填充），'
                        '有效通气区间为 0。仅有住院级信号（入室前已在通气 1,571 例、'
                        '入室后开始 505 例、既往通气史 412 例）与气道类型、治疗时点记录。',
    },
    '底线': ('两库都不具备"逐时准确的镇静剂量/通气状态"证据，且缺口方向不同：'
             'MIMIC 有通气时段但无镇静区间外的阴性证据；eICU 连通气区间都没有。'
             '因此任何"GCS 低是因为镇静"或"GCS 低反映神经功能"的因果解释'
             '都超出当前数据可支持范围。可支持的是：记录规则（ETT 编码）'
             '在 MIMIC 中系统性改变了 GCS 总分与语言分项的数值，'
             '且该规则在 eICU 不存在，构成跨库测量不等价。'),
}
(QA / 'assessability_linkage.json').write_text(
    json.dumps(R, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

# 按小时分层的可判定性覆盖
by_hour = pd.DataFrame({
    'hour': np.floor(qt / 60.0).astype(int),
    'has_sedative_active': (sed_active == 'yes').astype(int),
    'has_vent_status': pd.Series(vent_status).notna().astype(int).values,
    'unable': (gd.unable_flag == 1).astype(int).values,
    'is_invasive_vent': (pd.Series(vent_status) == 'InvasiveVent').astype(int).values,
}).groupby('hour').agg(
    n_timestamps=('unable', 'size'),
    pct_unable=('unable', lambda s: round(100 * s.mean(), 3)),
    pct_sedative_active=('has_sedative_active', lambda s: round(100 * s.mean(), 3)),
    pct_vent_status_known=('has_vent_status', lambda s: round(100 * s.mean(), 3)),
    pct_invasive_vent=('is_invasive_vent', lambda s: round(100 * s.mean(), 3)),
).reset_index()
by_hour.to_csv(QA / 'assessability_by_hour.csv', index=False)

print()
print('=== 第 7 问回答摘要 ===')
print(json.dumps(R['question_7_answer'], ensure_ascii=False, indent=2))
