#!/usr/bin/env python3
# =====================================================================
# qa_base_table.py —— 底表质量核查
#
# 覆盖任务要求的五项：
#   Q1 队列/结局一致性
#   Q2 键唯一性
#   Q3 各字段覆盖
#   Q4 时间差分布
#   Q5 是否误用了未来信息（含 merge_asof 独立复算，不只看自算 gap）
#
# 只读 base/mimic_gcs_timebase.csv 与上游交付；输出 qa/。
# =====================================================================
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REV = HERE.parent
UP = REV / 'dag_gcs_next_stage' / 'output'
BASE = HERE / 'base' / 'mimic_gcs_timebase.csv'
QA = HERE / 'qa'
QA.mkdir(parents=True, exist_ok=True)

F: dict = {}
T: dict[str, pd.DataFrame] = {}


def tab(name: str, df: pd.DataFrame) -> None:
    T[name] = df
    print(f'\n### {name}')
    print(df.to_string(index=False))


def main() -> int:
    t = pd.read_csv(BASE, low_memory=False)
    coh = pd.read_csv(UP / 'mimic_00_cohort_index.csv', low_memory=False)
    t['gcs_charttime'] = pd.to_datetime(t.gcs_charttime)

    # =================================================================
    # Q1 队列 / 结局一致性
    # =================================================================
    F['q1_rows'] = int(len(t))
    F['q1_unique_stay'] = int(t.stay_id.nunique())
    F['q1_unique_subject'] = int(t.subject_id.nunique())
    F['q1_cohort_records'] = int(len(coh))
    F['q1_cohort_deaths'] = int(coh.label.sum())
    F['q1_extra_stay_not_in_cohort'] = int(len(set(t.stay_id) - set(coh.stay_id)))
    F['q1_stay_without_any_gcs_in_window'] = int(len(set(coh.stay_id) - set(t.stay_id)))
    # 结局一致性：底表 label 必须与队列索引逐 stay 一致
    chk = t[['stay_id', 'label_hospital_mortality']].drop_duplicates().merge(
        coh[['stay_id', 'label']], on='stay_id', how='left')
    F['q1_label_mismatch'] = int((chk.label_hospital_mortality != chk.label).sum())
    # 时间窗：全部必须落在 [0,48)
    F['q1_time_h_min'] = float(t.gcs_time_h.min())
    F['q1_time_h_max'] = float(t.gcs_time_h.max())
    F['q1_rows_outside_window'] = int(((t.gcs_time_h < 0) | (t.gcs_time_h >= 48)).sum())
    tab('Q1 队列与结局一致性', pd.DataFrame([
        {'项目': '底表行数（GCS 评分时点）', '值': F['q1_rows']},
        {'项目': '涉及住院记录数', '值': F['q1_unique_stay']},
        {'项目': '涉及受试者数', '值': F['q1_unique_subject']},
        {'项目': '正式队列记录数', '值': F['q1_cohort_records']},
        {'项目': '队列内院内死亡数', '值': F['q1_cohort_deaths']},
        {'项目': '底表 label 与队列索引不一致数', '值': F['q1_label_mismatch']},
        {'项目': '落在 [0,48h) 之外的行数', '值': F['q1_rows_outside_window']},
        {'项目': '窗内无任何 GCS 记录的队列记录数（未静默排除，仅缺失）',
         '值': F['q1_stay_without_any_gcs_in_window']},
    ]))

    # =================================================================
    # Q2 键唯一性
    # =================================================================
    F['q2_row_key_dup'] = int(t.row_key.duplicated().sum())
    F['q2_stay_time_dup'] = int(t.duplicated(subset=['stay_id', 'gcs_charttime']).sum())
    F['q2_stay_seq_dup'] = int(t.duplicated(subset=['stay_id', 'gcs_seq']).sum())
    tab('Q2 键唯一性', pd.DataFrame([
        {'键': 'row_key (stay_id|charttime)', '重复数': F['q2_row_key_dup']},
        {'键': '(stay_id, gcs_charttime)', '重复数': F['q2_stay_time_dup']},
        {'键': '(stay_id, gcs_seq)', '重复数': F['q2_stay_seq_dup']},
    ]))

    # =================================================================
    # Q3 字段覆盖
    # =================================================================
    cov = pd.DataFrame({
        'column': t.columns,
        'non_null_n': [int(t[c].notna().sum()) for c in t.columns],
    })
    cov['non_null_pct'] = (cov.non_null_n / len(t) * 100).round(2)
    cov['group'] = cov.column.str.split('_').str[0]
    T['coverage'] = cov

    key_cov = []
    for c, label in [
        ('gcs_eyes_recorded', 'GCS 眼分项（派生层原样）'),
        ('gcs_motor_recorded', 'GCS 运动分项（派生层原样）'),
        ('gcs_verbal_recorded', 'GCS 语言分项（派生层原样）'),
        ('gcs_total_recorded', 'GCS 总分（派生层原样）'),
        ('rass_rass_observed', '同/既往最近一次观测 RASS'),
        ('rass_status', 'RASS 判定状态（含 unknown）'),
        ('sedative_iv_active', '镇静：该时点处于输注区间内'),
        ('sedative_emar_admin_prior', '镇静：既往可确认给药(eMAR)'),
        ('opioid_iv_active', '阿片：该时点处于输注区间内'),
        ('opioid_emar_admin_prior', '阿片：既往可确认给药(eMAR)'),
        ('nmb_iv_active', '肌松：该时点处于输注区间内'),
        ('nmb_emar_admin_prior', '肌松：既往可确认给药(eMAR)'),
        ('vent_active_any', '呼吸支持：该时点处于任一通气/支持区间内'),
        ('vent_active_invasive', '呼吸支持：有创通气（区间覆盖）'),
        ('phys_mbp_value', 'MAP（同/既往最近一次，精确时刻）'),
        ('phys_ph_value', 'pH（同/既往最近一次，精确时刻）'),
        ('phys_resp_rate_value', '呼吸频率（同/既往最近一次）'),
        ('phys_temperature_value', '体温（同/既往最近一次）'),
    ]:
        s = t[c]
        nonnull = int(s.notna().sum())
        key_cov.append({
            '字段': c, '含义': label,
            '非空数': nonnull, '非空比例%': round(nonnull / len(t) * 100, 2),
        })
    tab('Q3 关键字段覆盖', pd.DataFrame(key_cov))

    # GCS 来源归属分布（本阶段核心新增信息）
    org = []
    for comp in ('eyes', 'verbal', 'motor'):
        vc = t[f'gcs_{comp}_origin'].value_counts()
        for k, v in vc.items():
            org.append({'分项': comp, '来源': k, '时点数': int(v),
                        '占比%': round(v / len(t) * 100, 2)})
    tab('GCS 分项来源归属（当前记录 / 前值补入 / 缺失）', pd.DataFrame(org))

    rule = t.gcs_total_rule.value_counts().rename_axis('总分规则分支').reset_index(name='时点数')
    rule['占比%'] = (rule.时点数 / len(t) * 100).round(2)
    tab('GCS 总分规则分支', rule)

    # 总分=15 而分项之和<15 的成因拆解（修正此前"高估 8 分"的误读口径）
    sub = t[(t.gcs_total_recorded == 15)
            & (t.gcs_eyes_recorded.notna()) & (t.gcs_verbal_recorded.notna())
            & (t.gcs_motor_recorded.notna())
            & ((t.gcs_eyes_recorded + t.gcs_verbal_recorded + t.gcs_motor_recorded) < 15)]
    is_ett = sub.gcs_total_rule.isin(['ett_rule_forced_15_current', 'ett_rule_forced_15_carried'])
    is_default = sub.gcs_total_used_default_for_missing == 1
    dec = sub.assign(cause=np.select(
        [is_ett, is_default, sub.gcs_total_used_carryforward == 1],
        ['A. ETT 记录规则强制置 15',
         'B. 分项当日缺失，被默认正常值（6/5/4）补齐',
         'C. 前值补入的分项本身低于默认正常值'],
        default='D. 其他')).cause.value_counts()
    dec = dec.rename_axis('成因').reset_index(name='时点数')
    dec['占比%'] = (dec.时点数 / len(t) * 100).round(2)
    F['q3_total15_sum_lt15_rows'] = int(len(sub))
    F['q3_total15_sum_lt15_stays'] = int(sub.stay_id.nunique())
    F['q3_total15_sum_lt15_cause_split'] = dec.set_index('成因').时点数.to_dict()
    F['q3_total15_sum_lt15_cause_note'] = (
        '全部由 GCS 记录规则造成，不是经真值确认的临床意识误差；'
        'ETT 分支下语言分项 0 属"不可评估编码"，不等于"无反应"')
    tab('总分=15 且分项之和<15 的成因拆解', dec)

    # eMAR 给药分层构成
    emar = []
    for s, cn in (('sedative', '镇静'), ('opioid', '阿片'), ('nmb', '肌松')):
        emar.append({
            '类别': cn,
            '既往可确认给药时点数': int((t[f'{s}_emar_admin_prior'] == 1).sum()),
            '明确未给药时点数': int((t[f'{s}_emar_not_given_prior'] == 1).sum()),
            '未确认记录时点数': int((t[f'{s}_emar_other_unconfirmed_prior'] == 1).sum()),
            '仅医嘱(未确认)时点数': int((t[f'{s}_order_unconfirmed_prior'] == 1).sum()),
        })
    tab('Q3 eMAR/医嘱三层分离（不合并为"已暴露"）', pd.DataFrame(emar))

    # =================================================================
    # Q4 时间差分布
    # =================================================================
    gap_specs = [
        ('rass_gap_h', '→ 观测 RASS'),
        ('phys_mbp_gap_h', '→ MAP'),
        ('phys_ph_gap_h', '→ pH'),
        ('phys_resp_rate_gap_h', '→ 呼吸频率'),
        ('phys_temperature_gap_h', '→ 体温'),
        ('sedative_last_end_gap_h', '→ 镇静输注最近一次结束'),
        ('opioid_last_end_gap_h', '→ 阿片最近一次结束'),
        ('nmb_last_end_gap_h', '→ 肌松最近一次结束'),
        ('vent_last_end_gap_h', '→ 呼吸支持最近一次结束'),
        ('sedative_emar_admin_gap_h', '→ 镇静可确认给药'),
    ]
    rows = []
    for col, lab in gap_specs:
        s = t[col].dropna()
        if s.empty:
            continue
        rows.append({
            '对齐目标': lab, '可用时点数': int(len(s)),
            '中位(h)': round(float(s.median()), 3),
            'P25(h)': round(float(s.quantile(.25)), 3),
            'P75(h)': round(float(s.quantile(.75)), 3),
            'P95(h)': round(float(s.quantile(.95)), 3),
            '最大(h)': round(float(s.max()), 3),
            '≤0(同时刻)%': round(float((s <= 0).mean() * 100), 2),
        })
    tab('Q4 时间差分布（同/既往对齐）', pd.DataFrame(rows))

    # =================================================================
    # Q5 未来信息：硬断言 + merge_asof 独立复算
    # =================================================================
    # 5a 所有"回看型"特征的时间差必须 >= 0
    look_cols = [c for c in t.columns
                 if c.endswith('_gap_h') and not c.startswith(('sedative_iv', 'opioid_iv',
                                                               'nmb_iv', 'vent_active',
                                                               'sedative_emar', 'opioid_emar',
                                                               'nmb_emar', 'sedative_order',
                                                               'opioid_order', 'nmb_order'))]
    viol = {c: int((t[c].dropna() < 0).sum()) for c in look_cols}
    viol = {k: v for k, v in viol.items() if v}
    F['q5_lookback_negative_gap_violations'] = viol
    F['q5_lookback_columns_checked'] = len(look_cols)

    # 5b 区间型特征：选择依据是"起点 <= 时点"，故 start gap 必须 >= 0；
    #    结束时间可以晚于时点（这属于"治疗进行中"，不是使用未来信息）
    iv_cols = [c for c in t.columns if c.endswith('_active_start_gap_h')]
    iv_viol = {c: int((t[c].dropna() < 0).sum()) for c in iv_cols}
    iv_viol = {k: v for k, v in iv_viol.items() if v}
    F['q5_interval_start_negative_gap_violations'] = iv_viol
    # 区间型中 end < start 的异常区间计数（若上游区间本身倒挂）
    F['q5_rows_with_active_but_negative_start'] = int(
        sum((t[c].dropna() < 0).sum() for c in iv_cols))

    # 5c 状态自洽：only_later=1 必须伴随 unknown 状态
    incons = {}
    for pre in ('rass', 'cam_icu', 'rass_goal'):
        if f'{pre}_only_later' in t and f'{pre}_status' in t:
            bad = int(((t[f'{pre}_only_later'] == 1) & (t[f'{pre}_status'] != 'unknown_no_record_at_or_before')).sum())
            incons[pre] = bad
    F['q5_only_later_state_inconsistent'] = incons

    # 5d 独立复算：用 pandas merge_asof(direction='backward') 重新求"最近一条既往记录"
    asof = {}

    def asof_check(name, ev_df, idc, tc, vc, tbl_col):
        q = t[['stay_id', 'gcs_charttime', tbl_col]].copy()
        e = ev_df[[idc, tc, vc]].dropna(subset=[vc]).copy()
        e[tc] = pd.to_datetime(e[tc])
        q = q.sort_values('gcs_charttime')
        e = e.sort_values(tc)
        m = pd.merge_asof(q, e, left_on='gcs_charttime', right_on=tc,
                          by='stay_id', direction='backward')
        a, b = m[tbl_col], m[vc]
        both_na = a.isna() & b.isna()
        eq = np.isclose(a.fillna(-999.0).astype(float), b.fillna(-999.0).astype(float))
        bad = int((~(both_na | eq)).sum())
        # 未来信息检查：合并后命中的源时间必须 <= 时点
        fut = int((m[tc].notna() & (m[tc] > m.gcs_charttime)).sum())
        asof[name] = {'rows': int(len(m)), 'mismatch': bad, 'future_source_time': fut}

    ra = pd.read_csv(UP / 'mimic_09_rass_cam_icu.csv', low_memory=False,
                     usecols=['stay_id', 'charttime', 'assessment_field', 'valuenum'])
    asof_check('RASS(观测, valuenum)', ra[ra.assessment_field == 'rass_actual'],
               'stay_id', 'charttime', 'valuenum', 'rass_rass_observed')

    ph = pd.read_csv(HERE / 'output' / 'mimic_13_bg_ph.csv', low_memory=False,
                     usecols=['stay_id', 'charttime', 'ph'])
    asof_check('pH', ph, 'stay_id', 'charttime', 'ph', 'phys_ph_value')

    vs = pd.read_csv(HERE / 'output' / 'mimic_12_vitals_timeseries.csv', low_memory=False,
                     usecols=['stay_id', 'charttime', 'mbp'])
    asof_check('MAP', vs, 'stay_id', 'charttime', 'mbp', 'phys_mbp_value')

    F['q5_asof_independent_recheck'] = asof
    tab('Q5 未来信息核查', pd.DataFrame([
        {'检查': '回看型字段出现负时间差的列数', '结果': len(viol)},
        {'检查': '区间型字段出现"起点>时点"的列数', '结果': len(iv_viol)},
        {'检查': '检查的回看型列数', '结果': len(look_cols)},
        {'检查': 'merge_asof 独立复算不一致数（RASS/pH/MAP 合计）',
         '结果': sum(v['mismatch'] for v in asof.values())},
        {'检查': 'merge_asof 命中的源时间晚于评分时点的行数',
         '结果': sum(v['future_source_time'] for v in asof.values())},
    ]))

    # 覆盖分层（供后续分析选口径）
    strat = []
    for lab, col in [('RASS 同时刻', 'rass_status'), ('', None)]:
        pass
    rass_cov = pd.DataFrame([{
        '口径': '同时刻(gap=0)', '时点数': int((t.rass_status == 'same_time').sum())},
        {'口径': '既往 1h 内', '时点数': int((t.rass_within_1h == 1).sum())},
        {'口径': '既往 2h 内', '时点数': int((t.rass_within_2h == 1).sum())},
        {'口径': '既往 4h 内', '时点数': int((t.rass_within_4h == 1).sum())},
        {'口径': '无同时刻或既往记录（未知）', '时点数': int((t.rass_status == 'unknown_no_record_at_or_before').sum())},
        {'口径': '其中：存在更晚记录（非"从未测量"）', '时点数': int((t.rass_only_later == 1).sum())},
    ])
    rass_cov['占比%'] = (rass_cov.时点数 / len(t) * 100).round(2)
    tab('Q3 RASS 可判定性分层', rass_cov)

    vent_cov = pd.DataFrame([{
        '状态': s, '时点数': int((t[f'vent_active_{s}'] == 1).sum()),
        '占比%': round(float((t[f'vent_active_{s}'] == 1).mean() * 100), 2)}
        for s in ('invasive', 'noninvasive', 'hfnc', 'supp_o2', 'trach')])
    tab('Q3 呼吸支持分层（不合并为"通气"）', vent_cov)

    # 落盘（文件名用编号，避免中文与斜杠污染路径）
    SLUG = {
        'Q1 队列与结局一致性': 'q1_cohort_outcome',
        'Q2 键唯一性': 'q2_key_uniqueness',
        'Q3 关键字段覆盖': 'q3_key_field_coverage',
        'GCS 分项来源归属（当前记录 / 前值补入 / 缺失）': 'q3_gcs_component_origin',
        'GCS 总分规则分支': 'q3_gcs_total_rule_branch',
        '总分=15 且分项之和<15 的成因拆解': 'q3_total15_below_sum_causes',
        'Q3 eMAR/医嘱三层分离（不合并为"已暴露"）': 'q3_emar_three_layer',
        'Q4 时间差分布（同/既往对齐）': 'q4_gap_distribution',
        'Q5 未来信息核查': 'q5_future_information',
        'Q3 RASS 可判定性分层': 'q3_rass_assessability',
        'Q3 呼吸支持分层（不合并为"通气"）': 'q3_respiratory_support_strata',
        'coverage': 'q3_all_columns_coverage',
    }
    for name, df in T.items():
        slug = SLUG.get(name, name)
        df.to_csv(QA / f'qa_{slug}.csv', index=False)
    with pd.ExcelWriter(QA / 'qa_all_tables.xlsx') as xw:
        for name, df in T.items():
            df.to_excel(xw, sheet_name=SLUG.get(name, name)[:31], index=False)
    (QA / 'qa_facts.json').write_text(
        json.dumps(F, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print('\nQA 完成 ->', QA)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
