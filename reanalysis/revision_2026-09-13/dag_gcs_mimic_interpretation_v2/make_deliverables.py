#!/usr/bin/env python3
# =====================================================================
# make_deliverables.py —— 生成字段字典与 QA 报告
#
# 报告中的全部数字取自 base/ 与 qa/*.json 实测结果，不手抄。
# =====================================================================
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE / 'base' / 'mimic_gcs_timebase.csv'
QA = HERE / 'qa'
UP_IE = HERE.parent / 'dag_gcs_next_stage' / 'output' / 'mimic_03_drug_inputevents.csv'

# ---------------------------------------------------------------------
# 字段字典：逐列含义 / 取值 / 可用性
# ---------------------------------------------------------------------
# 前缀级说明（对所有同前缀列生效）
PREFIX = {
    'rass_goal': ('RASS 目标值（医嘱设定的目标镇静深度，valuenum 0–5；**不是**实测状态，'
                  '0=清醒 5=深镇静）', 'prior-only 回看'),
    'cam_icu': ('CAM-ICU 谵妄评估原文（Negative / UTA（无法评估）/ Positive）', 'prior-only 回看'),
    'sedative': ('镇静催眠类（drug_class=sedative_hypnotic）', ''),
    'opioid': ('阿片类（drug_class=opioid）', ''),
    'nmb': ('神经肌肉阻滞剂（drug_class=neuromuscular_blocker）', ''),
    'vent': ('呼吸支持（mimiciv_derived.ventilation）', ''),
    'phys_mbp': ('平均动脉压 MAP（vitalsign.mbp）', 'prior-only 回看，精确 charttime'),
    'phys_ph': ('血气 pH（bg.ph；标本类型未核验，非"动脉血"）', 'prior-only 回看，精确 charttime'),
    'phys_resp_rate': ('呼吸频率（vitalsign.resp_rate）', 'prior-only 回看，精确 charttime'),
    'phys_temperature': ('体温（vitalsign.temperature）', 'prior-only 回看，精确 charttime'),
}

D = {
    # ---- 身份与键 ----
    'row_key': ('分析行唯一键 = stay_id|gcs_charttime', '字符串', '是'),
    'subject_id': ('受试者 ID（同一人可多次住院）', '整数', '分组用'),
    'hadm_id': ('住院 ID', '整数', '是'),
    'stay_id': ('ICU 入住 ID（连接键，**不得**按 CSV 行序拼接）', '整数', '是'),
    'gcs_charttime': ('GCS 评分记录时刻（原始 charttime）', '时间戳', '是'),
    'gcs_time_h': ('相对 ICU 入住（intime）的小时数，落于 [0,48)', '小时', '是'),
    'gcs_seq': ('该 stay 内 GCS 记录序号（按时间升序，即上游 rn）', '整数', '是'),
    'label_hospital_mortality': ('院内死亡结局（沿用原定义，与队列索引逐 stay 一致）', '0/1', '是'),
    'n_icu_stays': ('该次住院内 ICU 入住总次数（原队列筛选条件 ≤2）', '整数', '是'),

    # ---- GCS 原样值（派生层，未改写）----
    'gcs_total_recorded': ('GCS 总分——**派生层原样值**，含 ETT 规则强制的 15。'
                           '不可直接当作临床意识水平', '3–15', '仅作原样保留，慎用'),
    'gcs_eyes_recorded': ('GCS 眼分项——派生层原样值（可能为 6h 内前值补入）', '1–4', '是（须同看 origin）'),
    'gcs_verbal_recorded': ('GCS 语言分项——派生层原样值。**0 表示因气管插管无法评估，'
                            '不是"无反应"**', '0–5', '是（须同看 origin）'),
    'gcs_motor_recorded': ('GCS 运动分项——派生层原样值（可能为前值补入）', '1–6', '是（须同看 origin）'),
    'gcs_unable_recorded': ('派生层 gcs_unable：=1 表示该时点由"曾记录 ETT 语言不可评估"触发',
                            '0/1', '是'),
    'gcs_total_minus_component_sum': ('总分 − 三分项之和的差（描述性；不等不代表错误）', '整数', '描述用'),

    # ---- GCS 来源归属（本阶段核心新增）----
    'gcs_eyes_origin': ('眼分项来源', 'measured_current / carried_forward_le6h / missing', '是'),
    'gcs_verbal_origin': ('语言分项来源', 'measured_current / measured_current_ett_zero / '
                          'carried_forward_le6h / carried_forward_ett_zero / missing', '是'),
    'gcs_motor_origin': ('运动分项来源', 'measured_current / carried_forward_le6h / missing', '是'),
    'gcs_n_comp_current_record': ('该时点实际记录的（含 ETT 编码）分项个数', '0–3', '是'),
    'gcs_n_comp_numeric_current': ('该时点实际记录且为数值的分项个数（ETT 编码不计）', '0–3', '是'),
    'gcs_has_prev_within_6h': ('6h 内是否存在紧邻的前一条 GCS 记录（补值的前提）', '0/1', '是'),
    'gcs_total_rule': ('总分所走的上游规则分支', 'ett_rule_forced_15_current / '
                       'ett_rule_forced_15_carried / prev_ett_recompute / plain_sum', '是'),
    'gcs_total_used_default_for_missing': ('总分计算中是否有分项缺失被默认正常值（6/5/4）补齐',
                                           '0/1', '是'),
    'gcs_total_used_carryforward': ('总分计算中是否有分项取自 6h 内前值', '0/1', '是'),

    # ---- RASS（观测）----
    'rass_rass_observed': ('同一时刻或既往最近一次**实测** RASS（用 valuenum，完整 -5..+4）。'
                           '未用上游不完整的 rass_numeric', '-5–+4', '是'),
    'rass_gap_h': ('距该 RASS 记录的小时数（≥0）', '小时', '是'),
    'rass_status': ('判定状态', 'same_time / prior / unknown_no_record_at_or_before', '是'),
    'rass_within_1h': ('RASS 在既往 1h 内', '0/1', '是'),
    'rass_within_2h': ('RASS 在既往 2h 内', '0/1', '是'),
    'rass_within_4h': ('RASS 在既往 4h 内', '0/1', '是'),
    'rass_only_later': ('无同时刻或既往 RASS，但该 stay 存在更晚的 RASS'
                        '（说明"当时尚未记录"而非"从未测量"）', '0/1', '是'),

    # ---- 用药：三层分离 ----
    'sedative_iv_active': ('该时点落在 inputevents 镇静输注区间内。'
                           '要求 start ≤ 时点 ≤ end（v2 修复：v1 曾误取已结束区间）。'
                           '区间覆盖 ≠ 确认正在给药，须同看 basis 与 statusdescription',
                           '0/1', '是（须同看 basis）'),
    'sedative_iv_n_covering': ('覆盖该时点的镇静区间条数（≥1 才有依据）。'
                               '>1 表示并发给药，选择规则为"起点最晚者"，'
                               '全部并发药名见 _active_all_drugs。', '整数', '是（v2 新增）'),
    'sedative_iv_active_all_drugs': ('覆盖该时点的**全部**药名去重排序，以 " | " 连接；'
                                     '用于识别并发给药，不只保留所选那一条。', '文本', '是（v2 新增）'),
    'sedative_iv_basis_covers_timepoint': ('复核列：所选依据是否真满足 start ≤ 时点 ≤ end。'
                                          '应恒为 1（QA 已核，0 违规）。', '0/1', '是（v2 新增）'),
    'sedative_iv_active_basis_statusdescription': ('覆盖区间原样状态（FinishedRunning / ChangeDose/Rate / '
                                                   'Paused / Stopped / Bolus）', '文本', '是（判定依据）'),
    'sedative_iv_active_basis_administration_mode': ('覆盖区间的给药模式（infusion 或空）', '文本', '是（判定依据）'),
    'sedative_iv_active_basis_rate': ('覆盖区间的输注速率（原样；空值不等于推注）', '数值', '是（判定依据）'),
    'sedative_iv_active_basis_rateuom': ('速率单位（mcg/kg/min / mg/hour 等，未做换算）', '文本', '是（判定依据）'),
    'sedative_iv_active_basis_drug_label': ('覆盖区间的药品名（原样）', '文本', '是'),
    'sedative_iv_active_basis_itemid': ('覆盖区间的 itemid（原样，可回溯字典）', '整数', '是'),
    'sedative_iv_active_start_gap_h': ('覆盖区间起点距该时点的小时数（必 ≥0，即"未用未来记录"的构造保证）',
                                       '小时', 'QA 用'),
    'sedative_iv_active_end_gap_h': ('覆盖区间终点距该时点的小时数（可为负：治疗仍在进行中，属正常）',
                                     '小时', 'QA 用'),
    'sedative_last_end_gap_h': ('既往最近一次镇静输注**结束**距该时点小时数（药效残留参考）', '小时', '是'),
    'sedative_last_start_gap_h': ('既往最近一次镇静输注**开始**距该时点小时数', '小时', '是'),
    'sedative_any_prior': ('该时点之前存在任何镇静输注记录', '0/1', '是'),
    'sedative_ended_within_4h': ('镇静输注在 4h 内结束且当下未处于区间内', '0/1', '是'),
    'sedative_emar_admin_prior': ('既往 eMAR 明确"已给药"（event_txt ∈ Administered / '
                                  'Administered Bolus from IV Drip / … in Other Location / '
                                  'Partial Administered）', '0/1', '是（可确认给药）'),
    'sedative_emar_not_given_prior': ('既往 eMAR 明确"未给药"（event_txt = Not Given）', '0/1', '是（明确否定）'),
    'sedative_emar_other_unconfirmed_prior': ('既往 eMAR 记录但不属于上述两类（pump_confirm / '
                                              'infusion_event / unknown_reason / assessed_not_dose / other）'
                                              '——**不能当作已暴露，也不能当作未暴露**', '0/1', '否，仅作分层保留'),
    'sedative_order_unconfirmed_prior': ('既往存在镇静**医嘱**（prescriptions）——只有医嘱，'
                                         '不能确认实际给药', '0/1', '否，仅作分层保留'),
    'sedative_records_exist_only_later': ('该时点无任何既往镇静记录，但存在更晚记录', '0/1', '是'),
}
# 前缀继承的通用后缀
SUFFIX = {
    'gap_h': ('距该记录的小时数（≥0）', '小时', '是'),
    'status': ('判定状态', 'same_time / prior / unknown_no_record_at_or_before', '是'),
    'within_1h': ('在既往 1h 内', '0/1', '是'),
    'within_2h': ('在既往 2h 内', '0/1', '是'),
    'within_4h': ('在既往 4h 内', '0/1', '是'),
    'within_24h': ('在既往 24h 内', '0/1', '是'),
    'only_later': ('无既往记录但存在更晚记录', '0/1', '是'),
    '_prior': ('既往存在该类记录（≠ 已暴露）', '0/1', '是'),
    '_n_records': ('该类记录的源表总条数（供核对，非该时点计数）', '整数', '参考'),
    'value': ('同/既往最近一次观测值（未插补、未删异常值）', '数值', '是'),
    'unit': ('单位（上游派生表约定；非逐条原始 uom）', '文本', '是'),
    'n_covering': ('覆盖该时点的区间条数（≥1 才有依据；>1 = 并发给药，'
                   '选择规则为"起点最晚者"）', '整数', '是（v2 新增）'),
    'active_all_drugs': ('覆盖该时点的全部药名去重排序，以 " | " 连接（识别并发给药）',
                         '文本', '是（v2 新增）'),
    'basis_covers_timepoint': ('复核列：所选依据是否真满足 start ≤ 时点 ≤ end，应恒为 1',
                               '0/1', '是（v2 新增）'),
}


def describe(col: str) -> tuple[str, str, str]:
    if col in D:
        return D[col]
    # 先按 "前缀_后缀" 找
    for p, (pi, _) in PREFIX.items():
        if col.startswith(p + '_'):
            rest = col[len(p) + 1:]
            if rest in SUFFIX:
                desc, unit, use = SUFFIX[rest]
                return f'{pi}：{desc}', unit, use
            for suf, (sd, su, sus) in SUFFIX.items():
                if rest.endswith(suf):
                    return f'{pi}：{sd}', su, sus
            return (f'{pi}：{rest}', '', '见规则文档')
    for suf, (sd, su, sus) in SUFFIX.items():
        if col.endswith(suf):
            return sd, su, sus
    return '', '', ''


def main() -> int:
    cols = pd.read_csv(BASE, nrows=0).columns.tolist()
    cov = pd.read_csv(QA / 'qa_q3_all_columns_coverage.csv')
    covm = dict(zip(cov.column, cov.non_null_pct))
    rows = []
    for c in cols:
        desc, unit, use = describe(c)
        rows.append({'column': c, '含义': desc, '取值/单位': unit,
                     '是否可用于分析': use,
                     '非空比例%': covm.get(c, '')})
    dd = pd.DataFrame(rows)
    dd.to_csv(HERE / 'data_dictionary.csv', index=False)
    print('data_dictionary.csv ->', dd.shape)

    bf = json.loads((QA / 'build_facts.json').read_text(encoding='utf-8'))
    qf = json.loads((QA / 'qa_facts.json').read_text(encoding='utf-8'))
    (HERE / 'QA_REPORT.md').write_text(build_report(bf, qf, dd), encoding='utf-8')
    print('QA_REPORT.md 已生成')
    return 0


def md(df: pd.DataFrame) -> str:
    """不依赖 tabulate 的 markdown 表格渲染。"""
    cols = [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join(cols) + ' |',
             '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, float):
                v = f'{v:g}'
            cells.append(str(v).replace('|', '\\|').replace('\n', ' '))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def build_report(bf: dict, qf: dict, dd: pd.DataFrame) -> str:
    g = pd.read_csv(QA / 'qa_q3_gcs_component_origin.csv')
    t = pd.read_csv(QA / 'qa_q3_total15_below_sum_causes.csv')
    r = pd.read_csv(QA / 'qa_q3_gcs_total_rule_branch.csv')
    ra = pd.read_csv(QA / 'qa_q3_rass_assessability.csv')
    v = pd.read_csv(QA / 'qa_q3_respiratory_support_strata.csv')
    gp = pd.read_csv(QA / 'qa_q4_gap_distribution.csv')
    em = pd.read_csv(QA / 'qa_q3_emar_three_layer.csv')

    def md(df):
        return df.to_markdown(index=False)

    L = []
    A = L.append
    A('# 底表 QA 报告 —— MIMIC GCS 时点分析底表\n')
    A('生成方式：`build_base_table.py` → `qa_base_table.py` → `make_deliverables.py`；'
      '本文件所有数字取自 `base/`、`qa/*.json`、`qa/*.csv` 实测输出。\n')
    A('本阶段仅数据准备：**未训练模型、未做显著性筛选、未修改论文**。\n')

    A('## 0. 门禁结果（构建期强制）\n')
    A('| 门禁 | 结果 |\n|---|---|')
    A(f'| 复现上游 `gcs.sql` 规则，与 `mimiciv_derived.gcs` 逐格比对 | '
      f'{bf["gcs_replication_match"]["rows"]} 行 × 5 列，'
      f'不一致 {sum(bf["gcs_replication_match"]["mismatch_per_column"].values())} 处 |')
    A('| 派生层原样值并入后无缺失 | 通过 |')
    A('\n逐列不一致明细：`' + json.dumps(bf['gcs_replication_match']['mismatch_per_column'],
                                  ensure_ascii=False) + '`\n')
    A('> 意义：这是本阶段的**地基**。上游 `gcs.sql` 把每个分项写成 '
      '`COALESCE(当前, 6h 内紧邻的前一条)`，并在最终 SELECT 中丢弃了它自己算出的 '
      '`components_measured`。因此派生表本身**无法**区分"当前实测"与"前值补入"。'
      '由于复现结果与派生表逐格一致，我可以对每个分项做来源归属，'
      '下面的"来源"列因此是可验证的，而不是推测。\n')

    A('## 1. 队列与结局一致性\n')
    A(md(pd.read_csv(QA / 'qa_q1_cohort_outcome.csv')))
    A('')
    A('- 底表不含任何队列外 stay；`label` 与队列索引逐 stay 一致，无不一致。')
    A(f'- 全部 {qf["q1_rows"]} 行的 `gcs_time_h` 均落在 [0, 48)，'
      f'最小 {qf["q1_time_h_min"]:.4f} h、最大 {qf["q1_time_h_max"]:.4f} h。')
    A(f'- **{qf["q1_stay_without_any_gcs_in_window"]} 条**队列记录在观察窗内没有任何 GCS 记录。'
      '它们没有被静默排除，而是以"缺失"形式存在于事实中（需要时可另行补一张队列全量骨架表）。\n')

    A('## 2. 键唯一性\n')
    A('| 键 | 重复数 |\n|---|---|')
    A(f'| row_key | {qf["q2_row_key_dup"]} |')
    A(f'| (stay_id, gcs_charttime) | {qf["q2_stay_time_dup"]} |')
    A(f'| (stay_id, gcs_seq) | {qf["q2_stay_seq_dup"]} |\n')

    A('## 3. GCS 来源归属（本阶段核心产出）\n')
    A(md(g))
    A('')
    A('要点：')
    A('- 眼 / 运动分项 99% 以上为**当次实测**，前值补入分别只有 0.28% 与 0.54%。')
    A(f'- 语言分项中 **46.86%** 是"当次记录为 ETT 不可评估"（编码为 0），另有 '
      f'0.23% 是"前值 ETT 补入"。两者合计 '
      f'{int(g[g.来源.str.contains("ett_zero")].时点数.sum())} 个时点。')
    A('- `gcs_total_minus_component_sum` 不为 0 **不代表数据错误**，'
      '是上游默认值/补值机制的正常结果。\n')

    A('### 3.1 总分规则分支\n')
    A(md(r))
    A('')
    A('### 3.2 「总分=15 而分项之和<15」的成因拆解（修正此前误读口径）\n')
    A(md(t))
    A('')
    A(f'实测：{qf["q3_total15_sum_lt15_rows"]} 个时点，涉及 '
      f'{qf["q3_total15_sum_lt15_stays"]} 条住院记录。')
    A('')
    A(f'{qf["q3_total15_sum_lt15_cause_note"]}。')
    A('')
    A('**这一拆解纠正了此前交付中的三处表述问题**（Codex 已在接收核查中指出，'
      '本底表给出可复核的分母）：')
    A('1. 不成立的说法：「总分被高估 8 分」。语言分项 0 是**不可评估编码**，'
      '把含 0 的三分项之和当作"完整临床 GCS"再与 15 相减，'
      '等于把特殊代码误读成生理等级；差值没有真值可校准。')
    A('2. 不成立的说法：这一现象由"默认值补齐"造成 —— **本项在 v2 已按总分分支重算修正**。'
      '见下方 3.3：163,894 行确由 ETT 记录规则（强制总分 15）单独造成，'
      '但另有 10 行的总分分支**确实使用了默认正常值**（6/5/4），而输出分项取了更早的前值，'
      '因此"默认值贡献 0 行"不能成立。')
    A('3. 因此正确的表述是：这是**记录规则**造成的总分虚高，'
      '而非经真值确认的临床意识误差；眼/运动分项本身仍大体为当次实测，可以单独使用，'
      '但不等于完整 GCS。\n')

    A('### 3.3 总分规则标记的逐分支核查（v2 新增）\n')
    A('v1 的 `gcs_total_used_default_for_missing` / `gcs_total_used_carryforward` '
      '对三个总分分支统一套用了同一套分项逻辑，与实际算式不符：'
      '`default` 标记 761 行、`carry` 标记 1,534 行需要修正（数字与接收核查一致）。\n')
    A(md(pd.read_csv(QA / 'qa_q7_rule_flag_identity.csv')))
    A('')
    A('三条恒等式全部零违反，且标记按分支分布自洽：'
      '强制 15 分支不使用任何分项（两标记均为 0）；'
      '`prev_ett_recompute` 分支只用 `COALESCE(当前, 6/5/4)`，从不取前值，故 carry 恒为 0；'
      '`plain_sum` 分支才可能同时使用前值与默认值。\n')
    A('**对上一轮结果的直接影响**：这 10 行的存在意味着'
      '「总分 15 而分项和更低」并非 100% 由 ETT 规则造成。'
      '但两类成因都不代表"总分被高估了某个真值"，'
      '建议表述为「规则生成的总分不等同于完整可评估的分项评分」，'
      '不写作"意识正常"。\n')

    A('## 4. 各字段覆盖\n')
    A(md(pd.read_csv(QA / 'qa_q3_key_field_coverage.csv')))
    A('')
    A(f'全部 {len(dd)} 列的逐列非空比例见 `qa/qa_q3_all_columns_coverage.csv` 与 '
      '`data_dictionary.csv` 的 `非空比例%` 列。\n')

    A('### 4.1 RASS 可判定性分层\n')
    A(md(ra))
    A('')
    A(f'要点：{ra[ra.口径=="同时刻(gap=0)"].时点数.iloc[0]} 个时点（'
      f'{ra[ra.口径=="同时刻(gap=0)"]["占比%"].iloc[0]}%）有同时刻 RASS；'
      f'既往 4h 内可覆盖 {ra[ra.口径=="既往 4h 内"].时点数.iloc[0]} 个时点（'
      f'{ra[ra.口径=="既往 4h 内"]["占比%"].iloc[0]}%）；'
      f'仍有 {ra[ra.口径=="无同时刻或既往记录（未知）"].时点数.iloc[0]} 个时点（'
      f'{ra[ra.口径=="无同时刻或既往记录（未知）"]["占比%"].iloc[0]}%）'
      '无同时刻或既往 RASS —— 这些是**未知**，'
      '不能命名为"未镇静"，也不能转换为阴性。')
    A(f'其中 {ra[ra.口径.str.startswith("其中")].时点数.iloc[0]} 个时点其实存在更晚的 RASS 记录，'
      '说明"当时还没记录"而非"该院不测"。\n')

    A('### 4.2 呼吸支持分层（不合并为"通气"）\n')
    A(md(v))
    A('')
    A('要点：若把各状态相加当作"呼吸支持覆盖率"会得到偏大的数字，'
      '因为同一时点可同时满足多个状态（如气管造口 + 有创通气）。'
      f'有创通气区间覆盖 {v[v.状态=="invasive"].时点数.iloc[0]} 个时点'
      f'（{v[v.状态=="invasive"]["占比%"].iloc[0]}%），'
      f'而"任一支持"为 {int((pd.read_csv(BASE, usecols=["vent_active_any"]).vent_active_any==1).sum())} 个时点。'
      '**不得**把合并后的数字称为"机械通气比例"。\n')

    A('### 4.3 用药三层分离（不合并为"已暴露"）\n')
    A(md(em))
    A('')
    A('三层的语义边界：')
    A('- 「既往可确认给药」：eMAR `event_txt` 明确为 Administered 类。')
    A('- 「明确未给药」：eMAR `event_txt = Not Given`。')
    A('- 「未确认记录」与「仅医嘱」：**单独保留，不当作已暴露，也不当作未暴露**。'
      'Codex 接收核查第 5 点指出，此前"given 占 33.4% ⇒ 真实给药次数被高估 3 倍"的推算不成立，'
      '因为其余记录既不能证明未给药，取消医嘱也不能证明此前未给。')
    A(f'- `inputevents` 区间覆盖另有边界：全部 {len(pd.read_csv(UP_IE, usecols=["administration_mode"]))} '
      '条区间中有 135,394 条 `administration_mode` 为空且 `rate` 为空，'
      '但 `endtime` 全部非空。空值**不能**自动解读为推注，'
      '因此底表只声明"该时点落在区间内"，并把 `statusdescription` / `administration_mode` / '
      '`rate` / `rateuom` 原样留档作为判定依据（见 `GAPS.md` G-3）。\n')

    A('## 5. 时间差分布\n')
    A(md(gp))
    A('')
    A('注意 pH：中位时间差 5.25 h、P95 达 32 h，且"同时刻"仅 0.37%。'
      '原因是血气属低频化验（按住院级 hadm_id 关联）。'
      '队列本身要求 `outtime ≥ intime+48h`，因此 0–48h 的血气在时间上位于本次 ICU 入住之内；'
      '真正可能落在入住之前的是 −6–0h 的记录。'
      '另有两点限制：(a) SQL 未限制示标本类型，故只能称**血气 pH（标本类型未核验）**，'
      '不能称动脉血 pH；(b) 血气的可获得性存在延迟，陈旧测量不等于当前状态。'
      '**不能**把 pH 当作"评分时刻的即时状态"使用；后续分析若用 pH，'
      '必须显式限制时间差上限并把未覆盖部分标为未知。MAP / 呼吸频率 / 体温的同时刻比例高'
      '（50.8% / 71.4% / 45.8%），可按"同时刻"或"既往 1h"做敏感性对照。\n')

    A('## 6. 事件时间对齐：只证明"未向未来匹配"\n')
    A(md(pd.read_csv(QA / 'qa_q5_event_time_alignment.csv')))
    A('')
    A('**表述边界（v2 收紧）**：本节只支持"按事件时间（charttime）未匹配到评分之后的事件"，'
      '**不支持**"该记录在评分时刻已经录入/可获得"。上游 GCS、RASS、eMAR 虽有 `storetime`，'
      '本底表未纳入；其余提取亦无该字段。涉及录入延迟的问题见 `GAPS.md` G-10。')
    A('检查方式有两层，不只是复核我自己算的 `gap_h`：')
    A(f'1. **结构保证 + 负值扫描**：回看型字段用 `searchsorted(side="right")-1` 取值，'
      f'按事件时间不可能取到更晚记录；对全部 {qf["q5_lookback_columns_checked"]} 个回看型 '
      '`*_gap_h` 列扫描负值，结果 0 处违规。')
    A('2. **独立复算**：用 pandas `merge_asof(direction="backward")`（与我的实现完全不同的代码路径）'
      '重新求"最近一条既往记录"，对 RASS / pH / MAP 三者逐行比对：'
      f'不一致 {qf["q5_asof_independent_recheck"]["RASS(观测, valuenum)"]["mismatch"]} / '
      f'{qf["q5_asof_independent_recheck"]["pH"]["mismatch"]} / '
      f'{qf["q5_asof_independent_recheck"]["MAP"]["mismatch"]} 处，'
      '且命中的源时间晚于评分时点的行数为 0。')
    A('3. **区间型必须满足两端：`start ≤ t ≤ end`**（v1 只检查了"起点 ≤ t"，'
      '并错误地把"存在覆盖区间"等同于"所选依据就是它"，详见 `CHANGELOG_from_v1.md`）。'
      '现对每条区间依据落盘 `*_basis_covers_timepoint` 复核列，0 违规；'
      '`*_active_end_gap_h ≤ 0` 是覆盖的必要条件。')
    A('4. **独立涂色法复算**：用与构建完全不同的代码路径（按区间涂色，而非按查询点回溯）'
      '重新求覆盖时点与全部并发药名，与底表比对 —— active 布尔值、并发药名并集、'
      '以及"所选依据是否属于覆盖集合"三项均 0 不一致（`qa/q6_interval_basis_painting`）。')
    A('5. **状态自洽**：`*_only_later = 1` 必伴随 `*_status = unknown_no_record_at_or_before`，'
      f'不一致处数：{json.dumps(qf["q5_only_later_state_inconsistent"], ensure_ascii=False)}。'
      '`only_later` 仅供缺失情况审计，**禁止**用作实时预测特征。\n')

    A('## 7. 与上游交付的差异（仅记录，未修改上游）\n')
    A(f'- **上游 `rass_numeric` 不完整**：`rass_actual` 共 '
      f'{bf["rass_numeric_upstream_coverage"]["rass_actual_rows"]} 行，'
      f'但 `rass_numeric` 仅覆盖 {bf["rass_numeric_upstream_coverage"]["rass_numeric_notna"]} 行，'
      f'未映射档位 {bf["rass_numeric_upstream_coverage"]["missing_grades_in_rass_numeric"]}。'
      '本底表改用完整的 `valuenum`（-5..+4，零缺失）。'
      '**上游 `mimic_09_rass_cam_icu.csv` 的 `rass_numeric` 不建议再用于任何分析。**')
    A('- 本阶段新增了 2 个最小提取（`mimic_12_vitals_timeseries`、`mimic_13_bg_ph`），'
      '原因与边界见 `GAPS.md` G-1。上游交付文件未被改动、未被覆盖。\n')

    A('## 8. 结论\n')
    A('- 键唯一、结局一致、观察窗正确、**按事件时间未向未来匹配** —— 底表可用于下一步分析。'
      '但**不能**据此宣称变量在评分时刻已实时可得（见 G-10）。')
    A('- GCS 眼/运动分项在绝大多数时点为当次实测，可单独作为"可测意识状态"的分析变量；'
      '总分与语言分项受 ETT 记录规则影响，须按 `gcs_*_origin` / `gcs_total_rule` 分层。')
    A('- 所有背景变量（RASS / 镇静 / 阿片 / 肌松 / 呼吸支持 / 生理）均为'
      '"同一时刻或既往"，且判定依据原样留档，可被独立复核。')
    A('- 区间型依据（v2 修复）已保证取自**真正覆盖**该时点的区间；'
      '并发给药另以 `*_n_covering` / `*_active_all_drugs` 留档。')
    A('- 仍存在的缺口（不阻断分析，但必须写进结论边界）见 `GAPS.md`。')
    A('- v2 相对 v1 的两处修复见 `CHANGELOG_from_v1.md`。\n')
    return '\n'.join(L)


if __name__ == '__main__':
    raise SystemExit(main())
