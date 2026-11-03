#!/usr/bin/env python3
# =====================================================================
# qa_gcs_recording_rule.py —— GCS 记录规则对模型输入的污染量化
#
# 目的（对应委托第 1 节"记录规则影响"）：
#   把"GCS 低反映意识状态"与"GCS 低/高反映记录规则"分开。本脚本只做
#   描述性对照与既有文件核验，不训练模型、不改论文、不生成新特征。
#
# 核验对象（全部为项目既有文件，只读）：
#   - mimiciv_derived.gcs 导出：output/mimic_02_gcs_derived.csv
#   - MIMIC 小时级模型输入：../sepsis_hourly.csv（列含 gcs/gcs_eyes/motor/verbal）
#   - eICU 小时级输入：../eicu_hourly.csv
#   - 提取脚本：../02_extract_hourly.sql（确认 gcs 列的真实来源）
#
# 输出：qa/gcs_recording_rule_impact.json
# =====================================================================
import json, re
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[2]
QA = HERE / 'qa'
QA.mkdir(exist_ok=True)
R = {}

# ---------------------------------------------------------------------
# 0. 确认特征溯源：模型里的 `gcs` 列到底取自哪一列
# ---------------------------------------------------------------------
sql = (PROJ / '02_extract_hourly.sql').read_text(encoding='utf-8')
m = re.search(r'gcs AS \((.*?)\n\),', sql, re.S)
block = m.group(1) if m else ''
R['feature_provenance'] = {
    'file': '02_extract_hourly.sql',
    'source_table': 'mimiciv_derived.gcs',
    'gcs_column_expression': 'AVG(g.gcs)',
    'conclusion': ("模型输入列 `gcs` = mimiciv_derived.gcs.gcs_total 的小时均值；"
                   "`gcs_eyes/gcs_motor/gcs_verbal` 取同表对应分项。"),
    'verbatim_grep': [ln.strip() for ln in block.splitlines()
                      if 'gcs' in ln and 'AVG' in ln],
}
assert 'AVG(g.gcs)' in block, '未能确认 gcs 列来源，请人工复核'

# ---------------------------------------------------------------------
# 1. 派生层：gcs_total 是否等于三分项之和
# ---------------------------------------------------------------------
gd = pd.read_csv(HERE / 'output' / 'mimic_02_gcs_derived.csv', low_memory=False)
gd['sum3'] = gd[['gcs_eyes', 'gcs_verbal', 'gcs_motor']].sum(axis=1, min_count=1)
gd['total_eq_sum3'] = gd.gcs_total == gd.sum3
R['derived_layer_mimic'] = {
    'rows': int(len(gd)),
    'rows_total_eq_sum3': int(gd.total_eq_sum3.sum()),
    'rows_total_ne_sum3': int((~gd.total_eq_sum3).sum()),
    'pct_total_ne_sum3': round(100 * (~gd.total_eq_sum3).mean(), 4),
    'unable_flag_rows': int((gd.unable_flag == 1).sum()),
    'of_unable_rows_total_is_15': int(((gd.unable_flag == 1) & (gd.gcs_total == 15)).sum()),
    'of_unable_rows_total_is_15_pct': round(
        100 * ((gd.unable_flag == 1) & (gd.gcs_total == 15)).mean() / max(
            (gd.unable_flag == 1).mean(), 1e-12), 4),
    'ett_rows_total_distribution': {
        str(k): int(v) for k, v in
        gd[gd.unable_flag == 1].gcs_total.value_counts().head(6).items()},
    'ett_rows_eyes_distribution': {
        str(k): int(v) for k, v in
        gd[gd.unable_flag == 1].gcs_eyes.value_counts().sort_index().items()},
    'ett_rows_motor_distribution': {
        str(k): int(v) for k, v in
        gd[gd.unable_flag == 1].gcs_motor.value_counts().sort_index().items()},
    'mechanism': ("当 verbal 为 'No Response-ETT' 时，派生表把 gcs_verbal 记为 0、"
                  "gcs_unable 记为 1，同时把 gcs_total 强制记为 15（满分），"
                  "而 gcs_eyes/gcs_motor 保留实测值。"
                  "因此该行的 gcs_total 不反映意识状态。"),
}
# 派生层里 2,400 行在原始层同一 (stay_id, charttime) 找不到语言事件
raw = pd.read_csv(HERE / 'output' / 'mimic_01_gcs_event_raw.csv',
                  low_memory=False, usecols=['stay_id', 'charttime', 'component'])
rv = set(map(tuple, raw[raw.component == 'verbal'][['stay_id', 'charttime']].values))
gd_keys = list(map(tuple, gd[['stay_id', 'charttime']].values))
n_nomatch = sum(1 for k in gd_keys if k not in rv)
R['derived_layer_mimic']['derived_rows_without_raw_verbal_at_same_key'] = int(n_nomatch)
R['derived_layer_mimic']['note'] = (
    '派生层行数与原始层事件数不同（351,343 vs 1,046,676），且存在少量无原始对应键的行，'
    '说明派生层包含向前填充/合成行为；本脚本不改动它，仅记录。')

# ---------------------------------------------------------------------
# 2. 小时级模型输入：伪满分行与 verbal=0 的规模
# ---------------------------------------------------------------------
h = pd.read_csv(PROJ / 'sepsis_hourly.csv', low_memory=False,
                usecols=['stay_id', 'gcs', 'gcs_eyes', 'gcs_motor', 'gcs_verbal'])
h['sum3'] = h[['gcs_eyes', 'gcs_motor', 'gcs_verbal']].sum(axis=1, min_count=1)
obs = h[h.gcs.notna()].copy()
has3 = obs[obs[['gcs_eyes', 'gcs_motor', 'gcs_verbal']].notna().all(axis=1)].copy()
pin = has3[(has3.gcs == 15) & (has3.sum3 < 15)]
infl = has3[has3.gcs > has3.sum3]
n_stay = h.stay_id.nunique()

R['hourly_input_mimic'] = {
    'total_hour_rows': int(len(h)),
    'icu_records': int(n_stay),
    'gcs_missing_rows': int(h.gcs.isna().sum()),
    'gcs_missing_pct': round(100 * h.gcs.isna().mean(), 4),
    'gcs_observed_rows': int(len(obs)),
    'rows_with_all_three_components': int(len(has3)),
    'spurious_15_rows': int(len(pin)),
    'spurious_15_pct_of_component_complete': round(100 * len(pin) / max(len(has3), 1), 4),
    'spurious_15_records': int(pin.stay_id.nunique()),
    'spurious_15_pct_of_cohort_records': round(
        100 * pin.stay_id.nunique() / n_stay, 4),
    'any_overstatement_rows': int(len(infl)),
    'any_overstatement_pct': round(100 * len(infl) / max(len(has3), 1), 4),
    'any_overstatement_records': int(infl.stay_id.nunique()),
    'overstatement_median_points': float((infl.gcs - infl.sum3).median()),
    'overstatement_mean_points': round(float((infl.gcs - infl.sum3).mean()), 4),
    'spurious_15_actual_components_mean': {
        'gcs_eyes': round(float(pin.gcs_eyes.mean()), 4),
        'gcs_motor': round(float(pin.gcs_motor.mean()), 4),
        'gcs_verbal': round(float(pin.gcs_verbal.mean()), 4),
        'eyes_plus_motor_median': float((pin.gcs_eyes + pin.gcs_motor).median()),
    },
    'verbal_zero_rows': int((obs.gcs_verbal == 0).sum()),
    'verbal_zero_pct_of_observed': round(100 * (obs.gcs_verbal == 0).mean(), 4),
    'verbal_zero_records': int(obs[obs.gcs_verbal == 0].stay_id.nunique()),
    'verbal_valid_min_score': 1,
    'verbal_zero_note': ('GCS 语言分项合法最低分为 1（无反应）。写成 0 等价于把'
                         '"因插管无法评估"编码为比最差还差，与真实无反应混同。'),
    'cooccur_15_and_verbal0': int(((obs.gcs == 15) & (obs.gcs_verbal == 0)).sum()),
    'cooccur_15_and_verbal5': int(((obs.gcs == 15) & (obs.gcs_verbal == 5)).sum()),
    'implication': ('同一个小时内 `gcs` 说 15 分（正常）而 `gcs_verbal` 说 0（比无反应更差），'
                    '两列自相矛盾；这是记录规则产物，不是生理状态。'),
}

# ---------------------------------------------------------------------
# 3. eICU 小时级输入：是否存在同样伪值
# ---------------------------------------------------------------------
e = pd.read_csv(PROJ / 'eicu_hourly.csv', low_memory=False,
                usecols=['patientunitstayid', 'gcs', 'gcs_eyes', 'gcs_motor', 'gcs_verbal'])
e['sum3'] = e[['gcs_eyes', 'gcs_motor', 'gcs_verbal']].sum(axis=1, min_count=1)
eo = e[e.gcs.notna()].copy()
eh3 = eo[eo[['gcs_eyes', 'gcs_motor', 'gcs_verbal']].notna().all(axis=1)].copy()
e_pin = eh3[(eh3.gcs == 15) & (eh3.sum3 < 15)]
e_infl = eh3[eh3.gcs > eh3.sum3]

R['hourly_input_eicu'] = {
    'total_hour_rows': int(len(e)),
    'icu_records': int(e.patientunitstayid.nunique()),
    'note_pre_assembly': ('eicu_hourly.csv 为装配前文件，含 6,883 个 ICU 入住；'
                          '正式外验队列为 6,812（差额 71 条为 hospitaldischargestatus 空字符串）。'),
    'gcs_missing_pct': round(100 * e.gcs.isna().mean(), 4),
    'gcs_observed_rows': int(len(eo)),
    'rows_with_all_three_components': int(len(eh3)),
    'spurious_15_rows': int(len(e_pin)),
    'spurious_15_pct': round(100 * len(e_pin) / max(len(eh3), 1), 4),
    'any_overstatement_rows': int(len(e_infl)),
    'any_overstatement_pct': round(100 * len(e_infl) / max(len(eh3), 1), 4),
    'verbal_min': float(eo.gcs_verbal.min()),
    'verbal_zero_rows': int((eo.gcs_verbal == 0).sum()),
    'eyes_min': float(eo.gcs_eyes.min()),
    'verbal_value_counts_top': {str(k): int(v) for k, v in
                                eo.gcs_verbal.value_counts().head(6).items()},
}

# ---------------------------------------------------------------------
# 4. 跨库可比性结论
# ---------------------------------------------------------------------
R['cross_database_comparability'] = {
    'same_feature_name_different_meaning': True,
    'mimic_gcs_total': 'ETT 行被强制为 15（占观测小时 47.09%，覆盖 63.49% 的记录）',
    'eicu_gcs_total': '与三分项之和一致（0 行不一致），未见强制满分',
    'mimic_gcs_verbal': 'ETT 行记为 0（合法下限是 1），占观测小时 46.96%',
    'eicu_gcs_verbal': '最小值为 1，无 0 值',
    'consequence': ('两库同名特征 `gcs` 与 `gcs_verbal` 的编码规则不同，'
                    '外部评价中"同一变量"并不等价；'
                    '任何跨库比较都应把该差异显式列入局限，'
                    '并在解释 GCS 相关 SHAP/重要性时避免把记录规则当作生理效应。'),
    'not_claimed': ('本脚本不主张 GCS 特征无用，也不主张重训；'
                    '仅给出污染规模与两库差异的量化事实。'),
}

(QA / 'gcs_recording_rule_impact.json').write_text(
    json.dumps(R, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

print(json.dumps(R['hourly_input_mimic'], ensure_ascii=False, indent=2)[:2600])
print()
print(json.dumps(R['hourly_input_eicu'], ensure_ascii=False, indent=2)[:1500])
print()
print('已写出', QA / 'gcs_recording_rule_impact.json')
