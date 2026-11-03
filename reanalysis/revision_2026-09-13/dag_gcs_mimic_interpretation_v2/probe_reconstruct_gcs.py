#!/usr/bin/env python3
# =====================================================================
# 去风险验证：在已导出 CSV 上复现 mimic-code gcs.sql 规则，
# 与 mimiciv_derived.gcs 实际输出逐格比对。
#
# 若复现一致，则"分项来源（当前记录 / 6h 前值补入 / 缺失）"的归属可信，
# 可用于构建底表。若不一致，则必须停止并把差异列为缺口。
#
# 只读既有 CSV，不连数据库、不写除 stdout 外的任何内容。
# =====================================================================
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NEXT = HERE.parent / 'dag_gcs_next_stage'
sys.dont_write_bytecode = True

RAW = NEXT / 'output' / 'mimic_01_gcs_event_raw.csv'
DER = NEXT / 'output' / 'mimic_02_gcs_derived.csv'

raw = pd.read_csv(RAW, low_memory=False,
                  usecols=['stay_id', 'charttime', 'itemid', 'value_raw', 'valuenum'])
der = pd.read_csv(DER, low_memory=False,
                  usecols=['stay_id', 'charttime', 'gcs_total', 'gcs_eyes',
                           'gcs_verbal', 'gcs_motor', 'gcs_unable'])

# ---- 1) 复现 base：按 (stay_id, charttime) 透视，规则逐字对齐上游 ----
raw['_motor'] = np.where(raw.itemid == 223901, raw.valuenum, np.nan)
raw['_verbal'] = np.where(raw.itemid == 223900,
                          np.where(raw.value_raw == 'No Response-ETT', 0, raw.valuenum),
                          np.nan)
raw['_eyes'] = np.where(raw.itemid == 220739, raw.valuenum, np.nan)
raw['_ett'] = np.where((raw.itemid == 223900) & (raw.value_raw == 'No Response-ETT'), 1, 0)

base = raw.groupby(['stay_id', 'charttime'], as_index=False).agg(
    gcsmotor=('_motor', 'max'),
    gcsverbal=('_verbal', 'max'),
    gcseyes=('_eyes', 'max'),
    endotrachflag=('_ett', 'max'),
)
base = base.sort_values(['stay_id', 'charttime'], kind='stable').reset_index(drop=True)
base['rn'] = base.groupby('stay_id').cumcount() + 1

print('原始事件行:', len(raw), ' base 行(distinct stay+charttime):', len(base))
print('派生层行:', len(der))

# ---- 2) 连接前一条 base 行（rn-1）且前一条时间 > 当前 - 6h ----
base['_t'] = pd.to_datetime(base.charttime)
prev = base[['stay_id', 'rn', 'gcsmotor', 'gcsverbal', 'gcseyes', '_t']].copy()
prev['rn'] = prev.rn + 1
prev = prev.rename(columns={'gcsmotor': 'p_motor', 'gcsverbal': 'p_verbal',
                            'gcseyes': 'p_eyes', '_t': 'p_t'})
b = base.merge(prev, on=['stay_id', 'rn'], how='left')
# 上游条件：b2.charttime > b.charttime - INTERVAL '6' HOUR
in6 = (b.p_t > b._t - pd.Timedelta(hours=6))
for c in ['p_motor', 'p_verbal', 'p_eyes']:
    b.loc[~in6, c] = np.nan

# ---- 3) 复现 gcs 总分的 CASE 逻辑（逐支对齐） ----
def total_row(r):
    if r.gcsverbal == 0:
        return 15
    if pd.isna(r.gcsverbal) and r.p_verbal == 0:
        return 15
    if r.p_verbal == 0:
        return (r.gcsmotor if not pd.isna(r.gcsmotor) else 6) \
             + (r.gcsverbal if not pd.isna(r.gcsverbal) else 5) \
             + (r.gcseyes if not pd.isna(r.gcseyes) else 4)
    m = r.gcsmotor if not pd.isna(r.gcsmotor) else (r.p_motor if not pd.isna(r.p_motor) else 6)
    v = r.gcsverbal if not pd.isna(r.gcsverbal) else (r.p_verbal if not pd.isna(r.p_verbal) else 5)
    e = r.gcseyes if not pd.isna(r.gcseyes) else (r.p_eyes if not pd.isna(r.p_eyes) else 4)
    return m + v + e

# 向量化等价实现（与逐行函数同义，避免 apply 过慢）
def vec_total(df):
    v, pv = df.gcsverbal, df.p_verbal
    m, pm = df.gcsmotor, df.p_motor
    e, pe = df.gcseyes, df.p_eyes
    branch_ett_cur = (v == 0)
    branch_ett_prev = (v.isna()) & (pv == 0)
    branch_prev_ett_notnow = (pv == 0) & ~branch_ett_cur & ~branch_ett_prev
    branch_normal = ~(branch_ett_cur | branch_ett_prev | branch_prev_ett_notnow)
    t = pd.Series(np.nan, index=df.index)
    t[branch_ett_cur | branch_ett_prev] = 15
    t[branch_prev_ett_notnow] = m.fillna(6) + v.fillna(5) + e.fillna(4)
    t[branch_normal] = m.fillna(pm).fillna(6) + v.fillna(pv).fillna(5) + e.fillna(pe).fillna(4)
    return t

# 逐行函数与向量化交叉校验（抽样 3000 行）
b['gcs_recon'] = vec_total(b)
samp = b.sample(min(3000, len(b)), random_state=0)
bad = 0
for _, r in samp.iterrows():
    if abs(total_row(r) - r.gcs_recon) > 1e-9:
        bad += 1
print('向量化 vs 逐行 抽样不一致数:', bad)

# ---- 4) 复现输出分项 = COALESCE(current, prev) ----
b['eyes_recon'] = b.gcseyes.fillna(b.p_eyes)
b['verbal_recon'] = b.gcsverbal.fillna(b.p_verbal)
b['motor_recon'] = b.gcsmotor.fillna(b.p_motor)

# ---- 5) 与派生层比对 ----
cmp = der.merge(b[['stay_id', 'charttime', 'gcs_recon', 'eyes_recon',
                   'verbal_recon', 'motor_recon', 'endotrachflag']],
                on=['stay_id', 'charttime'], how='outer', indicator=True)
print()
print('--- 键匹配 ---')
print(cmp._merge.value_counts().to_string())

both = cmp[cmp._merge == 'both']
def mm(a, bcol):
    x, y = both[a], both[bcol]
    neq = ~((x.isna() & y.isna()) | (np.isclose(x.fillna(-999), y.fillna(-999))))
    return int(neq.sum())

print()
print('--- 逐列不一致行数（both 部分 %d 行）---' % len(both))
print('gcs_total  :', mm('gcs_total', 'gcs_recon'))
print('gcs_eyes   :', mm('gcs_eyes', 'eyes_recon'))
print('gcs_verbal :', mm('gcs_verbal', 'verbal_recon'))
print('gcs_motor  :', mm('gcs_motor', 'motor_recon'))
print('gcs_unable :', mm('gcs_unable', 'endotrachflag'))

# 不一致样例
for a, bc in [('gcs_total', 'gcs_recon'), ('gcs_eyes', 'eyes_recon'),
              ('gcs_verbal', 'verbal_recon'), ('gcs_motor', 'motor_recon')]:
    x, y = both[a], both[bc]
    neq = ~((x.isna() & y.isna()) | (np.isclose(x.fillna(-999), y.fillna(-999))))
    if neq.any():
        print()
        print('### %s != %s 样例（最多 12 行）' % (a, bc))
        cols = ['stay_id', 'charttime', 'gcs_total', 'gcs_eyes', 'gcs_verbal',
                'gcs_motor', 'gcs_unable', 'gcs_recon', 'eyes_recon',
                'verbal_recon', 'motor_recon', 'endotrachflag']
        print(both[neq][cols].head(12).to_string(index=False))
