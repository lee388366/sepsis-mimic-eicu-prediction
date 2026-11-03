import numpy as np
import pandas as pd
from scipy import stats

# =====================================================================
# make_table1_external.py —— eICU 外验集 Table 1 基线特征表（存活 vs 死亡）
#
# 数据源:
#   eicu_training_data_imputed.npz  (6,812 例，eicu_05_impute.py 产物，
#                                    X_hcv 已插补无 NaN——基线表须用插补后数据，
#                                    与 MIMIC make_table1.py 口径一致)
#     X_sv  [n, 4]  = [age, gender_m, weight, height]
#     X_hcv [n, 14, 48] 变量顺序见 HCV_VARS（与 MIMIC 一致）
#     y     [n]
#   eicu_demographics.csv  (6,883 行，eicu_04b_extract_demographics.sql 产物，
#                           按 patientunitstayid 对齐；仅 6,812 例在最终外验集)
#
# 统计方法（与 make_table1.py 完全一致）:
#   连续变量 -> 中位数(IQR) + Mann-Whitney U 检验（GCS 系列同样先取患者级 48h 均值）
#   分类变量 -> n(%) + 卡方检验
#   P 值 <0.001 统一显示为 '<0.001'（避免 '0.000'）
# 输出: tables/table1_external.csv（表头列名已标注统计量类型）
# =====================================================================

d = np.load('eicu_training_data_imputed.npz')
X_sv = d['X_sv']                  # [n, 4]
X_hcv = d['X_hcv']                # [n, 14, 48]
y = d['y'].astype(int)
stay_id = d['stay_id']

N = len(y)
n_surv = (y == 0).sum()
n_dead = (y == 1).sum()
print(f'总数 {N} | 存活 {n_surv} | 死亡 {n_dead}')

# ---------- 种族（eicu_demographics.csv，按 patientunitstayid 对齐）----------
demog = pd.read_csv('eicu_demographics.csv')
race_by_stay = demog.set_index('patientunitstayid')['race_group']
races = race_by_stay.reindex(stay_id).fillna('Other/Unknown').values
RACE_ORDER = ['White', 'Black', 'Hispanic', 'Asian', 'Other/Unknown']
print(f'种族对齐: {len(races)} 例（demographics {len(demog)} 行 → npz {N} 例）')

SV_NAMES = ['Age (years)', 'Gender (male)', 'Weight (kg)', 'Height (cm)']
SV_CAT = {1: 'Gender (male)'}   # gender_m 是分类

HCV_VARS = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'spo2',
            'temperature', 'glucose', 'fio2', 'ph', 'gcs',
            'gcs_eyes', 'gcs_motor', 'gcs_verbal']
HCV_LABELS = ['Heart rate (bpm)', 'Systolic BP (mmHg)', 'Diastolic BP (mmHg)',
              'Mean BP (mmHg)', 'Respiratory rate (/min)', 'SpO2 (%)',
              'Temperature (°C)', 'Glucose (mg/dL)', 'FiO2 (fraction)',
              'pH', 'GCS total', 'GCS eyes', 'GCS motor', 'GCS verbal']
rows = []

def fmt_median_iqr(x):
    """median (IQR) 格式"""
    med = np.median(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    return f'{med:.1f} ({iqr:.1f})'

def fmt_n_pct(mask, total):
    """n (%) 格式"""
    n = int(mask.sum())
    return f'{n} ({100*n/total:.1f}%)'

def fmt_pvalue(p):
    """P 值格式化：<0.001 统一为 '<0.001'（避免 0.000），否则三位小数"""
    return '<0.001' if p < 0.001 else f'{p:.3f}'

# ---------- 静态变量 ----------
# 年龄
age = X_sv[:, 0]
rows.append(['Age (years)',
             fmt_median_iqr(age[y == 0]), fmt_median_iqr(age[y == 1]),
             fmt_pvalue(stats.mannwhitneyu(age[y==0], age[y==1]).pvalue)])

# 性别（分类）
g = X_sv[:, 1]
m_surv = g[y == 0]
m_dead = g[y == 1]
ct = np.array([[int((m_surv == 1).sum()), int((m_surv == 0).sum())],
               [int((m_dead == 1).sum()), int((m_dead == 0).sum())]])
chi2, p_chi, _, _ = stats.chi2_contingency(ct)
rows.append(['Gender (male)', fmt_n_pct(m_surv == 1, n_surv),
             fmt_n_pct(m_dead == 1, n_dead), fmt_pvalue(p_chi)])

# 种族（分类，eICU 五分类）
for r in RACE_ORDER:
    surv_mask = races[y == 0] == r
    dead_mask = races[y == 1] == r
    ct = np.array([[int(surv_mask.sum()), int((~surv_mask).sum())],
                   [int(dead_mask.sum()), int((~dead_mask).sum())]])
    chi2, p_race, _, _ = stats.chi2_contingency(ct)
    rows.append([f'Race: {r}', fmt_n_pct(surv_mask, n_surv),
                 fmt_n_pct(dead_mask, n_dead), fmt_pvalue(p_race)])

# 体重 / 身高
for idx in (2, 3):
    v = X_sv[:, idx]
    rows.append([SV_NAMES[idx],
                 fmt_median_iqr(v[y == 0]), fmt_median_iqr(v[y == 1]),
                 fmt_pvalue(stats.mannwhitneyu(v[y==0], v[y==1]).pvalue)])

# ---------- 临床变量（48h 均值）----------
for j, (var, label) in enumerate(zip(HCV_VARS, HCV_LABELS)):
    series = X_hcv[:, j, :]
    mean_series = series.mean(axis=1)
    rows.append([label,
                 fmt_median_iqr(mean_series[y == 0]),
                 fmt_median_iqr(mean_series[y == 1]),
                 fmt_pvalue(stats.mannwhitneyu(mean_series[y==0], mean_series[y==1]).pvalue)])

# ---------- 输出 ----------
df = pd.DataFrame(rows, columns=['Variable',
                                 'Survived (n=%d, median (IQR) or n (%%))' % n_surv,
                                 'Died (n=%d, median (IQR) or n (%%))' % n_dead,
                                 'P value']) 
pd.set_option('display.max_rows', None)
print('\n===== Table 1 (eICU external) =====')
print(df.to_string(index=False))
df.to_csv('tables/table1_external.csv', index=False)
print('\n已保存: tables/table1_external.csv')
