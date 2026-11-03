import numpy as np
import pandas as pd
import joblib

# =====================================================================
# eicu_04_assemble.py —— eICU 外验数据集组装（v2 修复版）
#
# 输入：
#   eicu_hourly.csv    （eicu_02_extract_hourly.sql 输出：48h×14变量）
#   eicu_sofa.csv      （eicu_03_sofa_qsofa.sql 输出：SOFA/qSOFA/标签）
#
# 输出：
#   eicu_training_data.npz
#     X_hcv [n, 14, 48]  小时级变量（顺序对齐 MIMIC 的 HCV_VARS）
#     X_sv  [n, 4]       [age, gender_m, weight, height]
#     y     [n]          住院死亡标签
#     stay_id [n]        eICU patientunitstayid
#     sofa_max_48h, qsofa6, qsofa24  评分
#
# v2 修复：
#   1. pivot_table 加 dropna=False（保留 48h 全 NaN 的患者，避免 KeyError）
#   2. 读取后剔除 label 为 NaN 的患者（hospitaldischargestatus 非 Alive/Expired）
#   3. age 处理 '> 89' -> 89
#
# 关键纪律：
#   - 变量顺序严格对齐 MIMIC 的 05_assemble_training_data.py
#   - gender 编码：Male->1, Female->0
#   - 标准化不在本脚本做（阶段 5 用 MIMIC 的 scaler）
# =====================================================================

HCV_VARS = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'spo2',
            'temperature', 'glucose', 'fio2', 'ph',
            'gcs', 'gcs_eyes', 'gcs_motor', 'gcs_verbal']
SV_NUM = ['age', 'weight', 'height']
SV_BIN = ['gender']

# ---------- 1. 读取（剔除 label NaN 的患者）----------
hourly = pd.read_csv('eicu_hourly.csv')
sofa = pd.read_csv('eicu_sofa.csv')

# 剔除结局无法判定的患者（hospitaldischargestatus 非 Alive/Expired）
sofa = sofa[sofa['label'].notna()].copy()
valid_ids = set(sofa['patientunitstayid'])
hourly = hourly[hourly['patientunitstayid'].isin(valid_ids)].copy()

print(f'eicu_hourly: {len(hourly)} 行, {hourly["patientunitstayid"].nunique()} 患者')
print(f'eicu_sofa: {len(sofa)} 例（剔除 label NaN 后）')

# ---------- 2. 排除行数不足 48 的患者 ----------
n_per = hourly.groupby('patientunitstayid').size()
bad = n_per[n_per != 48].index.tolist()
print(f'排除不足 48h 的患者: {len(bad)} 例')
hourly = hourly[~hourly['patientunitstayid'].isin(bad)].copy()
hourly = hourly.sort_values(['patientunitstayid', 'hr']).reset_index(drop=True)

keep = sorted(hourly['patientunitstayid'].unique())
print(f'最终患者数: {len(keep)}')

# ---------- 3. 组装 X_hcv [n, 14, 48] ----------
n = len(keep)
X_hcv = np.zeros((n, len(HCV_VARS), 48), dtype=np.float32)
for j, var in enumerate(HCV_VARS):
    pivot = hourly.pivot_table(index='patientunitstayid', columns='hr',
                               values=var, aggfunc='first',
                               dropna=False)          # ← v2: 保留全 NaN 患者
    X_hcv[:, j, :] = pivot.loc[keep].values

# ---------- 4. 组装 X_sv [n, 4] ----------
sv = (hourly[['patientunitstayid'] + SV_NUM + SV_BIN]
      .drop_duplicates('patientunitstayid')
      .set_index('patientunitstayid'))

# 处理 eICU age（'> 89' -> 89）
def parse_age(x):
    try:
        return 89 if str(x).strip() == '> 89' else float(x)
    except (ValueError, TypeError):
        return np.nan

sv['age'] = sv['age'].map(parse_age)
sv['gender_m'] = sv['gender'].map({'Male': 1, 'Female': 0})

# weight/height 转数值
sv['weight'] = pd.to_numeric(sv['weight'], errors='coerce')
sv['height'] = pd.to_numeric(sv['height'], errors='coerce')

X_sv = np.zeros((n, 4), dtype=np.float32)
X_sv[:, 0] = sv.loc[keep, 'age'].fillna(sv['age'].mean()).values
X_sv[:, 1] = sv.loc[keep, 'gender_m'].fillna(0).values
X_sv[:, 2] = sv.loc[keep, 'weight'].fillna(sv['weight'].mean()).values
X_sv[:, 3] = sv.loc[keep, 'height'].fillna(sv['height'].mean()).values

# ---------- 5. 标签 + 评分 ----------
sofa = sofa.set_index('patientunitstayid').loc[keep]
y = sofa['label'].astype(np.int8).values
sofa_score = sofa['sofa_max_48h'].astype(float).values
qsofa6 = sofa['qsofa6'].astype(int).values
qsofa24 = sofa['qsofa24'].astype(int).values

# ---------- 6. 保存 ----------
np.savez('eicu_training_data.npz',
         X_hcv=X_hcv,
         X_sv=X_sv,
         y=y,
         stay_id=np.array(keep, dtype=np.int64),
         sofa_max_48h=sofa_score,
         qsofa6=qsofa6,
         qsofa24=qsofa24)

print('保存完成: eicu_training_data.npz')
print(f'  X_hcv: {X_hcv.shape}  X_sv: {X_sv.shape}  y: {y.shape}')
print(f'  阳性(死亡): {y.sum()} ({100*y.mean():.1f}%)')
print(f'  SOFA 中位: {np.median(sofa_score):.1f} | qSOFA6≥2: {100*(qsofa6>=2).mean():.1f}%')