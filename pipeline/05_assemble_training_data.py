import numpy as np
import pandas as pd

# =====================================================================
# 05_assemble_training_data.py
# 组装训练数据：sepsis_hourly_imputed.csv(特征) + sepsis_labels.csv(标签)
#
# 口径：以 sepsis_labels.csv 的 stay_id 为准（rn=1，24,894 例），
#       从 hourly_imputed 中只保留这些患者的特征，保证特征-标签对齐。
#
# 输出 sepsis_training_data.npz：
#   X_hcv  [n, 14, 48]  float32   HCV 小时级变量（14 变量 × 48h）
#   X_sv   [n, 4]       float32   静态变量 [age, gender_m, weight, height]
#   y      [n]          int8      结局标签（1=死亡，0=存活）
#   stay_id [n]         int64     对齐用的 stay_id
#   subject_id [n]      int64     患者级 ID（受试者级分组切分用，
#                                 同一患者的多次入住保证在同一集合）
# =====================================================================

HCV_VARS = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'spo2',
            'temperature', 'glucose', 'fio2', 'ph',
            'gcs', 'gcs_eyes', 'gcs_motor', 'gcs_verbal']
SV_NUM = ['age', 'weight', 'height']      # 数值型静态变量
SV_BIN = ['gender']                        # 二分类静态变量（M/F）

# 1. 读取数据
hourly = pd.read_csv('sepsis_hourly_imputed_v2.csv')
labels = pd.read_csv('sepsis_labels.csv')

# 2. 按标签口径过滤（rn=1 对齐）
keep = labels['stay_id'].astype(int).values
hourly = hourly[hourly['stay_id'].isin(keep)].copy()
hourly = hourly.sort_values(['stay_id', 'hr']).reset_index(drop=True)

# 3. 校验：每个患者必须恰好 48 行
n_per_stay = hourly.groupby('stay_id').size()
assert (n_per_stay == 48).all(), '存在不足 48 小时的患者，请检查'
print(f'患者数: {len(keep)}，hourly 行数: {len(hourly)}，校验通过')

# 4. 组装 HCV 张量 [n, 14, 48]
n = len(keep)
X_hcv = np.zeros((n, len(HCV_VARS), 48), dtype=np.float32)
for j, var in enumerate(HCV_VARS):
    pivot = hourly.pivot(index='stay_id', columns='hr', values=var)
    X_hcv[:, j, :] = pivot.loc[keep].values

# 5. 组装静态变量 [n, 4]
sv = (hourly[['stay_id'] + SV_NUM + SV_BIN]
      .drop_duplicates('stay_id')
      .set_index('stay_id'))

# gender 编码：M -> 1, F -> 0（其余按缺失处理）
sv['gender_m'] = sv['gender'].map({'M': 1, 'F': 0})

# weight/height 缺失用全体均值填充（Liu 的做法，静态变量用均值）
for c in SV_NUM:
    sv[c] = sv[c].fillna(sv[c].mean())

X_sv = np.zeros((n, 4), dtype=np.float32)
X_sv[:, 0] = sv.loc[keep, 'age'].values
X_sv[:, 1] = sv.loc[keep, 'gender_m'].values
X_sv[:, 2] = sv.loc[keep, 'weight'].values
X_sv[:, 3] = sv.loc[keep, 'height'].values

# 6. 标签 [n]（从 labels 文件取，保证与特征同一顺序）
y_df = labels.set_index('stay_id').loc[keep]
y = y_df['label'].astype(np.int8).values

# 7. 保存
np.savez('sepsis_training_data.npz',
         X_hcv=X_hcv,
         X_sv=X_sv,
         y=y,
         stay_id=np.array(keep, dtype=np.int64),
         subject_id=labels.set_index('stay_id').loc[keep, 'subject_id'].values.astype(np.int64))

print('保存完成: sepsis_training_data.npz')
print(f'  X_hcv: {X_hcv.shape}  X_sv: {X_sv.shape}  y: {y.shape}')
print(f'  阳性(死亡): {y.sum()} ({100*y.mean():.1f}%)')