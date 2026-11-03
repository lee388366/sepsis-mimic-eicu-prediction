import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# =====================================================================
# eicu_05_impute.py —— eICU 外验数据插补（对齐 MIMIC 的 03_impute.py）
#
# 输入: eicu_training_data.npz（X_hcv 原始 NaN）
# 输出: eicu_training_data_imputed.npz（插补后）
#
# 策略（方案 A）：
#   固定值用 MIMIC 的 FIXED 生理常数（跨人群通用）
#   患者均值用 eICU 自己的均值（反映外验人群）
#   规则与 03_impute.py 完全一致：
#     - 48h 全无观测 → FIXED 固定值
#     - <24 个观测 → 该患者该变量均值
#     - >=24 个观测 → nearest 插值（含外推）
#
# 注意：X_hcv 形状 [n, 14, 48]，变量顺序 = MIMIC 的 HCV_VARS
# =====================================================================

# MIMIC 03_impute.py 的 FIXED 字典（生理正常值，照抄）
FIXED = {
    'heart_rate': 86, 'sbp': 118, 'dbp': 59, 'mbp': 77,
    'resp_rate': 19, 'spo2': 98, 'temperature': 36,
    'glucose': 128, 'fio2': 21, 'ph': 7.4,
    'gcs': 12, 'gcs_eyes': 4, 'gcs_motor': 5, 'gcs_verbal': 4,
}
HCV_VARS = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'spo2',
            'temperature', 'glucose', 'fio2', 'ph',
            'gcs', 'gcs_eyes', 'gcs_motor', 'gcs_verbal']
ALL_HOURS = np.arange(48)

# ---------- 1. 加载 ----------
d = np.load('eicu_training_data.npz')
X_hcv = d['X_hcv'].astype(np.float32)   # [n, 14, 48]
X_sv  = d['X_sv'].astype(np.float32)
y     = d['y'].astype(np.int64)
stay_id = d['stay_id']
print(f'加载: X_hcv {X_hcv.shape} | y 阳性率 {100*y.mean():.1f}%')

# ---------- 2. 逐患者逐变量插补 ----------
X_imp = X_hcv.copy()
n_pat = X_hcv.shape[0]

for i in range(n_pat):
    for j, var in enumerate(HCV_VARS):
        series = X_hcv[i, j, :]                  # 48 小时
        n_obs = np.sum(~np.isnan(series))

        if n_obs == 0:
            # ① 全无观测 → FIXED 固定值
            X_imp[i, j, :] = FIXED[var]

        elif n_obs < 24:
            # ② 不足一半 → 该患者该变量均值
            X_imp[i, j, :] = np.nanmean(series)

        else:
            # ③ 过半 → nearest 插值（含外推）
            obs_mask = ~np.isnan(series)
            hours_obs = ALL_HOURS[obs_mask]
            interp = interp1d(hours_obs, series[obs_mask],
                              kind='nearest',
                              bounds_error=False,
                              fill_value='extrapolate')
            X_imp[i, j, :] = interp(ALL_HOURS)

print('插补完成')

# ---------- 3. 校验：不应再有 NaN ----------
n_nan = np.isnan(X_imp).sum()
print(f'插补后残留 NaN: {n_nan}')
assert n_nan == 0, '仍有 NaN！'

# ---------- 4. 保存 ----------
np.savez('eicu_training_data_imputed.npz',
         X_hcv=X_imp,
         X_sv=X_sv,
         y=y,
         stay_id=stay_id,
         sofa_max_48h=d['sofa_max_48h'],
         qsofa6=d['qsofa6'],
         qsofa24=d['qsofa24'])

print('保存完成: eicu_training_data_imputed.npz')
print(f'  X_hcv: {X_imp.shape}（已插补，无 NaN）')

# ---------- 5. 各变量插补方式统计（供论文报告缺失处理）----------
print('\n各变量插补情况（全队列）:')
for j, var in enumerate(HCV_VARS):
    n_missing = np.isnan(X_hcv[:, j, :]).sum()
    rate = 100 * n_missing / (n_pat * 48)
    # 统计该变量有多少患者"全缺失"（用 FIXED 填充）
    n_full_miss = np.isnan(X_hcv[:, j, :]).all(axis=1).sum()
    print(f'  {var:15s}: 缺失率 {rate:5.1f}% | 全缺失(固定值)患者 {n_full_miss:4d} ({100*n_full_miss/n_pat:.1f}%)')