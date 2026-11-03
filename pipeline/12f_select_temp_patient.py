#!/usr/bin/env python3
# =====================================================================
# 12f_select_temp_patient.py —— 挑选"温度主导"的第二个死亡患者（S18 选患）
#
# 与 09_shap_waterfall.py 完全同源：同 RF、同 seed、同测试集。
# RF 超参与 SHAP 缓存由 shap_cache.py 统一管理（load_shap 自动校验缓存，
# 超参变更时自动重训重算）。GCS 簇取全部 12 列（Score/Eyes/Motor/Verbal
# 的 mean/min/max），与 12g 一致。
#
# 在测试集实际死亡患者中，按 温度相关特征(Temperature mean/min/max，列
# 18/19/20) 的 SHAP 绝对贡献之和排序，打印 Top 候选，供 S18 选患。
# =====================================================================
import numpy as np
from shap_cache import load_shap

res = load_shap()
sv_all = res['sv']; base_i = res['base']; prob = res['prob']
X = res['X']; y = res['y']; te_idx = res['te_idx']; stay_id = res['stay_id']
feat_names = res['feat_names']
y_te = y[te_idx]

temp_cols = [18, 19, 20]           # Temperature (mean/min/max)
gcs_cols = list(range(30, 42))     # GCS Score/Eyes/Motor/Verbal 的 mean/min/max（12 列）
print('Temperature 特征列:', [feat_names[c] for c in temp_cols])

sv_death = sv_all[:, :, 1]         # 死亡类 SHAP
deceased = np.where(y_te == 1)[0]

temp_contrib = np.abs(sv_death[deceased][:, temp_cols]).sum(axis=1)
gcs_contrib = np.abs(sv_death[deceased][:, gcs_cols]).sum(axis=1)

order = np.argsort(-temp_contrib)
print('\n=== 温度贡献 Top 20（死亡患者）===')
print('  测试集idx | 预测P(死) | f(x) | 温度SHAP | GCS SHAP | 温度比GCS | Temp mean | Temp min | Temp max')
for rank, k in enumerate(order[:20]):
    i = deceased[k]
    f_x = float(base_i[i]) + float(sv_death[i].sum())
    raw_temp = X[te_idx[i]][temp_cols]
    ratio = temp_contrib[k] / max(gcs_contrib[k], 1e-9)
    print(f'  {i:>7} | {prob[i]:.3f} | {f_x:.3f} | {temp_contrib[k]:.3f} | '
          f'{gcs_contrib[k]:.3f} | {ratio:.1f}x | {raw_temp[0]:.1f} | {raw_temp[1]:.1f} | {raw_temp[2]:.1f}')

top = int(deceased[order[0]])
print('\n=== 当前温度贡献 Top-1 死亡患者（S18 候选，供 12g 核对）===')
print(f'  测试集 idx={top} | stay_id={stay_id[te_idx[top]]} | f(x)={float(base_i[top]) + float(sv_death[top].sum()):.3f}')
print(f'  Temperature mean={X[te_idx[top]][18]:.1f}°C min={X[te_idx[top]][19]:.1f} max={X[te_idx[top]][20]:.1f}')
print(f'  GCS total mean={X[te_idx[top]][30]:.2f} | 温度簇SHAP={temp_contrib[order[0]]:.3f} | GCS簇SHAP={gcs_contrib[order[0]]:.3f}')
