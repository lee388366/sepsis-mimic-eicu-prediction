#!/usr/bin/env python3
# =====================================================================
# 12g_shap_waterfall_temp.py —— S18：温度贡献突出的第二个死亡病例 waterfall
#
# 与 09_shap_waterfall.py 同源同 RF（超参唯一来源 shap_cache.py::RF_PARAMS）。
# 缓存加载自动校验；加载后核对原定患者 (idx 3153) 是否仍是当前模型温度
# 贡献 Top-3 死亡患者，不是则改用当前 Top-1 并打印新的题注数值。
# 结尾打印可直接复制进 docx 的题注数值（f(x)/体温/GCS）。
#
# 输出: figures/shap_waterfall_temp.png
# =====================================================================
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from shap_cache import load_shap

PATIENT_INDEX = 3153   # 温度贡献 Top-1（见 12f 输出）；若超参变更导致更替会自动改用新患者
MAX_DISPLAY = 15

res = load_shap()
sv_all = res['sv']; base_all = res['base']
X = res['X']; y = res['y']; te_idx = res['te_idx']; stay_id = res['stay_id']
feat_names = res['feat_names']

temp_cols = [18, 19, 20]
gcs_cols = list(range(30, 42))
sv_death = sv_all[:, :, 1]
deceased = np.where(y[te_idx] == 1)[0]
tc = np.abs(sv_death[deceased][:, temp_cols]).sum(axis=1)
gc = np.abs(sv_death[deceased][:, gcs_cols]).sum(axis=1)
order = np.argsort(-tc)

if PATIENT_INDEX in deceased[order[:3]]:
    print(f'[S18校验] 原定患者 idx {PATIENT_INDEX} 仍是当前模型温度贡献 Top-3，继续使用。')
else:
    PATIENT_INDEX = int(deceased[order[0]])
    print('!!! 原定患者已不在温度贡献 Top-3，改用当前 Top-1 !!!')
    print(f'    新患者: idx {PATIENT_INDEX} (stay_id={stay_id[te_idx[PATIENT_INDEX]]})')
    print('    请更新本文件顶部 PATIENT_INDEX，并把下方题注数值复制进 docx。')

i = PATIENT_INDEX
print(f'展示患者: 测试集下标 {i} | 真实标签 = {"死亡" if y[te_idx[i]] == 1 else "存活"} | '
      f'stay_id={stay_id[te_idx[i]]}')

sv_i = shap.Explanation(
    values=sv_death[i],
    base_values=float(base_all[i]),
    data=X[te_idx[i]],
    feature_names=feat_names,
)
f_x = float(base_all[i]) + float(sv_death[i].sum())
print(f'基线 E[f(X)] = {float(base_all[i]):.4f} → 该患者 f(x) = {f_x:.4f}')

temp_contrib = np.abs(sv_death[i, temp_cols]).sum()
gcs_contrib = np.abs(sv_death[i, gcs_cols]).sum()
print(f'温度簇 SHAP 贡献 = {temp_contrib:.3f} | GCS 簇 = {gcs_contrib:.3f}')

print('\n=== S18 题注数值（复制进 docx 题注/正文）===')
print(f'  f(x) = {f_x:.2f}')
print(f'  mean temperature = {X[te_idx[i]][18]:.1f}°C (mean/min/max = '
      f'{X[te_idx[i]][18]:.1f}/{X[te_idx[i]][19]:.1f}/{X[te_idx[i]][20]:.1f})')
print(f'  total GCS (mean) ≈ {X[te_idx[i]][30]:.1f}')
print(f'  温度簇 SHAP = {temp_contrib:.3f} | GCS 簇 SHAP = {gcs_contrib:.3f}')

shap.plots.waterfall(sv_i, max_display=MAX_DISPLAY, show=False)
plt.gcf().set_size_inches(10, 7)
plt.savefig('figures/shap_waterfall_temp.png', dpi=300, bbox_inches='tight')
print('已保存: figures/shap_waterfall_temp.png')
