import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# =====================================================================
# 07_calibration_dca_external.py —— eICU 外部验证 RF 校准曲线 + DCA
#
# 数据源: eicu_predictions.npz（与 11_calibration_recalibration.py 同源）
#   - 'RandomForest': 外部测试集预测概率
#   - 'y_true':       外部真实标签
#
# 校准口径与 07_calibration_dca.py / 11_calibration_recalibration.py
# 完全一致 —— logistic recalibration: logit(Y) = a + b*logit(p)
#
# 输出: figures/calibration_dca_rf_external.png
#   （用于补充材料 Figure S3 的"外部验证 RF 校准 + DCA 合并图"）
# =====================================================================

d = np.load('eicu_predictions.npz')
y_true = d['y_true'].astype(int)
prob = d['RandomForest'].astype(float)
n = len(y_true)
prevalence = y_true.mean()
print(f'外部验证集(eICU): {n} 例 | 死亡率(先验): {prevalence:.3f}')


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_stats(y, p):
    """与 11_calibration_recalibration.py 完全同口径。返回 slope/intercept/CITL/Brier。"""
    lp = logit(p)
    X = sm.add_constant(lp)
    model = sm.Logit(y, X).fit(disp=0)
    a, b = model.params[0], model.params[1]
    ci = model.conf_int()
    z_slope = (b - 1.0) / model.bse[1]
    p_slope1 = 2 * (1 - scipy.stats.norm.cdf(abs(z_slope)))
    model_citl = sm.Logit(y, np.ones_like(y), offset=lp).fit(disp=0)
    a0 = model_citl.params[0]
    a0_ci = model_citl.conf_int()[0]
    return {
        'slope': b, 'slope_ci_low': ci[1][0], 'slope_ci_high': ci[1][1],
        'slope_p_eq1': p_slope1,
        'intercept': a, 'intercept_ci_low': ci[0][0], 'intercept_ci_high': ci[0][1],
        'citl': a0, 'citl_ci_low': a0_ci[0], 'citl_ci_high': a0_ci[1],
        'brier': brier_score_loss(y, p),
    }


cal = calibration_stats(y_true, prob)

fig, (ax_cal, ax_dca) = plt.subplots(1, 2, figsize=(12, 5.5))

fraction_positive, mean_predicted = calibration_curve(
    y_true, prob, n_bins=10, strategy='uniform')

ax_cal.plot(mean_predicted, fraction_positive, 'o-', color='#D62728',
            lw=2.2, ms=6, label='RandomForest (eICU)')
ax_cal.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Perfect calibration')
ax_cal.set_xlabel('Mean predicted probability', fontsize=13)
ax_cal.set_ylabel('Observed fraction of deaths', fontsize=13)
ax_cal.set_xlim(0, 1)
ax_cal.set_ylim(0, 1)

ax_cal.set_title(
    f'External Calibration Curve (RF, eICU)\n'
    f'slope={cal["slope"]:.3f}, intercept={cal["intercept"]:.3f}, CITL={cal["citl"]:.3f} | Brier={cal["brier"]:.3f}',
    fontsize=13)
ax_cal.legend(loc='best', fontsize=11, framealpha=0.9)
ax_cal.grid(alpha=0.3)


def dca_net_benefit(y_true, prob):
    thresholds = np.linspace(0.01, 0.99, 100)
    nb = []
    n = len(y_true)
    for t in thresholds:
        y_pred = (prob >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        nb.append(tp / n - fp / n * (t / (1 - t)))
    return thresholds, np.array(nb)


thr, nb_model = dca_net_benefit(y_true, prob)

thr_ref = np.linspace(0.01, 0.99, 100)
nb_treat_all = prevalence - (1 - prevalence) * thr_ref / (1 - thr_ref)

ax_dca.plot(thr, nb_model, color='#D62728', lw=2.2, label='RandomForest (eICU)')
ax_dca.plot(thr_ref, nb_treat_all, 'k-', lw=1.2, label='Treat all')
ax_dca.plot([0, 1], [0, 0], 'k--', lw=1.2, label='Treat none')
ax_dca.set_xlabel('Threshold probability', fontsize=13)
ax_dca.set_ylabel('Net benefit', fontsize=13)
ax_dca.set_xlim(0, 1)
ax_dca.set_ylim(-0.1, max(nb_treat_all.max(), nb_model.max()) * 1.15)
ax_dca.set_title('Decision Curve (RF, eICU)', fontsize=14)
ax_dca.legend(loc='best', fontsize=11, framealpha=0.9)
ax_dca.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/calibration_dca_rf_external.png', dpi=300, bbox_inches='tight')
print('已保存: figures/calibration_dca_rf_external.png')

print('\n===== eICU 外部 RF 校准指标（与 table10 Ext_ 列同口径）=====')
print(f'Brier Score          : {cal["brier"]:.4f}')
print(f'校准斜率 (slope)     : {cal["slope"]:.3f} (95%CI {cal["slope_ci_low"]:.3f}-{cal["slope_ci_high"]:.3f}, p={cal["slope_p_eq1"]:.3f})')
print(f'校准截距 (intercept) : {cal["intercept"]:.3f} (95%CI {cal["intercept_ci_low"]:.3f}-{cal["intercept_ci_high"]:.3f})')
print(f'CITL                 : {cal["citl"]:.3f} (95%CI {cal["citl_ci_low"]:.3f}-{cal["citl_ci_high"]:.3f})')
