# 11_calibration_recalibration.py
# 标准做法：logistic recalibration 求校准斜率/截距/CITL（TRIPOD 标准，Cox 1958）
#          + sklearn 官方 CalibrationDisplay 画校准曲线
#          + eICU 上做再校准（logistic + isotonic），验证集评估
# 依赖: sklearn 1.9 / statsmodels / scipy / matplotlib
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.calibration import CalibrationDisplay
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

SEED = 42
MODELS = ['LSTM', 'LogisticRegression', 'DecisionTree', 'RandomForest',
          'KNN', 'SVM', 'GaussianNB']

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def invlogit(x):
    return 1.0 / (1.0 + np.exp(-x))

def calibration_stats(y, p):
    """logistic recalibration: logit(Y) = a + b*logit(p)。
    返回斜率 b、截距 a、CITL(offset 模型)、各自 95%CI、斜率=1 检验 p 值、Brier。"""
    lp = logit(p)
    X = sm.add_constant(lp)
    model = sm.Logit(y, X).fit(disp=0)
    a, b = model.params[0], model.params[1]
    ci = model.conf_int()
    z_slope = (b - 1.0) / model.bse[1]
    p_slope1 = 2 * (1 - scipy.stats.norm.cdf(abs(z_slope)))
    # CITL: 斜率固定为 1 的 offset 模型（calibration-in-the-large）
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

def plot_all_calibration(y_true, probs, title, path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Perfect calibration')
    for name in MODELS:
        disp = CalibrationDisplay.from_predictions(
            y_true, probs[name], n_bins=10, strategy='uniform',
            name=name, ax=ax, ref_line=False)
    ax.set_xlabel('Mean predicted probability', fontsize=13)
    ax.set_ylabel('Observed fraction of deaths', fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ---------- 1. 加载内部/外部预测 ----------
mi = np.load('all_predictions.npz')
me = np.load('eicu_predictions.npz')
y_i, y_e = mi['y_true'].astype(int), me['y_true'].astype(int)
probs_i = {n: mi[n].astype(float) for n in MODELS}
probs_e = {n: me[n].astype(float) for n in MODELS}
print(f'内部 test: {len(y_i)} 例 | 外部 eICU: {len(y_e)} 例')

# ---------- 2. 内部 + 外部 校准指标表 ----------
rows = []
for n in MODELS:
    r = {'Model': n}
    r.update({f'Int_{k}': v for k, v in calibration_stats(y_i, probs_i[n]).items()})
    r.update({f'Ext_{k}': v for k, v in calibration_stats(y_e, probs_e[n]).items()})
    r['Int_AUC'] = roc_auc_score(y_i, probs_i[n])
    r['Ext_AUC'] = roc_auc_score(y_e, probs_e[n])
    rows.append(r)
cal = pd.DataFrame(rows).sort_values('Ext_AUC', ascending=False).reset_index(drop=True)
cal.to_csv('tables/table10_calibration_internal_external.csv', index=False)
print('\n===== 内外部校准指标（完整表见 tables/table10_calibration_internal_external.csv）=====')
print(cal.round(3).to_string(index=False))

# ---------- 3. 校准曲线图 ----------
plot_all_calibration(y_i, probs_i, 'Internal calibration (test set, n=3742)',
                     'figures/calibration_curves_internal.png')
plot_all_calibration(y_e, probs_e, 'External calibration (eICU, n=6812)',
                     'figures/calibration_curves_external.png')
print('\n已保存: figures/calibration_curves_internal.png / _external.png')

# ---------- 4. eICU 再校准（50/50 分层切分，验证集评估）----------
idx = np.arange(len(y_e))
tr, va = train_test_split(idx, test_size=0.5, stratify=y_e, random_state=SEED)
rec_rows = []
for n in MODELS:
    p = probs_e[n]
    # (a) logistic recalibration（标准，Cox）
    lp_tr = logit(p[tr])
    m = sm.Logit(y_e[tr], sm.add_constant(lp_tr)).fit(disp=0)
    a_, b_ = m.params[0], m.params[1]
    p_log = invlogit(a_ + b_ * logit(p[va]))
    # (b) isotonic recalibration
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(p[tr], y_e[tr])
    p_iso = iso.predict(p[va])
    rec_rows.append({
        'Model': n,
        'Brier_raw': brier_score_loss(y_e[va], p[va]),
        'Brier_logistic': brier_score_loss(y_e[va], p_log),
        'Brier_isotonic': brier_score_loss(y_e[va], p_iso),
        'recal_a': a_, 'recal_b': b_,
    })
rec = pd.DataFrame(rec_rows)
rec.to_csv('tables/table10_recalibration_eicu.csv', index=False)
print('\n===== eICU 再校准（验证集 50%）=====')
print(rec.round(4).to_string(index=False))
print('Brier 越小越好；比较 Brier_raw vs Brier_logistic/isotonic 看再校准是否改善')
print('已保存: tables/table10_recalibration_eicu.csv')