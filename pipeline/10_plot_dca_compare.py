import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =====================================================================
# 10_plot_dca_compare.py —— 多模型 DCA 对比（ML 模型临床效用）
#
# 在同一决策曲线上绘制：
#   - LSTM / RF / LR（输出概率的 ML 模型，DCA 框架天然适配）
#   - "全部患者治疗"（Treat all）与 "不治疗"（Treat none）基线
#
# 为什么不画 SOFA/qSOFA：
#   DCA 的 x 轴是"风险阈值概率"，SOFA/qSOFA 是整数评分而非概率，
#   混在同一概率轴上方法学上不严谨（且整数评分在连续阈值下产生
#   空洞区间/-inf）。SOFA/qSOFA 的临床效用由 CIC 图呈现：
#   - figures/cic_qsofa6_vs_lstm.png（qSOFA-6h vs LSTM）
#   - figures/cic_qsofa_24h.png / cic_qsofa_windows.png（qSOFA 单独）
#   - figures/clinical_impact_curve_lstm.png（LSTM 单独）
#
# 数据源: all_predictions.npz（同一 test set）
# 输出: figures/dca_compare.png + 净获益数值表（供正文引用）
# =====================================================================

m = np.load('all_predictions.npz')
y_true = m['y_true'].astype(int)

n = len(y_true)
prevalence = y_true.mean()
print(f'test set: {n} 例 | 死亡率 {prevalence:.3f}')

# 只含输出概率的 ML 模型
models = {
    'LSTM': m['LSTM'].astype(float),
    'RF':   m['RandomForest'].astype(float),
    'LR':   m['LogisticRegression'].astype(float),
}

def dca_net_benefit(y_true, score, n):
    """标准 Vickers DCA 净获益。阈值扫描 0.01-0.99，避开 t>=1 除零。"""
    thr = np.linspace(0.01, 0.99, 200)
    nb = np.zeros(len(thr))
    for i, t in enumerate(thr):
        if t >= 1.0:
            nb[i] = np.nan
            continue
        yp = (score >= t).astype(int)
        tp = np.sum((yp == 1) & (y_true == 1))
        fp = np.sum((yp == 1) & (y_true == 0))
        nb[i] = tp / n - fp / n * (t / (1 - t))
    return thr, nb

plt.figure(figsize=(9, 6.5))

# Treat all / Treat none 基线
thr_ref = np.linspace(0.01, 0.99, 200)
nb_all = prevalence - (1 - prevalence) * thr_ref / (1 - thr_ref)
plt.plot(thr_ref, nb_all, 'k-', lw=1.5, label='Treat all')
plt.plot([0, 1], [0, 0], 'k--', lw=1.2, label='Treat none')

COLORS = {'LSTM': '#D62728', 'RF': '#1F77B4', 'LR': '#2CA02C'}

nb_store = {}
for name, score in models.items():
    thr, nb = dca_net_benefit(y_true, score, n)
    nb_store[name] = (thr, nb)
    plt.plot(thr, nb, color=COLORS[name], lw=2.2, label=name)

plt.xlabel('Threshold probability', fontsize=13)
plt.ylabel('Net benefit', fontsize=13)
plt.xlim(0, 1)
all_valid = np.concatenate([nb_all] + [nb[np.isfinite(nb)] for _, nb in nb_store.values()])
plt.ylim(-0.1, np.max(all_valid) * 1.15)
plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/dca_compare.png', dpi=300, bbox_inches='tight')
print('已保存: figures/dca_compare.png')

# ---------- 净获益对比表（供正文引用）----------
print('\n===== 净获益对比（阈值 0.10/0.15/0.20/0.30）=====')
for t in [0.10, 0.15, 0.20, 0.30]:
    nb_t = prevalence - (1 - prevalence) * t / (1 - t)
    row = f'阈值 {t:.2f}: TreatAll={nb_t:.4f}'
    for name in models:
        thr, nb = nb_store[name]
        idx = np.argmin(np.abs(thr - t))
        row += f' | {name}={nb[idx]:.4f}'
    print(row)

# ---------- 临床相关区间 (0.05-0.40) 净获益对比（修正版）----------
print('\n===== 临床相关区间 (0.05-0.40) 净获益对比 =====')
clin_thr = np.linspace(0.05, 0.40, 36)

for name in ['LSTM', 'RF', 'LR']:
    thr, nb = nb_store[name]
    nb_clin = np.interp(clin_thr, thr, nb)
    best_i = np.argmax(nb_clin)
    print(f'  {name:5s}: 区间内净获益均值 = {nb_clin.mean():.4f} | '
          f'峰值 = {nb_clin.max():.4f} @ 阈值 {clin_thr[best_i]:.2f}')

nb_all_clin = prevalence - (1 - prevalence) * clin_thr / (1 - clin_thr)
print(f'  TreatAll: 区间内净获益均值 = {nb_all_clin.mean():.4f}')

# LSTM vs RF 的净获益差值（0.05-0.40）
thr_l, nb_l = nb_store['LSTM']
thr_r, nb_r = nb_store['RF']
nb_l_clin = np.interp(clin_thr, thr_l, nb_l)
nb_r_clin = np.interp(clin_thr, thr_r, nb_r)
diff_clin = nb_l_clin - nb_r_clin
print(f'\n  LSTM - RF 净获益差值（0.05-0.40）: 均值 {np.mean(diff_clin):+.4f} | '
      f'LSTM 占优区间 = {clin_thr[diff_clin > 0].min():.2f}–{clin_thr[diff_clin > 0].max():.2f}')

# LSTM 超过 Treat all 的临床区间
nb_all_clin = prevalence - (1 - prevalence) * clin_thr / (1 - clin_thr)
over = clin_thr[nb_l_clin > nb_all_clin]
if len(over):
    print(f'  LSTM 净获益 > Treat all 的区间: {over.min():.2f} – {over.max():.2f}')