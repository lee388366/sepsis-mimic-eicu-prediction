import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =====================================================================
# eicu_09_plot_dca_cic_external.py —— eICU 外验集 DCA + CIC
#
# 目的：MIMIC 内部已有 DCA/CIC（10_plot_dca_compare.py / 07_plot_clinical_impact.py
#       / 10_plot_qsofa_lstm_cic.py），本脚本在 eICU 外验集上补齐净获益分析。
#   - DCA：LSTM / RF / LR（概率型模型）vs Treat all / Treat none
#   - CIC：LSTM / RF 连续曲线 + qSOFA-6h vs LSTM 阶梯对比
#
# 数据源: eicu_predictions.npz（eicu_06_external_validate.py 生成）
#   - 7 个模型外验概率 + y_true + sofa + qsofa6 + stay_id（n=6,812）
#
# 关键纪律:
#   - LSTM 报告阈值沿用 MIMIC 验证集 F3 最优阈值（sequence_results.csv），
#     不在外验集上重新选阈值（防用外验调参的过拟合陷阱）。
#   - DCA 只画概率型模型：x 轴是阈值概率，SOFA/qSOFA 整数评分混入不严谨
#     （与 10_plot_dca_compare.py 同一方法学决定）。
#     SOFA/qSOFA 的临床效用由 CIC 阶梯图呈现。
#
# 输出: figures/dca_compare_external.png
#       figures/clinical_impact_curve_lstm_external.png
#       figures/clinical_impact_curve_rf_external.png
#       figures/cic_qsofa6_vs_lstm_external.png
#       终端打印净获益表 + CIC 数值表（供正文引用）
# =====================================================================

d = np.load('eicu_predictions.npz')
y_true = d['y_true'].astype(int)
n = len(y_true)
prevalence = y_true.mean()
print(f'eICU 外验集: {n} 例 | 死亡率 {prevalence:.3f}')

# LSTM 操作阈值沿用 MIMIC 验证集 F3 最优阈值（外验不重新选阈值，防过拟合）
seq = pd.read_csv('sequence_results.csv').set_index('model').loc['LSTM']
LSTM_THR = float(seq['best_threshold'])
print(f'LSTM 报告阈值（MIMIC 验证集 F3 最优，外验沿用）: {LSTM_THR:.3f}')

# ---------- 1. DCA：标准 Vickers 净获益 ----------
def dca_net_benefit(y_true, score):
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

models = {
    'LSTM': d['LSTM'].astype(float),
    'RF':   d['RandomForest'].astype(float),
    'LR':   d['LogisticRegression'].astype(float),
}
COLORS = {'LSTM': '#D62728', 'RF': '#1F77B4', 'LR': '#2CA02C'}

thr_ref = np.linspace(0.01, 0.99, 200)
nb_all = prevalence - (1 - prevalence) * thr_ref / (1 - thr_ref)

plt.figure(figsize=(9, 6.5))
plt.plot(thr_ref, nb_all, 'k-', lw=1.5, label='Treat all')
plt.plot([0, 1], [0, 0], 'k--', lw=1.2, label='Treat none')

nb_store = {}
for name, score in models.items():
    thr, nb = dca_net_benefit(y_true, score)
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
plt.savefig('figures/dca_compare_external.png', dpi=300, bbox_inches='tight')
print('已保存: figures/dca_compare_external.png')

# 外验 DCA 数值表（精确阈值，不用网格近似）
def nb_exact(score, t):
    """在精确阈值 t 处直接算 Vickers 净获益（不依赖网格取整）"""
    yp = (score >= t).astype(int)
    tp = np.sum((yp == 1) & (y_true == 1))
    fp = np.sum((yp == 1) & (y_true == 0))
    return tp / n - fp / n * (t / (1 - t))

print('\n===== eICU 外验净获益对比（阈值 0.10/0.15/0.20/0.25/0.30）=====')
for t in [0.10, 0.15, 0.20, 0.25, 0.30]:
    nb_t = prevalence - (1 - prevalence) * t / (1 - t)
    row = f'阈值 {t:.2f}: TreatAll={nb_t:.4f}'
    for name, score in models.items():
        row += f' | {name}={nb_exact(score, t):.4f}'
    print(row)

# ---------- 2. CIC：LSTM / RF 连续曲线 ----------
thresholds = np.linspace(0.01, 0.99, 99)

def cic_curves(prob):
    n_hr = np.zeros(len(thresholds))
    n_ev = np.zeros(len(thresholds))
    for i, t in enumerate(thresholds):
        mask = prob >= t
        n_hr[i] = mask.sum()
        n_ev[i] = (mask & (y_true == 1)).sum()
    return n_hr, n_ev

def plot_cic(prob, fname, color='#D62728'):
    n_hr, n_ev = cic_curves(prob)
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, n_hr, color=color, lw=2.2, label='Number high risk')
    plt.plot(thresholds, n_ev, color='#1F77B4', lw=2.2,
             label='Number high risk with event')
    plt.fill_between(thresholds, n_ev, n_hr, color='gray', alpha=0.15,
                     label='Over-triage gap (non-event among high risk)')
    plt.xlabel('High-risk threshold probability', fontsize=13)
    plt.ylabel('Number of patients classified as high risk', fontsize=13)
    plt.xlim(0, 1)
    plt.ylim(0, n_hr.max() * 1.05)
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'已保存: {fname}')

plot_cic(d['LSTM'].astype(float), 'figures/clinical_impact_curve_lstm_external.png', '#D62728')
plot_cic(d['RandomForest'].astype(float), 'figures/clinical_impact_curve_rf_external.png', '#1F77B4')

# ---------- 3. CIC：qSOFA-6h vs LSTM 阶梯对比 ----------
q6 = d['qsofa6'].astype(int)
lstm = d['LSTM'].astype(float)
n_death = y_true.sum()

q_thresholds = np.array([0, 1, 2, 3, 4]) - 0.5
q_hr, q_ev = [], []
for t in q_thresholds:
    mask = q6 >= t
    q_hr.append(mask.sum())
    q_ev.append((mask & (y_true == 1)).sum())

thr = np.linspace(0.01, 0.99, 99)
l_hr = np.array([(lstm >= t).sum() for t in thr])
l_ev = np.array([((lstm >= t) & (y_true == 1)).sum() for t in thr])

fig, ax = plt.subplots(figsize=(9, 6.5))
ax.plot(thr, l_hr, color='#D62728', lw=2.2, label='LSTM - number high risk')
ax.plot(thr, l_ev, color='#D62728', lw=2.2, ls='--', label='LSTM - high risk with event')
ax.fill_between(thr, l_ev, l_hr, color='#D62728', alpha=0.08)
ax.step(q_thresholds, q_hr, where='post', color='#000000', lw=2.0, label='qSOFA-6h - number high risk')
ax.step(q_thresholds, q_ev, where='post', color='#000000', lw=2.0, ls='--', label='qSOFA-6h - high risk with event')

ax.axvline(LSTM_THR, color='#D62728', ls=':', lw=1.5)
ax.annotate(f'LSTM op. point\n(thr={LSTM_THR:.2f})', xy=(LSTM_THR, (lstm >= LSTM_THR).sum()),
            xytext=(0.35, n * 0.5), fontsize=10, color='#D62728')
ax.axvline(1.5, color='#000000', ls=':', lw=1.5)
ax.annotate('qSOFA op. point\n(score≥2)', xy=(1.5, (q6 >= 2).sum()),
            xytext=(0.06, n * 0.5), fontsize=10)

ax.set_xlabel('Threshold probability / qSOFA score', fontsize=13)
ax.set_ylabel('Number of patients classified as high risk', fontsize=13)
ax.set_xlim(0, 1)
ax.set_ylim(0, n * 1.05)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/cic_qsofa6_vs_lstm_external.png', dpi=300, bbox_inches='tight')
print('已保存: figures/cic_qsofa6_vs_lstm_external.png')

# ---------- 4. 操作点数字（供正文引用）----------
def summarize(name, mask):
    hr = mask.sum()
    ev = (mask & (y_true == 1)).sum()
    miss = n_death - ev
    print(f'\n{name}:')
    print(f'  高危人数: {hr} ({100*hr/n:.1f}% test set)')
    print(f'  其中死亡: {ev} (精度 {ev/hr:.3f})')
    print(f'  漏掉死亡: {miss} ({100*miss/n_death:.1f}%)')
    print(f'  虚警(高危但存活): {hr-ev} ({100*(hr-ev)/hr:.1f}% of high-risk)')

summarize('qSOFA-6h (score≥2, eICU 外验)', q6 >= 2)
summarize(f'LSTM (thr={LSTM_THR:.2f}, eICU 外验)', lstm >= LSTM_THR)

# CIC 关键阈值表
print('\n===== eICU 外验 LSTM CIC 关键阈值（供正文引用）=====')
print(f'{"Threshold":<10}{"High risk (n)":<18}{"with event (n)":<18}{"Precision":<12}{"事件捕获率"}')
for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]:
    mask = lstm >= t
    hr = mask.sum()
    ev = (mask & (y_true == 1)).sum()
    prec = ev / hr if hr > 0 else 0.0
    captured = ev / n_death
    print(f'{t:<10.2f}{hr:<18}{ev:<18}{prec:<12.3f}{captured:.3f}')
