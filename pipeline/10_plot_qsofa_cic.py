import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =====================================================================
# 10_plot_qsofa_cic.py —— qSOFA 单独 CIC（临床影响曲线）
#
# 与 10_plot_qsofa_lstm_cic.py 的 qSOFA 部分一致，但 x 轴用 qSOFA
# 评分（0-3 分）本身，阶梯式展示"不同评分界值下筛出多少人/死多少人"。
#
# 数据源: qsofa_predictions.npz（qsofa6/qsofa12/qsofa24 + y_true）
#
# 输出: figures/cic_qsofa_24h.png（24h 主图）+ figures/cic_qsofa_windows.png
#       终端打印各窗各界值的分诊数字
# =====================================================================

q = np.load('qsofa_predictions.npz')
y_true = q['y_true'].astype(int)
n = len(y_true)
n_death = y_true.sum()
print(f'test set: {n} 例 | 死亡 {n_death} 例 | 死亡率 {y_true.mean():.3f}')

scores = {'6h': q['qsofa6'].astype(int),
          '12h': q['qsofa12'].astype(int),
          '24h': q['qsofa24'].astype(int)}

# ---------- 1. 计算各窗各界值（≥1/≥2/≥3）的 CIC 数值 ----------
cutoffs = [1, 2, 3]
for wname, s in scores.items():
    for c in cutoffs:
        mask = s >= c
        hr = int(mask.sum())
        ev = int((mask & (y_true == 1)).sum())
        print(f'qSOFA-{wname} ≥{c}: 高危 {hr:>5} ({100*hr/n:5.1f}%) | '
              f'死亡 {ev:>4} | 精度 {ev/hr if hr else 0:.3f} | 捕获 {100*ev/n_death:.1f}%')

# ---------- 2. 主图：24h 窗 CIC（x 轴 = qSOFA 分数）----------
s = scores['24h']
fig, ax = plt.subplots(figsize=(8, 6))

xs = np.array([0, 1, 2, 3, 4])
hr_vals = np.array([(s >= t).sum() for t in xs])
ev_vals = np.array([((s >= t) & (y_true == 1)).sum() for t in xs])

ax.step(xs, hr_vals, where='post', color='#000000', lw=2.2, label='Number high risk')
ax.step(xs, ev_vals, where='post', color='#1F77B4', lw=2.2, ls='--', label='Number high risk with event')
ax.fill_between(xs, ev_vals, hr_vals, step='post', color='gray', alpha=0.15,
                label='Over-triage gap')

for x, h, e in zip(xs, hr_vals, ev_vals):
    ax.text(x + 0.06, h + 40, f'{h}', fontsize=10, va='bottom')

ax.set_xlabel('qSOFA score cutoff (≥)', fontsize=13)
ax.set_ylabel('Number of patients classified as high risk', fontsize=13)
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xticklabels(['≥0', '≥1', '≥2', '≥3', '≥4'])
ax.set_xlim(0, 4.2)
ax.set_ylim(0, n * 1.08)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(alpha=0.3)
ax.set_title('qSOFA (24h window) Clinical Impact Curve', fontsize=13)
plt.tight_layout()
plt.savefig('figures/cic_qsofa_24h.png', dpi=300, bbox_inches='tight')
print('\n已保存: figures/cic_qsofa_24h.png')

# ---------- 3. 汇总图：6h/12h/24h 三窗并排（敏感性）----------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
for ax, (wname, s) in zip(axes, scores.items()):
    hr_vals = np.array([(s >= t).sum() for t in xs])
    ev_vals = np.array([((s >= t) & (y_true == 1)).sum() for t in xs])
    ax.step(xs, hr_vals, where='post', color='#000000', lw=2.0)
    ax.step(xs, ev_vals, where='post', color='#1F77B4', lw=2.0, ls='--')
    ax.fill_between(xs, ev_vals, hr_vals, step='post', color='gray', alpha=0.15)
    ax.set_title(f'qSOFA-{wname} window', fontsize=12)
    ax.set_xlabel('qSOFA score cutoff (≥)', fontsize=11)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(['0', '1', '2', '3', '4'])
    ax.grid(alpha=0.3)
axes[0].set_ylabel('Number classified as high risk', fontsize=12)
axes[0].legend(['High risk', 'With event'], loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig('figures/cic_qsofa_windows.png', dpi=300, bbox_inches='tight')
print('已保存: figures/cic_qsofa_windows.png')
