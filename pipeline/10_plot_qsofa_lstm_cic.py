import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =====================================================================
# 10_plot_qsofa_lstm_cic.py —— qSOFA-6h vs LSTM 临床影响曲线（CIC）对比
#
# 目的：直观展示"规则评分 vs 学习型时序模型"的临床分诊差异。
#   - qSOFA-6h：只能用"≥2 分"这一个操作点（0-3 整数分）
#   - LSTM：连续概率，可在任意阈值下选操作点（用 sequence_results.csv
#           验证集 F3 最优阈值，当前 0.25）
#
# 数据源：
#   qsofa_predictions.npz（qSOFA-6h 得分 + y_true + stay_id）
#   all_predictions.npz（LSTM 概率 + y_true + stay_id）
#   sequence_results.csv（LSTM 最优阈值，与 table2 同源）
#
# 输出：figures/cic_qsofa6_vs_lstm.png
#       终端打印两个模型关键操作点的分诊数字（供正文引用）
# =====================================================================

q = np.load('qsofa_predictions.npz')
a = np.load('all_predictions.npz')
seq = pd.read_csv('sequence_results.csv').set_index('model').loc['LSTM']
LSTM_THR = float(seq['best_threshold'])

# 断言：两个文件是同一批 test set
assert (q['y_true'] == a['y_true']).all(), 'qsofa 与 all_predictions 的 test set 不一致！'

y_true = q['y_true'].astype(int)
q6 = q['qsofa6'].astype(int)          # qSOFA-6h 得分 0-3
lstm = a['LSTM'].astype(float)        # LSTM 概率
n = len(y_true)
n_death = y_true.sum()
print(f'test set: {n} 例 | 死亡 {n_death} 例 | 死亡率 {y_true.mean():.3f}')

# ---------- 1. qSOFA-6h 的 CIC（仅一个操作点：≥2）----------
q_thresholds = np.array([0, 1, 2, 3, 4]) - 0.5   # → -0.5,0.5,1.5,2.5,3.5
q_hr, q_ev = [], []
for t in q_thresholds:
    mask = q6 >= t
    q_hr.append(mask.sum())
    q_ev.append((mask & (y_true == 1)).sum())

# ---------- 2. LSTM 的 CIC（连续曲线）----------
thr = np.linspace(0.01, 0.99, 99)
l_hr  = np.array([(lstm >= t).sum() for t in thr])
l_ev  = np.array([((lstm >= t) & (y_true == 1)).sum() for t in thr])

# ---------- 3. 绘图 ----------
fig, ax = plt.subplots(figsize=(9, 6.5))

ax.plot(thr, l_hr, color='#D62728', lw=2.2, label='LSTM - number high risk')
ax.plot(thr, l_ev, color='#D62728', lw=2.2, ls='--', label='LSTM - high risk with event')
ax.fill_between(thr, l_ev, l_hr, color='#D62728', alpha=0.08)

ax.step(q_thresholds, q_hr, where='post', color='#000000', lw=2.0, label='qSOFA-6h - number high risk')
ax.step(q_thresholds, q_ev, where='post', color='#000000', lw=2.0, ls='--', label='qSOFA-6h - high risk with event')

ax.axvline(LSTM_THR, color='#D62728', ls=':', lw=1.5)
ax.annotate(f'LSTM op. point\n(thr={LSTM_THR:.2f})', xy=(LSTM_THR, (lstm >= LSTM_THR).sum()),
            xytext=(0.35, 3300), fontsize=10, color='#D62728')
ax.axvline(1.5, color='#000000', ls=':', lw=1.5)
ax.annotate('qSOFA op. point\n(score≥2)', xy=(1.5, (q6 >= 2).sum()),
            xytext=(0.06, 3300), fontsize=10)

ax.set_xlabel('Threshold probability / qSOFA score', fontsize=13)
ax.set_ylabel('Number of patients classified as high risk', fontsize=13)
ax.set_xlim(0, 1)
ax.set_ylim(0, n * 1.05)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/cic_qsofa6_vs_lstm.png', dpi=300, bbox_inches='tight')
print('已保存: figures/cic_qsofa6_vs_lstm.png')

# ---------- 4. 关键操作点数字（供正文引用）----------
def summarize(name, mask):
    hr = mask.sum()
    ev = (mask & (y_true == 1)).sum()
    miss = n_death - ev
    print(f'\n{name}:')
    print(f'  高危人数: {hr} ({100*hr/n:.1f}% test set)')
    print(f'  其中死亡: {ev} (精度 {ev/hr:.3f})')
    print(f'  漏掉死亡: {miss} ({100*miss/n_death:.1f}%)')
    print(f'  虚警(高危但存活): {hr-ev} ({100*(hr-ev)/hr:.1f}% of high-risk)')

summarize('qSOFA-6h (score≥2)', q6 >= 2)
summarize(f'LSTM (thr={LSTM_THR:.2f})', lstm >= LSTM_THR)