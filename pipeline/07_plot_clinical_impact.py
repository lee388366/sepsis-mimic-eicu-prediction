import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =====================================================================
# 07_plot_clinical_impact.py —— 临床影响曲线（Clinical Impact Curve, CIC）
#
# 数据源: all_predictions.npz（合并文件：6 基线 + LSTM，与 ROC / PR / DCA 同一个文件）
#   - 'LSTM':        测试集预测概率（LSTM 时序模型）
#   - 'y_true':      测试集真实标签
#
# 定义: CIC 回答的是「如果我把预测概率 ≥ 阈值的患者都收治/干预，
#       会拉出多少人，其中真正的死亡者有多少」，是 DCA 净获益的
#       可视化补充（Kerr et al., BMC Med Res Methodol, 2016）。
#   - 红色曲线 = 每个阈值下被判为高危的患者数（Number high risk）
#   - 蓝色曲线 = 其中实际死亡的人数（Number high risk with event）
#   两条曲线越贴合，说明模型"筛人"越精准。
#
# 输出: figures/clinical_impact_curve_lstm.png（论文用，300 DPI）
#       figures/clinical_impact_curve_rf.png（RF 对照，同一脚本生成）
#       终端打印关键阈值下的数值表（供正文引用）
# =====================================================================

d = np.load('all_predictions.npz')
y_true = d['y_true'].astype(int)
prob = d['LSTM'].astype(float)
seq = pd.read_csv('sequence_results.csv').set_index('model').loc['LSTM']
REPORT_THR = float(seq['best_threshold'])
n = len(y_true)
prevalence = y_true.mean()

print(f'测试集: {n} 例 | 死亡率(先验): {prevalence:.3f}')

# ---------- 1. 计算 CIC ----------
thresholds = np.linspace(0.01, 0.99, 99)


def cic_curves(prob):
    """返回 (n_high_risk, n_high_risk_event)，阈值网格固定为 self.thresholds"""
    n_high_risk = np.zeros(len(thresholds))         # 预测概率 ≥ 阈值的人数
    n_high_risk_event = np.zeros(len(thresholds))   # 其中实际死亡的人数
    for i, t in enumerate(thresholds):
        mask = prob >= t
        n_high_risk[i] = mask.sum()
        n_high_risk_event[i] = (mask & (y_true == 1)).sum()
    return n_high_risk, n_high_risk_event


def plot_cic(prob, fname):
    """绘制单个模型的 CIC 并保存"""
    n_high_risk, n_high_risk_event = cic_curves(prob)
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, n_high_risk, color='#D62728', lw=2.2,
             label='Number high risk')
    plt.plot(thresholds, n_high_risk_event, color='#1F77B4', lw=2.2,
             label='Number high risk with event')
    plt.fill_between(thresholds, n_high_risk_event, n_high_risk,
                     color='gray', alpha=0.15,
                     label='Over-triage gap (non-event among high risk)')

    plt.xlabel('High-risk threshold probability', fontsize=13)
    plt.ylabel('Number of patients classified as high risk', fontsize=13)
    plt.xlim(0, 1)
    plt.ylim(0, n_high_risk.max() * 1.05)
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'已保存: {fname}')


# ---------- 2. 绘图：LSTM（主模型）+ RF（对照）----------
plot_cic(prob, 'figures/clinical_impact_curve_lstm.png')
plot_cic(d['RandomForest'].astype(float), 'figures/clinical_impact_curve_rf.png')

# ---------- 3. 关键阈值数值表（供正文引用，LSTM）----------
print('\n===== 关键阈值下的 CIC 数值（供正文引用）=====')
print(f'{"Threshold":<10}{"High risk (n)":<18}{"with event (n)":<18}{"Precision":<12}{"事件捕获率"}')
for t in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    mask = prob >= t
    hr = mask.sum()
    ev = (mask & (y_true == 1)).sum()
    prec = ev / hr if hr > 0 else 0.0
    captured = ev / y_true.sum()   # 事件捕获率 = 筛出的死亡者 / 全部死亡者
    print(f'{t:<10.2f}{hr:<18}{ev:<18}{prec:<12.3f}{captured:.3f}')

# ---------- 4. 主模型报告阈值（来自 06b 验证集 F3 阈值扫描，动态读取）----------
mask = prob >= REPORT_THR
hr = mask.sum()
ev = (mask & (y_true == 1)).sum()
print(f'\n===== 项目主模型报告阈值 ({REPORT_THR:.2f}) 摘要 =====')
print(f'被判为高危: {hr} 例 ({100*hr/n:.1f}% 的测试集)')
print(f'其中实际死亡: {ev} 例 (精度 {ev/hr:.3f})')
print(f'捕获了全部死亡事件中的 {100*ev/y_true.sum():.1f}%')
