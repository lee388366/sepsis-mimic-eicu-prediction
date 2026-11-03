import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =====================================================================
# 07_plot_confusion_matrix.py
# 最优模型 RandomForest 的混淆矩阵
#
# 数据源: baseline_predictions.npz
#   - 'RandomForest': 测试集预测概率
#   - 'y_true':       测试集真实标签
#
# 关键点: 最优阈值自动从 baseline_results.csv 读取（随 06 重跑自动同步，
#         避免硬编码导致与表格指标不一致）
# 输出: confusion_matrix_rf.png
# =====================================================================

d = np.load('baseline_predictions.npz')
y_true = d['y_true'].astype(int)
prob = d['RandomForest'].astype(float)

# 从 baseline_results.csv 自动读取 RF 在验证集上选出的最优 F3 阈值
df = pd.read_csv('baseline_results.csv')
THRESHOLD = float(df.loc[df['model'] == 'RandomForest', 'best_threshold'].iloc[0])
print(f'从 baseline_results.csv 读取到 RF 最优阈值: {THRESHOLD}')

y_pred = (prob >= THRESHOLD).astype(int)

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print('=== RandomForest 混淆矩阵 ===')
print(f'阈值: {THRESHOLD}')
print(f'TN(真阴): {tn}  FP(假阳): {fp}')
print(f'FN(漏报): {fn}  TP(真阳): {tp}')
print(f'阳性比例: {y_true.mean():.3f}')

# 用 sklearn 官方展示器画图（论文标准，带配色、标注、归一化选项）
fig, ax = plt.subplots(figsize=(6.5, 5.5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['Survived (0)', 'Died (1)'])
disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')

ax.set_title(f'RandomForest Confusion Matrix\n(threshold={THRESHOLD}, '
             f'test n={len(y_true)})', fontsize=13)
plt.tight_layout()
plt.savefig('figures/confusion_matrix_rf.png', dpi=300, bbox_inches='tight')
print('\n已保存: confusion_matrix_rf.png')