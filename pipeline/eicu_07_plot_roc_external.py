import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# =====================================================================
# eicu_07_plot_roc_external.py —— eICU 外验 ROC 曲线
#
# 数据源: eicu_predictions.npz（外验预测概率 + SOFA/qSOFA + y_true）
#
# 对比：全部 7 个 ML（LSTM / RF / SVM / LR / GaussianNB / KNN / DT）
#       + SOFA / qSOFA-6h（临床评分），与内部 ROC（roc_curves_qsofa.png）
#       及正文 Table 的 7 模型口径保持一致。
# 输出: figures/roc_curves_external.png
# =====================================================================

d = np.load('eicu_predictions.npz')
y_true = d['y_true'].astype(int)

models = {
    'LSTM':               d['LSTM'].astype(float),
    'RandomForest':       d['RandomForest'].astype(float),
    'SVM':                d['SVM'].astype(float),
    'LogisticRegression': d['LogisticRegression'].astype(float),
    'GaussianNB':         d['GaussianNB'].astype(float),
    'KNN':                d['KNN'].astype(float),
    'DecisionTree':       d['DecisionTree'].astype(float),
    'SOFA':               d['sofa'].astype(float),
    'qSOFA-6h':           d['qsofa6'].astype(float),
}
n = len(y_true)
print(f'eICU 外验集: {n} 例 | 死亡率 {y_true.mean():.3f}')

plt.figure(figsize=(8, 7))
plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Chance (AUC=0.50)')

# 配色与内部 ROC（10_qsofa_baseline.py）一致；ML 实线，评分虚线
COLORS = {'LSTM': '#D62728', 'RandomForest': '#1F77B4', 'SVM': '#2CA02C',
          'LogisticRegression': '#9467BD', 'GaussianNB': '#FF7F0E',
          'KNN': '#8C564B', 'DecisionTree': '#17BECF',
          'SOFA': '#000000', 'qSOFA-6h': '#17BECF'}
LS = {'LSTM': '-', 'RandomForest': '-', 'SVM': '-', 'LogisticRegression': '-',
      'GaussianNB': '-', 'KNN': '-', 'DecisionTree': '-',
      'SOFA': '--', 'qSOFA-6h': '--'}

# 排序：按 AUC 降序
aucs = {name: auc(*roc_curve(y_true, prob)[:2]) for name, prob in models.items()}
order = sorted(models.keys(), key=lambda k: -aucs[k])

for name in order:
    fpr, tpr, _ = roc_curve(y_true, models[name])
    plt.plot(fpr, tpr, color=COLORS[name], lw=2.2, ls=LS[name],
             label=f'{name} (AUC={aucs[name]:.4f})')

plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/roc_curves_external.png', dpi=300, bbox_inches='tight')
print('\n已保存: figures/roc_curves_external.png')

print('\nAUC 排序:')
for name in order:
    print(f'  {name:20s} AUC = {aucs[name]:.4f}')