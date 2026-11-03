import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# =====================================================================
# 07_plot_pr.py —— 7 个模型 PR 曲线对比（同图 + AP 标注）
#
# 数据源: all_predictions.npz（6 基线 + LSTM，合并预测）
#
# 说明: 类别不平衡时 PR 曲线比 ROC 更能反映模型在少数类(死亡)上的表现
# 输出: pr_curves.png
# =====================================================================

d = np.load('all_predictions.npz')
y_true = d['y_true'].astype(int)

# 基线：随机预测的 AP = 死亡率（19.3%）
BASE_RATE = y_true.mean()
print(f'死亡率(随机基线AP): {BASE_RATE:.3f}')

MODELS = {
    'RandomForest':       'RandomForest',
    'SVM':                'SVM',
    'LogisticRegression': 'LogisticRegression',
    'KNN':                'KNN',
    'GaussianNB':         'GaussianNB',
    'DecisionTree':       'DecisionTree',
    'LSTM':               'LSTM',
}
ORDER = ['LSTM', 'RandomForest', 'SVM', 'LogisticRegression', 'KNN', 'GaussianNB', 'DecisionTree']
COLORS = ['#D62728', '#1F77B4', '#2CA02C', '#9467BD', '#FF7F0E', '#8C564B', '#17BECF']

plt.figure(figsize=(8, 7))

# 随机基线（水平虚线：AP = 正例比例）
plt.axhline(y=BASE_RATE, color='k', linestyle='--', lw=1.2,
            label=f'Chance (AP={BASE_RATE:.3f})')

for i, name in enumerate(ORDER):
    prob = d[MODELS[name]].astype(float)
    precision, recall, _ = precision_recall_curve(y_true, prob)
    ap = average_precision_score(y_true, prob)
    plt.plot(recall, precision, color=COLORS[i], lw=2.2,
             label=f'{name} (AP={ap:.4f})')

plt.xlabel('Recall (True Positive Rate)', fontsize=13)
plt.ylabel('Precision (Positive Predictive Value)', fontsize=13)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/pr_curves.png', dpi=300, bbox_inches='tight')
print('已保存: pr_curves.png')

# 打印 AP 排序
print('\nAP 排序:')
for name in ORDER:
    prob = d[MODELS[name]].astype(float)
    ap = average_precision_score(y_true, prob)
    print(f'  {name:20s} AP = {ap:.4f}')