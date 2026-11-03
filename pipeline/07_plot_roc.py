import numpy as np
import matplotlib
matplotlib.use('Agg')          # 服务器/后台保存图片用，本地跑也没问题
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# =====================================================================
# 07_plot_roc.py —— 7 个模型 ROC 曲线对比（同图 + AUC 标注）
#
# 数据源: all_predictions.npz（6 基线 + LSTM，合并预测）
#   - 每个模型: 测试集预测概率（key 如 LogisticRegression, RandomForest...）
#   - y_true:   测试集真实标签
#
# 输出: roc_curves.png（论文用，300 DPI）
# =====================================================================

# 加载预测结果
d = np.load('all_predictions.npz')
y_true = d['y_true'].astype(int)

# 模型显示名 -> npz 里的 key
MODELS = {
    'LSTM':               'LSTM',
    'RandomForest':       'RandomForest',
    'SVM':                'SVM',
    'LogisticRegression': 'LogisticRegression',
    'KNN':                'KNN',
    'GaussianNB':         'GaussianNB',
    'DecisionTree':       'DecisionTree',
}
# 按 AUC 从高到低排（显示更美观）
ORDER = ['LSTM', 'RandomForest', 'SVM', 'LogisticRegression', 'GaussianNB', 'KNN', 'DecisionTree']

# 统一配色（与 eicu_07 一致）
COLORS = {
    'LSTM': '#D62728', 'RandomForest': '#1F77B4', 'SVM': '#2CA02C',
    'LogisticRegression': '#9467BD', 'GaussianNB': '#FF7F0E',
    'KNN': '#8C564B', 'DecisionTree': '#17BECF',
}

plt.figure(figsize=(8, 7))

# 画对角线（随机猜测基准）
plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Chance (AUC=0.50)')

for name in ORDER:
    prob = d[MODELS[name]].astype(float)
    fpr, tpr, _ = roc_curve(y_true, prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=COLORS[name], lw=2.2, ls='-',
             label=f'{name} (AUC={roc_auc:.4f})')

plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/roc_curves.png', dpi=300, bbox_inches='tight')
print('已保存: roc_curves.png')

# 顺便打印 AUC 排序
print('\nAUC 排序:')
for name in ORDER:
    prob = d[MODELS[name]].astype(float)
    fpr, tpr, _ = roc_curve(y_true, prob)
    print(f'  {name:20s} AUC = {auc(fpr, tpr):.4f}')