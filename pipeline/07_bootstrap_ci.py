import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# =====================================================================
# 07_bootstrap_ci.py —— 7 个模型 AUC / AP 的 95% 置信区间（Bootstrap）
#
# 数据源: all_predictions.npz（6 基线 + LSTM，合并预测）
# 方法:  有放回重采样 1000 次，取 2.5% / 97.5% 分位数
# 输出:  终端打印每个模型的 AUC / AP + 95% CI
# =====================================================================

np.random.seed(42)

d = np.load('all_predictions.npz')
y_true = d['y_true'].astype(int)

MODELS = ['LSTM', 'RandomForest', 'SVM', 'LogisticRegression', 'GaussianNB', 'KNN', 'DecisionTree']

N_BOOT = 1000
n = len(y_true)

print(f'{"模型":<18}{"AUC":<12}{"AUC 95%CI":<24}{"AP":<12}{"AP 95%CI"}')
for name in MODELS:
    prob = d[name].astype(float)
    auc_bs = np.zeros(N_BOOT)
    ap_bs = np.zeros(N_BOOT)
    for b in range(N_BOOT):
        idx = np.random.choice(n, n, replace=True)   # bootstrap 抽样
        auc_bs[b] = roc_auc_score(y_true[idx], prob[idx])
        ap_bs[b] = average_precision_score(y_true[idx], prob[idx])
    auc_ci = np.percentile(auc_bs, [2.5, 97.5])
    ap_ci = np.percentile(ap_bs, [2.5, 97.5])
    print(f'{name:<18}{auc_bs.mean():<12.4f}'
          f'{auc_ci[0]:.4f}–{auc_ci[1]:.4f}'
          f'{"":<6}{ap_bs.mean():<12.4f}{ap_ci[0]:.4f}–{ap_ci[1]:.4f}')