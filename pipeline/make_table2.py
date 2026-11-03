import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# =====================================================================
# make_table2.py —— 生成 tables/table2_model_performance.csv
#
# 数据源:
#   baseline_results.csv      (来自 06_train_baseline.py 的测试集指标)
#   baseline_predictions.npz  (来自 06_train_baseline.py 的预测概率)
# 方法: 与 07_bootstrap_ci.py 相同 —— 有放回重采样 1000 次，
#       取 AUC / AP 的 2.5% / 97.5% 分位数作为 95% CI。
#
# 输出: tables/table2_model_performance.csv
#   Model,AUC,AUC_95CI_low,AUC_95CI_high,AUPRC,AUPRC_95CI_low,AUPRC_95CI_high,
#   Optimal_threshold,Precision,Recall,F1,F3
# =====================================================================

SEED = 42
N_BOOT = 1000

# 与 06/07 完全相同的模型顺序和命名映射
MODEL_KEYS = ['LSTM', 'RandomForest', 'SVM', 'LogisticRegression', 'GaussianNB', 'KNN', 'DecisionTree']
MODEL_LABELS = {
    'LSTM': 'LSTM',
    'RandomForest':        'Random forest',
    'SVM':                 'SVM',
    'LogisticRegression':  'Logistic regression',
    'GaussianNB':          'Gaussian naive Bayes',
    'KNN':                 'K-nearest neighbors',
    'DecisionTree':        'Decision tree',
}

np.random.seed(SEED)

# 1. 测试集预测概率 + 真实标签
d = np.load('all_predictions.npz')
y_true = d['y_true'].astype(int)

# 2. baseline 指标
res = pd.read_csv('baseline_results.csv').set_index('model')
seq = pd.read_csv('sequence_results.csv').set_index('model')
res = pd.concat([res, seq])

# 3. 每模型 bootstrap CI
def bootstrap_ci(prob, y, metric):
    bs = np.zeros(N_BOOT)
    for b in range(N_BOOT):
        idx = np.random.choice(len(y), len(y), replace=True)
        bs[b] = metric(y[idx], prob[idx])
    return np.percentile(bs, [2.5, 97.5])

rows = []
for key in MODEL_KEYS:
    prob = d[key].astype(float)
    auc_ci = bootstrap_ci(prob, y_true, roc_auc_score)
    ap_ci  = bootstrap_ci(prob, y_true, average_precision_score)

    r = res.loc[key]
    rows.append({
        'Model': MODEL_LABELS[key],
        'AUC': round(r['AUC'], 3),
        'AUC_95CI_low': round(auc_ci[0], 3),
        'AUC_95CI_high': round(auc_ci[1], 3),
        'AUPRC': round(r['AUPRC'], 3),
        'AUPRC_95CI_low': round(ap_ci[0], 3),
        'AUPRC_95CI_high': round(ap_ci[1], 3),
        'Optimal_threshold': round(r['best_threshold'], 2),
        'Precision': round(r['Precision'], 3),
        'Recall': round(r['Recall'], 3),
        'F1': round(r['F1'], 3),
        'F3': round(r['F3'], 3),
    })

df = pd.DataFrame(rows)
df.to_csv('tables/table2_model_performance.csv', index=False, float_format='%.3f')

print(df.to_string(index=False))
print('\n已保存: tables/table2_model_performance.csv')
