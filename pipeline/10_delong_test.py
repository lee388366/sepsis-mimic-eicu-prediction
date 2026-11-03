import numpy as np
import pandas as pd
import scipy.stats

# =====================================================================
# 10_delong_test.py —— DeLong 检验（AUC 差异显著性）
#
# 实现来源: yandexdataschool/roc_comparison (compare_auc_delong_xu.py)
#           经过 Netflix vmaf 和大量论文验证的 fastDeLong (Sun & Xu 2014)。
#
# 用法: 对同一 test set 的任意两个模型/评分做 DeLong 检验，
#       得到两两 p 值矩阵，判断 AUC 差异是否显著。
#
# 数据源: all_predictions.npz + qsofa_predictions.npz + sofa_predictions.npz
# 输出: tables/table7_delong_pvalues.csv
# =====================================================================

# ---------- fastDeLong 官方实现（原样保留）----------
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def calc_pvalue(aucs, sigma):
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)

def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)

def delong_p(gt, s1, s2):
    """返回 p 值（由 log10(p) 反算）"""
    log10p = delong_roc_test(gt, s1, s2)[0]
    return 10 ** float(np.asarray(log10p).reshape(-1)[0])

# ---------- 1. 加载数据 ----------
m = np.load('all_predictions.npz')
q = np.load('qsofa_predictions.npz')
s = np.load('sofa_predictions.npz')
y_true = m['y_true'].astype(int)
assert (q['y_true'] == y_true).all() and (s['y_true'] == y_true).all()

scores = {
    'LSTM':   m['LSTM'].astype(float),
    'RF':     m['RandomForest'].astype(float),
    'SVM':    m['SVM'].astype(float),
    'LR':     m['LogisticRegression'].astype(float),
    'GNB':    m['GaussianNB'].astype(float),
    'KNN':    m['KNN'].astype(float),
    'DT':     m['DecisionTree'].astype(float),
    'SOFA':   s['sofa'].astype(float),
    'qSOFA6': q['qsofa6'].astype(float),
}
names = list(scores.keys())

# ---------- 2. AUC + 两两 DeLong ----------
from sklearn.metrics import roc_auc_score
aucs = {a: roc_auc_score(y_true, scores[a]) for a in names}
print('===== 各模型 AUC =====')
for a in names:
    print(f'  {a:8s} AUC = {aucs[a]:.4f}')

print('\n===== DeLong 两两 p 值 =====')
pmat = pd.DataFrame(index=names, columns=names, dtype=float)
for a in names:
    for b in names:
        if a == b:
            pmat.loc[a, b] = 1.0
        else:
            pmat.loc[a, b] = delong_p(y_true, scores[a], scores[b])
print(pmat.round(4).to_string())

pmat.to_csv('tables/table7_delong_pvalues.csv', index=True)
print('\n已保存: tables/table7_delong_pvalues.csv')

# ---------- 3. 核心引用 ----------
print('\n===== 与 LSTM 的 DeLong 检验 =====')
for b in ['RF', 'SOFA', 'qSOFA6']:
    p = delong_p(y_true, scores['LSTM'], scores[b])
    print(f'  LSTM vs {b:8s}: p = {p:.4g} {"***" if p < 0.001 else "*" if p < 0.05 else "ns"}')