import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             precision_score, recall_score, f1_score)
from split_utils import grouped_tts

# =====================================================================
# 10_qsofa_baseline.py —— qSOFA 核心对标（6h/12h/24h 敏感性分析）
#
# 临床背景：qSOFA 在床旁快速筛查中使用，观察窗不宜过长。
#   qSOFA（Singer 2016）：RR>=22 / SBP<=100 / GCS<15，各 1 分，总分 0-3。
#   每窗内：任一真实观测满足即计 1 分（缺失小时不计分）。
#
# 输出：tables/table5_qsofa_comparison.csv + table5_qsofa_windows.csv
#       figures/roc_curves_qsofa.png（按 AUC 全局降序）
#       qsofa_predictions.npz
#
# 口径（2026-08-25 修正）：
#   1. 主表 qSOFA 用 6h 窗（与论文叙事一致——6h 为推荐床旁筛查窗；
#      12h/24h 保留在 table5_qsofa_windows.csv 敏感性分析）。
#   2. ML 模型的 Precision/Recall/F1/F3 从 baseline_results.csv +
#      sequence_results.csv 读取（与 table2 同源、同阈值），不再留空。
# =====================================================================

SEED = 42
np.random.seed(SEED)

# ---------- 1. 加载 ----------
d = np.load('sepsis_training_data.npz')
keep = d['stay_id'].astype(np.int64)
y = d['y'].astype(np.int64)

hourly = pd.read_csv('sepsis_hourly.csv')
hourly = hourly[hourly['stay_id'].isin(keep)].sort_values(['stay_id', 'hr']).reset_index(drop=True)

def pivot_48(keep_ids, col):
    p = hourly.pivot_table(index='stay_id', columns='hr', values=col,
                           aggfunc='first', dropna=False)
    return p.loc[keep_ids].values.astype(float)   # [n,48] 含 NaN

rr  = pivot_48(keep, 'resp_rate')
sbp = pivot_48(keep, 'sbp')
gcs = pivot_48(keep, 'gcs')

# ---------- 2. 划分（受试者级分组，与 06/06b 一致）----------
tr_idx, va_idx, te_idx = grouped_tts(d['subject_id'].astype(np.int64), y)
print(f'train {len(tr_idx)} | val {len(va_idx)} | test {len(te_idx)}')

# ---------- 3. 计算多窗 qSOFA ----------
def calc_qsofa(rr_a, sbp_a, gcs_a, n_hours):
    """前 n_hours 小时内的 any-time qSOFA"""
    c_rr  = (rr_a[:, :n_hours] >= 22).any(axis=1).astype(int)
    c_sbp = (sbp_a[:, :n_hours] <= 100).any(axis=1).astype(int)
    c_gcs = (gcs_a[:, :n_hours] < 15).any(axis=1).astype(int)
    return c_rr + c_sbp + c_gcs

q6   = calc_qsofa(rr, sbp, gcs, 6)
q12  = calc_qsofa(rr, sbp, gcs, 12)
q24  = calc_qsofa(rr, sbp, gcs, 24)

y_te = y[te_idx]
n_te = len(y_te)

def fbeta(p, r, beta=3.0):
    return (1 + beta**2) * p * r / (beta**2 * p + r) if (beta**2 * p + r) > 0 else 0.0

def score_metrics(score, thr=2):
    auc = roc_auc_score(y_te, score)
    ap  = average_precision_score(y_te, score)
    pos = (score >= thr).astype(int)
    p = precision_score(y_te, pos, zero_division=0)
    r = recall_score(y_te, pos, zero_division=0)
    return auc, ap, p, r, f1_score(y_te, pos, zero_division=0), fbeta(p, r)

# ---------- 4. 窗口敏感性分析表 ----------
print('\n===== qSOFA 窗口敏感性分析（test set 3742 例）=====')
print(f'{"窗口":<8}{"AUC":<10}{"AUPRC":<10}{"阳性率(≥2)":<12}{"Prec":<8}{"Rec":<8}{"F3"}')
rows_win = []
for name, q in [('6h', q6), ('12h', q12), ('24h', q24)]:
    auc, ap, p_, r_, f1_, f3_ = score_metrics(q[te_idx])
    pos_rate = (q[te_idx] >= 2).mean()
    rows_win.append((name, auc, ap, pos_rate, p_, r_, f3_))
    print(f'{name:<8}{auc:<10.4f}{ap:<10.4f}{pos_rate:<12.3f}{p_:<8.3f}{r_:<8.3f}{f3_:.3f}')

# 主分析：6h（推荐床旁筛查窗；12h/24h 保留在敏感性分析表）
auc_q, ap_q, p2, r2, f1_2, f3_2 = score_metrics(q6[te_idx])

# ---------- 5. 与 7 个 ML 对比（主表，qSOFA 用 6h）----------
m = np.load('all_predictions.npz')
assert (m['y_true'] == y_te).all(), 'all_predictions 与本次划分的 test set 不一致！'

MODELS = ['LSTM', 'RandomForest', 'SVM', 'LogisticRegression', 'GaussianNB', 'KNN', 'DecisionTree']

# ML 操作点指标从结果文件读取（与 make_table2.py 同源、同阈值）
res = pd.read_csv('baseline_results.csv').set_index('model')
seq = pd.read_csv('sequence_results.csv').set_index('model')
res = pd.concat([res, seq])
def model_metrics(key):
    r = res.loc[key]
    return (r['best_threshold'], r['Precision'], r['Recall'], r['F1'], r['F3'])

rows = [{
    'Model': 'qSOFA-6h (rule-based)',
    'AUC': auc_q, 'AUPRC': ap_q, 'Optimal_threshold': 2,
    'Precision': p2, 'Recall': r2, 'F1': f1_2, 'F3': f3_2,
}]
for name in MODELS:
    prob = m[name].astype(float)
    thr, p_, r_, f1_, f3_ = model_metrics(name)
    rows.append({'Model': name, 'AUC': roc_auc_score(y_te, prob),
                 'AUPRC': average_precision_score(y_te, prob),
                 'Optimal_threshold': thr,
                 'Precision': p_, 'Recall': r_, 'F1': f1_, 'F3': f3_})

df = pd.DataFrame(rows).sort_values('AUC', ascending=False).reset_index(drop=True)
print('\n===== 模型对比（qSOFA-6h vs 7 ML, 同一 test set）=====')
print(df.to_string(index=False))
df.to_csv('tables/table5_qsofa_comparison.csv', index=False)
print('已保存: tables/table5_qsofa_comparison.csv')

# 敏感性分析表也存一份
df_win = pd.DataFrame(rows_win, columns=['Window', 'AUC', 'AUPRC', 'Pos_rate_ge2', 'Precision', 'Recall', 'F3'])
df_win.to_csv('tables/table5_qsofa_windows.csv', index=False)
print('已保存: tables/table5_qsofa_windows.csv')

# ---------- 6. ROC 图（按 AUC 全局降序）----------
plt.figure(figsize=(8, 7))
plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Chance (AUC=0.50)')

COLORS = {'LSTM': '#D62728', 'RandomForest': '#1F77B4', 'SVM': '#2CA02C',
          'LogisticRegression': '#9467BD', 'GaussianNB': '#FF7F0E',
          'KNN': '#8C564B', 'DecisionTree': '#17BECF',
          'SOFA': '#000000', 'qSOFA-6h': '#17BECF'}

# 所有模型（ML + SOFA + qSOFA）按 AUC 全局排序
# SOFA 从 sofa_predictions.npz 读（同一 test set），与 B 面板/正文 Table 口径一致
s = np.load('sofa_predictions.npz')
assert (s['y_true'] == y_te).all(), 'sofa_predictions 与本次划分的 test set 不一致！'
all_probs = {name: m[name].astype(float) for name in MODELS}
all_probs['SOFA'] = s['sofa'].astype(float)
all_probs['qSOFA-6h'] = q6[te_idx].astype(float)

auc_map = {name: roc_auc_score(y_te, prob) for name, prob in all_probs.items()}
order_all = sorted(auc_map.keys(), key=lambda k: -auc_map[k])

for name in order_all:
    fpr, tpr, _ = roc_curve(y_te, all_probs[name])
    color = COLORS.get(name, '#000000')
    ls = '--' if name in ('SOFA', 'qSOFA-6h') else '-'
    plt.plot(fpr, tpr, color=color, lw=1.8, ls=ls,
             label=f'{name} (AUC={auc_map[name]:.4f})')

plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/roc_curves_qsofa.png', dpi=300, bbox_inches='tight')
print('已保存: figures/roc_curves_qsofa.png')

np.savez('qsofa_predictions.npz', qsofa6=q6[te_idx], qsofa12=q12[te_idx],
         qsofa24=q24[te_idx], y_true=y_te, stay_id=d['stay_id'][te_idx])
print('已保存: qsofa_predictions.npz')