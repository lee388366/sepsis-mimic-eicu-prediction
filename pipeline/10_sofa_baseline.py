import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             precision_score, recall_score, f1_score)
from split_utils import grouped_tts

# =====================================================================
# 10_sofa_baseline.py —— SOFA 核心对标基线（规则评分 vs 7 个 ML）
#
# 数据源:
#   sofa_extract.csv         （04b_extract_sofa.sql 输出，24894 例）
#   sepsis_labels.csv        （04_labels.sql 输出，rn=1 口径）
#   all_predictions.npz      （7 模型合并预测，同一 test set）
#
# 方法:
#   - SOFA 连续得分算 AUC/AUPRC（与 7 ML 公平对比）
#   - 临床常用界值 ≥6/≥8/≥10/≥12 报 Se/Sp（文献共识，非官方）
#   - 时间窗纪律：SOFA 为 48h 最差值（max SOFA），与 ML 特征同窗
#
# 口径（2026-08-25 修正）：
#   1. ML 模型的 Precision/Recall/F1/F3 从 baseline_results.csv +
#      sequence_results.csv 读取（与 table2 同源、同阈值），不再留空。
#   2. SOFA 规则评分操作点取文献共识界值 ≥6（与 table5_sofa_cutoffs.csv 一致）。
#
# 输出:
#   tables/table5_sofa_comparison.csv  SOFA vs 7 ML 对比表
#   figures/roc_curves_sofa.png        8 条 ROC（7 ML + SOFA）
# =====================================================================

SEED = 42
np.random.seed(SEED)

# ---------- 1. 加载 SOFA + 标签，按 stay_id 对齐 ----------
sofa = pd.read_csv('sofa_extract.csv')
labels = pd.read_csv('sepsis_labels.csv')

# 清理 psql -f 导出带入的行数提示（如 "(24894 rows)"）和空行
sofa = sofa[sofa['stay_id'].astype(str).str.strip().str.isdigit()].copy()
# 统一 stay_id 类型（04b 导出 CSV 的 stay_id 是带引号的 str，labels 是 int）
sofa['stay_id'] = sofa['stay_id'].astype(str).str.strip().astype(int)
labels['stay_id'] = labels['stay_id'].astype(int)

# 对齐：只保留标签文件里的 stay（rn=1 口径，与训练一致）
merged = labels[['stay_id', 'label', 'subject_id']].merge(
    sofa[['stay_id', 'sofa_max_48h']], on='stay_id', how='left')
assert merged['sofa_max_48h'].notna().all(), '部分患者无 SOFA 数据！'

y = merged['label'].astype(np.int64).values
sofa_score = merged['sofa_max_48h'].astype(float).values
subject_id = merged['subject_id'].astype(np.int64).values
N = len(y)
print(f'数据: {N} 例 | SOFA 中位数 {np.median(sofa_score):.1f} | 死亡率 {100*y.mean():.1f}%')

# ---------- 2. 划分（受试者级分组，与 06/06b 一致）----------
tr_idx, va_idx, te_idx = grouped_tts(subject_id, y)
print(f'train {len(tr_idx)} | val {len(va_idx)} | test {len(te_idx)}')

y_te = y[te_idx]
sofa_te = sofa_score[te_idx]

# ---------- 3. SOFA 连续得分 + 界值分析 ----------
def fbeta(p, r, beta=3.0):
    return (1 + beta**2) * p * r / (beta**2 * p + r) if (beta**2 * p + r) > 0 else 0.0

auc_s = roc_auc_score(y_te, sofa_te)
ap_s  = average_precision_score(y_te, sofa_te)
print(f'\nSOFA 连续得分: AUC={auc_s:.4f} | AUPRC={ap_s:.4f}')

print('\n===== SOFA 界值敏感性（test set）=====')
print(f'{"界值":<6}{"高危(n)":<10}{"占比":<8}{"Prec":<8}{"Rec":<8}{"F1":<8}{"F3"}')
cutoff_rows = []
for c in [6, 8, 10, 12]:
    pos = (sofa_te >= c).astype(int)
    p = precision_score(y_te, pos, zero_division=0)
    r = recall_score(y_te, pos, zero_division=0)
    f1_ = f1_score(y_te, pos, zero_division=0)
    f3_ = fbeta(p, r)
    cutoff_rows.append((c, pos.sum(), pos.mean(), p, r, f1_, f3_))
    print(f'≥{c:<5}{pos.sum():<10}{pos.mean():<8.3f}{p:<8.3f}{r:<8.3f}{f1_:<8.3f}{f3_:.3f}')

# ---------- 4. 与 7 个 ML 模型对比 ----------
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

# SOFA 规则评分操作点取文献共识界值 ≥6（与 table5_sofa_cutoffs.csv 一致，Recall 最高）
_, _, _, p6, r6, f1_6, f3_6 = cutoff_rows[0]

rows = [{
    'Model': 'SOFA (rule-based)',
    'AUC': auc_s, 'AUPRC': ap_s, 'Optimal_threshold': 6,
    'Precision': p6, 'Recall': r6, 'F1': f1_6, 'F3': f3_6,
}]
for name in MODELS:
    prob = m[name].astype(float)
    thr, p_, r_, f1_, f3_ = model_metrics(name)
    rows.append({'Model': name, 'AUC': roc_auc_score(y_te, prob),
                 'AUPRC': average_precision_score(y_te, prob),
                 'Optimal_threshold': thr,
                 'Precision': p_, 'Recall': r_, 'F1': f1_, 'F3': f3_})

df = pd.DataFrame(rows).sort_values('AUC', ascending=False).reset_index(drop=True)
print('\n===== 模型对比（SOFA vs 7 ML, 同一 test set）=====')
print(df.to_string(index=False))
df.to_csv('tables/table5_sofa_comparison.csv', index=False)
print('已保存: tables/table5_sofa_comparison.csv')

# ---------- 5. ROC 对比图（8 条曲线）----------
plt.figure(figsize=(8, 7))
plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Chance (AUC=0.50)')
fpr, tpr, _ = roc_curve(y_te, sofa_te)
plt.plot(fpr, tpr, color='#8C564B', lw=2.2, ls='--',
         label=f'SOFA (AUC={auc_s:.4f})')
COLORS = {'LSTM': '#D62728', 'RandomForest': '#1F77B4', 'SVM': '#2CA02C',
          'LogisticRegression': '#9467BD', 'GaussianNB': '#FF7F0E',
          'KNN': '#8C564B', 'DecisionTree': '#17BECF'}
for name in MODELS:
    fpr, tpr, _ = roc_curve(y_te, m[name].astype(float))
    plt.plot(fpr, tpr, color=COLORS[name], lw=1.8,
             label=f'{name} (AUC={roc_auc_score(y_te, m[name].astype(float)):.4f})')
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/roc_curves_sofa.png', dpi=300, bbox_inches='tight')
print('已保存: figures/roc_curves_sofa.png')

# 界值表也存一份
df_cut = pd.DataFrame(cutoff_rows, columns=['Cutoff', 'High_risk_n', 'Pos_rate', 'Precision', 'Recall', 'F1', 'F3'])
df_cut.to_csv('tables/table5_sofa_cutoffs.csv', index=False)
print('已保存: tables/table5_sofa_cutoffs.csv')

np.savez('sofa_predictions.npz', sofa=sofa_te, y_true=y_te,
         stay_id=merged['stay_id'].values[te_idx])
print('已保存: sofa_predictions.npz')