import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# =====================================================================
# 07b_bootstrap_ci_subject.py —— 受试者级(subject-clustered) bootstrap
#
# 目的: 量化 Codex 指出的 "bootstrap 按记录抽样、未按受试者聚类" 的影响。
#   方法: 有放回抽取 *受试者*, 同受试者的所有住院记录一并进入/退出,
#         保留同一受试者多次住院的内部相关性。
#   对照: 同时跑 "记录级 bootstrap"(现行 07/make_table2 口径),
#         同 seed、同 1000 次, 直接比较 CI 宽度。
#
# 三类输出:
#   (1) 各模型 记录级 vs 受试者级 的 AUC/AP 点估计与 95%CI
#   (2) 配对差值: LSTM−RF 在同一次抽样内同时计算 AUC/AP 差值,
#       分别给出 记录级 与 受试者级 的差值 95%CI(判断"模型间差异"需看差值的CI,
#       不能靠两个独立CI是否重叠)
#   (3) 受试者重复住院结构: 受试者占比 与 记录占比 分别报告
#
# 数据源: all_predictions.npz(测试集预测) + sepsis_training_data.npz(stay_id→subject_id)
# 输出:   tables/table2_ci_record_vs_subject.csv(主对照)
#         tables/table2_ci_paired_diff_lstm_rf.csv(LSTM−RF 配对差值)
# =====================================================================

SEED = 42
N_BOOT = 1000
MODEL_KEYS = ['LSTM', 'RandomForest', 'SVM', 'LogisticRegression',
              'GaussianNB', 'KNN', 'DecisionTree']
PAIRED = ('LSTM', 'RandomForest')

np.random.seed(SEED)

# 1. 加载预测与受试者映射
d = np.load('all_predictions.npz')
y_true = d['y_true'].astype(int)
n = len(y_true)

tr = np.load('sepsis_training_data.npz')
stay2sub = dict(zip(tr['stay_id'], tr['subject_id']))
subject_id = np.array([stay2sub[s] for s in d['stay_id']])

vc = pd.Series(subject_id).value_counts()
n_sub = len(vc)
n_multi_subj = int((vc > 1).sum())
n_multi_records = int(vc[vc > 1].sum())
print(f'测试集: {n} 条住院记录 / {n_sub} 位受试者')
print(f'  多次住院的受试者:        {n_multi_subj} 人 ({100*n_multi_subj/n_sub:.1f}% 的受试者)')
print(f'  这些受试者贡献的记录:    {n_multi_records} 条 ({100*n_multi_records/n:.1f}% 的记录)')
print(f'  住院次数: 2次={int((vc==2).sum())}人, ≥3次={int((vc>=3).sum())}人, 最多={int(vc.max())}次')

# 2. 抽样实现 ----------------------------------------------------------
# 受试者级抽样索引批: 与 make_table2.py 完全同源 —— np.random.default_rng(SEED)
# 一次性预生成 N_BOOT 批; AUC/AP 与全部模型共用同一批, 保证模型间与指标间可比、
# 并可逐位交叉核验正式 table2。
rec_by_sub = {}
for i, s in enumerate(subject_id):
    rec_by_sub.setdefault(s, []).append(i)
subs = list(rec_by_sub.keys())
_rng = np.random.default_rng(SEED)
boot_idx_subject = []
for _ in range(N_BOOT):
    picked = _rng.choice(len(subs), len(subs), replace=True)
    boot_idx_subject.append(np.concatenate([rec_by_sub[subs[k]] for k in picked]))

# 记录级抽样索引批(独立 RNG, 仅供结构对照; 历史 MT19937 seed42 口径见
# 07_bootstrap_ci.py)
_rng_rec = np.random.default_rng(1)
boot_idx_record = [_rng_rec.choice(n, n, replace=True) for _ in range(N_BOOT)]

def _ci_from_batches(prob, batches, metric):
    bs = np.zeros(len(batches))
    for b, idx in enumerate(batches):
        bs[b] = metric(y_true[idx], prob[idx])
    return bs

def bs_record(prob):
    """记录级: AUC/AP 均用同一批记录级索引。"""
    auc = _ci_from_batches(prob, boot_idx_record, roc_auc_score)
    ap = _ci_from_batches(prob, boot_idx_record, average_precision_score)
    return auc, ap

def bs_subject(prob):
    """受试者级: AUC/AP 均用同一批受试者级索引(与 table2 同源)。"""
    auc = _ci_from_batches(prob, boot_idx_subject, roc_auc_score)
    ap = _ci_from_batches(prob, boot_idx_subject, average_precision_score)
    return auc, ap

def paired_diff(prob_a, prob_b, by_subject):
    """同一批抽样里同时算 A 与 B 的指标并作差 → 配对差值分布。

    AUC 差值 = AUC(A) − AUC(B); AP 差值同理。
    与单模型 CI 使用完全相同的索引批(受试者级或记录级)。
    返回值各为长度 N_BOOT 的数组。
    """
    batches = boot_idx_subject if by_subject else boot_idx_record
    da = np.zeros(N_BOOT); dp = np.zeros(N_BOOT)
    for b, idx in enumerate(batches):
        da[b] = roc_auc_score(y_true[idx], prob_a[idx]) - roc_auc_score(y_true[idx], prob_b[idx])
        dp[b] = (average_precision_score(y_true[idx], prob_a[idx])
                 - average_precision_score(y_true[idx], prob_b[idx]))
    return da, dp

def q(x, lo=2.5, hi=97.5):
    return f'{np.mean(x):.4f} ({np.percentile(x, lo):.4f}–{np.percentile(x, hi):.4f})'

# 3. 各模型 CI 对照 ------------------------------------------------
print(f'\n{"模型":<18}{"记录级 AUC":<22}{"受试者级 AUC":<22}'
      f'{"记录级 AP":<22}{"受试者级 AP"}')
rows = []
for key in MODEL_KEYS:
    prob = d[key].astype(float)
    auc_r, ap_r = bs_record(prob)
    auc_s, ap_s = bs_subject(prob)
    rows.append({
        'Model': key,
        'AUC_rec_low': round(np.percentile(auc_r, 2.5), 3),
        'AUC_rec_high': round(np.percentile(auc_r, 97.5), 3),
        'AUC_sub_low': round(np.percentile(auc_s, 2.5), 3),
        'AUC_sub_high': round(np.percentile(auc_s, 97.5), 3),
        'AP_rec_low': round(np.percentile(ap_r, 2.5), 3),
        'AP_rec_high': round(np.percentile(ap_r, 97.5), 3),
        'AP_sub_low': round(np.percentile(ap_s, 2.5), 3),
        'AP_sub_high': round(np.percentile(ap_s, 97.5), 3),
    })
    print(f'{key:<18}{q(auc_r):<22}{q(auc_s):<22}{q(ap_r):<22}{q(ap_s)}')

pd.DataFrame(rows).to_csv('tables/table2_ci_record_vs_subject.csv',
                          index=False, float_format='%.3f')

# 4. 配对差值: LSTM − RF -------------------------------------------
p_a = d[PAIRED[0]].astype(float)
p_b = d[PAIRED[1]].astype(float)
# 正式点估计: 在原测试集上直接计算(与抽样方式无关, 两种口径应一致);
# bootstrap 只用于给出该差值的置信区间。
pt_auc = roc_auc_score(y_true, p_a) - roc_auc_score(y_true, p_b)
pt_ap = average_precision_score(y_true, p_a) - average_precision_score(y_true, p_b)
da_r, dp_r = paired_diff(p_a, p_b, by_subject=False)
da_s, dp_s = paired_diff(p_a, p_b, by_subject=True)
print(f'\n配对差值 {PAIRED[0]} − {PAIRED[1]} (同一次抽样内同时计算):')
print(f'  原测试集点估计:  AUC Δ={pt_auc:.4f}   AP Δ={pt_ap:.4f}')
print(f'  AUC 差值 CI  记录级:   {np.percentile(da_r,2.5):.4f}–{np.percentile(da_r,97.5):.4f} (bootstrap均值 {np.mean(da_r):.4f})')
print(f'  AUC 差值 CI  受试者级: {np.percentile(da_s,2.5):.4f}–{np.percentile(da_s,97.5):.4f} (bootstrap均值 {np.mean(da_s):.4f})')
print(f'  AP  差值 CI  记录级:   {np.percentile(dp_r,2.5):.4f}–{np.percentile(dp_r,97.5):.4f}')
print(f'  AP  差值 CI  受试者级: {np.percentile(dp_s,2.5):.4f}–{np.percentile(dp_s,97.5):.4f}')
print(f'  受试者级 AUC 差值 95%CI 是否跨 0: {np.percentile(da_s,2.5) <= 0 <= np.percentile(da_s,97.5)}')

pd.DataFrame([{
    'Metric': 'AUC',
    # 点估计取原测试集直接差值(唯一正式值), CI 用受试者级 bootstrap
    'Point_estimate': round(pt_auc, 4),
    'CI_low_subject': round(np.percentile(da_s, 2.5), 4),
    'CI_high_subject': round(np.percentile(da_s, 97.5), 4),
    'CI_low_record': round(np.percentile(da_r, 2.5), 4),
    'CI_high_record': round(np.percentile(da_r, 97.5), 4),
}, {
    'Metric': 'AP',
    'Point_estimate': round(pt_ap, 4),
    'CI_low_subject': round(np.percentile(dp_s, 2.5), 4),
    'CI_high_subject': round(np.percentile(dp_s, 97.5), 4),
    'CI_low_record': round(np.percentile(dp_r, 2.5), 4),
    'CI_high_record': round(np.percentile(dp_r, 97.5), 4),
}]).to_csv('tables/table2_ci_paired_diff_lstm_rf.csv', index=False)

print('\n已保存: tables/table2_ci_record_vs_subject.csv')
print('        tables/table2_ci_paired_diff_lstm_rf.csv')
