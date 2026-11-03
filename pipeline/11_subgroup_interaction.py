# 11_subgroup_interaction.py
# 把"ML 相对 SOFA 的优势在低危/非休克组更大"从描述性 ΔAUC 升级为：
#   1) 组内 LSTM vs SOFA 的 DeLong 检验（复用已验证实现）
#   2) 组内 ΔAUC 的 bootstrap 95%CI
#   3) 组间 ΔAUC 差值 (Δlow − Δhigh) 的 bootstrap 95%CI（跨组是否显著不同）
#   4) logistic 交互检验（LSTM logit 与亚组的交互项）
#   5) Bonferroni 多重校正
# 依赖: sklearn / scipy / statsmodels
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

SEED = 42

# ---------- DeLong（原样取自 10_delong_test.py，已验证）----------
def compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N, dtype=float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1); i = j
    T2 = np.empty(N, dtype=float); T2[J] = T + 1
    return T2

def fastDeLong(pts, m):
    n = pts.shape[1] - m
    pos, neg = pts[:, :m], pts[:, m:]
    k = pts.shape[0]
    tx = np.empty([k, m], dtype=float); ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(pos[r, :]); ty[r, :] = compute_midrank(neg[r, :])
        tz[r, :] = compute_midrank(pts[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    return aucs, np.cov(v01) / m + np.cov(v10) / n

def delong_p(gt, s1, s2):
    order = (-gt).argsort(); m = int(gt.sum())
    pts = np.vstack((s1, s2))[:, order]
    aucs, cov = fastDeLong(pts, m)
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, cov), l.T))
    log10p = np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)
    return 10 ** float(np.asarray(log10p).reshape(-1)[0])

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def boot_ci_dauc(y, p1, p2, n=1000):
    """组内 ΔAUC = AUC(p1) − AUC(p2) 的 bootstrap 95%CI"""
    rng = np.random.default_rng(SEED); idx = np.arange(len(y)); ds = []
    for _ in range(n):
        i = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[i])) < 2: continue
        ds.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    return np.percentile(ds, [2.5, 97.5])

def boot_ci_ddiff(y, grp, p1, p2, lo, hi, n=1000):
    """跨组差异: (ΔAUC_low − ΔAUC_high) 的 bootstrap 95%CI"""
    rng = np.random.default_rng(SEED); idx = np.arange(len(y)); ds = []
    for _ in range(n):
        i = rng.choice(idx, size=len(idx), replace=True)
        ml, mh = grp[i] == lo, grp[i] == hi
        if ml.sum() < 2 or mh.sum() < 2: continue
        if len(np.unique(y[i][ml])) < 2 or len(np.unique(y[i][mh])) < 2: continue
        d1 = roc_auc_score(y[i][ml], p1[i][ml]) - roc_auc_score(y[i][ml], p2[i][ml])
        d2 = roc_auc_score(y[i][mh], p1[i][mh]) - roc_auc_score(y[i][mh], p2[i][mh])
        ds.append(d1 - d2)
    return np.percentile(ds, [2.5, 97.5])

def interaction_p(df, grp_col):
    """y ~ 1 + logit(LSTM) + 亚组 + logit(LSTM)×亚组，返回交互项 p 值"""
    g = pd.Categorical(df[grp_col]).codes.astype(float)   # 0/1
    lp = logit(df['LSTM'].values)
    X = np.column_stack([np.ones(len(df)), lp, g, lp * g])
    m = sm.Logit(df['y'].values, X).fit(disp=0)
    beta, se = m.params[3], m.bse[3]
    z = beta / se
    return 2 * (1 - scipy.stats.norm.cdf(abs(z))), beta, m.conf_int()[3]

# ---------- 1. 加载并合并（与 10_subgroup_performance.py 同口径）----------
m = np.load('all_predictions.npz'); q = np.load('qsofa_predictions.npz')
s = np.load('sofa_predictions.npz'); d = np.load('sepsis_training_data.npz')
se = pd.read_csv('sofa_extract.csv'); se['stay_id'] = se['stay_id'].astype(int)

df = pd.DataFrame({'stay_id': m['stay_id'], 'y': m['y_true'].astype(int),
                   'LSTM': m['LSTM'].astype(float), 'RF': m['RandomForest'].astype(float),
                   'SOFA': s['sofa'].astype(float), 'qsofa6': q['qsofa6'].astype(float)})
df = df.merge(pd.DataFrame({'stay_id': d['stay_id'], 'age': d['X_sv'][:, 0]}), on='stay_id')
df = df.merge(se[['stay_id', 'cardiovascular']], on='stay_id')
print(f'对齐后: {len(df)} 例 | 死亡率 {df["y"].mean():.3f}')

df['age_grp']    = np.where(df['age'] < 65, 'age<65', 'age>=65')
df['sofa_grp']   = np.where(df['SOFA'] < 8, 'SOFA<8', 'SOFA>=8')
df['shock_grp']  = np.where(df['cardiovascular'] >= 2, 'shock', 'nonshock')

DIMS = [('Age', 'age_grp'), ('Initial SOFA', 'sofa_grp'), ('Septic shock', 'shock_grp')]

# ---------- 2. 组内 DeLong + ΔAUC CI + 组间 ΔCI ----------
rows = []
for dim, col in DIMS:
    g_lo, g_hi = sorted(df[col].unique())   # 按字母序 low/high
    sub_lo, sub_hi = df[df[col] == g_lo], df[df[col] == g_hi]
    for gname, sub in [(g_lo, sub_lo), (g_hi, sub_hi)]:
        dauc_ci = boot_ci_dauc(sub['y'].values, sub['LSTM'].values, sub['SOFA'].values)
        rows.append({
            'Dimension': dim, 'Subgroup': gname, 'n': len(sub),
            'Death_rate': sub['y'].mean(),
            'AUC_LSTM': roc_auc_score(sub['y'], sub['LSTM']),
            'AUC_SOFA': roc_auc_score(sub['y'], sub['SOFA']),
            'dAUC_LSTM-SOFA': roc_auc_score(sub['y'], sub['LSTM']) - roc_auc_score(sub['y'], sub['SOFA']),
            'dAUC_CI_low': dauc_ci[0], 'dAUC_CI_high': dauc_ci[1],
            'DeLong_p': delong_p(sub['y'].values, sub['LSTM'].values, sub['SOFA'].values),
        })
    # 组间 ΔAUC 差 (low − high)：点估计 + bootstrap 95%CI
    dd = boot_ci_ddiff(df['y'].values, df[col].values, df['LSTM'].values,
                       df['SOFA'].values, g_lo, g_hi)
    dauc_lo = roc_auc_score(sub_lo['y'], sub_lo['LSTM']) - roc_auc_score(sub_lo['y'], sub_lo['SOFA'])
    dauc_hi = roc_auc_score(sub_hi['y'], sub_hi['LSTM']) - roc_auc_score(sub_hi['y'], sub_hi['SOFA'])
    rows.append({'Dimension': dim, 'Subgroup': f'diff({g_lo} - {g_hi})',
                 'dAUC_LSTM-SOFA': dauc_lo - dauc_hi,
                 'dAUC_CI_low': dd[0], 'dAUC_CI_high': dd[1],
                 'CI_excludes_0': (dd[0] > 0) or (dd[1] < 0)})
res = pd.DataFrame(rows)
res.to_csv('tables/table11_subgroup_dauc_ci.csv', index=False)
print('\n===== 组内 DeLong + ΔAUC 95%CI + 组间差（完整表见 csv）=====')
print(res.round(3).to_string(index=False))

# ---------- 3. logistic 交互检验（LSTM logit × 亚组）----------
print('\n===== 交互检验（logit(LSTM) × 亚组，Bonferroni 校正）=====')
it = []
for dim, col in DIMS:
    p, beta, ci = interaction_p(df, col)
    it.append({'Dimension': dim, 'interaction_p': p, 'beta': beta,
               'ci_low': ci[0], 'ci_high': ci[1]})
itdf = pd.DataFrame(it)
itdf['p_bonferroni'] = np.clip(itdf['interaction_p'] * len(DIMS), 0, 1)
itdf.to_csv('tables/table11_subgroup_interaction.csv', index=False)
print(itdf.round(4).to_string(index=False))
print('\n已保存: tables/table11_subgroup_dauc_ci.csv + table11_subgroup_interaction.csv')