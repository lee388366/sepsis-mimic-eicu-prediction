import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# =====================================================================
# eicu_08_subgroup_external.py —— eICU 外验种族/性别亚组性能 + bootstrap CI
#
# 数据：eicu_predictions.npz（eicu_06 产物）+ eicu_demographics.csv（eicu_04b）
# 与 MIMIC 的 10b_subgroup_race.py 同口径，实现"内外部验证亚组公平性"闭环
#
# v2 修改：性别只保留 Male/Female，过滤 eICU 中 1 例 Gender:Unknown（避免 nan 行）
#
# 输出: tables/table9b_subgroup_external.csv
# =====================================================================

SEED = 42
N_BOOT = 1000

# ---------- 1. 加载并对齐 ----------
m = np.load('eicu_predictions.npz')
demo = pd.read_csv('eicu_demographics.csv')
demo['patientunitstayid'] = demo['patientunitstayid'].astype(int)

df = pd.DataFrame({
    'stay_id': m['stay_id'],
    'y': m['y_true'].astype(int),
    'LSTM': m['LSTM'].astype(float),
    'RF': m['RandomForest'].astype(float),
    'SOFA': m['sofa'].astype(float),
    'qsofa6': m['qsofa6'].astype(float),
})
df = df.merge(demo.rename(columns={'patientunitstayid': 'stay_id'}), on='stay_id')
assert len(df) == len(m['stay_id'])

# 过滤 Gender:Unknown（eICU 中 1 例），只保留 Male/Female
n_unknown = (df['gender'] == 'Unknown').sum()
df = df[df['gender'].isin(['Male', 'Female'])].reset_index(drop=True)
print(f'对齐后样本: {len(df)} 例 | 死亡率 {df["y"].mean():.3f} | 过滤 Gender:Unknown {n_unknown} 例')

# ---------- 2. bootstrap 函数（AUC + 死亡率） ----------
def boot_auc(y, prob, n_boot=N_BOOT, seed=SEED):
    """bootstrap 重采样（有放回），返回 AUC 的 2.5%/97.5% 分位数"""
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot)
    n = len(y)
    idx = np.arange(n)
    for b in range(n_boot):
        s = rng.choice(idx, size=n, replace=True)
        if len(np.unique(y[s])) < 2:
            aucs[b] = np.nan
        else:
            aucs[b] = roc_auc_score(y[s], prob[s])
    lo, hi = np.nanpercentile(aucs, [2.5, 97.5])
    return lo, hi

def boot_rate(y, n_boot=N_BOOT, seed=SEED):
    """bootstrap 重采样，返回死亡率的 2.5%/97.5% 分位数"""
    rng = np.random.default_rng(seed)
    rates = np.empty(n_boot)
    n = len(y)
    idx = np.arange(n)
    for b in range(n_boot):
        s = rng.choice(idx, size=n, replace=True)
        rates[b] = y[s].mean()
    lo, hi = np.nanpercentile(rates, [2.5, 97.5])
    return lo, hi

# ---------- 3. 亚组定义 ----------
SUBS = [('Race', 'race_group'), ('Gender', 'gender')]
MODELS = ['LSTM', 'RF', 'SOFA', 'qsofa6']

rows = []
for dim, col in SUBS:
    for grp in sorted(df[col].dropna().unique()):
        sub = df[df[col] == grp]
        if len(sub) < 100:
            print(f'  ⚠️ {dim}:{grp} 样本仅 {len(sub)} 例，CI 会较宽，谨慎解读')
        dr_lo, dr_hi = boot_rate(sub['y'].values)
        row = {'Subgroup': f'{dim}: {grp}', 'n': len(sub),
               'Death_rate': sub['y'].mean(),
               'Death_rate_CI': f'{dr_lo:.3f}-{dr_hi:.3f}'}
        for mdl in MODELS:
            auc = roc_auc_score(sub['y'], sub[mdl])
            lo, hi = boot_auc(sub['y'].values, sub[mdl].values)
            row[f'{mdl}_AUC'] = auc
            row[f'{mdl}_95CI'] = f'{lo:.3f}-{hi:.3f}'
            row[f'{mdl}_CI_lo'] = lo
            row[f'{mdl}_CI_hi'] = hi
        rows.append(row)

res = pd.DataFrame(rows)

print('\n===== eICU 外验 社会人口学亚组 × 模型 AUC（含 95%CI）=====')
for _, r in res.iterrows():
    print(f"{r['Subgroup']:32s} n={int(r['n']):5d} 死亡={r['Death_rate']:.3f} [{r['Death_rate_CI']}] | "
          + ' | '.join(f"{mdl} {r[f'{mdl}_AUC']:.3f} [{r[f'{mdl}_95CI']}]" for mdl in MODELS))

res.to_csv('tables/table9b_subgroup_external.csv', index=False)
print('\n已保存: tables/table9b_subgroup_external.csv')

# ---------- 4. 跨种族公平性 ----------
print('\n===== 跨种族亚组 AUC 范围（外验，含 CI）=====')
race_rows = res[res['Subgroup'].str.startswith('Race')]
for mdl in MODELS:
    v = race_rows[f'{mdl}_AUC']
    white = race_rows[race_rows['Subgroup'] == 'Race: White'][f'{mdl}_AUC'].values[0]
    print(f'  {mdl:6s}: min={v.min():.3f} max={v.max():.3f} range={v.max()-v.min():.3f} | 白种人={white:.3f}')