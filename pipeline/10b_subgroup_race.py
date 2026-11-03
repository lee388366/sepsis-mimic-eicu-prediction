import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# =====================================================================
# 10b_subgroup_race.py —— 种族/性别亚组性能对比 + bootstrap 95% CI
#
# 给每个亚组 × 模型的 AUC 加 bootstrap 95% CI（与 07_bootstrap_ci.py 同风格），
# 并给每个亚组的死亡率（Death_rate）也加 bootstrap 95% CI，
# 解决小样本亚组（如 Asian n=99）"估计不稳但没法量化"的问题。
#
# 输出: tables/table8b_subgroup_race.csv（点估计 + 死亡率 CI）
#       + tables/table8b_subgroup_race_boot.csv（含 CI 完整版，论文可引用）
# =====================================================================

SEED = 42
N_BOOT = 1000

# ---------- 1. 加载并对齐 ----------
m = np.load('all_predictions.npz')
q = np.load('qsofa_predictions.npz')
s = np.load('sofa_predictions.npz')

df = pd.DataFrame({
    'stay_id': m['stay_id'],
    'y': m['y_true'].astype(int),
    'LSTM': m['LSTM'].astype(float),
    'RF': m['RandomForest'].astype(float),
    'SOFA': s['sofa'].astype(float),
    'qsofa6': q['qsofa6'].astype(float),
})

demo = pd.read_csv('demographics_by_stay.csv')
df = df.merge(demo, on='stay_id')
assert len(df) == len(m['stay_id'])
print(f'对齐后样本: {len(df)} 例 | 死亡率 {df["y"].mean():.3f}')

# ---------- 2. 亚组定义 ----------
SUBS = [('Race', 'race_group'), ('Gender', 'gender')]
MODELS = ['LSTM', 'RF', 'SOFA', 'qsofa6']

# ---------- 3. bootstrap 函数 ----------
def boot_auc(y, prob, n_boot=N_BOOT, seed=SEED):
    """bootstrap 重采样（有放回），返回 AUC 的 2.5%/97.5% 分位数"""
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot)
    n = len(y)
    idx = np.arange(n)
    for b in range(n_boot):
        s = rng.choice(idx, size=n, replace=True)
        # 采样后若单类样本 <2，跳过（AUC 无法定义）
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

# ---------- 4. 逐亚组计算 AUC + CI + 死亡率 CI ----------
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

# ---------- 5. 输出 ----------
print('\n===== 社会人口学亚组 × 模型 AUC（含 95%CI + 死亡率 CI）=====')
for _, r in res.iterrows():
    print(f"{r['Subgroup']:28s} n={int(r['n']):5d} | "
          f"死亡={r['Death_rate']:.3f} [{r['Death_rate_CI']}] | "
          + ' | '.join(f"{mdl} {r[f'{mdl}_AUC']:.3f} [{r[f'{mdl}_95CI']}]" for mdl in MODELS))

# 完整版（论文可引用）
res.to_csv('tables/table8b_subgroup_race.csv', index=False)
print('\n已保存: tables/table8b_subgroup_race.csv')

# 精简版（AUC + 死亡率 CI）
ci_cols = ['Subgroup', 'n', 'Death_rate', 'Death_rate_CI'] + \
          [f'{mdl}_AUC' for mdl in MODELS] + [f'{mdl}_95CI' for mdl in MODELS]
res[ci_cols].to_csv('tables/table8b_subgroup_race_boot.csv', index=False)
print('已保存: tables/table8b_subgroup_race_boot.csv')

# ---------- 6. 跨种族公平性：AUC range + 白种人 vs 其他 ----------
print('\n===== 跨种族亚组 AUC 范围（公平性证据，含 CI）=====')
race_rows = res[res['Subgroup'].str.startswith('Race')]
for mdl in MODELS:
    v = race_rows[f'{mdl}_AUC']
    white = race_rows[race_rows['Subgroup'] == 'Race: White'][f'{mdl}_AUC'].values[0]
    print(f'  {mdl:6s}: min={v.min():.3f} max={v.max():.3f} range={v.max()-v.min():.3f} '
          f'| 白种人={white:.3f}')