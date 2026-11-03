import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# =====================================================================
# 10_subgroup_performance.py —— 种族/性别亚组性能对比（TRIPOD+AI 23a 公平性）
#
# 社会人口学亚组（对照 10_subgroup_performance.py 的临床亚组）：
#   1. 种族:   White / Black / Asian / Hispanic / Other / Unknown-Refused
#   2. 性别:   M / F
#
# 每个亚组内对比 LSTM / RF / SOFA / qSOFA-6h 的 AUC，回答：
#   模型性能是否在关键社会人口学亚组间存在差异（公平性）。
#
# 数据来源：demographics_by_stay.csv（04c_extract_demographics.sql 产物）
#            + all_predictions.npz / sofa_predictions.npz / qsofa_predictions.npz
#
# 输出: tables/table8_subgroup_race_point.csv
#       （点估计版，无 CI；含 CI 的完整版由 10b_subgroup_race.py 输出到
#        table8b_subgroup_race.csv / table8b_subgroup_race_boot.csv，
#        两脚本输出文件名已解耦，运行顺序互不覆盖）
# =====================================================================

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

# 人口学明细（04c_extract_demographics.sql 产物，按 stay_id 索引）
demo = pd.read_csv('demographics_by_stay.csv')
df = df.merge(demo, on='stay_id')

# 对齐检查（防静默错位）
assert len(df) == len(m['stay_id']), f'merge 后样本不一致: {len(df)}'
print(f'对齐后样本: {len(df)} 例 | 死亡率 {df["y"].mean():.3f}')

# ---------- 2. 亚组定义 ----------
SUBS = [
    ('Race', 'race_group'),
    ('Gender', 'gender'),
]
MODELS = ['LSTM', 'RF', 'SOFA', 'qsofa6']

# ---------- 3. 计算各亚组 AUC ----------
rows = []
for dim, col in SUBS:
    for grp in sorted(df[col].dropna().unique()):
        sub = df[df[col] == grp]
        if len(sub) < 100:
            print(f'  ⚠️ {dim}:{grp} 样本仅 {len(sub)} 例，谨慎解读')
        row = {'Subgroup': f'{dim}: {grp}', 'n': len(sub),
               'Death_rate': sub['y'].mean()}
        for mdl in MODELS:
            row[f'{mdl}_AUC'] = roc_auc_score(sub['y'], sub[mdl])
        rows.append(row)

res = pd.DataFrame(rows)
print('\n===== 社会人口学亚组 × 模型 AUC =====')
print(res.round(4).to_string(index=False))
res.to_csv('tables/table8_subgroup_race_point.csv', index=False)
print('\n已保存: tables/table8_subgroup_race_point.csv（点估计版，CI 版见 10b_subgroup_race.py）')

# ---------- 4. 公平性检查：模型在各亚组间的 AUC 离散度 ----------
print('\n===== 各模型跨种族亚组 AUC 范围（公平性证据）=====')
for mdl in MODELS:
    col = f'{mdl}_AUC'
    race_rows = res[res['Subgroup'].str.startswith('Race')]
    v = race_rows[col]
    print(f'  {mdl:6s}: min={v.min():.4f} max={v.max():.4f} '
          f'range={v.max()-v.min():.4f} '
          f'| 白种人={race_rows[race_rows["Subgroup"]=="Race: White"][col].values[0]:.4f}')