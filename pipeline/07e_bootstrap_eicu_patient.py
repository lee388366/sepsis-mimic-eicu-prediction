import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# =====================================================================
# 07e_bootstrap_eicu_patient.py —— eICU 外部 患者级(uniquepid) bootstrap 对照
#
# 背景: eICU 6,812 条最终记录经 07d 核查含 284 位重复入住患者(占记录
#   8.93%, 多余记录 324 条)。因此外部 95%CI 也应评估是否按患者聚类。
#   本脚本对比:
#     (A) 记录级 bootstrap  (当前 table12/13 外部列口径)
#     (B) 患者级 bootstrap  (有放回抽取 uniquepid, 其全部 ICU 记录一并进入)
#   输出各模型 AUC/AP 的两种 CI, 供决定外部正式口径。
#
# 数据源: eicu_predictions.npz + 本地 postgres(eicu_crd.patient) 的 uniquepid
# 输出:   tables/table2_ci_eicu_record_vs_patient.csv
# =====================================================================

SEED = 42
N_BOOT = 1000
MODELS = ['LSTM', 'RandomForest', 'SVM', 'LogisticRegression',
          'GaussianNB', 'KNN', 'DecisionTree', 'SOFA', 'qSOFA-6h']
KEY_MAP = {'SOFA': 'sofa', 'qSOFA-6h': 'qsofa6'}

# ---- 1. 预测与标签 ----
d = np.load('eicu_predictions.npz')
y = d['y_true'].astype(int)
n = len(y)
stays = d['stay_id'].astype(np.int64).tolist()

# ---- 2. uniquepid 映射(本地 postgres) ----
ids = ','.join(map(str, stays))
sql = ("SELECT patientunitstayid, uniquepid FROM eicu_crd.patient "
       "WHERE patientunitstayid IN (" + ids + ") ORDER BY patientunitstayid;")
r = subprocess.run(['psql', '-h', 'localhost', '-U', 'postgres', '-d', 'eicu',
                    '-At', '-c', sql], capture_output=True, text=True)
if r.returncode != 0:
    raise RuntimeError(f'psql 失败: {r.stderr[:500]}')
pid_map = {int(k): v for k, v in (ln.split('|') for ln in r.stdout.strip().splitlines())}
pid_arr = np.array([pid_map[s] for s in stays])
vc = pd.Series(pid_arr).value_counts()
print(f'eICU 外部: {n} 条记录 / {len(vc)} 位患者')
print(f'重复入住患者 {int((vc>1).sum())} 人, 贡献 {int(vc[vc>1].sum())} 条记录 '
      f'({100*vc[vc>1].sum()/n:.2f}%)')

rec_by_pid = {}
for i, p in enumerate(pid_arr):
    rec_by_pid.setdefault(p, []).append(i)
pids = list(rec_by_pid.keys())

# ---- 3. 抽样 ----
_rng_rec = np.random.default_rng(1)
boot_rec = [_rng_rec.choice(n, n, replace=True) for _ in range(N_BOOT)]
_rng_pid = np.random.default_rng(SEED)
boot_pid = []
for _ in range(N_BOOT):
    picked = _rng_pid.choice(len(pids), len(pids), replace=True)
    boot_pid.append(np.concatenate([rec_by_pid[pids[k]] for k in picked]))

def ci_from(prob, batches, metric):
    bs = np.zeros(len(batches))
    for b, idx in enumerate(batches):
        bs[b] = metric(y[idx], prob[idx])
    return np.mean(bs), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

print(f'\n{"模型":<18}{"记录级AUC CI":<22}{"患者级AUC CI":<22}'
      f'{"记录级AP CI":<22}{"患者级AP CI"}')
rows = []
for m in MODELS:
    prob = (d[m] if m not in KEY_MAP else d[KEY_MAP[m]]).astype(float)
    auc_rec = ci_from(prob, boot_rec, roc_auc_score)   # (mean, lo, hi)
    auc_pid = ci_from(prob, boot_pid, roc_auc_score)
    ap_rec = ci_from(prob, boot_rec, average_precision_score)
    ap_pid = ci_from(prob, boot_pid, average_precision_score)
    pt_auc = roc_auc_score(y, prob)
    pt_ap = average_precision_score(y, prob)
    rows.append({
        'Model': m,
        'AUC_point': round(pt_auc, 4),
        'AUC_rec_low': round(auc_rec[1], 3), 'AUC_rec_high': round(auc_rec[2], 3),
        'AUC_pid_low': round(auc_pid[1], 3), 'AUC_pid_high': round(auc_pid[2], 3),
        'AP_point': round(pt_ap, 4),
        'AP_rec_low': round(ap_rec[1], 3), 'AP_rec_high': round(ap_rec[2], 3),
        'AP_pid_low': round(ap_pid[1], 3), 'AP_pid_high': round(ap_pid[2], 3),
    })
    print(f'{m:<18}'
          f'{pt_auc:.3f} ({auc_rec[1]:.3f}–{auc_rec[2]:.3f})   '
          f'{pt_auc:.3f} ({auc_pid[1]:.3f}–{auc_pid[2]:.3f})   '
          f'{pt_ap:.3f} ({ap_rec[1]:.3f}–{ap_rec[2]:.3f})  '
          f'{pt_ap:.3f} ({ap_pid[1]:.3f}–{ap_pid[2]:.3f})')

df = pd.DataFrame(rows)
df.to_csv('tables/table2_ci_eicu_record_vs_patient.csv', index=False,
          float_format='%.3f')
print('\n已保存: tables/table2_ci_eicu_record_vs_patient.csv')
