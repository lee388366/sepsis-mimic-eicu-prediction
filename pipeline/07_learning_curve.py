import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from split_utils import grouped_tts

# =====================================================================
# 07_learning_curve.py —— RandomForest 学习曲线（诊断过拟合）
#
# 与 06/08/09 完全同源：同特征聚合、同 seed、同 RF 超参
# 做法: 训练样本量从 2000 逐步增至 17433（分 6 档），
#       每档画 训练集 AUC 与 验证集 AUC，检验两者是否收敛
# 输出: learning_curve_rf.png
# =====================================================================

SEED = 42
np.random.seed(SEED)

# ---------- 1-2. 与 06 相同的特征聚合（46 维）----------
d = np.load('sepsis_training_data.npz')
X_hcv = d['X_hcv'].astype(np.float32)
X_sv = d['X_sv'].astype(np.float32)
y = d['y'].astype(np.int64)
N = len(y)

CLINICAL = ['Heart Rate', 'Systolic BP', 'Diastolic BP', 'Mean BP', 'Respiratory Rate',
            'SpO2', 'Temperature', 'Glucose', 'FiO2', 'pH', 'GCS Score',
            'GCS Eyes', 'GCS Motor', 'GCS Verbal']
feat_parts = []
for j in range(len(CLINICAL)):
    series = X_hcv[:, j, :]
    feat_parts += [series.mean(1), series.min(1), series.max(1)]
X = np.column_stack(feat_parts + [X_sv[:, 0], X_sv[:, 1], X_sv[:, 2], X_sv[:, 3]]).astype(np.float32)

# ---------- 3. 划分（受试者级分组，同 06）----------
tr_idx, va_idx, te_idx = grouped_tts(d['subject_id'].astype(np.int64), y)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X[tr_idx])
X_va = scaler.transform(X[va_idx])
y_tr, y_va = y[tr_idx], y[va_idx]

# ---------- 4. 学习曲线：训练样本量分档 ----------
TRAIN_SIZES = [2000, 4000, 7000, 10000, 14000, len(tr_idx)]   # 最后一档 = 全部训练集
N_REPEAT = 3    # 每个档位重复 3 次取均值

train_aucs = np.zeros((len(TRAIN_SIZES), N_REPEAT))
val_aucs = np.zeros((len(TRAIN_SIZES), N_REPEAT))

print(f'训练集: {len(tr_idx)} 例 | 验证集: {len(va_idx)} 例')
for si, n_train in enumerate(TRAIN_SIZES):
    for rep in range(N_REPEAT):
        # 手动打乱 + 取前 n_train 个（支持 n_train = 全量）
        rng = np.random.default_rng(SEED * 100 + si * 10 + rep)
        perm = rng.permutation(len(tr_idx))
        sub_idx = perm[:n_train]
        rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                    class_weight='balanced', n_jobs=-1, random_state=SEED)
        rf.fit(X_tr[sub_idx], y_tr[sub_idx])
        train_aucs[si, rep] = roc_auc_score(y_tr[sub_idx], rf.predict_proba(X_tr[sub_idx])[:, 1])
        val_aucs[si, rep] = roc_auc_score(y_va, rf.predict_proba(X_va)[:, 1])
    print(f'  n={n_train:>6}: 训练AUC={train_aucs[si].mean():.4f}±{train_aucs[si].std():.4f}  '
          f'验证AUC={val_aucs[si].mean():.4f}±{val_aucs[si].std():.4f}')

# ---------- 5. 画图 ----------
plt.figure(figsize=(8, 6))
tr_mean, tr_std = train_aucs.mean(1), train_aucs.std(1)
va_mean, va_std = val_aucs.mean(1), val_aucs.std(1)

plt.fill_between(TRAIN_SIZES, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color='#D62728')
plt.fill_between(TRAIN_SIZES, va_mean - va_std, va_mean + va_std, alpha=0.15, color='#1F77B4')
plt.plot(TRAIN_SIZES, tr_mean, 'o-', color='#D62728', lw=2.2, label='Training set')
plt.plot(TRAIN_SIZES, va_mean, 's-', color='#1F77B4', lw=2.2, label='Validation set')

plt.xlabel('Number of training samples', fontsize=13)
plt.ylabel('AUC', fontsize=13)
plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/learning_curve_rf.png', dpi=300, bbox_inches='tight')
print('已保存: learning_curve_rf.png')