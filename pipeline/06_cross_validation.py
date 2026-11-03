import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from split_utils import grouped_kfold

# =====================================================================
# 06_cross_validation.py —— RandomForest 10 折交叉验证（评估泛化稳定性）
#
# 与 06_train_baseline.py 完全同源：同特征聚合、同 seed、同 RF 超参
# 做法: 全量数据 10 折分层交叉验证，每折报告验证集 AUC / AP
# 输出: 终端打印 每折 AUC / AP + 10 折均值 ± SD（供正文引用）
#       并保存 tables/table3_cv_10fold.csv（每折明细 + mean/SD 汇总行）
# =====================================================================

SEED = 42
np.random.seed(SEED)

# ---------- 1-2. 与 06 相同的特征聚合（46 维）----------
d = np.load('sepsis_training_data.npz')
X_hcv = d['X_hcv'].astype(np.float32)
X_sv = d['X_sv'].astype(np.float32)
y = d['y'].astype(np.int64)
subject_id = d['subject_id'].astype(np.int64)
N = len(y)

CLINICAL = ['Heart Rate', 'Systolic BP', 'Diastolic BP', 'Mean BP', 'Respiratory Rate',
            'SpO2', 'Temperature', 'Glucose', 'FiO2', 'pH', 'GCS Score',
            'GCS Eyes', 'GCS Motor', 'GCS Verbal']
feat_parts = []
for j in range(len(CLINICAL)):
    series = X_hcv[:, j, :]
    feat_parts += [series.mean(1), series.min(1), series.max(1)]
X = np.column_stack(feat_parts + [X_sv[:, 0], X_sv[:, 1], X_sv[:, 2], X_sv[:, 3]]).astype(np.float32)
print(f'数据: {N} 例 | 特征: {X.shape[1]} 维 | 阳性率: {y.mean():.3f}')

# ---------- 3. 10 折受试者级分层交叉验证 ----------
# 注意：StandardScaler 必须在每一折的 TRAIN 部分单独 fit，
#        避免数据泄漏（用整集 fit 会让验证折的信息渗入训练）。
#       StratifiedGroupKFold 保证同一 subject 只在同一折（零跨集泄漏）。
print('\n=== 10 折受试者级分层交叉验证 ===')
auc_list = []
ap_list = []
for fold, (tr_idx, va_idx) in enumerate(grouped_kfold(X, y, subject_id, n_splits=10), 1):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr_idx])
    X_va = scaler.transform(X[va_idx])
    y_tr, y_va = y[tr_idx], y[va_idx]

    rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                class_weight='balanced', n_jobs=-1, random_state=SEED)
    rf.fit(X_tr, y_tr)
    prob_va = rf.predict_proba(X_va)[:, 1]

    auc_f = roc_auc_score(y_va, prob_va)
    ap_f = average_precision_score(y_va, prob_va)
    auc_list.append((fold, auc_f))
    ap_list.append((fold, ap_f))
    print(f'  Fold {fold:>2}: AUC = {auc_f:.4f} | AP = {ap_f:.4f}')

auc_arr = np.array([v for _, v in auc_list])
ap_arr = np.array([v for _, v in ap_list])

# ---------- 4. 汇总 ----------
print('\n===== 10 折汇总（供正文引用）=====')
print(f'AUC: {auc_arr.mean():.4f} ± {auc_arr.std(ddof=1):.4f} (范围 {auc_arr.min():.4f}–{auc_arr.max():.4f})')
print(f'AP : {ap_arr.mean():.4f} ± {ap_arr.std(ddof=1):.4f} (范围 {ap_arr.min():.4f}–{ap_arr.max():.4f})')

# ---------- 5. 保存 10 折结果 ----------
df_cv = pd.DataFrame({
    'Model': 'RandomForest',
    'fold': [f for f, _ in auc_list],
    'AUC':  [v for _, v in auc_list],
    'AP':   [v for _, v in ap_list],
})
df_cv.loc[len(df_cv)] = ['RandomForest', 'mean', auc_arr.mean(), ap_arr.mean()]
df_cv.loc[len(df_cv)] = ['RandomForest', 'SD',   auc_arr.std(ddof=1), ap_arr.std(ddof=1)]
df_cv.to_csv('tables/table3_cv_10fold.csv', index=False)
print('已保存: tables/table3_cv_10fold.csv')