import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score, accuracy_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from split_utils import grouped_tts

# =====================================================================
# 06_train_baseline.py —— 脓毒症 48h 死亡预测 · 6 个经典 ML 基线
#
# 优化点：
#   1. KNN: n_neighbors=7, weights='distance'
#   2. SVM: CalibratedClassifierCV(SVC(), ensemble=False) 替代
#      SVC(probability=True)（解决弃用警告 + 提速）
#   3. 在验证集上阈值扫描（0.10~0.85），选最优 F3 阈值，
#      再用该阈值在测试集上报告 Precision/Recall/F1/F3
#
# 输出: baseline_results.csv + baseline_predictions.npz
# =====================================================================

SEED = 42
np.random.seed(SEED)

def fbeta_score(p, r, beta=3.0):
    return (1 + beta**2) * p * r / (beta**2 * p + r) if (beta**2 * p + r) > 0 else 0.0

def select_best_threshold(y_val, prob_val):
    """在验证集上扫描阈值，返回最优 F3 及对应阈值"""
    best_f3, best_th = 0.0, 0.5
    for th in np.arange(0.10, 0.86, 0.01):
        yp = (prob_val >= th).astype(int)
        p = precision_score(y_val, yp, zero_division=0)
        r = recall_score(y_val, yp, zero_division=0)
        f3 = fbeta_score(p, r)
        if f3 > best_f3:
            best_f3, best_th = f3, th
    return best_th, best_f3

# ---------- 1. 加载 ----------
d = np.load('sepsis_training_data.npz')
X_hcv = d['X_hcv'].astype(np.float32)
X_sv = d['X_sv'].astype(np.float32)
y = d['y'].astype(np.int64)
stay_id_all = d['stay_id'].astype(np.int64)   # 全量 stay_id（对齐用）
subject_id = d['subject_id'].astype(np.int64) # 受试者级分组切分
N = len(y)
print(f'数据: {N} 例 | X_hcv {X_hcv.shape} | 死亡率 {100*y.mean():.1f}%')

# ---------- 2. 特征聚合（与 08/09 一致：14 变量 × mean/min/max + 4 静态 = 46 维）----------
CLINICAL = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'spo2',
            'temperature', 'glucose', 'fio2', 'ph',
            'gcs', 'gcs_eyes', 'gcs_motor', 'gcs_verbal']
feat_parts = []
for j in range(len(CLINICAL)):
    series = X_hcv[:, j, :]
    feat_parts.append(series.mean(axis=1))
    feat_parts.append(series.min(axis=1))
    feat_parts.append(series.max(axis=1))
X = np.column_stack(feat_parts + [X_sv[:, 0], X_sv[:, 1], X_sv[:, 2], X_sv[:, 3]]).astype(np.float32)
print(f'聚合特征: {X.shape}（46 维）')

# ---------- 3. 划分 train/val/test（受试者级分组 70/15/15，零跨集泄漏）----------
tr_idx, va_idx, te_idx = grouped_tts(subject_id, y)

# ---------- 4. 标准化（只 fit 训练集）----------
scaler = StandardScaler()
X_tr = scaler.fit_transform(X[tr_idx])
X_va = scaler.transform(X[va_idx])
X_te = scaler.transform(X[te_idx])
y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]
print(f'train {len(tr_idx)} | val {len(va_idx)} | test {len(te_idx)}')

# ---------- 5. 模型定义 ----------
models = {
    'LogisticRegression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED),
    'DecisionTree':       DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=SEED),
    'RandomForest':       RandomForestClassifier(n_estimators=200, max_depth=15,
                                                 class_weight='balanced', n_jobs=-1, random_state=SEED),
    'KNN':                KNeighborsClassifier(n_neighbors=7, weights='distance'),
    'SVM':                CalibratedClassifierCV(
                              SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=SEED),
                              ensemble=False),
    'GaussianNB':         GaussianNB(),
}

# ---------- 6. 训练 + 验证选优阈值 ----------
results = []
preds_all = {}
for name, clf in models.items():
    clf.fit(X_tr, y_tr)
    va_prob = clf.predict_proba(X_va)[:, 1] if hasattr(clf, 'predict_proba') else clf.decision_function(X_va)
    va_auc = roc_auc_score(y_va, va_prob)

    # 验证集阈值扫描（选最优 F3 阈值）
    best_th, best_val_f3 = select_best_threshold(y_va, va_prob)
    print(f'  {name:20s} val AUC={va_auc:.4f} | 最优阈值={best_th:.2f} | val F3={best_val_f3:.4f}')

    # 测试集评估（用验证集选出的阈值）
    te_prob = clf.predict_proba(X_te)[:, 1] if hasattr(clf, 'predict_proba') else clf.decision_function(X_te)
    bin_pred = (te_prob >= best_th).astype(int)

    p = precision_score(y_te, bin_pred, zero_division=0)
    r = recall_score(y_te, bin_pred, zero_division=0)

    results.append({
        'model': name,
        'AUC': roc_auc_score(y_te, te_prob),
        'AUPRC': average_precision_score(y_te, te_prob),
        'best_threshold': round(best_th, 2),
        'Accuracy': accuracy_score(y_te, bin_pred),
        'Precision': p,
        'Recall': r,
        'F1': f1_score(y_te, bin_pred, zero_division=0),
        'F3': fbeta_score(p, r),
    })
    preds_all[name] = te_prob
    joblib.dump(clf, f'models/model_{name}.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')  # 保存标准化器

# ---------- 7. 输出 ----------
df_res = pd.DataFrame(results).sort_values('AUC', ascending=False).reset_index(drop=True)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')
print('\n===== 测试集结果（按 AUC 排序，阈值来自验证集扫描）=====')
print(df_res.to_string(index=False))
df_res.to_csv('baseline_results.csv', index=False)

np.savez('baseline_predictions.npz',
         **{name.replace(' ', '_'): prob for name, prob in preds_all.items()},
         y_true=y_te,
         stay_id=stay_id_all[te_idx])   # 测试集 stay_id，供追溯
print('\n已保存: baseline_results.csv + baseline_predictions.npz')