#!/usr/bin/env python3
"""shap_cache.py —— S14/S18 共用：RF 超参唯一来源 + 测试集 SHAP 缓存（带来源校验）。

背景：12f/12g 原先直接读写 tmp_shap_testset.npz，一旦缓存过期、被删或 RF
超参被改，会静默画出旧模型/旧患者的图。本模块把 RF 超参收敛到 RF_PARAMS
一处，并让 load_shap() 在加载缓存前校验缓存内记录的 seed/超参/特征结构：

- 校验通过 → 直接加载，秒级；
- 校验失败或缓存缺失 → 自动重新训练 RF 并重算测试集 SHAP，再写回缓存。

因此"更换 RF 超参"只需改本文件的 RF_PARAMS 一处，再跑 12f/12g 即可，
无需手动删缓存。
"""
import json
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap
from split_utils import grouped_tts

SEED = 42
RF_PARAMS = dict(n_estimators=200, max_depth=15,
                 class_weight='balanced', n_jobs=-1, random_state=SEED)
CACHE = 'tmp_shap_testset.npz'

CLINICAL = {
    'heart_rate':   'Heart Rate',
    'sbp':          'Systolic BP',
    'dbp':          'Diastolic BP',
    'mbp':          'Mean BP',
    'resp_rate':    'Respiratory Rate',
    'spo2':         'SpO2',
    'temperature':  'Temperature',
    'glucose':      'Glucose',
    'fio2':         'FiO2',
    'ph':           'pH',
    'gcs':          'GCS Score',
    'gcs_eyes':     'GCS Eyes',
    'gcs_motor':    'GCS Motor',
    'gcs_verbal':   'GCS Verbal',
}
SV_CLINICAL = ['Age', 'Gender (M)', 'Weight', 'Height']


def build_features(d):
    """与 09/12f/12g 完全同源的 46 维特征聚合（14 变量 × 3 统计量 + 4 静态）。"""
    X_hcv = d['X_hcv'].astype(np.float32)
    X_sv = d['X_sv'].astype(np.float32)
    y = d['y'].astype(np.int64)
    feat_names, feat_parts = [], []
    for j, (var, name) in enumerate(CLINICAL.items()):
        series = X_hcv[:, j, :]
        feat_parts.append(series.mean(axis=1)); feat_names.append(f'{name} (mean)')
        feat_parts.append(series.min(axis=1));  feat_names.append(f'{name} (min)')
        feat_parts.append(series.max(axis=1));  feat_names.append(f'{name} (max)')
    for j, name in enumerate(SV_CLINICAL):
        feat_parts.append(X_sv[:, j]); feat_names.append(name)
    X = np.column_stack(feat_parts).astype(np.float32)
    return X, feat_names, y


def _want_meta():
    return {
        'seed': SEED,
        'n_estimators': RF_PARAMS['n_estimators'],
        'max_depth': RF_PARAMS['max_depth'],
        'class_weight': RF_PARAMS['class_weight'],
        'random_state': RF_PARAMS['random_state'],
    }


def _meta_ok(meta, feat_names):
    for k, v in _want_meta().items():
        if meta.get(k) != v:
            print(f'[shap_cache] 超参不一致: {k} 缓存={meta.get(k)} 当前={v}')
            return False
    if meta.get('feature_names') != feat_names:
        print('[shap_cache] 特征结构不一致，重新计算')
        return False
    return True


def load_shap(force=False):
    """返回 dict：
        sv         (n_te, 46, 2) 全部类别 SHAP values
        base       (n_te,)       死亡类基线 E[f(X)]
        prob       (n_te,)       死亡预测概率
        X          (N, 46)       原始（未标准化）聚合特征
        y          (N,)          标签
        te_idx     (n_te,)       测试集下标
        stay_id    (N,)          stay_id
        feat_names (46,)         特征名
        meta       dict          缓存来源信息
        cached     bool          是否直接命中缓存
    """
    d = np.load('sepsis_training_data.npz')
    X, feat_names, y = build_features(d)
    np.random.seed(SEED)                      # 保证与历史 split 完全一致
    tr_idx, va_idx, te_idx = grouped_tts(d['subject_id'].astype(np.int64), y)

    if not force:
        try:
            z = np.load(CACHE, allow_pickle=True)
            meta = json.loads(z['meta'][0])
            if _meta_ok(meta, feat_names):
                print(f'[shap_cache] 缓存有效（创建于 {meta["created"]}），直接加载 {CACHE}')
                return dict(sv=z['sv'], base=z['base'], prob=z['prob'], X=X, y=y,
                            te_idx=te_idx, stay_id=d['stay_id'], feat_names=feat_names,
                            meta=meta, cached=True)
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            pass
        print('[shap_cache] 无有效缓存或超参已变更 → 重新训练 RF 并重算测试集 SHAP（约 3–4 分钟）')

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr_idx])
    X_te = scaler.transform(X[te_idx])
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_tr, y[tr_idx])
    explainer = shap.TreeExplainer(rf)
    sv_all = explainer(X_te)
    sv = sv_all.values
    base = sv_all.base_values[:, 1] if np.ndim(sv_all.base_values) == 2 else sv_all.base_values
    prob = rf.predict_proba(X_te)[:, 1]

    meta = dict(_want_meta(),
                feature_names=feat_names,
                n_features=int(X.shape[1]),
                n_train=int(len(tr_idx)), n_val=int(len(va_idx)), n_test=int(len(te_idx)),
                created=time.strftime('%Y-%m-%d %H:%M:%S'))
    np.savez(CACHE, sv=sv, base=base, prob=prob,
             meta=np.array([json.dumps(meta, ensure_ascii=False)]))
    print(f'[shap_cache] 已缓存: {CACHE}（{meta["created"]}）')
    return dict(sv=sv, base=base, prob=prob, X=X, y=y, te_idx=te_idx, stay_id=d['stay_id'],
                feat_names=feat_names, meta=meta, cached=False)
