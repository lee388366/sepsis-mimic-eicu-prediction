#!/usr/bin/env python3
"""s15_old_vs_new_prediction_agreement.py —— 新旧模型预测在内部测试集上的一致性

用途
----
讨论段需要说明「训练集独立插补 + 重新训练」相对旧分析的差别究竟有多大。
本脚本把旧预测（项目根 all_predictions/baseline_predictions/sequence_predictions）
与新主分析预测（data/preds_A.npz 的 test 子集）按 stay_id 对齐，逐模型给出
Pearson r、平均绝对差与最大绝对差，并把结果落盘供稿件数字溯源。

输出
----
tables/A/table_old_vs_new_pred_agreement.csv
qa/old_vs_new_prediction_agreement.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent.parent.parent          # 项目根（旧预测所在）
TAB = HERE / 'tables' / 'A'
QA = HERE / 'qa'
MODELS = ['LSTM', 'RandomForest', 'SVM', 'LogisticRegression', 'GaussianNB', 'KNN', 'DecisionTree']


def main() -> int:
    new = np.load(HERE / 'data' / 'preds_A.npz', allow_pickle=False)
    te = new['split'].astype(str) == 'test'
    sid_new = new['stay_id'].astype(np.int64)[te]
    y_new = new['y_true'].astype(int)[te]

    old_p: dict[str, np.ndarray] = {}
    for f, keys in [('baseline_predictions.npz', None), ('sequence_predictions.npz', None),
                    ('all_predictions.npz', None)]:
        p = ROOT / f
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        sid_old = d['stay_id'].astype(np.int64)
        idx = {int(s): i for i, s in enumerate(sid_old)}
        for k in d.files:
            if k in ('y_true', 'stay_id') or k in old_p:
                continue
            arr = np.asarray(d[k], dtype=float)
            if arr.ndim != 1 or len(arr) != len(sid_old):
                continue
            old_p[k] = np.array([arr[idx[int(s)]] for s in sid_new])

    rows = []
    for m in MODELS:
        if m not in old_p:
            continue
        a, b = old_p[m], np.asarray(new[f'prob_{m}'], float)[te]
        r = float(np.corrcoef(a, b)[0, 1])
        rows.append({'model': m, 'pearson_r': r, 'mean_abs_diff': float(np.mean(np.abs(a - b))),
                     'max_abs_diff': float(np.max(np.abs(a - b))),
                     'auc_old': float(np.mean(a)), 'n_test': int(len(a))})
    df = pd.DataFrame(rows).sort_values('pearson_r')
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / 'table_old_vs_new_pred_agreement.csv', index=False)
    QA.mkdir(parents=True, exist_ok=True)
    (QA / 'old_vs_new_prediction_agreement.json').write_text(json.dumps({
        'note': ('旧预测为 pre-split static imputation 分析的模型输出；新预测为 '
                 'train-only 预处理 + 重新训练的模型输出。两者在内部测试集同一批 '
                 'stay_id 上对齐比较，仅用于说明差异大小，不作为性能证据。'),
        'n_test': int(len(sid_new)),
        'models': rows,
        'min_pearson_r_excluding_LSTM': float(
            df[df.model != 'LSTM']['pearson_r'].min()) if (df.model != 'LSTM').any() else None,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(df.to_string(index=False))
    print(f'\n-> {TAB/"table_old_vs_new_pred_agreement.csv"}\n-> {QA/"old_vs_new_prediction_agreement.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
