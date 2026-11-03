import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold

# =====================================================================
# split_utils.py —— 受试者级（subject-level）分组切分工具
#
# 背景：MIMIC-IV 同一 subject_id 可多次入住 ICU（多次住院视为独立样本，
#       队列 24894 例来自 21608 位患者）。若按样本随机切分，同一患者的
#       多次入院会同时落入 train 与 val/test，构成跨集泄漏（原划分
#       train∩test 含 619 位患者），高估泛化性能。
#
# 方案：以 subject_id 为分组单位做切分——同一患者的全部 stay 只出现在
#       一个集合中，并尽量保持组级分层（以每个受试者首次样本的标签为
#       代表标签进行分层切分）。
#
# 使用：所有 MIMIC 训练/分析脚本统一 import 本模块，保证划分一致。
# =====================================================================

SEED = 42


def grouped_tts(subject_id, y, test_size=0.3, val_ratio=0.5, seed=SEED):
    """受试者级分组切分 → (tr_idx, va_idx, te_idx)

    比例：train:rest = 1-test_size : test_size（默认 70/30），
          rest 内 val:test = val_ratio:(1-val_ratio)（默认 50/50）→ 70/15/15。

    分层：以每个 subject 的首次样本标签为代表标签，在受试者层面分层。
    保证：同一 subject_id 的所有样本只出现在一个集合中（零跨集泄漏）。
    """
    subj = np.asarray(subject_id)
    y = np.asarray(y)
    idx = np.arange(len(subj))

    # 每个 subject 的代表标签（取该患者首次样本的标签）
    df = pd.DataFrame({'subj': subj, 'y': y})
    rep = df.groupby('subj')['y'].first()
    subjects = rep.index.values
    labels = rep.values
    subj_pos = np.arange(len(subjects))

    # 第一层：train vs rest（受试者层面分层）
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_s, rest_s = next(sss1.split(subjects, labels))
    tr_s, rest_s = subj_pos[tr_s], subj_pos[rest_s]

    # 第二层：rest 内 val vs test（受试者层面分层）
    rest_subj = subjects[rest_s]
    rest_lab = labels[rest_s]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    va_s, te_s = next(sss2.split(rest_subj, rest_lab))
    va_s, te_s = rest_s[va_s], rest_s[te_s]

    tr_set = set(subjects[tr_s])
    va_set = set(subjects[va_s])
    te_set = set(subjects[te_s])
    assert not (tr_set & va_set) and not (tr_set & te_set) and not (va_set & te_set), \
        '受试者级切分出现跨集泄漏，请检查'

    tr_idx = idx[np.isin(subj, list(tr_set))]
    va_idx = idx[np.isin(subj, list(va_set))]
    te_idx = idx[np.isin(subj, list(te_set))]
    assert len(tr_idx) + len(va_idx) + len(te_idx) == len(idx)
    return tr_idx, va_idx, te_idx


def grouped_kfold(X, y, subject_id, n_splits=10, seed=SEED, shuffle=True):
    """受试者级分层 K 折 → 迭代 (tr_idx, va_idx)

    用 StratifiedGroupKFold：同一 subject_id 的所有样本只出现在同一折，
    且尽量保持每折标签分布一致（组级分层）。
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    return sgkf.split(X, y, groups=np.asarray(subject_id))
