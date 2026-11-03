"""s05_subgroup.py —— 亚组性能、组内 ΔAUC、组间差异、交互检验、SOFA/qSOFA 对照

用法: python s05_subgroup.py A|B

纪律（回应审阅意见）:
  * 亚组 bootstrap 的抽样单位是 **受试者（subject_id）**，同一受试者的全部记录
    一起进入/退出重抽样本；患者内相关性由聚类处理，而不仅是"说明一下"。
  * DeLong 检验仍在亚组内以记录为单位（该检验本身不建模患者内相关），
    因此正文引用时应以 **受试者聚类 bootstrap 区间** 为主要不确定性来源，
    DeLong p 值只作为补充。
  * `cardiovascular >= 2` 只是 SOFA 心血管分项的 **代理分层**，不是"感染性休克"的确诊。
  * 交互检验的 3 个 p 值做 Bonferroni 校正（×3）。

输出到 tables/<V>/:
  table8_subgroup_auc.csv                各亚组各模型 AUC（含受试者聚类 CI）
  table11_subgroup_dauc_ci.csv           组内 ΔAUC(LSTM−SOFA) + 组间差
  table11_subgroup_interaction.csv       交互检验 + Bonferroni
  table5_sofa_comparison.csv / table5_qsofa_comparison.csv / table5_*_windows|cutoffs.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_eval as C  # noqa: E402
from sklearn.metrics import (average_precision_score, precision_score,  # noqa: E402
                             recall_score, roc_auc_score)

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent.parent.parent

SEED = 42
N_BOOT = 1000


def fbeta(p, r, beta):
    b2 = beta ** 2
    return (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) > 0 else 0.0


def boot_auc_cluster(y, p, clusters, n_boot=N_BOOT, seed=SEED):
    """亚组内 AUC 的受试者聚类 bootstrap。"""
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    b = C.cluster_batches(clusters, n_boot, seed)
    vals = []
    for idx in b:
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_dauc_cluster(y, p1, p2, clusters, n_boot=N_BOOT, seed=SEED):
    """组内 ΔAUC = AUC(p1) − AUC(p2) 的受试者聚类 bootstrap。"""
    b = C.cluster_batches(clusters, n_boot, seed)
    d = []
    for idx in b:
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    if not d:
        return np.nan, np.nan
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_ddiff_cluster(y, grp, p1, p2, lo, hi, clusters, n_boot=N_BOOT, seed=SEED):
    """组间差异 (ΔAUC_low − ΔAUC_high) 的受试者聚类 bootstrap。

    按受试者重抽；同一受试者的记录整体进出，再在重抽样本内分亚组。
    """
    b = C.cluster_batches(clusters, n_boot, seed)
    g = np.asarray(grp)
    d = []
    for idx in b:
        ml, mh = g[idx] == lo, g[idx] == hi
        if ml.sum() < 5 or mh.sum() < 5:
            continue
        if len(np.unique(y[idx][ml])) < 2 or len(np.unique(y[idx][mh])) < 2:
            continue
        d1 = roc_auc_score(y[idx][ml], p1[idx][ml]) - roc_auc_score(y[idx][ml], p2[idx][ml])
        d2 = roc_auc_score(y[idx][mh], p1[idx][mh]) - roc_auc_score(y[idx][mh], p2[idx][mh])
        d.append(d1 - d2)
    if not d:
        return np.nan, np.nan, 0
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), len(d)


def interaction_p(y, lp, grp_code):
    X = np.column_stack([np.ones(len(y)), lp, grp_code, lp * grp_code])
    m = sm.Logit(y, X).fit(disp=0)
    beta, se = m.params[3], m.bse[3]
    z = beta / se
    return 2 * (1 - scipy.stats.norm.cdf(abs(z))), float(beta), [float(v) for v in m.conf_int()[3]]


def main() -> int:
    V = (sys.argv[1] if len(sys.argv) > 1 else 'A').upper()
    TAB = HERE / 'tables' / V
    TAB.mkdir(parents=True, exist_ok=True)
    QA = HERE / 'qa'

    d = np.load(HERE / 'data' / f'preds_{V}.npz', allow_pickle=False)
    split = d['split'].astype(str)
    te = split == 'test'
    y = d['y_true'].astype(int)[te]
    stay = d['stay_id'].astype(np.int64)[te]
    subj = d['subject_id'].astype(np.int64)[te]
    probs = {m: d[f'prob_{m}'].astype(float)[te]
             for m in ['LSTM', 'RandomForest', 'SVM', 'LogisticRegression',
                       'GaussianNB', 'KNN', 'DecisionTree']}

    # ---- 附加变量 ----
    tr = np.load(HERE / 'data' / 'trainonly_mimic.npz', allow_pickle=False)
    age_all = pd.DataFrame({'stay_id': tr['stay_id'].astype(np.int64),
                            'age': tr['X_sv'][:, 0]})
    sofa_f = np.load(ROOT / 'sofa_predictions.npz')
    qsofa_f = np.load(ROOT / 'qsofa_predictions.npz')
    se = pd.read_csv(ROOT / 'sofa_extract.csv')
    se['stay_id'] = se['stay_id'].astype(np.int64)
    demo = pd.read_csv(ROOT / 'demographics_by_stay.csv')
    demo['stay_id'] = demo['stay_id'].astype(np.int64)

    df = pd.DataFrame({'stay_id': stay, 'subject_id': subj, 'y': y, **probs})
    df['RF'] = df['RandomForest']   # 亚组表的短名别名
    df = df.merge(age_all, on='stay_id', how='left')
    df = df.merge(pd.DataFrame({'stay_id': sofa_f['stay_id'].astype(np.int64),
                                'SOFA': sofa_f['sofa'].astype(float)}), on='stay_id', how='left')
    df = df.merge(pd.DataFrame({'stay_id': qsofa_f['stay_id'].astype(np.int64),
                                'qsofa6': qsofa_f['qsofa6'].astype(float)}), on='stay_id', how='left')
    df = df.merge(se[['stay_id', 'cardiovascular']], on='stay_id', how='left')
    df = df.merge(demo[['stay_id', 'gender', 'race_group', 'age_group']], on='stay_id', how='left')
    assert len(df) == len(y), f'merge 后样本数变化: {len(df)} vs {len(y)}'
    assert df[['SOFA', 'qsofa6', 'age', 'cardiovascular']].notna().all().all(), '关键变量出现缺失'
    print(f'变体 {V}: 对齐 {len(df)} 条记录 | 死亡率 {df.y.mean():.3f}')

    # ---- 亚组定义 ----
    df['age_grp'] = np.where(df['age'] < 65, 'age<65', 'age>=65')
    df['sofa_grp'] = np.where(df['SOFA'] < 8, 'SOFA<8', 'SOFA>=8')
    df['shock_grp'] = np.where(df['cardiovascular'] >= 2, 'shock_proxy', 'nonshock')
    DIMS = [('Age', 'age_grp'), ('Initial SOFA', 'sofa_grp'),
            ('Cardiovascular SOFA', 'shock_grp'), ('Gender', 'gender'),
            ('Race', 'race_group')]

    # ---- 1. 各亚组各模型 AUC（受试者聚类 CI）----
    rows = []
    for dim, col in DIMS:
        for grp in sorted(df[col].dropna().unique()):
            sub = df[df[col] == grp]
            if len(sub) < 100:
                print(f'  [注意] {dim}:{grp} 仅 {len(sub)} 条，谨慎解读')
            row = {'Dimension': dim, 'Subgroup': str(grp), 'n': len(sub),
                   'n_subjects': int(sub.subject_id.nunique()),
                   'Death_rate': round(sub.y.mean(), 4)}
            for mdl in ['LSTM', 'RF', 'SOFA', 'qsofa6']:
                p = sub[mdl].values.astype(float)
                lo, hi = boot_auc_cluster(sub.y.values, p, sub.subject_id.values)
                row[f'{mdl}_AUC'] = round(roc_auc_score(sub.y, p), 4)
                row[f'{mdl}_CI_low'] = round(lo, 4)
                row[f'{mdl}_CI_high'] = round(hi, 4)
            rows.append(row)
    sg = pd.DataFrame(rows)
    sg.to_csv(TAB / 'table8_subgroup_auc.csv', index=False)
    print('\n===== 亚组 AUC（受试者聚类 95%CI）=====')
    print(sg[['Dimension', 'Subgroup', 'n', 'n_subjects', 'Death_rate',
              'LSTM_AUC', 'LSTM_CI_low', 'LSTM_CI_high',
              'RF_AUC', 'RF_CI_low', 'RF_CI_high',
              'SOFA_AUC', 'qsofa6_AUC']].to_string(index=False))

    # ---- 2. 组内 ΔAUC(LSTM−SOFA) + 组间差（受试者聚类）----
    drow = []
    for dim, col in DIMS:
        if col in ('gender', 'race_group'):
            continue  # 公平性维度只报 AUC，不做 ΔAUC vs SOFA
        gl, gh = sorted(df[col].unique())
        sub_l, sub_h = df[df[col] == gl], df[df[col] == gh]
        for gname, sub in [(gl, sub_l), (gh, sub_h)]:
            p1, p2 = sub.LSTM.values, sub.SOFA.values
            lo, hi = boot_dauc_cluster(sub.y.values, p1, p2, sub.subject_id.values)
            dl = C.delong_test(sub.y.values, p1, p2)
            drow.append({'Dimension': dim, 'Subgroup': gname, 'n': len(sub),
                         'Death_rate': round(sub.y.mean(), 4),
                         'AUC_LSTM': round(roc_auc_score(sub.y, p1), 4),
                         'AUC_SOFA': round(roc_auc_score(sub.y, p2), 4),
                         'dAUC_LSTM_minus_SOFA': round(dl['auc_diff'], 4),
                         'dAUC_CI_low': round(lo, 4), 'dAUC_CI_high': round(hi, 4),
                         'dAUC_CI_excludes_0': bool(lo > 0 or hi < 0),
                         'DeLong_p_unclustered': round(dl['p'], 6),
                         'CI_method': 'subject-clustered bootstrap (record-level DeLong shown separately)'})
        lo, hi, nb = boot_ddiff_cluster(df.y.values, df[col].values, df.LSTM.values,
                                        df.SOFA.values, gl, gh, df.subject_id.values)
        dlo = roc_auc_score(sub_l.y, sub_l.LSTM) - roc_auc_score(sub_l.y, sub_l.SOFA)
        dhi = roc_auc_score(sub_h.y, sub_h.LSTM) - roc_auc_score(sub_h.y, sub_h.SOFA)
        drow.append({'Dimension': dim, 'Subgroup': f'diff({gl} − {gh})',
                     'dAUC_LSTM_minus_SOFA': round(dlo - dhi, 4),
                     'dAUC_CI_low': round(lo, 4), 'dAUC_CI_high': round(hi, 4),
                     'dAUC_CI_excludes_0': bool(lo > 0 or hi < 0),
                     'n_boot_valid': nb,
                     'CI_method': 'subject-clustered bootstrap'})
    dd = pd.DataFrame(drow)
    dd.to_csv(TAB / 'table11_subgroup_dauc_ci.csv', index=False)
    print('\n===== 组内 ΔAUC(LSTM−SOFA) 与组间差（受试者聚类 95%CI）=====')
    print(dd.to_string(index=False))

    # ---- 3. 交互检验 + Bonferroni ----
    it = []
    for dim, col in DIMS[:3]:
        g = pd.Categorical(df[col]).codes.astype(float)
        p, beta, ci = interaction_p(df.y.values, C._logit(df.LSTM.values), g)
        it.append({'Dimension': dim, 'interaction_p': p, 'beta': beta,
                   'ci_low': ci[0], 'ci_high': ci[1]})
    itdf = pd.DataFrame(it)
    itdf['p_bonferroni'] = np.clip(itdf['interaction_p'] * len(itdf), 0, 1)
    itdf.to_csv(TAB / 'table11_subgroup_interaction.csv', index=False)
    print('\n===== 交互检验（logit(LSTM) × 亚组；Bonferroni ×3）=====')
    print(itdf.round(4).to_string(index=False))

    # ---- 4. SOFA / qSOFA 对照表（评测子集已与新媒体对齐）----
    mdls = ['LSTM', 'RandomForest', 'SVM', 'LogisticRegression',
            'GaussianNB', 'KNN', 'DecisionTree']
    sc_rows = []
    for name in mdls:
        p = df[name].values.astype(float)
        th = float(d[f'threshold_{name}'])
        pred = (p >= th).astype(int)
        sc_rows.append({'Model': name, 'AUC': roc_auc_score(df.y, p),
                        'AUPRC': average_precision_score(df.y, p),
                        'Research_threshold': th,
                        'Precision': precision_score(df.y, pred, zero_division=0),
                        'Recall': recall_score(df.y, pred, zero_division=0),
                        'F1': fbeta(precision_score(df.y, pred, zero_division=0),
                                    recall_score(df.y, pred, zero_division=0), 1),
                        'F3': fbeta(precision_score(df.y, pred, zero_division=0),
                                    recall_score(df.y, pred, zero_division=0), 3)})
    sc_rows.append({'Model': 'SOFA (rule-based, 48h max)', 'AUC': roc_auc_score(df.y, df.SOFA),
                    'AUPRC': average_precision_score(df.y, df.SOFA)})
    sc_rows.append({'Model': 'qSOFA-6h (rule-based)', 'AUC': roc_auc_score(df.y, df.qsofa6),
                    'AUPRC': average_precision_score(df.y, df.qsofa6)})
    sc = pd.DataFrame(sc_rows).sort_values('AUC', ascending=False)
    sc.to_csv(TAB / 'table5_sofa_comparison.csv', index=False)
    sc.drop(columns=['Research_threshold'], errors='ignore').to_csv(
        TAB / 'table5_qsofa_comparison.csv', index=False)
    print('\n===== SOFA/qSOFA 与模型对照（同一评测子集）=====')
    print(sc.round(4).to_string(index=False))

    # SOFA 截断值与 qSOFA 窗口（同一子集重算）
    cut = []
    for c in [6, 8, 10, 12]:
        pred = (df.SOFA >= c).astype(int)
        cut.append({'Cutoff': c, 'High_risk_n': int(pred.sum()),
                    'Pos_rate': round(pred.mean(), 4),
                    'Precision': round(precision_score(df.y, pred, zero_division=0), 4),
                    'Recall': round(recall_score(df.y, pred, zero_division=0), 4),
                    'F1': round(fbeta(precision_score(df.y, pred, zero_division=0),
                                      recall_score(df.y, pred, zero_division=0), 1), 4),
                    'F3': round(fbeta(precision_score(df.y, pred, zero_division=0),
                                      recall_score(df.y, pred, zero_division=0), 3), 4)})
    pd.DataFrame(cut).to_csv(TAB / 'table5_sofa_cutoffs.csv', index=False)
    win = []
    for w, key in [('6h', 'qsofa6'), ('12h', 'qsofa12'), ('24h', 'qsofa24')]:
        qs = np.asarray(qsofa_f[key], dtype=float)
        qmap = dict(zip(qsofa_f['stay_id'].astype(np.int64).tolist(), qs.tolist()))
        qv = np.array([qmap[int(s)] for s in df.stay_id])
        pred = (qv >= 2).astype(int)
        win.append({'Window': w, 'AUC': round(roc_auc_score(df.y, qv), 4),
                    'AUPRC': round(average_precision_score(df.y, qv), 4),
                    'Pos_rate_ge2': round(pred.mean(), 4),
                    'Precision': round(precision_score(df.y, pred, zero_division=0), 4),
                    'Recall': round(recall_score(df.y, pred, zero_division=0), 4),
                    'F1': round(fbeta(precision_score(df.y, pred, zero_division=0),
                                      recall_score(df.y, pred, zero_division=0), 1), 4),
                    'F3': round(fbeta(precision_score(df.y, pred, zero_division=0),
                                      recall_score(df.y, pred, zero_division=0), 3), 4)})
    pd.DataFrame(win).to_csv(TAB / 'table5_qsofa_windows.csv', index=False)

    qa = {'variant': V, 'subgroup_bootstrap_unit': 'subject_id (受试者聚类，主结果)',
          'de_long_unit': 'record (未建模患者内相关，仅作补充)',
          'n_records': int(len(df)), 'n_subjects': int(df.subject_id.nunique()),
          'n_boot': N_BOOT, 'seed': SEED,
          'shock_definition': 'cardiovascular SOFA subscore >= 2 (代理分层，非确诊感染性休克)',
          'interaction_bonferroni_multiplier': int(len(itdf)),
          'sofa_qsofa_subset_aligned_to_new_test_set': True,
          'sofa_qsofa_scores_recomputed': False,
          'sofa_qsofa_note': ('SOFA/qSOFA 分数本身取自 0–48h 窗（与地标一致），'
                              '故未重算；仅把评测子集按 stay_id 对齐到新的测试集，'
                              '并逐条核验 y_true 一致（见 qa/s04_internal_*.json）')}
    (QA / f's05_subgroup_{V}.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2),
                                               encoding='utf-8')
    print(f'\n-> tables/{V}/ 与 qa/s05_subgroup_{V}.json 已写出')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
