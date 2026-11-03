"""s04_internal_performance.py —— 内部测试集主性能 + 聚类 CI + 校准 + DCA + 阈值操作特征

用法: python s04_internal_performance.py A|B

数据源: data/preds_<V>.npz（s03 新训练产物，含全部分区的预测）
        <ROOT>/sofa_predictions.npz, qsofa_predictions.npz（仅用于 ID 一致性核对）

输出（全部到 tables/ 与 qa/）:
  table2_model_performance.csv                 主性能表（含受试者聚类 CI）
  table2_ci_record_vs_subject.csv              记录级 vs 受试者级 CI 对照（敏感性）
  table2_ci_paired_diff_lstm_rf.csv            LSTM−RF 配对差值
  table7_delong_pvalues.csv                    模型间 DeLong 检验
  table10_calibration_internal.csv             校准指标（渐近 CI + 聚类 bootstrap CI）
  calibration_primary_patient_clustered.csv    主校准区间（受试者聚类）
  table_dca_internal.csv                       DCA 净获益曲线
  threshold_operating_characteristics_primary.csv
  threshold_interpretation.md
  qa/s04_internal_<V>.json

纪律:
  - 内部 CI 一律使用 **受试者聚类** bootstrap（主结果）；记录级仅作敏感性对照。
  - 配对差值点估计取原测试集直接计算；CI 来自同一次抽样内同时计算的差值分布。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_eval as C  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss,  # noqa: E402
                             confusion_matrix, precision_score, recall_score,
                             roc_auc_score)

HERE = Path(__file__).resolve().parent.parent
REV = HERE.parent
ROOT = REV.parent.parent

SEED = 42
N_BOOT = 1000
MODELS = ['LSTM', 'RandomForest', 'SVM', 'LogisticRegression',
          'GaussianNB', 'KNN', 'DecisionTree']
PAIRED = ('LSTM', 'RandomForest')
THRESH_GRID = np.round(np.arange(0.01, 0.99, 0.01), 4)


def fbeta(p, r, beta=3.0):
    b2 = beta ** 2
    return (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) > 0 else 0.0


def main() -> int:
    V = (sys.argv[1] if len(sys.argv) > 1 else 'A').upper()
    assert V in ('A', 'B'), '变体必须是 A 或 B'
    TAB = HERE / 'tables' / V
    QA = HERE / 'qa'
    TAB.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)

    d = np.load(HERE / 'data' / f'preds_{V}.npz', allow_pickle=False)
    split = d['split'].astype(str)
    te = split == 'test'
    y = d['y_true'].astype(int)[te]
    stay = d['stay_id'].astype(np.int64)[te]
    subj = d['subject_id'].astype(np.int64)[te]
    tinf = d['t_inf_h'].astype(float)[te]
    probs = {m: d[f'prob_{m}'].astype(float)[te] for m in MODELS}
    thr_sel = {m: float(d[f'threshold_{m}']) for m in MODELS}

    n = len(y)
    n_sub = len(np.unique(subj))
    vc = pd.Series(subj).value_counts()
    print(f'=== 变体 {V} | 测试集 {n} 条记录 / {n_sub} 位受试者 | 死亡 {int(y.sum())} ({y.mean():.3%}) ===')
    print(f'  多次住院受试者 {int((vc>1).sum())} 人 ({100*(vc>1).mean():.1f}% 的受试者)，'
          f'贡献记录 {int(vc[vc>1].sum())} 条 ({100*vc[vc>1].sum()/n:.1f}% 的记录)')

    batches_sub = C.cluster_batches(subj, N_BOOT, SEED)
    rng_rec = np.random.default_rng(1)
    batches_rec = [rng_rec.integers(0, n, n) for _ in range(N_BOOT)]

    # ---------------- 1. 主性能 + 聚类 CI ----------------
    rows, ci_rows, cal_rows = [], [], []
    boot_store = {}
    for m in MODELS:
        p = probs[m]
        auc_pt = roc_auc_score(y, p)
        ap_pt = average_precision_score(y, p)
        brier_pt = brier_score_loss(y, p)
        auc_s, ap_s = C.auc_ap_batches(y, p, batches_sub)
        auc_r, ap_r = C.auc_ap_batches(y, p, batches_rec)
        boot_store[m] = {'auc_sub': auc_s, 'ap_sub': ap_s}
        th = thr_sel[m]
        pred = (p >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({
            'Model': m, 'n': n, 'deaths': int(y.sum()),
            'AUC': round(auc_pt, 4),
            'AUC_CI_low_subject': round(np.percentile(auc_s, 2.5), 4),
            'AUC_CI_high_subject': round(np.percentile(auc_s, 97.5), 4),
            'AUPRC': round(ap_pt, 4),
            'AP_CI_low_subject': round(np.percentile(ap_s, 2.5), 4),
            'AP_CI_high_subject': round(np.percentile(ap_s, 97.5), 4),
            'Brier': round(brier_pt, 4),
            'F3_research_threshold': th,
            'Accuracy': round((tp + tn) / n, 4),
            'Precision_PPV': round(precision_score(y, pred, zero_division=0), 4),
            'Sensitivity': round(recall_score(y, pred, zero_division=0), 4),
            'Specificity': round(tn / (tn + fp), 4),
            'NPV': round(tn / (tn + fn) if (tn + fn) else np.nan, 4),
            'F1': round(fbeta(precision_score(y, pred, zero_division=0),
                              recall_score(y, pred, zero_division=0), 1.0), 4),
            'F3': round(fbeta(precision_score(y, pred, zero_division=0),
                              recall_score(y, pred, zero_division=0), 3.0), 4),
        })
        ci_rows.append({
            'Model': m,
            'AUC_rec_low': round(np.percentile(auc_r, 2.5), 4),
            'AUC_rec_high': round(np.percentile(auc_r, 97.5), 4),
            'AUC_sub_low': round(np.percentile(auc_s, 2.5), 4),
            'AUC_sub_high': round(np.percentile(auc_s, 97.5), 4),
            'AUC_rec_width': round(np.percentile(auc_r, 97.5) - np.percentile(auc_r, 2.5), 4),
            'AUC_sub_width': round(np.percentile(auc_s, 97.5) - np.percentile(auc_s, 2.5), 4),
            'AP_rec_low': round(np.percentile(ap_r, 2.5), 4),
            'AP_rec_high': round(np.percentile(ap_r, 97.5), 4),
            'AP_sub_low': round(np.percentile(ap_s, 2.5), 4),
            'AP_sub_high': round(np.percentile(ap_s, 97.5), 4),
        })

        # -------- 校准：渐近 CI（固定口径）+ 受试者聚类 bootstrap CI --------
        cal = C.calibration_stats(y, p)
        a_full, b_full = C.fit_logistic_recal(y, p)
        # (i) 固定再校准方程下的评价不确定性：方程固定，样本重抽
        br_fix, sl_fix, ic_fix, citl_fix = [], [], [], []
        # (ii) 每次重抽内重新拟合方程（总不确定性）
        br_ref = []
        br_raw_boot = []
        for idx in batches_sub:
            yy, pp = y[idx], p[idx]
            br_raw_boot.append(brier_score_loss(yy, pp))
            br_fix.append(brier_score_loss(yy, C.invlogit(a_full + b_full * C._logit(pp))))
            st = C.calibration_stats(yy, pp)
            sl_fix.append(st['slope']); ic_fix.append(st['intercept']); citl_fix.append(st['citl'])
            try:
                a2, b2 = C.fit_logistic_recal(yy, pp)
                br_ref.append(brier_score_loss(yy, C.invlogit(a2 + b2 * C._logit(pp))))
            except Exception:
                br_ref.append(np.nan)
        cal_rows.append({
            'Model': m, 'n': n, 'deaths': int(y.sum()),
            'brier_raw': round(cal['brier'], 4),
            'brier_raw_CI_low_subject': round(float(np.percentile(br_raw_boot, 2.5)), 4),
            'brier_raw_CI_high_subject': round(float(np.percentile(br_raw_boot, 97.5)), 4),
            'slope': round(cal['slope'], 4),
            'slope_CI_low_asymptotic': round(cal['slope_ci_low'], 4),
            'slope_CI_high_asymptotic': round(cal['slope_ci_high'], 4),
            'slope_CI_low_subject': round(np.nanpercentile(sl_fix, 2.5), 4),
            'slope_CI_high_subject': round(np.nanpercentile(sl_fix, 97.5), 4),
            'intercept': round(cal['intercept'], 4),
            'intercept_CI_low_asymptotic': round(cal['intercept_ci_low'], 4),
            'intercept_CI_high_asymptotic': round(cal['intercept_ci_high'], 4),
            'intercept_CI_low_subject': round(np.nanpercentile(ic_fix, 2.5), 4),
            'intercept_CI_high_subject': round(np.nanpercentile(ic_fix, 97.5), 4),
            'CITL': round(cal['citl'], 4),
            'CITL_CI_low_asymptotic': round(cal['citl_ci_low'], 4),
            'CITL_CI_high_asymptotic': round(cal['citl_ci_high'], 4),
            'CITL_CI_low_subject': round(np.nanpercentile(citl_fix, 2.5), 4),
            'CITL_CI_high_subject': round(np.nanpercentile(citl_fix, 97.5), 4),
            'recal_a': round(a_full, 4), 'recal_b': round(b_full, 4),
            'brier_logistic_fixed_equation': round(brier_score_loss(
                y, C.invlogit(a_full + b_full * C._logit(p))), 4),
            'brier_logistic_refit_CI_low_subject': round(np.nanpercentile(br_ref, 2.5), 4),
            'brier_logistic_refit_CI_high_subject': round(np.nanpercentile(br_ref, 97.5), 4),
        })

    perf = pd.DataFrame(rows).sort_values('AUC', ascending=False).reset_index(drop=True)
    perf.to_csv(TAB / 'table2_model_performance.csv', index=False)
    pd.DataFrame(ci_rows).to_csv(TAB / 'table2_ci_record_vs_subject.csv', index=False)
    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(TAB / 'table10_calibration_internal.csv', index=False)
    cal_df.to_csv(HERE / f'calibration_primary_patient_clustered_{V}.csv', index=False)

    print('\n===== 主性能（受试者聚类 95%CI）=====')
    print(perf[['Model', 'AUC', 'AUC_CI_low_subject', 'AUC_CI_high_subject',
                'AUPRC', 'AP_CI_low_subject', 'AP_CI_high_subject',
                'Brier']].to_string(index=False))
    print('\n===== 记录级 vs 受试者级 CI 宽度 =====')
    print(pd.DataFrame(ci_rows)[['Model', 'AUC_rec_width', 'AUC_sub_width']].to_string(index=False))

    # ---------------- 2. 配对差值 LSTM − RF ----------------
    pa, pb = probs[PAIRED[0]], probs[PAIRED[1]]
    pt_auc = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    pt_ap = average_precision_score(y, pa) - average_precision_score(y, pb)
    da_s, dp_s = C.paired_bootstrap(y, pa, pb, batches_sub)
    da_r, dp_r = C.paired_bootstrap(y, pa, pb, batches_rec)
    dl = C.delong_test(y, pa, pb)
    pd.DataFrame([
        {'Metric': 'AUC', 'Point_estimate': round(pt_auc, 4),
         'CI_low_subject': round(np.percentile(da_s, 2.5), 4),
         'CI_high_subject': round(np.percentile(da_s, 97.5), 4),
         'CI_low_record': round(np.percentile(da_r, 2.5), 4),
         'CI_high_record': round(np.percentile(da_r, 97.5), 4),
         'Crosses_zero_subject': bool(np.percentile(da_s, 2.5) <= 0 <= np.percentile(da_s, 97.5)),
         'DeLong_p': round(dl['p'], 4)},
        {'Metric': 'AP', 'Point_estimate': round(pt_ap, 4),
         'CI_low_subject': round(np.percentile(dp_s, 2.5), 4),
         'CI_high_subject': round(np.percentile(dp_s, 97.5), 4),
         'CI_low_record': round(np.percentile(dp_r, 2.5), 4),
         'CI_high_record': round(np.percentile(dp_r, 97.5), 4),
         'Crosses_zero_subject': bool(np.percentile(dp_s, 2.5) <= 0 <= np.percentile(dp_s, 97.5)),
         'DeLong_p': np.nan},
    ]).to_csv(TAB / 'table2_ci_paired_diff_lstm_rf.csv', index=False)
    print(f'\n===== 配对差值 {PAIRED[0]} − {PAIRED[1]} =====')
    print(f'  点估计（原测试集）: AUC Δ={pt_auc:.4f}  AP Δ={pt_ap:.4f}')
    print(f'  AUC Δ 受试者级 95%CI: {np.percentile(da_s,2.5):.4f}–{np.percentile(da_s,97.5):.4f}'
          f' | 是否跨 0: {np.percentile(da_s,2.5) <= 0 <= np.percentile(da_s,97.5)}')
    print(f'  AP  Δ 受试者级 95%CI: {np.percentile(dp_s,2.5):.4f}–{np.percentile(dp_s,97.5):.4f}')
    print(f'  DeLong: Δ={dl["auc_diff"]:.4f} z={dl["z"]:.3f} p={dl["p"]:.4f}')

    # ---------------- 3. 全模型间 DeLong ----------------
    delong_rows = []
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            r = C.delong_test(y, probs[a], probs[b])
            delong_rows.append({'Model_A': a, 'Model_B': b,
                                'AUC_A': round(r['auc_a'], 4), 'AUC_B': round(r['auc_b'], 4),
                                'AUC_diff': round(r['auc_diff'], 4),
                                'z': round(r['z'], 4), 'p_uncorrected': round(r['p'], 4)})
    dl_df = pd.DataFrame(delong_rows)
    dl_df['p_bonferroni'] = (dl_df.p_uncorrected * len(dl_df)).clip(upper=1).round(4)
    dl_df.to_csv(TAB / 'table7_delong_pvalues.csv', index=False)
    print(f'\nDeLong 检验 {len(dl_df)} 组（已保存，含 Bonferroni 校正 p）')

    # ---------------- 4. DCA + 临床影响曲线 ----------------
    dca_rows = []
    for m in MODELS:
        nb = C.dca_net_benefit(y, probs[m], THRESH_GRID)
        fl, hits = C.clinical_impact(y, probs[m], THRESH_GRID)
        for t, v, f, h in zip(THRESH_GRID, nb, fl, hits):
            dca_rows.append({'Model': m, 'threshold_prob': t, 'net_benefit': round(v, 5),
                             'flagged_n': f, 'deaths_among_flagged': h})
    for t, v in zip(THRESH_GRID, C.dca_treat_all(y, THRESH_GRID)):
        dca_rows.append({'Model': 'TreatAll', 'threshold_prob': t,
                         'net_benefit': round(v, 5), 'flagged_n': n, 'deaths_among_flagged': int(y.sum())})
    for t in THRESH_GRID:
        dca_rows.append({'Model': 'TreatNone', 'threshold_prob': t,
                         'net_benefit': 0.0, 'flagged_n': 0, 'deaths_among_flagged': 0})
    pd.DataFrame(dca_rows).to_csv(TAB / 'table_dca_internal.csv', index=False)
    # 关键阈值点的净获益（正文常用 0.10–0.30 区间）
    key_t = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    kb = {m: C.dca_net_benefit(y, probs[m], np.array(key_t)) for m in MODELS}
    kb['TreatAll'] = C.dca_treat_all(y, np.array(key_t))
    kb['TreatNone'] = np.zeros(len(key_t))
    print('\n===== DCA 净获益（选定阈值）=====')
    print(pd.DataFrame(kb, index=[f'{t:.2f}' for t in key_t]).round(4).to_string())

    # ---------------- 5. 阈值操作特征（研究性操作点）----------------
    th_rows = []
    for m in MODELS:
        p = probs[m]
        th = thr_sel[m]
        pred = (p >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        nb_at = float(C.dca_net_benefit(y, p, np.array([th]))[0])
        nbref = float(C.dca_treat_all(y, np.array([th]))[0])
        th_rows.append({
            'Model': m, 'dataset': f'internal_cohort_{V}', 'threshold': th,
            'threshold_origin': 'validation-set F3-optimal (research operating point)',
            'flagged_n': int(tp + fp), 'flagged_percent': round(100 * (tp + fp) / n, 2),
            'deaths_captured': int(tp), 'death_recall_sensitivity': round(tp / (tp + fn), 4),
            'specificity': round(tn / (tn + fp), 4),
            'PPV': round(tp / (tp + fp) if (tp + fp) else np.nan, 4),
            'NPV': round(tn / (tn + fn) if (tn + fn) else np.nan, 4),
            'false_positive_n': int(fp),
            'false_positive_percent_of_all': round(100 * fp / n, 2),
            'one_minus_PPV': round(1 - tp / (tp + fp) if (tp + fp) else np.nan, 4),
            'brier_at_threshold': round(brier_score_loss(y, p), 4),
            'net_benefit_at_threshold': round(nb_at, 5),
            'net_benefit_treat_all_at_threshold': round(nbref, 5),
            'net_benefit_gain_vs_treat_all': round(nb_at - nbref, 5),
        })
    th_df = pd.DataFrame(th_rows)
    th_df.to_csv(TAB / 'threshold_operating_characteristics_primary.csv', index=False)
    print('\n===== 阈值操作特征（F3 研究性操作点）=====')
    print(th_df[['Model', 'threshold', 'flagged_n', 'flagged_percent', 'deaths_captured',
                 'death_recall_sensitivity', 'specificity', 'PPV', 'one_minus_PPV',
                 'net_benefit_at_threshold', 'net_benefit_gain_vs_treat_all']].to_string(index=False))

    # ---------------- 6. SOFA / qSOFA ID 一致性核对 ----------------
    idchk = {'checked': True}
    for tag, fn in [('sofa', 'sofa_predictions.npz'), ('qsofa', 'qsofa_predictions.npz')]:
        f = ROOT / fn
        if not f.exists():
            idchk[tag] = 'missing file'
            continue
        s = np.load(f)
        s_stay = s['stay_id'].astype(np.int64)
        s_y = s['y_true'].astype(int)
        # cohort B 的测试集是 cohort A 测试集的子集，按 stay_id 取交集核对
        m = np.isin(s_stay, stay)
        ok_id = bool(np.array_equal(np.sort(s_stay[m]), np.sort(stay)))
        y_map = dict(zip(s_stay.tolist(), s_y.tolist()))
        ok_y = bool(all(y_map[int(s)] == int(yy) for s, yy in zip(stay, y) if int(s) in y_map))
        idchk[tag] = {'n_file': int(len(s_stay)), 'n_matched': int(m.sum()),
                      'stay_id_identical': ok_id, 'y_true_identical': ok_y}
        sc = s['sofa'] if tag == 'sofa' else s['qsofa6']
        sc = np.asarray(sc, dtype=float)[m]
        idchk[tag]['auc'] = round(float(roc_auc_score(y, sc)), 4)
        sc_boot = [roc_auc_score(y[i], sc[i]) for i in batches_sub]
        idchk[tag]['auc_CI_low_subject'] = round(float(np.percentile(sc_boot, 2.5)), 4)
        idchk[tag]['auc_CI_high_subject'] = round(float(np.percentile(sc_boot, 97.5)), 4)
    print('\n===== SOFA/qSOFA ID 一致性 =====')
    print(json.dumps(idchk, ensure_ascii=False, indent=2))

    # ---------------- 7. 阈值含义说明 ----------------
    (HERE / f'threshold_interpretation_{V}.md').write_text(f"""# 阈值解释（变体 {V}）

## 阈值来源
所有阈值均为 **验证集上 F3 最优化得到的"研究性操作点"（research operating point）**。
F3 是召回优先的数学加权评分（β=3），它 **不代表** 经过临床验证的漏诊/误报成本比。
本研究没有定义具体干预动作、复核流程或可接受工作负荷。

## 各模型操作特征
```
{th_df[['Model', 'threshold', 'flagged_n', 'flagged_percent', 'deaths_captured',
        'death_recall_sensitivity', 'specificity', 'PPV', 'one_minus_PPV',
        'false_positive_percent_of_all', 'net_benefit_at_threshold',
        'net_benefit_gain_vs_treat_all']].to_string(index=False)}
```

## 必须遵守的表述
- "高危人数"与"死亡捕获率" **不等于临床获益**；本表不构成干预效果证据。
- 表中 `one_minus_PPV` 是 **1−PPV**，不是"假阳性率"；假阳性率的分母是全部记录
  （见 `false_positive_percent_of_all`）。二者不可混称。
- DCA 的阈值概率是"愿意为避免一次死亡而接受的治疗例数"的隐含成本比，
  与 F3 操作点不是同一概念；不可互相替代。
- 本队列为 **入 ICU 后 48 小时地标** 人群，不是入院即刻筛查人群。
""", encoding='utf-8')

    # ---------------- 8. QA 汇总 ----------------
    qa = {
        'variant': V, 'seed': SEED, 'n_boot': N_BOOT,
        'bootstrap_unit_internal': 'subject_id (受试者聚类，主结果)',
        'bootstrap_unit_sensitivity': 'record (记录级，仅对照)',
        'test_n': n, 'test_subjects': int(n_sub), 'test_deaths': int(y.sum()),
        'test_prevalence': round(float(y.mean()), 6),
        'models_trained_anew': MODELS,
        'source_pred_file': f'data/preds_{V}.npz',
        'sofa_qsofa_id_check': idchk,
        'paired_LSTM_minus_RF': {
            'AUC_point_estimate': round(float(pt_auc), 4),
            'AUC_CI_subject': [round(float(np.percentile(da_s, 2.5)), 4),
                               round(float(np.percentile(da_s, 97.5)), 4)],
            'AUC_CI_crosses_zero': bool(np.percentile(da_s, 2.5) <= 0 <= np.percentile(da_s, 97.5)),
            'AP_point_estimate': round(float(pt_ap), 4),
            'AP_CI_subject': [round(float(np.percentile(dp_s, 2.5)), 4),
                              round(float(np.percentile(dp_s, 97.5)), 4)],
            'DeLong_p': round(float(dl['p']), 4),
        },
        'calibration_ci_definition': (
            'slope/intercept/CITL 的 subject 聚类 bootstrap 区间为"每次重抽内重新拟合'
            '逻辑校准方程"的总不确定性；brier_logistic_fixed_equation 为固定方程应用结果，'
            '其区间定义见 brier_logistic_refit_* 字段'),
        'threshold_origin': 'validation-set F3-optimal research operating point',
        'no_old_predictions_used': True,
    }
    (QA / f's04_internal_{V}.json').write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez(HERE / 'data' / f'boot_internal_{V}.npz',
             **{f'auc_sub_{m}': boot_store[m]['auc_sub'] for m in MODELS},
             **{f'ap_sub_{m}': boot_store[m]['ap_sub'] for m in MODELS})
    print(f'\n-> tables/ 与 qa/s04_internal_{V}.json 已写出')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
