#!/usr/bin/env python3
"""s10_figures.py —— 用新预测产物重生成论文图（输出到隔离目录，绝不覆盖旧图）

用法: python s10_figures.py A|B

输出目录: figures_trainonly/<V>/
  roc_curves.png                  内部测试集 ROC（7 模型）
  roc_curves_external.png         eICU 跨库 ROC（7 模型 + SOFA + qSOFA）
  pr_curves.png                   内部 PR
  calibration_curves_internal.png 内部校准曲线（7 模型）
  calibration_curves_external.png eICU 校准曲线（7 模型）
  calibration_primary.png         内部主校准图（含受试者聚类 CI 误差条）
  calibration_sensitivity.png     eICU 校准图（患者级划半、再校准前后）
  dca_compare.png / dca_compare_external.png
  clinical_impact_lstm.png / clinical_impact_lstm_external.png
  model_performance_bar.png       AUC + 受试者聚类 95%CI
  confusion_matrix_rf.png         RF 在 F3 研究性操作点的混淆矩阵
  lstm_attention.png              48 小时平均注意力
  attention_faithfulness.png      遮挡忠实性
  shap_summary_bar.png / shap_summary.png   RF SHAP

注意：脚本只读取新产物（preds_<V>.npz / eicu_probs_<V>.npz / tables/<V>/*.csv /
      qa/s08_explain_<V>.json / data/shap_values_<V>.npz），不读取任何旧预测。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent.parent.parent

MODELS = ['LSTM', 'RandomForest', 'SVM', 'LogisticRegression',
          'GaussianNB', 'KNN', 'DecisionTree']
COLORS = {'LSTM': '#D62728', 'RandomForest': '#1F77B4', 'SVM': '#2CA02C',
          'LogisticRegression': '#9467BD', 'GaussianNB': '#FF7F0E',
          'KNN': '#8C564B', 'DecisionTree': '#17BECF',
          'SOFA': '#7F7F7F', 'qSOFA-6h': '#BCBD22'}
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
                     'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})


# ---- 印刷档位（正文 Figure 2/3/4 的 6 个面板整页宽排版）-------------------
# 换算：印刷字号(pt) = 基准字号 × 显示宽度(in) ÷ 图像自然宽度(in)
#   原并排排版：11pt × 3.18 / 5.71 ≈ 6.1pt（偏小）
#   现整页宽：  10pt × 6.50 / 7.0  ≈ 9.3pt（接近正文 10pt）
# figsize 加宽同时放大约 20% 的相对留白，故相对排布比原图更宽松，不会更拥挤。
PRINT_RC = {'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
            'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9}
PRINT_FIG = {'roc': (7.5, 6.4), 'calib': (7.5, 6.4), 'dca': (7.5, 6.0)}
# 正文 3 张图（Figure 2/3/4）采用 A|B 双面板合图，按最终印刷尺寸设计：
# figsize 7.6in 宽 + 基准字号 10pt，整页宽 6.5in 显示时印刷约 9.3pt；
# 两面板共用图例置于图下方，避免图例遮挡曲线，且图与图注同页。
PRINT_FIG_2PANEL = (7.6, 3.9)


class _PrintProfile:
    """with print_profile(): 使用整页宽图档位。"""

    def __enter__(self):
        self._old = {k: plt.rcParams[k] for k in PRINT_RC}
        plt.rcParams.update(PRINT_RC)
        return self

    def __exit__(self, *exc):
        plt.rcParams.update(self._old)
        return False


def print_profile() -> _PrintProfile:
    return _PrintProfile()


def dca(y, p, ts):
    n = len(y)
    nb = np.array([((p >= t) & (y == 1)).sum() / n
                   - ((p >= t) & (y == 0)).sum() / n * (t / (1 - t)) for t in ts])
    prev = y.mean()
    return nb, prev - (1 - prev) * ts / (1 - ts)


def f3(v) -> str:
    """与 tables/*.csv（pandas/numpy 格式化）一致的 3 位小数显示。

    注意：np.float64.__format__ 与 Python float 在 .xxx5 边界上不同
    （例：0.793535 → numpy '0.793'，float '0.794'）。图内标注必须与
    表格显示一致，故统一走 np.float64。
    """
    return f'{np.float64(v):.3f}'

def main() -> int:
    V = (sys.argv[1] if len(sys.argv) > 1 else 'A').upper()
    FIG = HERE / 'figures_trainonly' / V
    FIG.mkdir(parents=True, exist_ok=True)
    TAB = HERE / 'tables' / V

    d = np.load(HERE / 'data' / f'preds_{V}.npz', allow_pickle=False)
    te = d['split'].astype(str) == 'test'
    y = d['y_true'].astype(int)[te]
    P = {m: d[f'prob_{m}'].astype(float)[te] for m in MODELS}
    thr = {m: float(d[f'threshold_{m}']) for m in MODELS}
    order = sorted(MODELS, key=lambda m: -roc_auc_score(y, P[m]))

    e = np.load(HERE / 'data' / f'eicu_probs_{V}.npz', allow_pickle=False)
    ye = e['y_true'].astype(int)
    PE = {m: e[f'prob_{m}'].astype(float) for m in MODELS}
    PE['SOFA'] = e['sofa'].astype(float)          # 规则评分，不是概率
    PE['qSOFA-6h'] = e['qsofa6'].astype(float)    # 规则评分，不是概率
    Pe = {m: e[f'prob_logrecal_{m}'].astype(float) for m in MODELS}
    order_e = sorted(MODELS + ['SOFA', 'qSOFA-6h'], key=lambda m: -roc_auc_score(ye, PE[m]))
    order_e_models = [m for m in order_e if m in MODELS]
    # 患者级再校准：方程由拟合半估计，图上只用"患者不重叠的评价半"
    eval_mask = (e['recal_eval_mask'].astype(bool)
                 if 'recal_eval_mask' in e.files else np.ones(len(ye), dtype=bool))
    rec_mask = (e['record_eval_mask'].astype(bool)
                if 'record_eval_mask' in e.files else np.ones(len(ye), dtype=bool))

    # ---------------- ROC（内部）----------------
    # 内部 ROC 同时画 SOFA / qSOFA（与 Table 2 一致；规则评分用虚线区分）
    si = np.load(ROOT / 'sofa_predictions.npz')
    qi = np.load(ROOT / 'qsofa_predictions.npz')
    stay_te = d['stay_id'][te].astype(np.int64)
    si_m = np.isin(si['stay_id'].astype(np.int64), stay_te)
    qi_m = np.isin(qi['stay_id'].astype(np.int64), stay_te)
    s_stay = si['stay_id'].astype(np.int64)[si_m]
    q_stay = qi['stay_id'].astype(np.int64)[qi_m]
    assert np.array_equal(np.sort(s_stay), np.sort(stay_te)), 'SOFA 测试集 ID 不一致'
    assert np.array_equal(np.sort(q_stay), np.sort(stay_te)), 'qSOFA 测试集 ID 不一致'
    o_s = {int(a): i for i, a in enumerate(s_stay)}
    o_q = {int(a): i for i, a in enumerate(q_stay)}
    P['SOFA'] = si['sofa'].astype(float)[si_m][[o_s[int(a)] for a in stay_te]]
    P['qSOFA-6h'] = qi['qsofa6'].astype(float)[qi_m][[o_q[int(a)] for a in stay_te]]
    order_i = sorted(MODELS + ['SOFA', 'qSOFA-6h'], key=lambda m: -roc_auc_score(y, P[m]))
    with print_profile():
        fig, ax = plt.subplots(figsize=PRINT_FIG['roc'])
        for m in order_i:
            fpr, tpr, _ = roc_curve(y, P[m])
            ls = '--' if m in ('SOFA', 'qSOFA-6h') else '-'
            ax.plot(fpr, tpr, color=COLORS[m], lw=1.9, ls=ls,
                    label=f'{m} (AUC={f3(roc_auc_score(y, P[m]))})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('1 − Specificity'); ax.set_ylabel('Sensitivity')
        ax.set_title(f'Internal test set (48-h landmark cohort {V}, n={len(y)})')
        ax.legend(loc='lower right'); ax.grid(alpha=0.3)
        fig.savefig(FIG / 'roc_curves.png'); plt.close(fig)

    # ---------------- ROC（eICU）----------------
    with print_profile():
        fig, ax = plt.subplots(figsize=PRINT_FIG['roc'])
        for m in order_e:
            fpr, tpr, _ = roc_curve(ye, PE[m])
            ls = '--' if m in ('SOFA', 'qSOFA-6h') else '-'
            ax.plot(fpr, tpr, color=COLORS[m], lw=1.9, ls=ls,
                    label=f'{m} (AUC={f3(roc_auc_score(ye, PE[m]))})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('1 − Specificity'); ax.set_ylabel('Sensitivity')
        ax.set_title(f'Exploratory cross-database evaluation (eICU, n={len(ye)})')
        ax.legend(loc='lower right'); ax.grid(alpha=0.3)
        fig.savefig(FIG / 'roc_curves_external.png'); plt.close(fig)

    # ---------------- PR（内部）----------------
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for m in order:
        pr, rc, _ = precision_recall_curve(y, P[m])
        ax.plot(rc, pr, color=COLORS[m], lw=1.9,
                label=f'{m} (AUPRC={f3(average_precision_score(y, P[m]))})')
    ax.axhline(y.mean(), color='k', ls='--', lw=1,
               label=f'Prevalence={y.mean():.3f}')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(f'Precision–recall, internal test set (cohort {V})')
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
    fig.savefig(FIG / 'pr_curves.png'); plt.close(fig)

    # ---------------- 校准曲线 ----------------
    # 注意：SOFA / qSOFA 是规则评分而非概率，不能画校准曲线（会报 y_prob 越界）。
    for tag, yy, PP, fn, title, mm in [
            ('internal', y, P, 'calibration_curves_internal.png',
             f'Calibration, internal test set (cohort {V})', order),
            ('external', ye, PE, 'calibration_curves_external.png',
             'Calibration, eICU exploratory cross-database evaluation', order_e_models)]:
        with print_profile():
            fig, ax = plt.subplots(figsize=PRINT_FIG['calib'])
            for m in mm:
                fr, mp = calibration_curve(yy, PP[m], n_bins=10, strategy='uniform')
                ax.plot(mp, fr, 'o-', color=COLORS[m], lw=1.8, ms=4, label=m)
            ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect')
            ax.set_xlabel('Mean predicted probability')
            ax.set_ylabel('Observed fraction of deaths')
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(title); ax.legend(loc='lower right'); ax.grid(alpha=0.3)
            fig.savefig(FIG / fn); plt.close(fig)

    # ---------------- 校准主图（含受试者聚类误差条）----------------
    cal = pd.read_csv(HERE / f'calibration_primary_patient_clustered_{V}.csv')
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    for m in order:
        fr, mp = calibration_curve(y, P[m], n_bins=10, strategy='uniform')
        ax.plot(mp, fr, 'o-', color=COLORS[m], lw=1.8, ms=4,
                label=f'{m} (slope={cal.loc[cal.Model == m, "slope"].iloc[0]:.2f}, '
                      f'CITL={cal.loc[cal.Model == m, "CITL"].iloc[0]:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed fraction of deaths')
    ax.set_title('Primary calibration (subject-clustered CIs in supplementary table)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.savefig(FIG / 'calibration_primary.png'); plt.close(fig)

    recal = pd.read_csv(TAB / 'table10_recalibration_eicu_patient_split.csv')
    recal_rec = pd.read_csv(TAB / 'table10_recalibration_eicu_record_split.csv')
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    for ax, m in zip(axes[:2], ['LSTM', 'RandomForest']):
        # 只用患者级不重叠的评价半
        yv = ye[eval_mask]
        fr, mp = calibration_curve(yv, PE[m][eval_mask], n_bins=10, strategy='uniform')
        fr2, mp2 = calibration_curve(yv, Pe[m][eval_mask], n_bins=10, strategy='uniform')
        ax.plot(mp, fr, 'o-', color='#7F7F7F', lw=1.8, ms=4, label='Raw (evaluation half)')
        ax.plot(mp2, fr2, 's-', color=COLORS[m], lw=1.8, ms=4,
                label='Logistic recalibrated\n(fit on disjoint patients)')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect')
        b0 = recal.loc[recal.Model == m, 'Brier_raw'].iloc[0]
        b1 = recal.loc[recal.Model == m, 'Brier_logistic'].iloc[0]
        ax.set_title(f'{m}: patient-level recalibration\nBrier {b0:.3f} \u2192 {b1:.3f} '
                     f'(n={int(eval_mask.sum())} records)')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Observed fraction of deaths')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
    # 第 3 面板：旧口径（记录级划半）作对照，用于说明抽样单位的影响
    ax = axes[2]
    m = 'LSTM'
    yv = ye[rec_mask]
    fr, mp = calibration_curve(yv, PE[m][rec_mask], n_bins=10, strategy='uniform')
    fr2, mp2 = calibration_curve(yv, Pe[m][rec_mask], n_bins=10, strategy='uniform')
    ax.plot(mp, fr, 'o-', color='#7F7F7F', lw=1.8, ms=4, label='Raw (record-split half)')
    ax.plot(mp2, fr2, 's-', color=COLORS[m], lw=1.8, ms=4, label='Logistic recalibrated')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect')
    b0 = recal_rec.loc[recal_rec.Model == m, 'Brier_raw'].iloc[0]
    b1 = recal_rec.loc[recal_rec.Model == m, 'Brier_logistic'].iloc[0]
    ax.set_title(f'{m} (sensitivity: record-level split)\nBrier {b0:.3f} \u2192 {b1:.3f} '
                 f'(n={int(rec_mask.sum())} records)')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed fraction of deaths')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
    fig.suptitle('eICU: recalibration. Panels 1-2 = patient-level split (primary, '
                 'fit/evaluation patients disjoint); panel 3 = record-level split (sensitivity)')
    fig.savefig(FIG / 'calibration_sensitivity.png'); plt.close(fig)

    # ---------------- DCA ----------------
    # SOFA / qSOFA 是规则评分，不能当作"阈值概率"参与 DCA
    ts = np.linspace(0.01, 0.60, 120)
    for tag, yy, PP, fn, title, mm in [
            ('internal', y, P, 'dca_compare.png',
             f'Decision curve analysis, internal test set (cohort {V})', order),
            ('external', ye, PE, 'dca_compare_external.png',
             'Decision curve analysis, eICU exploratory evaluation', order_e_models)]:
        with print_profile():
            fig, ax = plt.subplots(figsize=PRINT_FIG['dca'])
            for m in mm:
                nb, _ = dca(yy, PP[m], ts)
                ax.plot(ts, nb, color=COLORS[m], lw=1.8, label=m)
            nb, nb_all = dca(yy, PP[order[0]], ts)
            ax.plot(ts, nb_all, 'k-', lw=1.4, label='Treat all')
            ax.plot([0, 0.6], [0, 0], 'k--', lw=1.2, label='Treat none')
            ax.set_ylim(-0.12, max(0.06, float(np.max(nb_all)) * 1.05))
            ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit')
            ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
            fig.savefig(FIG / fn); plt.close(fig)

    # ---------------- 正文合图：Figure 2 / 3 / 4（A|B 双面板，共用图例）----------------
    # 说明：这三张图以「最终印刷尺寸」设计（figsize 7.6in 宽、基准字号 10pt），
    # 在稿件中整页宽（6.5in）显示时图内文字印刷约 9.3pt，与正文 10pt 接近；
    # 共用图例放在图体下方，既不遮挡曲线，也保证图与图注同页。
    with print_profile():
        fig, axes = plt.subplots(1, 2, figsize=PRINT_FIG_2PANEL)
        for ax, (yy, PP, mm, ttl) in zip(axes, [
                (y, P, order_i, f'A  Internal test set (cohort {V}, n={len(y)})'),
                (ye, PE, order_e, f'B  eICU exploratory evaluation (n={len(ye)})')]):
            for m in mm:
                fpr, tpr, _ = roc_curve(yy, PP[m])
                ls = '--' if m in ('SOFA', 'qSOFA-6h') else '-'
                ax.plot(fpr, tpr, color=COLORS[m], lw=1.6, ls=ls,
                        label=f'{m} (AUC={f3(roc_auc_score(yy, PP[m]))})')
            ax.plot([0, 1], [0, 1], 'k--', lw=1)
            ax.set_xlabel('1 − Specificity'); ax.set_ylabel('Sensitivity')
            ax.set_title(ttl); ax.grid(alpha=0.3)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc='lower center', ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, -0.04))
        fig.subplots_adjust(bottom=0.34, top=0.88, wspace=0.25)
        fig.savefig(FIG / 'fig2_roc_combined.png'); plt.close(fig)

    with print_profile():
        fig, axes = plt.subplots(1, 2, figsize=PRINT_FIG_2PANEL)
        for ax, (yy, PP, mm, ttl) in zip(axes, [
                (y, P, order, f'A  Internal test set (cohort {V})'),
                (ye, PE, order_e_models, 'B  eICU exploratory evaluation')]):
            for m in mm:
                fr, mp = calibration_curve(yy, PP[m], n_bins=10, strategy='uniform')
                ax.plot(mp, fr, 'o-', color=COLORS[m], lw=1.6, ms=3.5, label=m)
            ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect')
            ax.set_xlabel('Mean predicted probability')
            ax.set_ylabel('Observed fraction of deaths')
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(ttl); ax.grid(alpha=0.3)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc='lower center', ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, -0.04))
        fig.subplots_adjust(bottom=0.34, top=0.88, wspace=0.25)
        fig.savefig(FIG / 'fig3_calibration_combined.png'); plt.close(fig)

    with print_profile():
        fig, axes = plt.subplots(1, 2, figsize=PRINT_FIG_2PANEL)
        for ax, (yy, PP, mm, ttl) in zip(axes, [
                (y, P, order, f'A  Internal test set (cohort {V})'),
                (ye, PE, order_e_models, 'B  eICU exploratory evaluation')]):
            for m in mm:
                nb, _ = dca(yy, PP[m], ts)
                ax.plot(ts, nb, color=COLORS[m], lw=1.6, label=m)
            nb, nb_all = dca(yy, PP[order[0]], ts)
            ax.plot(ts, nb_all, 'k-', lw=1.2, label='Treat all')
            ax.plot([0, 0.6], [0, 0], 'k--', lw=1.0, label='Treat none')
            ax.set_ylim(-0.12, max(0.06, float(np.max(nb_all)) * 1.05))
            ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit')
            ax.set_title(ttl); ax.grid(alpha=0.3)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc='lower center', ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, -0.04))
        fig.subplots_adjust(bottom=0.34, top=0.88, wspace=0.25)
        fig.savefig(FIG / 'fig4_dca_combined.png'); plt.close(fig)

    # ---------------- 临床影响曲线 ----------------
    for tag, yy, PP, fn, title in [
            ('internal', y, P, 'clinical_impact_lstm.png',
             f'Clinical impact curve, LSTM (cohort {V})'),
            ('external', ye, PE, 'clinical_impact_lstm_external.png',
             'Clinical impact curve, LSTM, eICU')]:
        p = PP['LSTM']
        ttl = np.linspace(0.01, 0.60, 120)
        flag = np.array([(p >= t).sum() for t in ttl])
        hit = np.array([((p >= t) & (yy == 1)).sum() for t in ttl])
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        ax.plot(ttl, flag, color='#D62728', lw=2, label='Number flagged high risk')
        ax.plot(ttl, hit, color='#1F77B4', lw=2, label='Number of deaths captured')
        ax.set_xlabel('Threshold probability'); ax.set_ylabel('Number of records')
        ax.set_title(title); ax.legend(fontsize=9); ax.grid(alpha=0.3)
        fig.savefig(FIG / fn); plt.close(fig)

    # ---------------- 性能条形图（含受试者聚类 CI）----------------
    perf = pd.read_csv(TAB / 'table2_model_performance.csv')
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    xx = np.arange(len(perf))
    lo = perf.AUC - perf.AUC_CI_low_subject
    hi = perf.AUC_CI_high_subject - perf.AUC
    ax.bar(xx, perf.AUC, yerr=[lo, hi], capsize=4,
           color=[COLORS[m] for m in perf.Model], alpha=0.85)
    ax.set_xticks(xx); ax.set_xticklabels(perf.Model, rotation=30, ha='right')
    ax.axhline(0.5, color='k', ls='--', lw=1, label='Chance')
    ax.set_ylabel('AUROC (subject-clustered 95% CI)')
    ax.set_title(f'Internal test set discrimination (cohort {V})')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')
    fig.savefig(FIG / 'model_performance_bar.png'); plt.close(fig)

    # ---------------- 混淆矩阵（RF 在 F3 研究性操作点）----------------
    rf = P['RandomForest']
    pred = (rf >= thr['RandomForest']).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    im = ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=12)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Predicted low risk', 'Predicted high risk'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Survived', 'Died'])
    ax.set_title(f'RandomForest at research operating point\n'
                 f'threshold={thr["RandomForest"]:.2f} (validation-set F3-optimal)')
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(FIG / 'confusion_matrix_rf.png'); plt.close(fig)

    # ---------------- LSTM 注意力 + 遮挡忠实性 ----------------
    qaf = HERE / 'qa' / f's08_explain_{V}.json'
    if qaf.exists():
        q8 = json.loads(qaf.read_text())
        hours = q8['attention']['mean_att_top5_hours']
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        ax.bar(range(48), [1 / 48] * 48, color='#CCCCCC',
               label='Uniform attention (1/48)')
        ax.bar([h for h in range(48) if h in hours], [1 / 48] * len(hours),
               color='#D62728', label='Top-5 hours by mean attention')
        ax.set_xlabel('ICU hour (0–47)'); ax.set_ylabel('Mean attention weight')
        ax.set_title(f'Mean attention across 48 h, LSTM (cohort {V}) — '
                     f'top-5 hours: {hours}')
        ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')
        fig.savefig(FIG / 'lstm_attention.png'); plt.close(fig)

        f = pd.read_csv(TAB / 'table8c_attention_faithfulness.csv')
        ba = q8['occlusion']['baseline_auc']
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
        for ax, base, cols, ttl in [
                (axes[0], ba, ['AUC_global_top', 'AUC_per_sample_top',
                               'AUC_random_mean'], 'AUC drop'),
                (axes[1], q8['attention']['baseline_auprc'],
                 ['AUPRC_global_top', 'AUPRC_per_sample_top', 'AUPRC_random_mean'],
                 'AUPRC drop')]:
            ks = f.k_hours.values
            rnd = base - f[cols[2]].values
            sd = (f['AUC_random_sd'] if 'AUC' in cols[0] else f['AUPRC_random_sd']).values
            ax.fill_between(ks, rnd - sd, rnd + sd, color='gray', alpha=0.2,
                            label='Random occlusion (mean ± SD)')
            ax.plot(ks, rnd, 'o-', color='gray', lw=1.6)
            ax.plot(ks, base - f[cols[0]].values, 's--', color='#FF7F0E', lw=1.8,
                    label='Global top-k attention')
            ax.plot(ks, base - f[cols[1]].values, 'o-', color='#D62728', lw=2.2,
                    label='Per-sample top-k attention')
            ax.set_xticks(ks)
            ax.set_xlabel('Number of masked hours (k)')
            ax.set_ylabel('Performance drop'); ax.set_title(ttl)
            ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
        fig.suptitle('Attention occlusion faithfulness (model-behaviour analysis)')
        fig.savefig(FIG / 'attention_faithfulness.png'); plt.close(fig)

    # ---------------- SHAP ----------------
    sp = HERE / 'data' / f'shap_values_{V}.npz'
    if sp.exists():
        z = np.load(sp, allow_pickle=True)
        sv = z['shap']; nm = z['feature_names']
        top = pd.read_csv(TAB / 'shap_top_features.csv').head(20)
        fig, ax = plt.subplots(figsize=(7.0, 6.6))
        ax.barh(range(len(top))[::-1], top.mean_abs_shap.values,
                color='#1F77B4', alpha=0.85)
        ax.set_yticks(range(len(top))[::-1]); ax.set_yticklabels(top.feature.values, fontsize=8.5)
        ax.set_xlabel('Mean |SHAP value| (standardised feature units)')
        ax.set_title(f'RandomForest SHAP importance, internal test set (cohort {V})')
        ax.grid(alpha=0.3, axis='x')
        fig.savefig(FIG / 'shap_summary_bar.png'); plt.close(fig)

        # 简易 beeswarm（按重要性前 15 特征）
        k = 15
        idx = [int(np.where(nm == f)[0][0]) for f in top.feature.values[:k]]
        Xs = z['X_std'][:, idx]
        Sv = sv[:, idx]
        fig, ax = plt.subplots(figsize=(7.4, 6.6))
        rng = np.random.default_rng(42)
        for r in range(k):
            v = Sv[:, r]
            c = Xs[:, r]
            c = (c - np.nanmin(c)) / (np.nanmax(c) - np.nanmin(c) + 1e-9)
            ax.scatter(v, np.full(len(v), k - 1 - r) + rng.normal(0, 0.08, len(v)),
                       c=c, cmap='coolwarm', s=3, alpha=0.5, linewidths=0)
        ax.set_yticks(range(k)); ax.set_yticklabels(top.feature.values[:k], fontsize=8.5)
        ax.axvline(0, color='k', lw=1)
        ax.set_xlabel('SHAP value (impact on log-odds)')
        ax.set_title(f'SHAP summary, RandomForest (cohort {V})')
        ax.grid(alpha=0.3, axis='x')
        fig.savefig(FIG / 'shap_summary.png'); plt.close(fig)

    print(f'-> 图已写入 {FIG}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
