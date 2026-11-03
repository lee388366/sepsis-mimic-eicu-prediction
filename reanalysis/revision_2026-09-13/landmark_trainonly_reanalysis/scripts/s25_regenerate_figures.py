#!/usr/bin/env python3
"""s25_regenerate_figures.py —— 按本轮重分析产物重绘 10 张补充图（不改文件名）

重绘清单（稿件引用名 <- 本轮数据源）
--------------------------------------
fig16b_roc_internal_sofa.png     内部 ROC：ML 模型 + SOFA/qSOFA-6h
fig16c_roc_internal_models_only.png 内部 ROC：仅 7 个 ML 模型
fig17_shap_dependence.png        SHAP 依赖图（top-3 特征）
fig20_shap_waterfall.png         SHAP waterfall：死亡记录中预测概率最高者
shap_waterfall_temp.png          SHAP waterfall：温度簇 SHAP 贡献最大的死亡记录
fig3_rf_calib_dca.png            RF 外部校准曲线 + 决策曲线（合并面板）
fig8a_rf_cic_internal.png        RF 临床影响曲线（内部测试集）
fig8b_rf_cic_external.png        RF 临床影响曲线（eICU）
fig_race_subgroup_forest.png     种族亚组森林图（内部 + 外部）
fig12_learning_curve.png         RF 学习曲线（按新插补重算）

可复用旧产物的部分（已在脚本内注明理由）
------------------------------------------
SOFA / qSOFA-6h 内部评分：规则评分，用**插补前观测**计算，不依赖模型或插补参数；
且已核验旧 sofa_predictions.npz / qsofa_predictions.npz 的 stay_id 集合与本轮测试集
**完全相同**（均为 3,742 条），因此直接复用而非重算。

输出: figures_regenerated/<name>.png（隔离目录）；安装由 s24 的姊妹步骤 --install 完成。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent.parent
SITE = HERE.parent.parent
ROOT = SITE.parent
DATA = HERE / 'data'
TAB = HERE / 'tables' / 'A'
MD = HERE / 'models_trainonly' / 'A'
OUT = HERE / 'figures_regenerated'
FIGS = SITE / 'figs'
BK = HERE / '00_manuscript_sync' / 'before' / 'figs'

SEED = 42
MINUS = '\u2212'
RC = {'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
      'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight'}
COLORS = {'LSTM': '#D62728', 'RandomForest': '#1F77B4', 'SVM': '#2CA02C',
          'LogisticRegression': '#9467BD', 'GaussianNB': '#FF7F0E',
          'KNN': '#8C564B', 'DecisionTree': '#17BECF',
          'SOFA': '#7F7F7F', 'qSOFA-6h': '#BCBD22'}
DISP = {'LSTM': 'LSTM', 'RandomForest': 'RF', 'SVM': 'SVM',
        'LogisticRegression': 'LR', 'GaussianNB': 'GaussianNB',
        'KNN': 'KNN', 'DecisionTree': 'DT', 'SOFA': 'SOFA', 'qSOFA-6h': 'qSOFA-6h'}
MODELS = ['LSTM', 'RandomForest', 'SVM', 'LogisticRegression',
          'GaussianNB', 'KNN', 'DecisionTree']

HCV = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'spo2', 'temperature',
       'glucose', 'fio2', 'ph', 'gcs', 'gcs_eyes', 'gcs_motor', 'gcs_verbal']
CLIN = {'heart_rate': 'Heart Rate', 'sbp': 'Systolic BP', 'dbp': 'Diastolic BP',
        'mbp': 'Mean BP', 'resp_rate': 'Respiratory Rate', 'spo2': 'SpO2',
        'temperature': 'Temperature', 'glucose': 'Glucose', 'fio2': 'FiO2',
        'ph': 'pH', 'gcs': 'GCS Score', 'gcs_eyes': 'GCS Eyes',
        'gcs_motor': 'GCS Motor', 'gcs_verbal': 'GCS Verbal'}


def f3(v) -> str:
    return f'{float(v):.3f}'.replace('-', MINUS)


def load_all():
    z = np.load(DATA / 'trainonly_mimic.npz', allow_pickle=False)
    p = np.load(DATA / 'preds_A.npz', allow_pickle=True)
    e = np.load(DATA / 'eicu_probs_A.npz', allow_pickle=True)
    s = np.load(DATA / 'shap_values_A.npz', allow_pickle=True)
    split = p['split'].astype(str)
    return z, p, e, s, split


# ------------------------------------------------------------------ S12 ROC
def fig_roc_internal(z, p, split):
    te = np.flatnonzero(split == 'test')
    y = p['y_true'][te].astype(int)
    sofa = np.load(ROOT / 'sofa_predictions.npz')
    qs = np.load(ROOT / 'qsofa_predictions.npz')
    sofa_map = dict(zip(sofa['stay_id'], sofa['sofa']))
    qs_map = dict(zip(qs['stay_id'], qs['qsofa6']))
    stays = p['stay_id'][te]
    s_te = np.array([sofa_map[k] for k in stays], float)
    q_te = np.array([qs_map[k] for k in stays], float)

    curves = [(m, p[f'prob_{m}'][te].astype(float)) for m in MODELS]
    curves += [('SOFA', s_te), ('qSOFA-6h', q_te)]

    # (A) 含 SOFA/qSOFA 的版本
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 0.70, 0.90, 1], [0, 0.70, 0.90, 1], color='gray',
            ls=(0, (5, 4)), lw=1.0)
    for m, prob in curves:
        fpr, tpr, _ = roc_curve(y, prob)
        a = roc_auc_score(y, prob)
        lw = 2.6 if m == 'LSTM' else 1.9
        ax.plot(fpr, tpr, color=COLORS[m], lw=lw,
                label=f'{DISP[m]} (AUC = {a:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.0, label='Random (AUC = 0.500)')
    ax.set_title(f'Internal test set (n={len(y):,}) \u2014 ROC curve by comparator')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'fig16b_roc_internal_sofa.png')
    plt.close(fig)

    # (B) 仅 ML 模型
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for m, prob in curves[:len(MODELS)]:
        fpr, tpr, _ = roc_curve(y, prob)
        a = roc_auc_score(y, prob)
        lw = 2.6 if m == 'LSTM' else 1.9
        ax.plot(fpr, tpr, color=COLORS[m], lw=lw,
                label=f'{DISP[m]} (AUC = {a:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.0)
    ax.set_title(f'Internal test set (n={len(y):,}) \u2014 ROC curves by model comparator')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'fig16c_roc_internal_models_only.png')
    plt.close(fig)
    print('[OK ] fig16b/16c（内部 ROC，含与不含 SOFA/qSOFA）')


# ------------------------------------------------------- S13/S14/S18 SHAP
def _explanation(s, idx):
    """用保存的 SHAP 值 + RF 的 expected_value 重建 shap.Explanation。"""
    import joblib
    rf = joblib.load(MD / 'model_RandomForest.joblib')
    base = shap.TreeExplainer(rf, feature_perturbation='tree_path_dependent').expected_value
    base = np.asarray(base)
    base = float(base[1]) if base.ndim else float(base)
    names = [str(x) for x in s['feature_names']]
    return shap.Explanation(
        values=s['shap'][idx], base_values=base,
        data=s['X_std'][idx], feature_names=names)


def fig_shap(z, s):
    names = [str(x) for x in s['feature_names']]
    sv = s['shap']
    imp = np.abs(sv).mean(axis=0)
    top3 = np.argsort(imp)[::-1][:3]
    raw = z['X_flat']                       # 46 维未标准化特征（显示用 x 轴）
    te = np.flatnonzero(z['split'].astype(str) == 'test')

    # --- 依赖图（top-3）---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, fi in zip(axes, top3):
        shap.dependence_plot(fi, sv, raw[te], feature_names=names,
                             ax=ax, show=False)
        ax.set_title(f'{names[fi]}\nmean|SHAP| = {imp[fi]:.4f}', fontsize=11, pad=16)
        # shap 把「着色特征名」放在色条顶端，会与子图标题重叠；移到色条下方。
        cax = fig.axes[-1]
        lab = cax.get_ylabel()
        if lab:
            cax.set_ylabel('')
            cax.text(0.5, -0.22, lab, transform=cax.transAxes,
                     ha='center', va='top', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'fig17_shap_dependence.png')
    plt.close(fig)
    top_names = [names[i] for i in top3]

    # --- waterfall 1：死亡记录中预测概率最高者 ---
    y = s['y_true'].astype(int)
    prob = np.load(DATA / 'preds_A.npz')['prob_RandomForest']
    prob_te = prob[np.flatnonzero(np.load(DATA / 'preds_A.npz')['split'].astype(str) == 'test')]
    dec = np.flatnonzero(y == 1)

    def waterfall(idx, fname):
        """shap 自带 f(x) 标题；说明性文字用 suptitle 置于其上，避免标题互相覆盖。"""
        plt.figure()
        shap.plots.waterfall(_explanation(s, idx), max_display=15, show=False)
        fig = plt.gcf()
        fig.suptitle(
            f'Test record (stay_id={int(s["stay_id"][idx])}): '
            f'model output = {prob_te[idx]:.3f}, observed outcome: died in hospital',
            fontsize=10)
        fig.savefig(OUT / fname, bbox_inches='tight')
        plt.close('all')

    i1 = int(dec[np.argmax(prob_te[dec])])
    waterfall(i1, 'fig20_shap_waterfall.png')

    # --- waterfall 2：温度簇 SHAP 贡献最大的死亡记录（沿用旧选患规则）---
    temp_cols = [18, 19, 20]
    gcs_cols = list(range(30, 42))
    t_c = np.abs(sv[dec][:, temp_cols]).sum(axis=1)
    g_c = np.abs(sv[dec][:, gcs_cols]).sum(axis=1)
    i2 = int(dec[np.argmax(t_c)])
    waterfall(i2, 'shap_waterfall_temp.png')

    meta = {
        'top3_features': top_names,
        'mean_abs_shap_top3': [float(imp[i]) for i in top3],
        'waterfall_1': {'rule': '死亡记录中 RF 预测概率最高',
                        'test_index': i1, 'f_x': float(prob_te[i1]),
                        'stay_id': int(s['stay_id'][i1])},
        'waterfall_2': {'rule': '死亡记录中温度簇(mean/min/max) |SHAP| 之和最大',
                        'test_index': i2, 'f_x': float(prob_te[i2]),
                        'temperature_shap': float(t_c.max()),
                        'gcs_shap': float(g_c[np.argmax(t_c)]),
                        'stay_id': int(s['stay_id'][i2])},
    }
    print(f'[OK ] fig17（依赖图 top3: {", ".join(top_names)}）')
    print(f'[OK ] fig20 / shap_waterfall_temp（f(x)={meta["waterfall_1"]["f_x"]:.3f} / '
          f'{meta["waterfall_2"]["f_x"]:.3f}）')
    return meta


# ------------------------------------------------- S3 RF 外部校准 + DCA
def fig_rf_calib_dca(e):
    y = e['y_true'].astype(int)
    prob = e['prob_RandomForest'].astype(float)
    n, prev = len(y), y.mean()
    lp = np.log(np.clip(prob, 1e-6, 1 - 1e-6) / (1 - np.clip(prob, 1e-6, 1 - 1e-6)))
    import statsmodels.api as sm
    md = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
    slope, intercept = md.params[1], md.params[0]
    citl = sm.Logit(y, np.ones_like(y), offset=lp).fit(disp=0).params[0]
    brier = brier_score_loss(y, prob)

    fig, (ax_cal, ax_dca) = plt.subplots(1, 2, figsize=(12, 5.5))
    fp, mp = calibration_curve(y, prob, n_bins=10, strategy='uniform')
    ax_cal.plot(mp, fp, 'o-', color='#D62728', lw=2.2, ms=6,
                label='RandomForest (eICU)')
    ax_cal.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Perfect calibration')
    ax_cal.set_xlabel('Mean predicted probability')
    ax_cal.set_ylabel('Observed fraction of deaths')
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.set_title(f'External Calibration Curve (RF, eICU)\n'
                     f'slope={f3(slope)}, intercept={f3(intercept)}, '
                     f'CITL={f3(citl)} | Brier={f3(brier)}', fontsize=11)
    ax_cal.text(0.03, 0.95, f'Model: RandomForest | Prevalence (eICU): {prev:.3f}',
                transform=ax_cal.transAxes, fontsize=9, va='top')
    ax_cal.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax_cal.grid(alpha=0.3)

    dca = pd.read_csv(TAB / 'dca_eicu.csv')
    rf = dca[(dca.Model == 'RandomForest') & (dca.prob_type == 'raw')]
    ax_dca.plot(rf.threshold_prob, rf.net_benefit, color='#1F77B4', lw=2.2,
                label='RandomForest')
    t = rf.threshold_prob.values
    ta = prev - (1 - prev) * t / (1 - t)
    ax_dca.plot(t, ta, color='#7F7F7F', lw=1.8, ls='--', label='Treat all')
    ax_dca.axhline(0, color='k', lw=1.0, label='Treat none')
    ax_dca.set_xlim(0, 1)
    ax_dca.set_ylim(-0.05, 0.25)
    ax_dca.set_xlabel('Threshold probability')
    ax_dca.set_ylabel('Net benefit')
    ax_dca.set_title('Decision Curve Analysis (External / eICU)', fontsize=11)
    ax_dca.text(0.03, 0.95, f'Model: RandomForest | Prevalence (eICU): {prev:.3f}',
                transform=ax_dca.transAxes, fontsize=9, va='top')
    ax_dca.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax_dca.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'fig3_rf_calib_dca.png')
    plt.close(fig)
    print(f'[OK ] fig3_rf_calib_dca（slope={f3(slope)} Brier={f3(brier)}）')


# --------------------------------------------------- S7/S8 RF 临床影响曲线
def _cic(prob, y, fname, title):
    thr = np.linspace(0.01, 0.99, 99)
    hr = np.array([(prob >= t).sum() for t in thr], float)
    ev = np.array([((prob >= t) & (y == 1)).sum() for t in thr], float)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thr, hr, color='#D62728', lw=2.2, label='Number high risk')
    ax.plot(thr, ev, color='#1F77B4', lw=2.2,
            label='Number high risk with event')
    ax.fill_between(thr, ev, hr, color='gray', alpha=0.15,
                    label='Over-triage gap (non-event among high risk)')
    ax.set_xlabel('High-risk threshold probability')
    ax.set_ylabel('Number of patients classified as high risk')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, hr.max() * 1.05)
    ax.set_title(title, fontsize=12)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / fname)
    plt.close(fig)


def fig_rf_cic(z, p, e, split):
    te = np.flatnonzero(split == 'test')
    _cic(p['prob_RandomForest'][te].astype(float), p['y_true'][te].astype(int),
         'fig8a_rf_cic_internal.png',
         f'Random forest clinical impact curve \u2014 internal test set '
         f'(n={len(te):,})')
    _cic(e['prob_RandomForest'].astype(float), e['y_true'].astype(int),
         'fig8b_rf_cic_external.png',
         f'Random forest clinical impact curve \u2014 eICU-CRD (n={len(e["y_true"]):,})')
    print('[OK ] fig8a/fig8b（RF 临床影响曲线）')


# ------------------------------------------------------- S19 种族亚组森林图
def fig_race_forest(p, e, split):
    ins = pd.read_csv(TAB / 'table8_subgroup_auc.csv')
    ext = pd.read_csv(TAB / 'table9b_subgroup_external.csv')
    te = np.flatnonzero(split == 'test')
    y_te = p['y_true'][te].astype(int)
    ov_in = {'LSTM': float(roc_auc_score(y_te, p['prob_LSTM'][te])),
             'RF': float(roc_auc_score(y_te, p['prob_RandomForest'][te]))}
    tb = pd.read_csv(TAB / 'table2_model_performance.csv').set_index('Model')
    ov_in.update({
        'LSTM_lo': float(tb.loc['LSTM', 'AUC_CI_low_subject']),
        'LSTM_hi': float(tb.loc['LSTM', 'AUC_CI_high_subject']),
        'RF_lo': float(tb.loc['RandomForest', 'AUC_CI_low_subject']),
        'RF_hi': float(tb.loc['RandomForest', 'AUC_CI_high_subject'])})
    te9 = pd.read_csv(TAB / 'table9_external_validation.csv').set_index('Model')
    ov_ex = {
        'LSTM': float(te9.loc['LSTM', 'AUC_ext']),
        'RF': float(te9.loc['RandomForest', 'AUC_ext']),
        'LSTM_lo': float(te9.loc['LSTM', 'AUC_CI_low_patient']),
        'LSTM_hi': float(te9.loc['LSTM', 'AUC_CI_high_patient']),
        'RF_lo': float(te9.loc['RandomForest', 'AUC_CI_low_patient']),
        'RF_hi': float(te9.loc['RandomForest', 'AUC_CI_high_patient'])}

    def rows(df, c, ov):
        out = [('Overall', ov['RF'], ov['RF_lo'], ov['RF_hi'],
                ov['LSTM'], ov['LSTM_lo'], ov['LSTM_hi'], int(df[c['n']].sum()))]
        for _, r in df.sort_values('Subgroup').iterrows():
            out.append((r['Subgroup'],
                        float(r[c['rf']]), float(r[c['rf_lo']]), float(r[c['rf_hi']]),
                        float(r[c['lstm']]), float(r[c['lstm_lo']]), float(r[c['lstm_hi']]),
                        int(r[c['n']])))
        return out

    A = rows(
        ins[ins.Dimension == 'Race'],
        {'n': 'n', 'rf': 'RF_AUC', 'rf_lo': 'RF_CI_low', 'rf_hi': 'RF_CI_high',
         'lstm': 'LSTM_AUC', 'lstm_lo': 'LSTM_CI_low', 'lstm_hi': 'LSTM_CI_high'},
        ov_in)
    B = rows(
        ext[ext.Dimension == 'Race'],
        {'n': 'n', 'rf': 'RandomForest_AUC',
         'rf_lo': 'RandomForest_CI_low_patient', 'rf_hi': 'RandomForest_CI_high_patient',
         'lstm': 'LSTM_AUC', 'lstm_lo': 'LSTM_CI_low_patient',
         'lstm_hi': 'LSTM_CI_high_patient'},
        ov_ex)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, data, ttl in (
            (axes[0], A, 'A \u2014 Internal test set (MIMIC-IV)'),
            (axes[1], B, 'B \u2014 Exploratory eICU-CRD evaluation')):
        ypos = np.arange(len(data))[::-1]
        for k, (name, rf, rf_lo, rf_hi, lstm, lstm_lo, lstm_hi, n) in enumerate(data):
            yy = ypos[k]
            ax.errorbar(rf, yy + 0.16,
                        xerr=[[max(rf - rf_lo, 0)], [max(rf_hi - rf, 0)]],
                        fmt='o', color='#1F77B4', ms=6, capsize=3, lw=1.6,
                        label='RF' if k == 0 else None)
            ax.errorbar(lstm, yy - 0.16,
                        xerr=[[max(lstm - lstm_lo, 0)], [max(lstm_hi - lstm, 0)]],
                        fmt='s', color='#D62728', ms=6, capsize=3, lw=1.6,
                        label='LSTM' if k == 0 else None)
        ax.set_yticks(ypos)
        ax.set_yticklabels([f'{d[0]}  (n={d[7]:,})' for d in data], fontsize=9)
        ax.axvline(0.70, color='gray', ls=':', lw=1.4)
        ax.set_xlabel('AUC')
        ax.set_title(ttl, fontsize=11)
        ax.grid(alpha=0.3, axis='x')
        lo = min(min(d[2] for d in data), min(d[5] for d in data), 0.70)
        hi = max(max(d[4] for d in data), max(d[6] for d in data), 0.70)
        pad = max(0.02, (hi - lo) * 0.10)
        ax.set_xlim(lo - pad, hi + pad)
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_race_subgroup_forest.png')
    plt.close(fig)
    print('[OK ] fig_race_subgroup_forest（种族亚组森林图，内部+外部）')


# --------------------------------------------------------- S10 学习曲线
def fig_learning_curve(z):
    X = z['X_flat'].astype(np.float32)
    y = z['y'].astype(int)
    tr, va = z['tr_idx'], z['va_idx']
    sc = StandardScaler().fit(X[tr])
    Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
    ytr, yva = y[tr], y[va]
    sizes = [2000, 4000, 7000, 10000, 14000, len(tr)]
    reps = 3
    tr_auc = np.zeros((len(sizes), reps))
    va_auc = np.zeros((len(sizes), reps))
    for si, n in enumerate(sizes):
        for rep in range(reps):
            rng = np.random.default_rng(SEED * 100 + si * 10 + rep)
            sub = rng.permutation(len(tr))[:n]
            rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                        class_weight='balanced', n_jobs=-1,
                                        random_state=SEED)
            rf.fit(Xtr[sub], ytr[sub])
            tr_auc[si, rep] = roc_auc_score(ytr[sub], rf.predict_proba(Xtr[sub])[:, 1])
            va_auc[si, rep] = roc_auc_score(yva, rf.predict_proba(Xva)[:, 1])
        print(f'      n={n:>6}: train={tr_auc[si].mean():.4f}±{tr_auc[si].std():.4f} '
              f'val={va_auc[si].mean():.4f}±{va_auc[si].std():.4f}')
    tm, ts = tr_auc.mean(1), tr_auc.std(1)
    vm, vs = va_auc.mean(1), va_auc.std(1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(sizes, tm - ts, tm + ts, alpha=0.15, color='#D62728')
    ax.fill_between(sizes, vm - vs, vm + vs, alpha=0.15, color='#1F77B4')
    ax.plot(sizes, tm, 'o-', color='#D62728', lw=2.2, label='Training set')
    ax.plot(sizes, vm, 's-', color='#1F77B4', lw=2.2, label='Validation set')
    ax.set_xlabel('Number of training records')
    ax.set_ylabel('AUC')
    ax.set_title('Random forest learning curves', fontsize=12)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'fig12_learning_curve.png')
    plt.close(fig)
    print('[OK ] fig12_learning_curve（RF 学习曲线，已按新插补重算）')
    return {'sizes': [int(x) for x in sizes],
            'train_auc_mean': [float(x) for x in tm],
            'val_auc_mean': [float(x) for x in vm]}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    BK.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(RC)
    z, p, e, s, split = load_all()
    fig_roc_internal(z, p, split)
    meta = fig_shap(z, s)
    fig_rf_calib_dca(e)
    fig_rf_cic(z, p, e, split)
    fig_race_forest(p, e, split)
    lc = fig_learning_curve(z)
    import json
    (HERE / 'qa' / 's25_figure_regen.json').write_text(json.dumps(
        {'generated': sorted(x.name for x in OUT.glob('*.png')),
         'shap_meta': meta, 'learning_curve': lc,
         'reused_from_old_pipeline': {
             'sofa_qsofa_internal': '规则评分，用插补前观测；旧 npz 的 stay_id 集合已核验与新测试集完全一致'}},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n共 {len(list(OUT.glob("*.png")))} 张图 -> {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
