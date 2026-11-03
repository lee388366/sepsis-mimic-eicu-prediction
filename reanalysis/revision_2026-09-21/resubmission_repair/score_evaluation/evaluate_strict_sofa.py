"""Isolated evaluation of corrected SOFA against frozen variant-A predictions.

No training, threshold selection, label changes, or original-output writes.
Preserves original subject-bootstrap algorithm, seed=42 and 1000 replicates.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import statsmodels.api as sm

B = Path(__file__).resolve().parent
ROOT = B.parents[2]
R = ROOT / 'revision_2026-09-13/landmark_trainonly_reanalysis'
sys.path.insert(0, str(R / 'scripts'))
import common_eval as C

COMP = ['respiration', 'coagulation', 'liver', 'cardiovascular', 'cns', 'renal']
N_BOOT = 1000
SEED = 42

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def ci(v):
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]

def delong(y, a, b):
    auc, cov = C.delong_auc_cov(y, np.vstack([a, b]))
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    assert var > 0
    z = float((auc[0] - auc[1]) / np.sqrt(var))
    return dict(auc_a=float(auc[0]), auc_b=float(auc[1]),
                difference=float(auc[0] - auc[1]), z=z,
                p=float(2 * stats.norm.sf(abs(z))), se=float(np.sqrt(var)))

def boot_metrics(y, scores, ids):
    out = {k: [] for k in scores}
    aps = {k: [] for k in scores}
    for ix in C.cluster_batches(ids, N_BOOT, SEED):
        if len(np.unique(y[ix])) < 2:
            continue
        for k, p in scores.items():
            out[k].append(roc_auc_score(y[ix], p[ix]))
            aps[k].append(average_precision_score(y[ix], p[ix]))
    assert all(len(x) > 950 for x in out.values())
    return ({k: np.asarray(v) for k, v in out.items()},
            {k: np.asarray(v) for k, v in aps.items()})

def evaluate(score_path, output, legacy_baseline=False):
    output.mkdir(parents=True, exist_ok=True)
    inputs = [score_path, R/'data/preds_A.npz', R/'data/trainonly_mimic.npz',
              ROOT.parent/'qsofa_predictions.npz', ROOT.parent/'sofa_extract.csv',
              R/'scripts/common_eval.py', Path(__file__), ROOT.parent/'demographics_by_stay.csv']
    before = {str(p): sha(p) for p in inputs}
    d = np.load(inputs[1], allow_pickle=False)
    tr = np.load(inputs[2], allow_pickle=False)
    q = np.load(inputs[3], allow_pickle=False)
    raw = pd.read_csv(score_path)
    if not legacy_baseline:
        boundary = ['chart_outside_n', 'lab_outside_n', 'pf_pair_outside_n', 'uo_outside_n']
        assert raw[boundary].notna().all().all() and (raw[boundary] == 0).all().all()
    assert raw.stay_id.is_unique and len(raw) == 24894
    assert set(raw.stay_id) == set(tr['stay_id']) == set(d['stay_id'])
    assert raw.sofa_max_48h.notna().all()
    for col in COMP:
        observed = raw[col].dropna()
        assert observed.between(0, 4).all() and np.equal(observed, np.floor(observed)).all()
    assert np.array_equal(raw[COMP].fillna(0).sum(axis=1), raw.sofa_max_48h)
    mask = d['split'].astype(str) == 'test'
    ids = d['stay_id'][mask].astype(np.int64)
    y = d['y_true'][mask].astype(int)
    subj = d['subject_id'][mask].astype(np.int64)
    assert len(y) == 3742 and y.sum() == 712 and len(np.unique(subj)) == 3242
    assert np.array_equal(q['stay_id'], ids) and np.array_equal(q['y_true'], y)
    subject_alignment = pd.Series(tr['subject_id'], index=tr['stay_id']).loc[d['stay_id']].to_numpy()
    assert np.array_equal(subject_alignment, d['subject_id'])
    # 1:1 alignment; missing components become zero only in the legacy scoring total.
    score = raw.set_index('stay_id').loc[ids]
    age = pd.Series(tr['X_sv'][:, 0], index=tr['stay_id']).loc[ids].to_numpy()
    assert np.isfinite(age).all()
    probabilities = {'LSTM': d['prob_LSTM'][mask], 'RF': d['prob_RandomForest'][mask],
                     'SOFA': score.sofa_max_48h.to_numpy(float), 'qsofa6': q['qsofa6'].astype(float)}
    assert all(np.isfinite(v).all() for v in probabilities.values())
    auc_boot, ap_boot = boot_metrics(y, probabilities, subj)
    overall = []
    for name, p in probabilities.items():
        overall.append(dict(Model=name, n=len(y), deaths=int(y.sum()), n_subjects=len(np.unique(subj)),
                            AUC=float(roc_auc_score(y, p)), AUC_CI_low=ci(auc_boot[name])[0],
                            AUC_CI_high=ci(auc_boot[name])[1], AUPRC=float(average_precision_score(y, p)),
                            AP_CI_low=ci(ap_boot[name])[0], AP_CI_high=ci(ap_boot[name])[1]))
    pd.DataFrame(overall).to_csv(output/'overall.csv', index=False)
    dl = {name: delong(y, probabilities['LSTM'], p) for name, p in probabilities.items() if name != 'LSTM'}
    paired = {name: dict(difference=dl[name]['difference'], CI_low=ci(auc_boot['LSTM']-auc_boot[name])[0],
                        CI_high=ci(auc_boot['LSTM']-auc_boot[name])[1]) for name in dl}
    ops = []
    for cut in [6, 8, 10, 12]:
        tn, fp, fn, tp = [int(x) for x in confusion_matrix(y, probabilities['SOFA'] >= cut, labels=[0, 1]).ravel()]
        ops.append(dict(cutoff=cut, tn=tn, fp=fp, fn=fn, tp=tp, sensitivity=tp/(tp+fn),
                        specificity=tn/(tn+fp), ppv=tp/(tp+fp), npv=tn/(tn+fn), flagged=tp+fp))
    pd.DataFrame(ops).to_csv(output/'sofa_operating_points.csv', index=False)
    # Table 1 preserves median(IQR width) and existing tests.
    yy = tr['y'].astype(int)
    allscore = raw.set_index('stay_id').loc[tr['stay_id']]
    fixed_labels = pd.Series(d['y_true'], index=d['stay_id']).loc[tr['stay_id']].to_numpy()
    assert np.array_equal(yy, fixed_labels)
    sofa = allscore.sofa_max_48h.to_numpy()
    cv = allscore.cardiovascular.fillna(0).to_numpy()
    fmt = lambda v: f'{np.median(v):.1f} ({np.percentile(v,75)-np.percentile(v,25):.1f})'
    count = lambda v: f'{int(v.sum())} ({100*v.mean():.1f}%)'
    c0, c1 = cv[yy == 0] >= 2, cv[yy == 1] >= 2
    table1 = [dict(Variable='48-hour worst-component SOFA summary', Survived=fmt(sofa[yy==0]),
                  Died=fmt(sofa[yy==1]), p=float(stats.mannwhitneyu(sofa[yy==0], sofa[yy==1]).pvalue)),
              dict(Variable='Cardiovascular SOFA >=2', Survived=count(c0), Died=count(c1),
                   p=float(stats.chi2_contingency([[c0.sum(), (~c0).sum()], [c1.sum(), (~c1).sum()]])[1]))]
    pd.DataFrame(table1).to_csv(output/'table1_score_rows.csv', index=False)
    groups = [('Age', np.where(age < 65, 'age<65', 'age>=65')),
              ('48-hour SOFA summary', np.where(probabilities['SOFA'] < 8, 'SOFA<8', 'SOFA>=8')),
              ('Cardiovascular SOFA', np.where(score.cardiovascular.fillna(0).to_numpy() >= 2, 'shock_proxy', 'nonshock'))]
    demo = pd.read_csv(inputs[-1]).set_index('stay_id').loc[ids]
    assert demo[['gender', 'race_group']].notna().all().all()
    groups += [('Gender', demo.gender.to_numpy()), ('Race', demo.race_group.to_numpy())]
    subgroup, differences, interactions = [], [], []
    for dim, g in groups:
        levels = sorted(np.unique(g))
        point = {}
        for name in levels:
            sel = g == name
            pp = {k: p[sel] for k, p in probabilities.items()}
            ab, _ = boot_metrics(y[sel], pp, subj[sel])
            row = dict(Dimension=dim, Subgroup=name, n=int(sel.sum()), deaths=int(y[sel].sum()),
                       n_subjects=len(np.unique(subj[sel])), Death_rate=float(y[sel].mean()))
            for mdl, p in pp.items():
                row.update({mdl+'_AUC':float(roc_auc_score(y[sel], p)),
                            mdl+'_CI_low':ci(ab[mdl])[0], mdl+'_CI_high':ci(ab[mdl])[1]})
            subgroup.append(row)
            if dim in ('Gender', 'Race'):
                continue
            dl_sub = delong(y[sel], pp['LSTM'], pp['SOFA'])
            point[name] = dl_sub['difference']
            difci = ci(ab['LSTM'] - ab['SOFA'])
            differences.append(dict(Dimension=dim, Subgroup=name, n=int(sel.sum()),
                                    difference=point[name], CI_low=difci[0], CI_high=difci[1],
                                    DeLong_p_unclustered=dl_sub['p']))
        if dim in ('Gender', 'Race'):
            print('completed subgroup:', dim, flush=True)
            continue
        gl, gh = levels
        dd = []
        for ix in C.cluster_batches(subj, N_BOOT, SEED):
            ml, mh = g[ix] == gl, g[ix] == gh
            if min(ml.sum(), mh.sum()) < 5 or any(len(np.unique(y[ix][m])) < 2 for m in (ml,mh)):
                continue
            vals = [roc_auc_score(y[ix][m], probabilities['LSTM'][ix][m]) -
                    roc_auc_score(y[ix][m], probabilities['SOFA'][ix][m]) for m in (ml,mh)]
            dd.append(vals[0] - vals[1])
        difci = ci(dd)
        differences.append(dict(Dimension=dim, Subgroup=f'diff({gl} - {gh})',
                                difference=point[gl]-point[gh], CI_low=difci[0], CI_high=difci[1], n_boot_valid=len(dd)))
        group_code = (g == gh).astype(float)
        lp = C._logit(probabilities['LSTM'])
        fit = sm.Logit(y, np.column_stack([np.ones(len(y)), lp, group_code, lp*group_code])).fit(disp=0)
        p = float(2*stats.norm.sf(abs(fit.params[3]/fit.bse[3])))
        interactions.append(dict(Dimension=dim, interaction_p=p, beta=float(fit.params[3]),
                                 ci_low=float(fit.conf_int()[3,0]), ci_high=float(fit.conf_int()[3,1]),
                                 p_bonferroni=min(1., p*3)))
        print('completed subgroup:', dim, flush=True)
    pd.DataFrame(subgroup).to_csv(output/'subgroup_auc.csv', index=False)
    pd.DataFrame(differences).to_csv(output/'subgroup_difference.csv', index=False)
    pd.DataFrame(interactions).to_csv(output/'interaction.csv', index=False)
    # Plot-ready predictions stay local, carry stable IDs and labels for independent checks.
    np.savez_compressed(output/'plot_inputs.npz', stay_id=ids, subject_id=subj, y_true=y,
                        **probabilities, cardiovascular=score.cardiovascular.fillna(0).to_numpy())
    after = {str(p): sha(p) for p in inputs}
    assert before == after, 'an input changed during evaluation'
    summary = dict(scope='corrected SOFA evaluation only; all fitted ML models remain frozen',
                   seed=SEED, replicates=N_BOOT, ci='subject-clustered percentile bootstrap',
                   delong_scope='record-level supplementary inference',
                   interaction_scope='original unclustered logistic interaction, Bonferroni across three dimensions',
                   source_sha256=before, input_hashes_unchanged=True, legacy_baseline=legacy_baseline, overall=overall,
                   paired_differences=paired, delong=dl, operating_points=ops, table1=table1,
                   interactions=interactions, seed_subgroup_sensitivity='NOT_RECOMPUTED: individual seed predictions unavailable')
    (output/'evaluation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({'overall':overall,'table1':table1,'operating_points':ops}, indent=2), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', type=Path, default=B.parent/'score_lineage/data/mimic_sofa_strict48.csv')
    parser.add_argument('--output', type=Path, default=B/'results')
    parser.add_argument('--allow-legacy-baseline', action='store_true')
    args = parser.parse_args()
    evaluate(args.scores.resolve(), args.output.resolve(), args.allow_legacy_baseline)
