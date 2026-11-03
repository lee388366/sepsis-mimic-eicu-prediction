from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
H=Path(__file__).resolve().parent;R=H.parents[2];S=H.parent/'score_lineage'
f=S/'data/mimic_sofa_strict48.csv';n=pd.read_csv(f).set_index('stay_id');o=pd.read_csv(R.parent/'sofa_extract.csv').set_index('stay_id').loc[n.index]
p=np.load(R/'revision_2026-09-13/landmark_trainonly_reanalysis/data/preds_A.npz',allow_pickle=False)
c=['respiration','coagulation','liver','cardiovascular','cns','renal'];eligible=n.eligible_uo_endpoints>0
checks={'csv_hash_matches_science_qa':hashlib.sha256(f.read_bytes()).hexdigest()==json.loads((S/'strict48_score_qa.json').read_text())['new_csv_sha256'],'frozen_cohort_unique_ids':n.index.is_unique and len(n)==24894 and set(n.index)==set(p['stay_id']),'finite_integer_components_0_4':bool(np.isfinite(n[c]).all().all() and (n[c]==np.floor(n[c])).all().all() and n[c].ge(0).all().all() and n[c].le(4).all().all()),'component_sum':bool((n[c].sum(axis=1)==n.sofa_max_48h).all()),'hours_integer_0_48':bool(n[[x+'_hours' for x in c]].ge(0).all().all() and n[[x+'_hours' for x in c]].le(48).all().all()),'missing_component_scored_zero':all(bool((n.loc[n[x+'_hours']==0,x]==0).all()) for x in c),'boundaries_zero':bool((n[[x for x in n.columns if x.endswith('outside_n')]]==0).all().all()),'uo_coverage_bounds':bool((n.loc[eligible,'uo_coverage_min']>=22).all() and (n.loc[eligible,'uo_coverage_max']<=30).all()),'no_eligible_uo_has_no_quantity':bool(n.loc[~eligible,'uo_24hr_min'].isna().all())}
delta=n.sofa_max_48h-o.sofa_max_48h
recomputed={'changed_total_n':int(delta.ne(0).sum()),'increased_total_n':int(delta.gt(0).sum()),'decreased_total_n':int(delta.lt(0).sum()),'component_change_n':{x:int(n[x].ne(o[x]).sum()) for x in c},'old_null_components_n':{x:int(o[x].isna().sum()) for x in c},'component_missing_n':{x:int(n[x+'_hours'].eq(0).sum()) for x in c},'uo_eligible_n':int(eligible.sum()),'high_renal_uo_only_n':int(n.high_renal_uo_only.sum())}
q=json.loads((S/'strict48_score_qa.json').read_text())
for key in ['changed_total_n','increased_total_n','decreased_total_n','component_change_n','component_missing_n']:assert recomputed[key]==q[key]
u=pd.read_csv(S/'data/uo_determining_windows.csv').set_index('stay_id');j=n.join(u);high=j.high_renal_uo_only==1
checks['uo_determining_equal_scored_min']=bool(np.allclose(j.loc[eligible,'uo_24hr_min'],j.loc[eligible,'determining_uo']))
recomputed['high_renal_uo_only_determining_only_single_interval_n']=int((j.loc[high,'determining_max_intervals']==1).sum())
assert recomputed['high_renal_uo_only_determining_only_single_interval_n']==103
assert all(checks.values())
out={'scope':'Independent read of extracted local aggregate-score files; no database queries','input_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'checks':checks,'recomputed':recomputed,'caution':'One interval means two observed timestamps with assumed collection coverage. This is not proof the entire clinical renal score would decrease if sparse windows were excluded; tied alternative low-output windows could retain the component.'}
(H/'strict_score_independent_qa.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
