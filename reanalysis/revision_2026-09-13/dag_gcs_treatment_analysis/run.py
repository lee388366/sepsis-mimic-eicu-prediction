from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
P=Path(__file__).resolve().parent;R=P.parent
v1=R/'dag_gcs_mimic_interpretation/base/mimic_gcs_timebase.csv';v2=R/'dag_gcs_mimic_interpretation_v2/base/mimic_gcs_timebase.csv'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(v1)=='121d6ec433c9c6262c7ccd3b0b878e914fffeff944d35495fb1d7aa1deb9c384'
assert sha(v2)=='000531dc6b48766f966d235f6d754477da6d7a40d675f299c23adf130f62294e'
o=pd.read_csv(v1,low_memory=False);d=pd.read_csv(v2,low_memory=False)
assert d.row_key.equals(o.row_key)
changes={c:int((~(d[c].eq(o[c])|(d[c].isna()&o[c].isna()))).sum()) for c in o}
changes={k:v for k,v in changes.items() if v}
assert len(changes)==26
needed=[c for c in o if c.startswith(('rass','phys','gcs_')) and c not in ['gcs_total_used_default_for_missing','gcs_total_used_carryforward']]+['row_key','subject_id','stay_id','label_hospital_mortality']
assert not (set(needed)&set(changes))
for pre in ['sedative_iv','opioid_iv','nmb_iv']:
 m=d[pre+'_active']==1
 assert d.loc[m,pre+'_active_start_gap_h'].ge(0).all() and d.loc[m,pre+'_active_end_gap_h'].le(0).all()
 assert d.loc[m,pre+'_n_covering'].ge(1).all()
orig=d[['gcs_eyes_origin','gcs_motor_origin','gcs_verbal_origin']]
plain=d.gcs_total_rule.eq('plain_sum');prev=d.gcs_total_rule.eq('prev_ett_recompute')
exp=(plain&orig.eq('missing').any(axis=1))|(prev&~orig.eq('measured_current').all(axis=1))
assert d.gcs_total_used_default_for_missing.eq(exp.astype(int)).all()
exp=plain&orig.apply(lambda x:x.str.startswith('carried')).any(axis=1)
assert d.gcs_total_used_carryforward.eq(exp.astype(int)).all()
valid=d.gcs_eyes_origin.eq('measured_current')&d.gcs_motor_origin.eq('measured_current')&d.gcs_eyes_recorded.isin(range(1,5))&d.gcs_motor_recorded.isin(range(1,7))
a=d.loc[valid].sort_values(['stay_id','gcs_charttime']).drop_duplicates('stay_id').copy()
a['language']=np.select([a.gcs_verbal_origin.eq('measured_current_ett_zero'),a.gcs_verbal_origin.eq('measured_current')&a.gcs_verbal_recorded.isin(range(1,6))],['ETT','numeric'],default='other')
rows=[]
for win in [4,1]:
 admin=a.sedative_emar_admin_gap_h.between(0,win)
 iv=a.sedative_iv_active.eq(1)
 unconfirmed=a.sedative_emar_other_unconfirmed_gap_h.between(0,win)|a.sedative_order_unconfirmed_gap_h.between(0,win)
 a['treatment']=np.select([admin,iv,unconfirmed],['A_recent_admin','B_interval_only','C_unconfirmed_record'],default='D_status_unknown')
 for (lang,tr),s in a.groupby(['language','treatment']):
  z=s.loc[s.rass_gap_h.between(0,1),'rass_rass_observed'].dropna();z0=s.loc[s.rass_gap_h.eq(0),'rass_rass_observed'].dropna()
  rec=dict(window_h=win,language=lang,treatment=tr,stays=len(s),subjects=s.subject_id.nunique(),rass_n=len(z),rass_median=z.median(),rass_q25=z.quantile(.25),rass_q75=z.quantile(.75),rass_same_n=len(z0),rass_same_median=z0.median(),eye1_n=int(s.gcs_eyes_recorded.eq(1).sum()),motor1_n=int(s.gcs_motor_recorded.eq(1).sum()),invasive_interval_n=int(s.vent_active_invasive.sum()),sedative_not_given_n=int(s.sedative_emar_not_given_gap_h.between(0,win).sum()))
  for drug in ['sedative','opioid','nmb']:
   rec[drug+'_admin_n']=int(s[drug+'_emar_admin_gap_h'].between(0,win).sum())
   rec[drug+'_interval_n']=int(s[drug+'_iv_active'].sum())
   rec[drug+'_overlap_intervals_n']=int(s[drug+'_iv_n_covering'].gt(1).sum())
   rec[drug+'_distinct_drugs_gt1_n']=int(s[drug+'_iv_active_all_drugs'].fillna('').str.split(' | ',regex=False).str.len().gt(1).sum())
  rows.append(rec)
t=pd.DataFrame(rows);t.to_csv(P/'treatment_context_summary.csv',index=False)
# First-analysis counts and medians remain exactly identical.
prevsum=pd.read_csv(R/'dag_gcs_interpretation_analysis/first_current_EM_summary.csv')
for lang,name in [('ETT','Current ETT code'),('numeric','Current numeric verbal'),('other','Other verbal origin')]:
 s=a[a.language==lang];r=prevsum[prevsum.stratum==name].iloc[0];z=s.loc[s.rass_gap_h.between(0,1),'rass_rass_observed']
 assert len(s)==r.stays and z.notna().sum()==r.rass_prior1h_n and z.median()==r.rass_prior1h_median
facts={'v1_sha':sha(v1),'v2_sha':sha(v2),'v2_shape':list(d.shape),'changed_columns':changes,'new_columns':sorted(set(d)-set(o)),'first_analysis_stays':len(a),'first_result_verified':True,'all_selected_interval_bounds_valid':True,'rule_flags_verified':True,'python_pandas':pd.__version__}
(P/'verification.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2))
print(t.loc[(t.window_h==4)&(t.language!='other'),['language','treatment','stays','rass_n','rass_median','rass_same_n','rass_same_median','eye1_n','motor1_n','sedative_admin_n','sedative_interval_n','sedative_overlap_intervals_n']].to_string(index=False))
print('One hour:');print(t.loc[(t.window_h==1)&(t.language!='other'),['language','treatment','stays','rass_n','rass_median']].to_string(index=False))
