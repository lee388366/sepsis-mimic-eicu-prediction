from pathlib import Path
import pandas as pd, numpy as np, json, hashlib, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
SRC=HERE.parent/'dag_gcs_mimic_interpretation'
UP=HERE.parent/'dag_gcs_next_stage/output'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
manifest=json.loads((SRC/'provenance.json').read_text())
checks=[]
for key in ['upstream_delivery_readonly','new_extraction_this_stage','code','outputs']:
 for r in manifest[key]:
  checks.append({'file':r['path'],'match':sha(Path(r['path']))==r['sha256']})
assert all(r['match'] for r in checks)
d=pd.read_csv(SRC/'base/mimic_gcs_timebase.csv',low_memory=False)
c=pd.read_csv(UP/'mimic_00_cohort_index.csv')
assert not d.duplicated(['stay_id','gcs_charttime']).any()
assert d.gcs_time_h.ge(0).all() and d.gcs_time_h.lt(48).all()
cl=c.set_index('stay_id')['label']
assert d.label_hospital_mortality.eq(d.stay_id.map(cl)).all()
facts={'hash_files_checked':len(checks),'shape':list(d.shape),'cohort_stays':len(c),'base_stays':int(d.stay_id.nunique()),'no_gcs_stays':int(len(set(c.stay_id)-set(d.stay_id))),'interval_metadata_ended':{}}
for pre,a in [('sedative_iv','sedative_iv_active'),('opioid_iv','opioid_iv_active'),('nmb_iv','nmb_iv_active'),('vent','vent_active_any')]:
 bad=(d[a]==1)&(d[pre+'_active_end_gap_h']>0)
 facts['interval_metadata_ended'][pre]={'rows':int(bad.sum()),'stays':int(d.loc[bad,'stay_id'].nunique()),'active_rows':int(d[a].sum())}
facts['total15_below_sum_rows']=int(((d.gcs_total_recorded==15)&(d.gcs_total_minus_component_sum>0)).sum())
facts['gcs_total_rules']=d.gcs_total_rule.value_counts().to_dict()
valid=(d.gcs_eyes_origin=='measured_current')&(d.gcs_motor_origin=='measured_current')&d.gcs_eyes_recorded.isin(range(1,5))&d.gcs_motor_recorded.isin(range(1,7))
a=d.loc[valid].sort_values(['stay_id','gcs_charttime']).drop_duplicates('stay_id').copy()
a['stratum']=np.select([a.gcs_verbal_origin.eq('measured_current_ett_zero'),a.gcs_verbal_origin.eq('measured_current')&a.gcs_verbal_recorded.isin(range(1,6))],['Current ETT code','Current numeric verbal'],default='Other verbal origin')
facts['selected_stays']=len(a)
facts['selected_subjects']=int(a.subject_id.nunique())
facts['additional_no_current_EM']=int(d.stay_id.nunique()-len(a))
facts['selected_time_h_quantiles']=a.gcs_time_h.quantile([0,.25,.5,.75,.95,1]).to_dict()
rows=[]
for st,s in a.groupby('stratum'):
 r={'stratum':st,'stays':len(s),'subjects':s.subject_id.nunique(),'deaths':int(s.label_hospital_mortality.sum()),'mortality':s.label_hospital_mortality.mean()}
 for col in ['gcs_eyes_recorded','gcs_motor_recorded']:
  for v in sorted(s[col].unique()):r[f'{col}_{int(v)}_n']=int(s[col].eq(v).sum())
 for lim,label in [(1,'prior1h'),(0,'same_time')]:
  z=s.loc[s.rass_gap_h.between(0,lim)&s.rass_rass_observed.isin(range(-5,5)),'rass_rass_observed']
  r['rass_'+label+'_n']=len(z)
  r['rass_'+label+'_median']=z.median()
  for v in range(-5,5):r[f'rass_{label}_{v}_n']=int(z.eq(v).sum())
 for v,lim in [('mbp',1),('resp_rate',1),('temperature',1),('ph',2)]:
  z=s.loc[s[f'phys_{v}_gap_h'].between(0,lim),f'phys_{v}_value'].dropna()
  r[f'{v}_n']=len(z)
  for q,t in [(.25,'q25'),(.5,'median'),(.75,'q75')]:r[f'{v}_{t}']=z.quantile(q)
 rows.append(r)
summary=pd.DataFrame(rows);summary.to_csv(HERE/'first_current_EM_summary.csv',index=False)
cts=[]
for lim,label in [(1,'prior1h'),(0,'same_time')]:
 z=a[a.rass_gap_h.between(0,lim)&a.rass_rass_observed.isin(range(-5,5))]
 for field in ['gcs_eyes_recorded','gcs_motor_recorded']:
  r=z.groupby(['stratum',field,'rass_rass_observed']).size().reset_index(name='stays').rename(columns={field:'component_score'})
  r['component']=field;r['rass_window']=label;cts.append(r)
pd.concat(cts).to_csv(HERE/'EM_RASS_counts.csv',index=False)
# Subject cluster bootstrap; retain all admissions within a resampled subject.
subjects,si=np.unique(a.subject_id,return_inverse=True)
groups=[(st,e) for st in sorted(a.stratum.unique()) for e in range(1,5)]
N=np.zeros((len(subjects),len(groups)));D=N.copy()
for j,(st,e) in enumerate(groups):
 m=(a.stratum==st)&(a.gcs_eyes_recorded==e)
 np.add.at(N[:,j],si[m],1);np.add.at(D[:,j],si[m],a.loc[m,'label_hospital_mortality'].to_numpy())
rng=np.random.default_rng(20260913);bs=[]
for b in range(1000):
 ids=rng.integers(0,len(subjects),len(subjects));nn=N[ids].sum(axis=0)
 bs.append(np.divide(D[ids].sum(axis=0),nn,out=np.full(len(groups),np.nan),where=nn>0))
bs=np.array(bs);mort=[]
for j,(st,e) in enumerate(groups):
 n=int(N[:,j].sum());deaths=int(D[:,j].sum())
 lo,hi=np.nanquantile(bs[:,j],[.025,.975]) if n>=20 else (np.nan,np.nan)
 mort.append(dict(stratum=st,eye=e,stays=n,deaths=deaths,mortality=deaths/n if n>=20 else np.nan,ci_low=lo,ci_high=hi))
mort=pd.DataFrame(mort);mort.to_csv(HERE/'eye_mortality_subject_bootstrap.csv',index=False)
# Figure: per-eye score RASS distribution; each stay occurs once.
fig,axs=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
for ax,st in zip(axs,['Current ETT code','Current numeric verbal']):
 z=a[(a.stratum==st)&a.rass_gap_h.between(0,1)&a.rass_rass_observed.isin(range(-5,5))]
 ct=pd.crosstab(z.gcs_eyes_recorded,z.rass_rass_observed).reindex(index=range(1,5),columns=range(-5,5),fill_value=0)
 pct=ct.div(ct.sum(axis=1),axis=0)*100
 im=ax.imshow(pct,aspect='auto',vmin=0,vmax=100,cmap='Blues')
 ax.set_xticks(range(10),range(-5,5));ax.set_yticks(range(4),[f'E{e} (n={ct.loc[e].sum():,})' for e in range(1,5)])
 ax.set_xlabel('RASS at same time or within preceding 1 h');ax.set_title(st)
 for i in range(4):
  for j in range(10):
   val=pct.iloc[i,j]
   if np.isfinite(val):ax.text(j,i,f'{val:.0f}',ha='center',va='center',fontsize=8,color='white' if val>55 else 'black')
fig.colorbar(im,ax=axs,label='Within-eye-score distribution (%)',shrink=.8)
fig.suptitle('First current eye + motor record per ICU stay\nDescriptive concordance; not a sedation effect',fontsize=13)
fig.savefig(HERE/'EM_RASS.png',dpi=170);fig.savefig(HERE/'EM_RASS.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
for st,offset,color in [('Current ETT code',-.06,'#c66a2b'),('Current numeric verbal',.06,'#286b9c')]:
 z=mort[(mort.stratum==st)&(mort.stays>=20)]
 ax.errorbar(z.eye+offset,z.mortality*100,yerr=np.vstack([(z.mortality-z.ci_low)*100,(z.ci_high-z.mortality)*100]),fmt='o-',capsize=4,label=st,color=color)
ax.set_xticks(range(1,5),['E1','E2','E3','E4']);ax.set_xlabel('Eye score at first current eye + motor record')
ax.set_ylabel('In-hospital mortality (%)');ax.legend();ax.set_title('Unadjusted association, cohort requiring ICU stay >=48 h\n95% subject-cluster bootstrap intervals; no causal interpretation')
fig.savefig(HERE/'eye_mortality.png',dpi=170);fig.savefig(HERE/'eye_mortality.pdf');plt.close(fig)
facts['environment']={'python':sys.version,'pandas':pd.__version__,'numpy':np.__version__}
facts['hash_checks']=checks
(HERE/'verified_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in facts.items() if k not in ['hash_checks','environment']},ensure_ascii=False,indent=2))
print(summary[['stratum','stays','rass_prior1h_n','rass_same_time_n','rass_prior1h_median','mbp_median','ph_n']].to_string(index=False))
print(mort.to_string(index=False))
