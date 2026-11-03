from pathlib import Path
import ast, hashlib, json
import numpy as np
import pandas as pd
from scipy.stats import norm
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PARENT=ROOT.parent
R=ROOT/'revision_2026-09-13/landmark_trainonly_reanalysis'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
# Extract only the pure DeLong covariance function, without importing training or evaluation pipelines.
pure_source=R/'scripts/common_eval.py'
mod=ast.parse(pure_source.read_text())
fn=next(x for x in mod.body if isinstance(x,ast.FunctionDef) and x.name=='delong_auc_cov')
ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(pure_source),'exec'),ns)
z=np.load(R/'data/preds_A.npz',allow_pickle=False)
mask=z['split']=='test'; yy=z['y_true'][mask]; ids=z['stay_id'][mask]
s=np.load(PARENT/'sofa_predictions.npz',allow_pickle=False)
q=np.load(PARENT/'qsofa_predictions.npz',allow_pickle=False)
checks={}
for label,zz in [('sofa',s),('qsofa',q)]:
 checks[label+'_same_ids_order']=bool(np.array_equal(ids,zz['stay_id']))
 checks[label+'_same_outcomes']=bool(np.array_equal(yy,zz['y_true']))
 assert checks[label+'_same_ids_order'] and checks[label+'_same_outcomes']
pa=z['prob_LSTM'][mask]
results=[]
for name,pb in [('RF',z['prob_RandomForest'][mask]),('SVM',z['prob_SVM'][mask]),('LR',z['prob_LogisticRegression'][mask]),('SOFA',s['sofa']),('qSOFA6',q['qsofa6'])]:
 auc,cov=ns['delong_auc_cov'](yy,np.vstack([pa,pb]));v=cov[0,0]+cov[1,1]-2*cov[0,1];zz=(auc[0]-auc[1])/np.sqrt(v)
 results.append(dict(comparator=name,auc_lstm=float(auc[0]),auc_comparator=float(auc[1]),difference=float(auc[0]-auc[1]),z=float(zz),p_two_sided_stable=float(2*norm.sf(abs(zz))),p_original_cdf=float(2*(1-norm.cdf(abs(zz))))))
# Existing score-cache checks only; no raw clinical records read.
comp=['respiration','coagulation','liver','cardiovascular','cns','renal']
m=pd.read_csv(PARENT/'sofa_extract.csv',usecols=['stay_id','sofa_max_48h']+comp)
e=pd.read_csv(PARENT/'eicu_sofa.csv')
score_stats={}
for label,df in [('mimic',m),('eicu',e)]:
 score_stats[label]={'n_cache':len(df),'sum_components_equal_total':bool((df[comp].sum(axis=1)==df['sofa_max_48h']).all()),'total_range':[float(df.sofa_max_48h.min()),float(df.sofa_max_48h.max())],'component_value_counts':{c:{str(k):int(v) for k,v in df[c].value_counts(dropna=False).sort_index().items()} for c in comp}}
score_stats['eicu']['label_missing_n']=int(e.label.isna().sum())
ea=e[e.label.notna()].copy()
ez=np.load(R/'data/trainonly_eicu.npz',allow_pickle=False)
ex=ea.set_index('patientunitstayid').loc[ez['stay_id']]
checks['eicu_cache_ids_unique']=bool(e.patientunitstayid.is_unique)
checks['eicu_trainonly_score_equal_current_csv']=bool(np.array_equal(ex.sofa_max_48h.to_numpy(),ez['sofa_max_48h']))
checks['eicu_trainonly_qsofa_equal_current_csv']=bool(np.array_equal(ex.qsofa6.to_numpy(),ez['qsofa6']))
checks['eicu_trainonly_outcomes_equal_current_csv']=bool(np.array_equal(ex.label.to_numpy(),ez['y']))
mm=m.set_index('stay_id').loc[ids]
checks['mimic_test_sofa_equal_current_csv']=bool(np.array_equal(mm.sofa_max_48h.to_numpy(),s['sofa']))
ops=[]
for cut in [6,8,10,12]:
 pred=s['sofa']>=cut;tp=int(np.sum(pred & (yy==1)));fp=int(np.sum(pred & (yy==0)));tn=int(np.sum(~pred &(yy==0)));fnn=int(np.sum(~pred &(yy==1)))
 ops.append(dict(cutoff=cut,tp=tp,fp=fp,tn=tn,fn=fnn,sensitivity=tp/(tp+fnn),specificity=tn/(tn+fp),unflagged_fraction=float((~pred).mean())))
files=[pure_source,R/'data/preds_A.npz',PARENT/'sofa_predictions.npz',PARENT/'qsofa_predictions.npz',PARENT/'sofa_extract.csv',PARENT/'eicu_sofa.csv',PARENT/'04b_extract_sofa.sql',PARENT/'eicu_03_sofa_qsofa.sql',PARENT/'eicu_03_sofa_qsofa_copy.sql',R/'data/trainonly_eicu.npz']
out={'scope':'deterministic frozen-score and prediction checks; no training, no raw clinical extraction, no bootstrap','n_test':int(mask.sum()),'deaths_test':int(yy.sum()),'checks':checks,'delong_record_level_supplementary':results,'score_cache_summaries':score_stats,'sofa_operating_counts':ops,'source_sha256':{str(f):sha(f) for f in files}}
(HERE/'frozen_score_checks.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
print(json.dumps({'checks':checks,'delong':results,'score_cache_summaries':score_stats,'ops':ops},indent=2))
