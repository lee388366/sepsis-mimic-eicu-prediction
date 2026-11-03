from pathlib import Path
import numpy as np,pandas as pd,json,hashlib,argparse,math
from scipy.stats import rankdata
ap=argparse.ArgumentParser();ap.add_argument('score');ap.add_argument('results');a=ap.parse_args();B=Path('revision_2026-09-21/resubmission_repair');R=Path('revision_2026-09-13/landmark_trainonly_reanalysis');P=Path(a.results)
with np.load(R/'data/eicu_probs_A.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
s=pd.read_csv(a.score,dtype={'uniquepid':str}).set_index('patientunitstayid');ids=d['stay_id'];assert s.index.is_unique and set(s.index)==set(ids);s=s.loc[ids];y=d['y_true'];uid=d['uniquepid'].astype(str);assert np.array_equal(y,s.label) and np.array_equal(uid,s.uniquepid)
components=['respiration','coagulation','liver','cardiovascular','cns','renal']
base=s[components].fillna(0).sum(axis=1)
assert np.array_equal(base.to_numpy(),s.sofa_max_48h.to_numpy())
assert np.array_equal((base-s.respiration.fillna(0)+s.respiration_1h.fillna(0)).to_numpy(),s.sofa_support_1h.to_numpy())
assert np.array_equal((base-s.renal.fillna(0)+s.renal_uo_conservative.fillna(0)).to_numpy(),s.sofa_uo_conservative.to_numpy())
models=['LSTM','RandomForest','SVM','LogisticRegression','GaussianNB','KNN','DecisionTree'];scores={m:d['prob_'+m] for m in models};scores['SOFA']=s.sofa_max_48h.to_numpy();scores['qsofa6']=d['qsofa6']
def auc(y,p):
 n1=y.sum();n0=len(y)-n1;return float((rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*n0))
def aprec(y,p):
 ix=np.argsort(-p,kind='stable');ss=p[ix];yy=y[ix];ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(yy)-1];tp=np.cumsum(yy)[ends];return float(np.sum(np.diff(np.r_[0,tp])*(tp/(ends+1)))/yy.sum())
def draws(uid,n):
 order=list(dict.fromkeys(uid));loc={u:[] for u in order}
 for j,u in enumerate(uid):loc[u].append(j)
 arr=[np.array(loc[u],dtype=np.int64) for u in order];rng=np.random.default_rng(42)
 for _ in range(n):yield np.concatenate([arr[j] for j in rng.integers(0,len(arr),len(arr))])
boot={m:[] for m in scores};bap={m:[] for m in scores}
for ix in draws(uid,1000):
 for m,p in scores.items():boot[m].append(auc(y[ix],p[ix]));bap[m].append(aprec(y[ix],p[ix]))
boot={m:np.array(v) for m,v in boot.items()};table=pd.read_csv(P/'overall.csv').set_index('Model');out={}
for m,p in scores.items():
 q=np.percentile(boot[m],[2.5,97.5]);aq=np.percentile(bap[m],[2.5,97.5]);v={'AUC':auc(y,p),'AUC_CI_low':q[0],'AUC_CI_high':q[1],'AP':aprec(y,p),'AP_CI_low':aq[0],'AP_CI_high':aq[1]}
 for k,x in v.items():assert abs(x-table.loc[m,k])<1e-12,(m,k,x,table.loc[m,k])
 out[m]=v
ops=pd.read_csv(P/'operating_points.csv').set_index('cutoff');counts=[]
for cut in [6,8,10,12]:
 p=scores['SOFA']>=cut;v={'tn':int(((y==0)&~p).sum()),'fp':int(((y==0)&p).sum()),'fn':int(((y==1)&~p).sum()),'tp':int(((y==1)&p).sum())};v['sensitivity']=v['tp']/1178;v['specificity']=v['tn']/5634
 for k,x in v.items():assert abs(x-ops.loc[cut,k])<1e-12,(cut,k)
 counts.append(dict(cutoff=cut,**v))
e=json.loads((P/'evaluation.json').read_text());pairs={}
for m in ['LSTM','RandomForest']:
 q=np.percentile(boot[m]-boot['SOFA'],[2.5,97.5]);v=[out[m]['AUC']-out['SOFA']['AUC'],*q];expected=e['paired_comparisons'][m]
 for k,x in zip(['difference','patient_clustered_CI_low','patient_clustered_CI_high'],v):assert abs(x-expected[k])<1e-12
 pairs[m]=v
old=pd.read_csv(R/'tables/A/table9_external_validation.csv').set_index('Model')
for m in models:
 for k,oldk in [('AUC','AUC_ext'),('AUC_CI_low','AUC_CI_low_patient'),('AUC_CI_high','AUC_CI_high_patient')]:assert abs(round(out[m][k],4)-old.loc[m,oldk])<1e-12
sens={};st=pd.read_csv(P/'source_rule_sensitivity.csv').set_index('variant')
for col in ['sofa_support_1h','sofa_uo_conservative']:
 pp=s[col].to_numpy();v=np.array([auc(y[ix],pp[ix]) for ix in draws(uid,1000)]);q=np.percentile(v,[2.5,97.5]);dq=np.percentile(v-boot['SOFA'],[2.5,97.5]);sv={'AUC':auc(y,pp),'AUC_CI_low':q[0],'AUC_CI_high':q[1],'AUC_difference_from_primary':auc(y,pp)-out['SOFA']['AUC'],'difference_CI_low':dq[0],'difference_CI_high':dq[1],'changed_records':int((pp!=scores['SOFA']).sum())}
 for k,x in sv.items():assert abs(x-st.loc[col,k])<1e-12,(col,k,x,st.loc[col,k])
 sens[col]=sv
sub=[];demo=pd.read_csv(Path.cwd().parent/'eicu_demographics.csv').set_index('patientunitstayid').loc[ids];sg=pd.read_csv(P/'subgroup_sofa_full_precision.csv')
for _,row in sg.iterrows():
 col='gender' if row.Dimension=='Gender' else 'race_group';mask=demo[col].to_numpy()==row.Subgroup;sy=y[mask];sp=scores['SOFA'][mask];su=uid[mask];v=[auc(sy[ix],sp[ix]) for ix in draws(su,300) if len(np.unique(sy[ix]))==2];q=np.percentile(v,[2.5,97.5]);sv={'AUC':auc(sy,sp),'AUC_CI_low':q[0],'AUC_CI_high':q[1]}
 for k,x in sv.items():assert abs(x-row[k])<1e-12,(row.Subgroup,k)
 sub.append(dict(Dimension=row.Dimension,Subgroup=row.Subgroup,**sv))
result=dict(status='independently verified actual output',score_sha256=hashlib.sha256(Path(a.score).read_bytes()).hexdigest(),n=len(y),patients=len(set(uid)),deaths=int(y.sum()),bootstrap='independent rank AUROC/AP, first-seen cluster order, seed42,1000 percentile draws',overall=out,operating_points=counts,paired_differences=pairs,source_rule_sensitivities=sens,subgroups=sub,frozen_seven_models_reproduce_original_table=True)
internal=pd.read_csv(B/'score_evaluation/results/overall.csv').set_index('Model');delta=out['SOFA']['AUC']-internal.loc['SOFA','AUC'];assert abs(delta-e['new_sofa_auc_minus_corrected_internal'])<1e-12
oldsub=pd.read_csv(R/'tables/A/table9b_subgroup_external.csv');newsub=pd.read_csv(P/'table9b_subgroup_external.csv');cols=[c for c in oldsub if not c.startswith('SOFA_')];pd.testing.assert_frame_equal(oldsub[cols],newsub[cols],check_exact=True)
with np.load(P/'plot_inputs.npz',allow_pickle=False) as plot:
 for m in models:assert np.array_equal(d['prob_'+m],plot[m])
 assert np.array_equal(d['qsofa6'],plot['qsofa6']) and np.array_equal(d['stay_id'],plot['stay_id']) and np.array_equal(d['y_true'],plot['y_true']) and np.array_equal(uid,plot['uniquepid'].astype(str))
for model in ['LSTM','RandomForest']:
 aa=[];bb=[];m=int(y.sum());n=len(y)-m
 for name in [model,'SOFA']:
  pos=scores[name][y==1];neg=scores[name][y==0];sn=np.sort(neg);sp=np.sort(pos)
  aa.append((np.searchsorted(sn,pos,'left')+np.searchsorted(sn,pos,'right'))/(2*n));bb.append(1-(np.searchsorted(sp,neg,'left')+np.searchsorted(sp,neg,'right'))/(2*m))
 cov=np.cov(aa)/m+np.cov(bb)/n;var=cov[0,0]+cov[1,1]-2*cov[0,1];diff=np.mean(aa[0])-np.mean(aa[1]);pv=math.erfc(abs(diff)/np.sqrt(var)/np.sqrt(2));expected=e['paired_comparisons'][model]['record_level_DeLong_p'];assert math.isclose(pv,expected,rel_tol=1e-10,abs_tol=1e-30);pairs[model].append(pv)
for _,r in ops.iterrows():
 for key,num,den in [('ppv',r.tp,r.tp+r.fp),('npv',r.tn,r.tn+r.fn)]:assert abs(r[key]-num/den)<1e-12
 assert r.flagged==r.tp+r.fp
result.update(new_external_minus_corrected_internal=delta,non_SOFA_subgroup_columns_exactly_preserved=True,plot_inputs_frozen_predictions_exactly_preserved=True,evaluator_sha256=hashlib.sha256((P.parent/'evaluate_eicu_sofa.py').read_bytes()).hexdigest(),record_level_DeLong_p_independently_verified=True,ppv_npv_and_flagged_verified=True)
(B/'eicu_score_evaluation/independent_new_results.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['score_sha256','n','patients','deaths','operating_points','paired_differences']},indent=2));print('SOFA',out['SOFA'])
