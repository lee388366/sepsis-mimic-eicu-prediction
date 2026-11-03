"""Evaluate reconstructed eICU SOFA against frozen predictions; no model fitting.
Patient clustered bootstrap follows the original algorithm, 1000 draws, seed 42.
Legacy mode reproduces original score results without claiming score validity.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix
from scipy.stats import norm
B=Path(__file__).resolve().parent; REPAIR=B.parent; ROOT=REPAIR.parents[1]
R=ROOT/'revision_2026-09-13/landmark_trainonly_reanalysis'
sys.path.insert(0,str(R/'scripts'));import common_eval as C
COMP=['respiration','coagulation','liver','cardiovascular','cns','renal']
MODELS=['LSTM','RandomForest','SVM','LogisticRegression','GaussianNB','KNN','DecisionTree']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ci(v):return [float(x) for x in np.percentile(v,[2.5,97.5])]
def external_subgroups(ids,uid,y,sofa,legacy):
 # Preserve the original subgroup definitions and 300 patient-bootstrap draws.
 # Only the SOFA columns change; this source table is not the LSTM/RF-only S19.
 original=pd.read_csv(R/'tables/A/table9b_subgroup_external.csv')
 demo=pd.read_csv(ROOT.parent/'eicu_demographics.csv')
 assert demo.patientunitstayid.is_unique
 dm=dict(zip(demo.patientunitstayid.astype(np.int64),zip(demo.gender,demo.race_group)))
 assert set(map(int,ids)).issubset(dm)
 updated=original.copy();details=[];seen=[]
 for dim,pos in [('Gender',0),('Race',1)]:
  arr=np.array([dm[int(s)][pos] for s in ids])
  for group in sorted(set(arr.tolist())):
   sel=arr==group
   if sel.sum()<100:continue
   ix=original.index[(original.Dimension==dim)&(original.Subgroup==group)]
   assert len(ix)==1,(dim,group)
   j=ix[0];row=original.loc[j];sy=y[sel];sp=sofa[sel];su=uid[sel]
   assert int(row.n)==int(sel.sum()) and int(row.n_patients)==len(np.unique(su))
   assert abs(float(row.Death_rate)-round(float(sy.mean()),4))<1e-12
   draws=C.cluster_batches(su,300,42)
   vals=[roc_auc_score(sy[k],sp[k]) for k in draws if len(np.unique(sy[k]))>1]
   low,high=ci(vals);auc=float(roc_auc_score(sy,sp))
   for col,v in [('SOFA_AUC',auc),('SOFA_CI_low_patient',low),('SOFA_CI_high_patient',high)]:
    if legacy:assert abs(round(v,4)-float(row[col]))<1e-12,(dim,group,col,v,row[col])
    updated.at[j,col]=round(v,4)
   details.append(dict(Dimension=dim,Subgroup=group,n=int(sel.sum()),n_patients=len(np.unique(su)),deaths=int(sy.sum()),AUC=auc,AUC_CI_low=low,AUC_CI_high=high,valid_bootstrap_draws=len(vals)))
   seen.append(j)
 assert sorted(seen)==list(original.index)
 unchanged=[c for c in original if not c.startswith('SOFA_')]
 pd.testing.assert_frame_equal(original[unchanged],updated[unchanged],check_exact=True)
 return updated,pd.DataFrame(details)
def run(score_path,output,legacy=False):
 inputs=[score_path,R/'data/eicu_probs_A.npz',R/'data/trainonly_eicu.npz',REPAIR/'score_evaluation/results/evaluation.json',R/'scripts/common_eval.py',Path(__file__),REPAIR/'score_evaluation/results/overall.csv']
 inputs.extend([ROOT.parent/'eicu_demographics.csv',R/'tables/A/table9b_subgroup_external.csv'])
 before={str(p):sha(p) for p in inputs}
 with np.load(inputs[1],allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 with np.load(inputs[2],allow_pickle=False) as z:tr={k:z[k].copy() for k in ['stay_id','uniquepid','y','sofa_max_48h','qsofa6']}
 ids=d['stay_id'];y=d['y_true'];uid=d['uniquepid'].astype(str)
 assert len(ids)==6812 and len(np.unique(ids))==6812 and y.sum()==1178 and len(np.unique(uid))==6488
 assert np.array_equal(ids,tr['stay_id']) and np.array_equal(y,tr['y']) and np.array_equal(uid,tr['uniquepid'].astype(str))
 assert np.array_equal(d['sofa'],tr['sofa_max_48h']) and np.array_equal(d['qsofa6'],tr['qsofa6'])
 s=pd.read_csv(score_path,dtype={'uniquepid':str})
 if legacy:s=s[s.label.notna()].copy()
 assert s.patientunitstayid.is_unique and len(s)==6812 and set(s.patientunitstayid)==set(ids)
 s=s.set_index('patientunitstayid').loc[ids]
 assert np.array_equal(s.label.to_numpy(int),y)
 if not legacy:assert np.array_equal(s.uniquepid.to_numpy(str),uid)
 for c in COMP:
  v=s[c].dropna().to_numpy();assert np.isfinite(v).all() and ((v>=0)&(v<=4)&(v==np.floor(v))).all()
 sofa=s.sofa_max_48h.to_numpy(float);assert np.isfinite(sofa).all() and np.array_equal(s[COMP].fillna(0).sum(axis=1).to_numpy(),sofa)
 boundary=['lab_outside_n','pf_pair_outside_n','uo_outside_n','vaso_outside_n','gcs_outside_n','map_outside_n','support_outside_n'] if not legacy else []
 if not legacy:
  assert set(boundary).issubset(s.columns),'Complete source-window QA columns required'
  assert s[boundary].notna().all().all() and (s[boundary]==0).all().all()
 scores={m:d['prob_'+m].astype(float) for m in MODELS};scores.update(SOFA=sofa,qsofa6=d['qsofa6'].astype(float))
 sensitivity_scores={}
 if not legacy:
  for col in ['sofa_support_1h','sofa_uo_conservative']:
   assert col in s.columns,col
   v=s[col].to_numpy(float)
   assert np.isfinite(v).all() and ((v>=0)&(v<=24)&(v==np.floor(v))).all()
   primary_component,alternate_component=('respiration','respiration_1h') if col=='sofa_support_1h' else ('renal','renal_uo_conservative')
   assert alternate_component in s.columns
   alternate=s[alternate_component].dropna().to_numpy(float)
   assert np.isfinite(alternate).all() and ((alternate>=0)&(alternate<=4)&(alternate==np.floor(alternate))).all()
   assert np.array_equal(v,sofa-s[primary_component].fillna(0).to_numpy(float)+s[alternate_component].fillna(0).to_numpy(float))
   sensitivity_scores[col]=v
 batches=C.cluster_batches(uid,1000,42);boot={k:[] for k in scores};boot_ap={k:[] for k in scores}
 for ix in batches:
  assert len(np.unique(y[ix]))==2
  for k,p in scores.items():boot[k].append(roc_auc_score(y[ix],p[ix]));boot_ap[k].append(average_precision_score(y[ix],p[ix]))
 boot={k:np.array(v) for k,v in boot.items()}
 rows=[]
 for k,p in scores.items():
  ac=ci(boot[k]);apc=ci(boot_ap[k]);rows.append(dict(Model=k,n_records=len(y),n_patients=len(np.unique(uid)),deaths=int(y.sum()),AUC=roc_auc_score(y,p),AUC_CI_low=ac[0],AUC_CI_high=ac[1],AP=average_precision_score(y,p),AP_CI_low=apc[0],AP_CI_high=apc[1]))
 overall=pd.DataFrame(rows)
 sensitivity_rows=[]
 for k,p in sensitivity_scores.items():
  draws=[roc_auc_score(y[ix],p[ix]) for ix in batches];low,high=ci(draws)
  dl_low,dl_high=ci(np.asarray(draws)-boot['SOFA'])
  sensitivity_rows.append(dict(variant=k,n_records=len(y),changed_records=int((p!=sofa).sum()),AUC=float(roc_auc_score(y,p)),AUC_CI_low=low,AUC_CI_high=high,AUC_difference_from_primary=float(roc_auc_score(y,p)-roc_auc_score(y,sofa)),difference_CI_low=dl_low,difference_CI_high=dl_high))
 if legacy:
  assert np.array_equal(sofa,d['sofa'])
  old=pd.read_csv(R/'tables/A/table9_external_validation.csv');rr=old[old.Model.str.startswith('SOFA')].iloc[0];ss=overall[overall.Model=='SOFA'].iloc[0]
  for a,b in [('AUC','AUC_ext'),('AUC_CI_low','AUC_CI_low_patient'),('AUC_CI_high','AUC_CI_high_patient')]:assert abs(ss[a]-rr[b])<=0.000051,(a,ss[a],rr[b])
 dl={}
 for k in ['LSTM','RandomForest']:
  a,cov=C.delong_auc_cov(y,np.vstack([scores[k],sofa]));var=float(cov[0,0]+cov[1,1]-2*cov[0,1]);assert var>0
  z=float((a[0]-a[1])/np.sqrt(var));dif=boot[k]-boot['SOFA'];q=ci(dif)
  dl[k]={'difference':float(a[0]-a[1]),'patient_clustered_CI_low':q[0],'patient_clustered_CI_high':q[1],'record_level_DeLong_p':float(2*norm.sf(abs(z)))}
 ops=[]
 for cut in [6,8,10,12]:
  tn,fp,fn,tp=map(int,confusion_matrix(y,sofa>=cut,labels=[0,1]).ravel());ops.append(dict(cutoff=cut,tn=tn,fp=fp,fn=fn,tp=tp,sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),ppv=tp/(tp+fp) if tp+fp else None,npv=tn/(tn+fn) if tn+fn else None,flagged=tp+fp))
 internal=pd.read_csv(REPAIR/'score_evaluation/results/overall.csv');internal_auc=float(internal[internal.Model=='SOFA'].iloc[0].AUC)
 internal_json=json.loads(inputs[3].read_text());internal_json_auc=float(next(r['AUC'] for r in internal_json['overall'] if r['Model']=='SOFA'));assert abs(internal_auc-internal_json_auc)<1e-12
 delta=float(roc_auc_score(y,sofa)-internal_auc)
 out={'status':'legacy reproduction; invalid historical implementation' if legacy else 'corrected-score evaluation; source extraction review required separately','n':len(y),'patients':len(np.unique(uid)),'deaths':int(y.sum()),'bootstrap':{'cluster':'uniquepid','replicates':1000,'seed':42,'method':'percentile; original cluster ordering and sampling algorithm'},'source_window_qa':boundary,'missing_components':{c:int(s[c].isna().sum()) for c in COMP},'sofa_changed_records':int((sofa!=d['sofa']).sum()),'old_sofa_auc':float(roc_auc_score(y,d['sofa'])),'new_sofa_auc':float(roc_auc_score(y,sofa)),'new_sofa_auc_minus_corrected_internal':delta,'paired_comparisons':dl,'input_sha256':before,'core_model_predictions_unchanged':True}
 subgroup,subgroup_detail=external_subgroups(ids,uid,y,sofa,legacy)
 out['external_subgroups']={'dimensions':['Gender','Race'],'rows':len(subgroup),'bootstrap_replicates':300,'seed':42,'non_SOFA_columns_unchanged':True}
 out['source_rule_sensitivities']=sensitivity_rows
 output.mkdir(parents=True,exist_ok=True);overall.to_csv(output/'overall.csv',index=False);pd.DataFrame(ops).to_csv(output/'operating_points.csv',index=False)
 subgroup.to_csv(output/'table9b_subgroup_external.csv',index=False);subgroup_detail.to_csv(output/'subgroup_sofa_full_precision.csv',index=False)
 if sensitivity_rows:pd.DataFrame(sensitivity_rows).to_csv(output/'source_rule_sensitivity.csv',index=False)
 np.savez_compressed(output/'plot_inputs.npz',stay_id=ids,uniquepid=uid,y_true=y,SOFA=sofa,qsofa6=d['qsofa6'],**{m:d['prob_'+m] for m in MODELS})
 for p in inputs:assert sha(p)==before[str(p)],'Input changed during evaluation'
 (output/'evaluation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
 print(json.dumps({k:out[k] for k in ['status','n','patients','deaths','sofa_changed_records','old_sofa_auc','new_sofa_auc','new_sofa_auc_minus_corrected_internal']},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--score',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--legacy-baseline',action='store_true');a=ap.parse_args();run(a.score,a.output,a.legacy_baseline)
