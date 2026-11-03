#!/usr/bin/env python3
"""Prepare provenance, then replace SOFA ROC from independently evaluated local input.
No fitting/resampling/database access. Outputs only beside this script.
Run --prepare-only before the corrected input exists. Drawing requires --draw.
Drawing now requires --eicu-sofa rebuilt and an explicitly frozen eICU input hash.
Historical keep/omit modes require --allow-historical-intermediate and cannot be released.
"""
from pathlib import Path
from zipfile import ZipFile
import argparse, hashlib, importlib.util, json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
H=Path(__file__).resolve().parent
ROOT=H.parents[2]
R=ROOT/'revision_2026-09-13/landmark_trainonly_reanalysis'
INPUT=H.parent/'score_evaluation/results/plot_inputs.npz'
MODELS=['LSTM','RandomForest','SVM','LogisticRegression','GaussianNB','KNN','DecisionTree']
DISP={'RandomForest':'RF','LogisticRegression':'LR','DecisionTree':'DT'}
RC={'font.size':11,'axes.titlesize':12,'axes.labelsize':11,'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def trace(y,p):
 f,t,_=roc_curve(y,p)
 return {'auc':float(roc_auc_score(y,p)),'coordinates_sha256':arrsha(np.column_stack([f,t]))}
def rebuilt_eicu(e,PE,path,expected_hash):
 assert path is not None and path.exists(),'Frozen eICU plot input required'
 assert expected_hash and sha(path)==expected_hash,'eICU input hash is missing or mismatched'
 z=np.load(path,allow_pickle=False)
 assert np.array_equal(z['stay_id'],e['stay_id']),'eICU stay order mismatch'
 assert np.array_equal(z['uniquepid'].astype(str),e['uniquepid'].astype(str))
 assert np.array_equal(z['y_true'],e['y_true']),'eICU outcome mismatch'
 assert len(np.unique(z['stay_id']))==6812
 for m in MODELS:assert np.array_equal(z[m],e['prob_'+m]),m+' differs from frozen eICU probability'
 assert np.array_equal(z['qsofa6'],e['qsofa6'])
 score=z['SOFA'].astype(float)
 assert score.shape==e['y_true'].shape and np.isfinite(score).all()
 assert ((score>=0)&(score<=24)&(score==np.floor(score))).all()
 out={m:v.copy() for m,v in PE.items()};out['SOFA']=score
 for m in MODELS+['qSOFA-6h']:assert trace(e['y_true'],out[m])==trace(e['y_true'],PE[m])
 return out,{'path':str(path),'sha256':expected_hash,'old_sofa_auc':trace(e['y_true'],PE['SOFA'])['auc'],'new_sofa_auc':trace(e['y_true'],score)['auc'],'seven_ml_and_qsofa_exactly_frozen':True}
def load_source():
 p=np.load(R/'data/preds_A.npz',allow_pickle=False);e=np.load(R/'data/eicu_probs_A.npz',allow_pickle=False)
 mask=p['split'].astype(str)=='test';ids=p['stay_id'][mask];y=p['y_true'][mask]
 P={m:p['prob_'+m][mask].astype(float) for m in MODELS}
 for fn,key,m in [('sofa_predictions.npz','sofa','SOFA'),('qsofa_predictions.npz','qsofa6','qSOFA-6h')]:
  z=np.load(ROOT.parent/fn,allow_pickle=False);ix={int(v):i for i,v in enumerate(z['stay_id'])}
  P[m]=z[key][[ix[int(v)] for v in ids]].astype(float)
 PE={m:e['prob_'+m].astype(float) for m in MODELS}
 PE.update(SOFA=e['sofa'].astype(float));PE['qSOFA-6h']=e['qsofa6'].astype(float)
 assert len(y)==3742 and y.sum()==712 and len(e['y_true'])==6812
 return p,e,ids,y,P,PE
spec=importlib.util.spec_from_file_location('frozen_s10',R/'scripts/s10_figures.py')
orig=importlib.util.module_from_spec(spec);spec.loader.exec_module(orig)
def s12(y,P,path,include_scores):
 plt.rcdefaults();plt.rcParams.update(RC)
 fig,ax=plt.subplots(figsize=(6.4,5.6))
 if include_scores:ax.plot([0,.70,.90,1],[0,.70,.90,1],color='gray',ls=(0,(5,4)),lw=1)
 for m in MODELS+(['SOFA','qSOFA-6h'] if include_scores else []):
  f,t,_=roc_curve(y,P[m]);a=roc_auc_score(y,P[m])
  ax.plot(f,t,color=orig.COLORS[m],lw=2.6 if m=='LSTM' else 1.9,label=f'{DISP.get(m,m)} (AUC = {a:.3f})')
 kw={'label':'Random (AUC = 0.500)'} if include_scores else {}
 ax.plot([0,1],[0,1],'k--',lw=1,**kw)
 ax.set_title(f'Internal test set (n={len(y):,}) — '+('ROC curve by comparator' if include_scores else 'ROC curves by model comparator'))
 ax.set_xlabel('False positive rate');ax.set_ylabel('True positive rate')
 ax.legend(loc='lower right',fontsize=9,framealpha=.9);ax.grid(alpha=.3)
 fig.tight_layout();fig.savefig(path);plt.close(fig)
def prepare(p,e,ids,y,P,PE):
 original_names=['fig2_roc_combined.png','fig16b_roc_internal_sofa.png','fig16c_roc_internal_models_only.png']
 originals={};embedding={}
 for n in original_names:
  src=ROOT/'figs'/n;backup=H/'before/figs'/n;backup.parent.mkdir(parents=True,exist_ok=True)
  if not backup.exists():shutil.copy2(src,backup)
  assert sha(src)==sha(backup),'Formal figure changed since frozen backup: '+n
  originals['figs/'+n]=sha(backup)
 docs=['previous_submission/03_补充材料_待核对/Supplementary_Information.docx','Sepsis_Mortality_Supplementary_Information.docx','previous_submission/08_退回补正_20260921/Main_Manuscript_corrected_20260921.docx']
 for name in docs:
  with ZipFile(ROOT/name) as z:
   hits=[]
   for member in z.namelist():
    if member.startswith('word/media/'):
     h=hashlib.sha256(z.read(member)).hexdigest()
     for source,value in originals.items():
      if h==value:hits.append({'member':member,'source':source,'original_sha256':h})
   embedding[name]={'docx_sha256':sha(ROOT/name),'matches':hits}
 # Reproduce S12B directly from the seven frozen ML probabilities; no score used.
 output=H/'qa_S12B_frozen_reproduction.png';s12(y,P,output,False)
 old=np.asarray(Image.open(ROOT/'figs/fig16c_roc_internal_models_only.png').convert('RGBA'))
 new=np.asarray(Image.open(output).convert('RGBA'))
 pixel_equal=old.shape==new.shape and np.array_equal(old,new)
 assert pixel_equal,'S12B reproduction pixel mismatch; inspect before drawing'
 audit={'status':'PREPARED; corrected SOFA plots not drawn','originals':originals,'embedding':embedding,
 'sources':{str(x.relative_to(ROOT)) if x.is_relative_to(ROOT) else str(x):sha(x) for x in [R/'data/preds_A.npz',R/'data/eicu_probs_A.npz',ROOT.parent/'qsofa_predictions.npz',ROOT.parent/'sofa_predictions.npz',R/'scripts/s10_figures.py',R/'scripts/s25_regenerate_figures.py']},
 'S12B':{'n':len(y),'deaths':int(y.sum()),'models':MODELS,'pixel_equal_to_actual_formal_png':pixel_equal,'formal_png_sha256':sha(ROOT/'figs/fig16c_roc_internal_models_only.png'),'reproduction_sha256':sha(output),'traces':{m:trace(y,P[m]) for m in MODELS}},
 'Figure2B':{'n':len(e['y_true']),'deaths':int(e['y_true'].sum()),'source':'data/eicu_probs_A.npz; frozen ML probabilities and qsofa6; historical sofa is baseline only','default':'drawing requires explicit rebuilt eICU plot input path and frozen SHA256','traces':{m:trace(e['y_true'],PE[m]) for m in PE}},
 'input_contract':{'path':str(INPUT),'required_keys':['stay_id','subject_id','y_true','LSTM','RF','SOFA','qsofa6'],'checks':'stay_id and labels exactly match frozen test order; LSTM/RF/qsofa6 exactly match frozen values; 7 ML probabilities read from preds_A; SOFA integer 0..24; input file hash recorded'},
 'script_sha256':sha(Path(__file__))}
 # Preparation does not change integration mappings before an output exists.
 write(H/'sofa_roc_preflight.json',audit)
 return audit

def draw(p,e,ids,y,P,PE,audit,args):
 eicu_input=None
 if args.eicu_sofa=='rebuilt':PE,eicu_input=rebuilt_eicu(e,PE,args.eicu_input,args.eicu_input_sha256)
 else:assert args.allow_historical_intermediate,'Historical eICU SOFA is blocked: supply frozen rebuilt input, or explicitly mark historical intermediate'
 assert args.input.exists(),'Corrected plot_inputs.npz not available; no ROC figure written'
 input_hash=sha(args.input)
 z=np.load(args.input,allow_pickle=False)
 assert np.array_equal(z['stay_id'],ids) and np.array_equal(z['y_true'],y)
 mask=p['split'].astype(str)=='test'
 assert np.array_equal(z['subject_id'],p['subject_id'][mask])
 for key,m in [('LSTM','LSTM'),('RF','RandomForest'),('qsofa6','qSOFA-6h')]:assert np.array_equal(z[key],P[m]),key+' differs from frozen source'
 newsofa=z['SOFA'].astype(float)
 assert len(newsofa)==len(y) and np.isfinite(newsofa).all() and ((newsofa>=0)&(newsofa<=24)&(newsofa==np.floor(newsofa))).all()
 oldP={k:v.copy() for k,v in P.items()};P['SOFA']=newsofa
 for m in MODELS+['qSOFA-6h']:assert trace(y,P[m])==trace(y,oldP[m])
 # Preserve prior isolated repaired figures before replacing them.
 for n in ['fig2_roc_combined.png','Figure_2.pdf']:
  target=H/n;bk=H/'before_strict_sofa'/n;bk.parent.mkdir(exist_ok=True)
  if target.exists() and not bk.exists():shutil.copy2(target,bk)
 order_i=sorted(oldP,key=lambda m:-roc_auc_score(y,oldP[m]))
 order_e=sorted(PE,key=lambda m:-roc_auc_score(e['y_true'],PE[m]))
 with orig.print_profile():
  fig,axes=plt.subplots(1,2,figsize=orig.PRINT_FIG_2PANEL)
  for panel,ax,yy,probs,order,title in [('A',axes[0],y,P,order_i,f'A  Internal test set (cohort A, n={len(y)})'),('B',axes[1],e['y_true'],PE,order_e,f'B  eICU exploratory evaluation (n={len(e["y_true"])})')]:
   for m in order:
    if panel=='B' and m=='SOFA' and args.eicu_sofa=='omit':continue
    f,t,_=roc_curve(yy,probs[m]);ax.plot(f,t,color=orig.COLORS[m],lw=1.6,ls='--' if m in ['SOFA','qSOFA-6h'] else '-',label=m)
   ax.plot([0,1],[0,1],'k--',lw=1);ax.set_xlabel('1 − Specificity');ax.set_ylabel('Sensitivity');ax.set_title(title);ax.grid(alpha=.3)
  h,_=axes[0].get_legend_handles_labels()
  oldlegend=fig.legend(h,[f'{m} (AUC={orig.f3(roc_auc_score(y,oldP[m]))})' for m in order_i],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.04))
  fig.subplots_adjust(bottom=.34,top=.88,wspace=.25);fig.canvas.draw();bbox=fig.get_tightbbox(fig.canvas.get_renderer()).padded(.1);oldlegend.remove()
  fig.legend(h,order_i,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.04))
  fig.savefig(H/'fig2_roc_combined.png',bbox_inches=bbox)
  fig.savefig(H/'Figure_2.pdf',bbox_inches=bbox,metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
 s12(y,P,H/'fig16b_roc_internal_sofa.png',True)
 checks={'internal_traces':{m:trace(y,P[m]) for m in P},'eicu_traces':{m:trace(e['y_true'],PE[m]) for m in PE if not (m=='SOFA' and args.eicu_sofa=='omit')},'seven_ml_and_qsofa_coordinates_unchanged':True,'S12B_formal_untouched':sha(ROOT/'figs/fig16c_roc_internal_models_only.png')==audit['originals']['figs/fig16c_roc_internal_models_only.png']}
 if args.eicu_sofa=='keep':
  old=np.asarray(Image.open(H/'before_strict_sofa/fig2_roc_combined.png'));new=np.asarray(Image.open(H/'fig2_roc_combined.png'))
  checks['Figure2B_data_panel_pixel_equal']=old.shape==new.shape and np.array_equal(old[:900,old.shape[1]//2:],new[:900,new.shape[1]//2:])
  assert checks['Figure2B_data_panel_pixel_equal']
 outputs={n:{'sha256':sha(H/n),'pixels':list(Image.open(H/n).size) if n.endswith('.png') else None} for n in ['fig2_roc_combined.png','Figure_2.pdf','fig16b_roc_internal_sofa.png']}
 manpath=H/'manifest.json';man=json.loads(manpath.read_text());saved=dict(man['originals'])
 man['originals']['figs/fig16b_roc_internal_sofa.png']=audit['originals']['figs/fig16b_roc_internal_sofa.png']
 assert all(man['originals'][k]==v for k,v in saved.items())
 assert sha(args.input)==input_hash
 for source,value in audit['originals'].items():assert sha(ROOT/source)==value
 man['outputs'].update(outputs);man['checks']['strict48_roc_repair']=checks
 man['checks']['Figure2_all_18_curve_coordinate_hashes']=[dict(panel=panel,model=m,n=len(y) if panel=='A' else len(e['y_true']),**entry) for panel,rows in [('A',checks['internal_traces']),('B',checks['eicu_traces'])] for m,entry in rows.items()]
 man['checks']['Figure2_shared_legend']=order_i
 man['checks']['strict48_roc_input']={'path':str(args.input),'sha256':sha(args.input),'eicu_sofa':args.eicu_sofa,'eicu_rebuilt_input':eicu_input}
 if eicu_input:assert sha(args.eicu_input)==eicu_input['sha256']
 write(manpath,man)
 write(H/'sofa_roc_repair_manifest.json',{'status':'REBUILT SCORES DRAWN; visual and integration QA pending' if eicu_input else 'HISTORICAL INTERMEDIATE; not cleared for final release','input':man['checks']['strict48_roc_input'],'originals':man['originals'],'outputs':outputs,'checks':checks,'script_sha256':sha(Path(__file__))})
 print(json.dumps({'outputs':outputs,'checks':checks},indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
 group.add_argument('--prepare-only',action='store_true');group.add_argument('--draw',action='store_true')
 parser.add_argument('--input',type=Path,default=INPUT);parser.add_argument('--eicu-sofa',choices=['rebuilt','keep','omit'],default='rebuilt')
 parser.add_argument('--eicu-input',type=Path);parser.add_argument('--eicu-input-sha256')
 parser.add_argument('--allow-historical-intermediate',action='store_true');args=parser.parse_args()
 if args.draw and args.eicu_sofa=='rebuilt':
  assert args.eicu_input and args.eicu_input_sha256,'Explicit frozen eICU input path and SHA256 required before any drawing'
  assert args.eicu_input.exists() and sha(args.eicu_input)==args.eicu_input_sha256,'Frozen eICU input does not exist or its hash mismatches'
 if args.draw and args.eicu_sofa!='rebuilt':assert args.allow_historical_intermediate,'Historical comparator drawing blocked'
 p,e,ids,y,P,PE=load_source();audit=prepare(p,e,ids,y,P,PE)
 if args.draw:draw(p,e,ids,y,P,PE,audit,args)
 else:print('Prepared; no corrected SOFA figures drawn. See sofa_roc_preflight.json')
