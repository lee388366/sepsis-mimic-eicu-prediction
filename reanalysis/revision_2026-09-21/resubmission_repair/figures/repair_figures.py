#!/usr/bin/env python3
"""Display-only repair from frozen A-variant arrays. No fitting or resampling.
Writes exclusively to this script's folder. Existing formal figures remain unchanged.
"""
from pathlib import Path
import hashlib,json,shutil,importlib.util
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.metrics import roc_curve,roc_auc_score
from PIL import Image
HERE=Path(__file__).resolve().parent
if (HERE/'sofa_roc_repair_manifest.json').exists():
 raise SystemExit('Blocked: this historical display-repair script would restore obsolete SOFA ROC curves and overwrite their manifest. Use repair_sofa_roc.py with explicit rebuilt input; isolate any further S7/S8/S20-only changes.')
ROOT=HERE.parents[2]
SRC=ROOT/'revision_2026-09-13/landmark_trainonly_reanalysis'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
manifest={'scope':'Display only, frozen arrays; no model training, no bootstrap, no manuscript writes','sources':{},'originals':{},'outputs':{},'checks':{}}
def read(p):manifest['sources'][str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)]=sha(p);return p
names=['fig18_shap_bar.png','fig19_shap_beeswarm.png','fig2_roc_combined.png','fig8a_rf_cic_internal.png','fig8b_rf_cic_external.png']
for n in names:
 p=ROOT/'figs'/n;b=HERE/'before'/'figs'/n;b.parent.mkdir(parents=True,exist_ok=True)
 if not b.exists():shutil.copy2(p,b)
 manifest['originals'][str(p.relative_to(ROOT))]=sha(p)
pdf=ROOT/'previous_submission/02_独立图件/Figure_2.pdf';b=HERE/'before/Figure_2.pdf'
if not b.exists():shutil.copy2(pdf,b)
manifest['originals'][str(pdf.relative_to(ROOT))]=sha(pdf)
p=np.load(read(SRC/'data/preds_A.npz'),allow_pickle=True)
e=np.load(read(SRC/'data/eicu_probs_A.npz'),allow_pickle=True)
s=np.load(read(SRC/'data/shap_values_A.npz'),allow_pickle=True)
top=pd.read_csv(read(SRC/'tables/A/shap_top_features.csv')).head(20)
te=p['split'].astype(str)=='test';y=p['y_true'][te];ye=e['y_true']
assert len(y)==3742 and y.sum()==712 and len(ye)==6812 and ye.sum()==1178
assert np.array_equal(s['stay_id'],p['stay_id'][te])
sv=s['shap'];nm=s['feature_names'];idx=np.array([np.flatnonzero(nm==f)[0] for f in top.feature])
assert np.allclose(np.abs(sv[:,idx]).mean(0),top.mean_abs_shap,atol=1e-7)
base=p['prob_RandomForest'][te]-sv.sum(1)
assert np.ptp(base)<1e-6
manifest['checks']['shap_probability_additivity_base_range']=[float(base.min()),float(base.max())]
plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':11,'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight'})
def save(fig,n,**kw):
 fig.savefig(HERE/n,**kw);manifest['outputs'][n]={'sha256':sha(HERE/n)}
 if n.endswith('.png'):manifest['outputs'][n]['pixels']=list(Image.open(HERE/n).size)

# S20A: unchanged source order/values, output-probability units.
fig,ax=plt.subplots(figsize=(7,6.6))
pos=np.arange(20)[::-1]
ax.barh(pos,top.mean_abs_shap,color='#1F77B4',alpha=.85)
ax.set_yticks(pos,top.feature,fontsize=8.5)
ax.set_xlabel('Mean |SHAP value| (predicted probability)')
ax.set_title('RandomForest SHAP importance, internal test set (cohort A)')
ax.grid(alpha=.3,axis='x')
assert [t.get_text() for t in ax.get_yticklabels()]==top.feature.tolist()
save(fig,'fig18_shap_bar.png');plt.close(fig)

# S20B: retain each original point and seed-42 jitter, align its y-axis label.
k=15;Xs=s['X_std'][:,idx[:k]];Sv=sv[:,idx[:k]]
# Narrow the data panel to fit the new colour bar within the old image footprint.
fig,ax=plt.subplots(figsize=(6.582,6.6));rng=np.random.default_rng(42)
rows=[]
for r in range(k):
 v=Sv[:,r];c=Xs[:,r];c=(c-np.nanmin(c))/(np.nanmax(c)-np.nanmin(c)+1e-9)
 ys=np.full(len(v),k-1-r)+rng.normal(0,.08,len(v))
 scatter=ax.scatter(v,ys,c=c,cmap='coolwarm',vmin=0,vmax=1,s=3,alpha=.5,linewidths=0)
 assert np.allclose(scatter.get_offsets()[:,0],sv[:,idx[r]])
 rows.append({'row_from_top':r+1,'label':str(top.feature.iloc[r]),'y_center':k-1-r,'shap_values_sha256':arrsha(v),'colour_feature':str(nm[idx[r]])})
ax.set_yticks(np.arange(k)[::-1],top.feature.iloc[:k],fontsize=8.5)
assert [t.get_text() for t in ax.get_yticklabels()]==top.feature.iloc[:k].tolist()
ax.axvline(0,color='k',lw=1)
ax.set_xlabel('SHAP value (change in predicted probability)')
ax.set_title('SHAP summary, RandomForest (cohort A)');ax.grid(alpha=.3,axis='x')
cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='coolwarm'),ax=ax,fraction=.033,pad=.035)
cb.set_ticks([0,1],labels=['Low','High']);cb.set_label('Feature value (within each feature)',fontsize=9)
save(fig,'fig19_shap_beeswarm.png');plt.close(fig)
manifest['checks']['S20A_displayed_features']=20;manifest['checks']['S20B_displayed_features']=15
manifest['checks']['S20B_point_label_bindings']=rows

# Main Figure 2: use original curve algorithm, order, colours and layout.
spec=importlib.util.spec_from_file_location('orig_s10',SRC/'scripts/s10_figures.py');orig=importlib.util.module_from_spec(spec);spec.loader.exec_module(orig)
read(SRC/'scripts/s10_figures.py')
P={m:p['prob_'+m][te].astype(float) for m in orig.MODELS}
PE={m:e['prob_'+m].astype(float) for m in orig.MODELS}
PE.update({'SOFA':e['sofa'].astype(float),'qSOFA-6h':e['qsofa6'].astype(float)})
for filename,key,model in [('sofa_predictions.npz','sofa','SOFA'),('qsofa_predictions.npz','qsofa6','qSOFA-6h')]:
 z=np.load(read(ROOT.parent/filename));ids=z['stay_id'].astype(np.int64);look={int(v):i for i,v in enumerate(ids)}
 P[model]=z[key][[look[int(a)] for a in p['stay_id'][te]]].astype(float)
order_i=sorted(P,key=lambda m:-roc_auc_score(y,P[m]));order_e=sorted(PE,key=lambda m:-roc_auc_score(ye,PE[m]))
traces=[]
with orig.print_profile():
 fig,axes=plt.subplots(1,2,figsize=orig.PRINT_FIG_2PANEL)
 for panel,ax,(yy,PP,mm,title) in zip(['A','B'],axes,[(y,P,order_i,f'A  Internal test set (cohort A, n={len(y)})'),(ye,PE,order_e,f'B  eICU exploratory evaluation (n={len(ye)})')]):
  for m in mm:
   fpr,tpr,_=roc_curve(yy,PP[m]);ls='--' if m in ['SOFA','qSOFA-6h'] else '-'
   line,=ax.plot(fpr,tpr,color=orig.COLORS[m],lw=1.6,ls=ls,label=f'{m} (AUC={orig.f3(roc_auc_score(yy,PP[m]))})')
   traces.append({'panel':panel,'model':m,'n':len(yy),'auc':float(roc_auc_score(yy,PP[m])),'coordinates_sha256':arrsha(np.column_stack([line.get_xdata(),line.get_ydata()]))})
  ax.plot([0,1],[0,1],'k--',lw=1);ax.set_xlabel('1 − Specificity');ax.set_ylabel('Sensitivity');ax.set_title(title);ax.grid(alpha=.3)
 h,l=axes[0].get_legend_handles_labels()
 oldlegend=fig.legend(h,l,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.04))
 fig.subplots_adjust(bottom=.34,top=.88,wspace=.25)
 fig.canvas.draw();old_bbox=fig.get_tightbbox(fig.canvas.get_renderer()).padded(.1)
 oldlegend.remove()
 newlegend=fig.legend(h,order_i,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.04))
 assert all('AUC' not in t.get_text() for t in newlegend.get_texts())
 # The explicit former bounding box preserves export dimensions despite shorter legend.
 save(fig,'fig2_roc_combined.png',bbox_inches=old_bbox)
 save(fig,'Figure_2.pdf',bbox_inches=old_bbox,metadata={'CreationDate':None,'ModDate':None})
 plt.close(fig)
manifest['checks']['Figure2_all_18_curve_coordinate_hashes']=traces
manifest['checks']['Figure2_shared_legend']=order_i

# S7/S8: existing 99 thresholds/counts and styles, a clearer shaded-area label only.
plt.rcdefaults();plt.rcParams.update({'font.size':11,'savefig.dpi':300})
for z,mask,n,title in [(p,te,'fig8a_rf_cic_internal.png','Internal test set'),(e,np.ones(len(ye),bool),'fig8b_rf_cic_external.png','eICU-CRD')]:
 prob=z['prob_RandomForest'][mask];yy=z['y_true'][mask];t=np.linspace(.01,.99,99)
 flagged=np.array([(prob>=v).sum() for v in t]);events=np.array([((prob>=v)&(yy==1)).sum() for v in t])
 assert np.all(events<=flagged)
 fig,ax=plt.subplots(figsize=(8,6))
 ax.plot(t,flagged,color='#d62728',lw=2.2,label='Flagged records')
 ax.plot(t,events,color='#1f77b4',lw=2.2,label='Flagged records with in-hospital death')
 ax.fill_between(t,events,flagged,color='gray',alpha=.15,label='Flagged records without in-hospital death')
 ax.set(xlabel='Threshold probability',ylabel='Number of records',xlim=(0,1),ylim=(0,flagged.max()*1.05),title=f'Random forest clinical impact curve\n{title} (n={len(yy):,} records)')
 ax.legend(fontsize=9);ax.grid(alpha=.25);fig.tight_layout()
 save(fig,n,bbox_inches='tight');plt.close(fig)
 manifest['checks'][n]={'n':len(yy),'deaths':int(yy.sum()),'thresholds':99,'count_curve_sha256':arrsha(np.column_stack([t,flagged,events]))}
# Read-only source / original preservation verification.
for rel,h in manifest['originals'].items():assert sha(ROOT/rel)==h
for rel,h in manifest['sources'].items():assert sha(ROOT/rel if not Path(rel).is_absolute() else rel)==h
manifest['checks']['all_original_formal_figures_and_sources_unchanged']=True
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'outputs':manifest['outputs'],'checks':'passed'},indent=2))
