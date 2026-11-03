#!/usr/bin/env python3
"""Fixed-window retrospective recording trajectories; no model fitting or raw mutation."""
from pathlib import Path
import os, sys, json, hashlib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REV = HERE.parent
UP = REV / 'dag_gcs_next_stage/output'
BASE = REV / 'dag_gcs_mimic_interpretation_v2/base/mimic_gcs_timebase.csv'
H = 3_600_000_000_000
CUTS = [6, 12, 24, 36, 48]
DRUGS = {'sedative':'sedative_hypnotic','opioid':'opioid','nmb':'neuromuscular_blocker'}
ADMIN = {'Administered','Administered Bolus from IV Drip','Administered in Other Location','Partial Administered'}
STATES = ['A','B','C','D']
CHANGES = ['higher','unchanged','lower','mixed']
GROUPS = ['R_to_R','R_to_U','U_to_R','U_to_U']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def ns(series):
    return pd.to_datetime(series).to_numpy().astype('datetime64[ns]').astype(np.int64)


def last_before(events, q_id, q_t, col):
    """Strictly earlier event, stable ordering. Does not skip invalid/missing latest values."""
    ev = events.sort_values(['stay_id',col],kind='stable').reset_index(drop=True)
    idx = np.full(len(q_id),-1,dtype=np.int64)
    if ev.empty: return ev, idx
    bounds = ev.groupby('stay_id',sort=False).indices
    qbounds = pd.Series(np.arange(len(q_id))).groupby(q_id,sort=False).apply(list)
    t = ev[col].to_numpy(dtype=np.int64)
    for sid, rows in qbounds.items():
        bb = bounds.get(sid)
        if bb is None: continue
        rows = np.asarray(rows,dtype=np.int64)
        j = np.searchsorted(t[bb],q_t[rows],side='left')-1
        good = j>=0
        idx[rows[good]] = bb[j[good]]
    return ev,idx


def event_age(events,q_id,q_t,col):
    ev, idx = last_before(events,q_id,q_t,col)
    age = np.full(len(q_id),np.nan)
    good = idx>=0
    if good.any(): age[good]=(q_t[good]-ev[col].to_numpy()[idx[good]])/H
    return age


def interval_summary(events,q_id,q_t):
    """Status immediately before cutoff: start < cutoff, end >= cutoff."""
    count=np.zeros(len(q_id),dtype=np.int32)
    distinct=np.zeros(len(q_id),dtype=np.int16)
    bounds=events.groupby('stay_id',sort=False).indices
    qbounds=pd.Series(np.arange(len(q_id))).groupby(q_id,sort=False).apply(list)
    start=events['_start'].to_numpy();end=events['_end'].to_numpy();labels=events['drug_label'].to_numpy()
    for sid,rows in qbounds.items():
        bb=bounds.get(sid)
        if bb is None:continue
        rows=np.asarray(rows,dtype=int)
        cover=(start[bb][None,:]<q_t[rows,None])&(end[bb][None,:]>=q_t[rows,None])
        count[rows]=cover.sum(axis=1)
        for k,row in enumerate(rows):
            if count[row]: distinct[row]=len(set(labels[bb][cover[k]].tolist()))
    return count,distinct


def classify(admin_age,iv,other_age,order_age,window):
    admin=np.isfinite(admin_age)&(admin_age>=0)&(admin_age<=window)
    unconfirmed=(np.isfinite(other_age)&(other_age>=0)&(other_age<=window))|(np.isfinite(order_age)&(order_age>=0)&(order_age<=window))
    return np.select([admin,np.asarray(iv)>0,unconfirmed],STATES[:3],default='D')


def build_grid():
    source_paths=[BASE,UP/'mimic_00_cohort_index.csv',UP/'mimic_03_drug_inputevents.csv',UP/'mimic_04_drug_emar.csv',UP/'mimic_05_drug_prescriptions.csv',UP/'mimic_09_rass_cam_icu.csv']
    hashes={str(p):sha(p) for p in source_paths}
    assert hashes[str(BASE)]=='000531dc6b48766f966d235f6d754477da6d7a40d675f299c23adf130f62294e'
    coh=pd.read_csv(UP/'mimic_00_cohort_index.csv',usecols=['stay_id','subject_id','intime','outtime','label']).sort_values('stay_id').reset_index(drop=True)
    assert len(coh)==24894 and not coh.stay_id.duplicated().any()
    coh['_intime']=ns(coh.intime);coh['_outtime']=ns(coh.outtime)
    assert ((coh._outtime-coh._intime)>=48*H).all()
    g=coh.loc[coh.index.repeat(len(CUTS)),['stay_id','subject_id','_intime','_outtime']].reset_index(drop=True)
    g['cut_h']=np.tile(CUTS,len(coh));g['_cut']=g._intime+g.cut_h*H
    qid=g.stay_id.to_numpy();qt=g._cut.to_numpy()
    d=pd.read_csv(BASE,low_memory=False)
    assert not d.duplicated(['stay_id','gcs_charttime']).any()
    good=d.gcs_eyes_origin.eq('measured_current')&d.gcs_motor_origin.eq('measured_current')&d.gcs_eyes_recorded.isin(range(1,5))&d.gcs_motor_recorded.isin(range(1,7))
    em=d.loc[good].copy();em['_t']=ns(em.gcs_charttime)
    em,idx=last_before(em,qid,qt,'_t')
    hit=idx>=0;safe=np.maximum(idx,0)
    age=np.where(hit,(qt-em['_t'].to_numpy()[safe])/H,np.nan)
    hit=hit&(age<=4)&(age>0)
    g['em_available']=hit;g['em_age_h']=np.where(hit,age,np.nan)
    g['_em_t']=np.where(hit,em['_t'].to_numpy()[safe],0)
    for out,col in [('E','gcs_eyes_recorded'),('M','gcs_motor_recorded')]:g[out]=np.where(hit,em[col].to_numpy()[safe],np.nan)
    origin=em.gcs_verbal_origin.to_numpy()[safe]
    verbal=em.gcs_verbal_recorded.to_numpy()[safe]
    lang=np.select([origin=='measured_current_ett_zero',(origin=='measured_current')&np.isin(verbal,range(1,6))],['ETT','numeric'],default='other')
    g['language']=np.where(hit,lang,'missing_EM')
    # Score-time treatment context is copied only for actual selected current E/M rows.
    for drug in DRUGS:
        for window in [4,1]:
            tx=classify(em[f'{drug}_emar_admin_gap_h'].to_numpy()[safe],em[f'{drug}_iv_active'].to_numpy()[safe],em[f'{drug}_emar_other_unconfirmed_gap_h'].to_numpy()[safe],em[f'{drug}_order_unconfirmed_gap_h'].to_numpy()[safe],window)
            g[f'{drug}_score_tx{window}']=np.where(hit,tx,'missing_EM')
    # Independent RASS cutoff selection.
    rass=pd.read_csv(UP/'mimic_09_rass_cam_icu.csv',usecols=['stay_id','charttime','assessment_field','valuenum'])
    rass=rass[rass.assessment_field.eq('rass_actual')].copy()
    assert not rass.duplicated(['stay_id','charttime']).any()
    assert rass.valuenum.isin(range(-5,5)).all()
    rass['_t']=ns(rass.charttime)
    ra,ri=last_before(rass,qid,qt,'_t');rh=ri>=0;rs=np.maximum(ri,0)
    rage=np.where(rh,(qt-ra['_t'].to_numpy()[rs])/H,np.nan)
    rh=rh&(rage>0)&(rage<=1)
    g['rass']=np.where(rh,ra.valuenum.to_numpy()[rs],np.nan)
    g['rass_age_h']=np.where(rh,rage,np.nan)
    g['em_rass_abs_gap_h']=np.where(hit&rh,np.abs(g.em_age_h-g.rass_age_h),np.nan)
    # Raw recording sources, independent of whether E/M is available at cutoff.
    ie=pd.read_csv(UP/'mimic_03_drug_inputevents.csv',usecols=['stay_id','starttime','endtime','drug_label','drug_class'])
    ie['_start']=ns(ie.starttime);ie['_end']=ns(ie.endtime)
    assert ie[['stay_id','starttime','endtime','drug_label','drug_class']].notna().all().all() and (ie._end>=ie._start).all()
    ev=pd.read_csv(UP/'mimic_04_drug_emar.csv',usecols=['stay_id','charttime','event_txt','drug_class'])
    ev=ev[ev.stay_id.notna()].copy();ev['stay_id']=ev.stay_id.astype(int);ev['_t']=ns(ev.charttime)
    pr=pd.read_csv(UP/'mimic_05_drug_prescriptions.csv',usecols=['stay_id','starttime','drug_class'])
    pr['_t']=ns(pr.starttime)
    for drug,cls in DRUGS.items():
        ii=ie[ie.drug_class.eq(cls)].reset_index(drop=True)
        counts,distinct=interval_summary(ii,qid,qt)
        g[f'{drug}_n_intervals']=counts;g[f'{drug}_n_labels']=distinct
        e=ev[ev.drug_class.eq(cls)]
        ag=event_age(e[e.event_txt.isin(ADMIN)],qid,qt,'_t')
        ng=event_age(e[e.event_txt.eq('Not Given')],qid,qt,'_t')
        og=event_age(e[~e.event_txt.isin(ADMIN|{'Not Given'})],qid,qt,'_t')
        pg=event_age(pr[pr.drug_class.eq(cls)],qid,qt,'_t')
        for win in [4,1]:
            g[f'{drug}_tx{win}']=classify(ag,counts,og,pg,win)
            g[f'{drug}_not_given{win}']=np.isfinite(ng)&(ng>0)&(ng<=win)
    assert len(g)==len(coh)*len(CUTS) and not g.duplicated(['stay_id','cut_h']).any()
    assert g.loc[g.em_available,'em_age_h'].between(0,4,inclusive='right').all()
    assert g.loc[g.rass.notna(),'rass_age_h'].between(0,1,inclusive='right').all()
    facts={'sources_sha256':hashes,'plan_sha256':sha(HERE/'PLAN.md'),'cohort_stays':len(coh),'cohort_subjects':coh.subject_id.nunique(),'grid_rows':len(g),'base_stays':d.stay_id.nunique(),'no_GCS_stays':len(set(coh.stay_id)-set(d.stay_id)),'rass_duplicate_keys':0,'inputevents_bad_intervals':0,'python':sys.version,'pandas':pd.__version__,'numpy':np.__version__}
    return g,facts


def response_change(a,b):
    de=b.E.to_numpy()-a.E.to_numpy();dm=b.M.to_numpy()-a.M.to_numpy()
    return np.select([(de>=0)&(dm>=0)&((de>0)|(dm>0)),(de==0)&(dm==0),(de<=0)&(dm<=0)&((de<0)|(dm<0))],CHANGES[:3],default='mixed')


def pair_data(g,start,end,fresh=4,drug_window=4,context='cutoff'):
    a=g[g.cut_h.eq(start)].set_index('stay_id').sort_index()
    b=g[g.cut_h.eq(end)].set_index('stay_id').sort_index()
    assert a.index.equals(b.index)
    ma=a.em_available&a.em_age_h.le(fresh);mb=b.em_available&b.em_age_h.le(fresh)
    m=ma&mb
    out=pd.DataFrame({'subject_id':a.subject_id,'start_present':ma,'end_present':mb},index=a.index)
    out['paired']=m
    out.loc[m,'change']=response_change(a.loc[m],b.loc[m])
    out['E0']=a.E.where(ma);out['M0']=a.M.where(ma);out['E1']=b.E.where(mb);out['M1']=b.M.where(mb)
    out['initial_ceiling']=(out.E0.eq(4)&out.M0.eq(6))
    for drug in DRUGS:
        c=f'{drug}_tx{drug_window}' if context=='cutoff' else f'{drug}_score_tx{drug_window}'
        out[drug+'0']=a[c];out[drug+'1']=b[c]
        r0=a[c].isin(['A','B']);r1=b[c].isin(['A','B'])
        out[drug+'_transition']=np.where(r0,'R','U')+'_to_'+np.where(r1,'R','U')
    out['lang0']=a.language.where(ma,'missing_EM');out['lang1']=b.language.where(mb,'missing_EM')
    out['rass0']=a.rass;out['rass1']=b.rass
    return out


def make_summaries(g):
    coverage=[];grades=[];treatment=[];agealign=[]
    complete=g.groupby('stay_id').em_available.all()
    complete_ids=set(complete[complete].index)
    for subset,s in [('all_cohort',g),('EM_all_5_cutoffs',g[g.stay_id.isin(complete_ids)])]:
        for cut,t in s.groupby('cut_h'):
            r={'subset':subset,'cut_h':cut,'stays':len(t),'em_n':int(t.em_available.sum()),'rass_n':int(t.rass.notna().sum()),'both_n':int((t.em_available&t.rass.notna()).sum()),'em_last2h_n':int(t.em_age_h.le(2).sum()),'ceiling_E4M6_n':int((t.E.eq(4)&t.M.eq(6)).sum())}
            for field in ['em_age_h','rass_age_h','em_rass_abs_gap_h']:
                z=t[field].dropna()
                for q,label in [(.25,'p25'),(.5,'median'),(.75,'p75'),(.95,'p95')]:r[field+'_'+label]=z.quantile(q)
            coverage.append(r)
            for comp,levels in [('E',range(1,5)),('M',range(1,7)),('rass',range(-5,5))]:
                for level in list(levels)+['missing']:
                    n=int(t[comp].isna().sum()) if level=='missing' else int(t[comp].eq(level).sum())
                    grades.append(dict(subset=subset,cut_h=cut,component=comp,grade=level,n=n,denominator=len(t)))
            if subset=='all_cohort':
                for drug in DRUGS:
                    for win in [4,1]:
                        for st in STATES:
                            treatment.append(dict(cut_h=cut,drug=drug,record_window_h=win,state=st,n=int(t[f'{drug}_tx{win}'].eq(st).sum()),denominator=len(t),overlapping_intervals_n=int(t[f'{drug}_n_intervals'].gt(1).sum()),different_labels_n=int(t[f'{drug}_n_labels'].gt(1).sum()),not_given_n=int(t[f'{drug}_not_given{win}'].sum())))
                        mm=t.em_available
                        agealign.append(dict(cut_h=cut,drug=drug,record_window_h=win,em_n=int(mm.sum()),state_different_at_EM_n=int((t.loc[mm,f'{drug}_tx{win}']!=t.loc[mm,f'{drug}_score_tx{win}']).sum())))
    pd.DataFrame(coverage).to_csv(HERE/'coverage_and_ages.csv',index=False)
    pd.DataFrame(grades).to_csv(HERE/'grade_distributions.csv',index=False)
    tx=pd.DataFrame(treatment)
    covercols=['overlapping_intervals_n','different_labels_n','not_given_n']
    tx[['cut_h','drug','record_window_h','denominator']+covercols].drop_duplicates().to_csv(HERE/'treatment_record_coverage.csv',index=False)
    tx.drop(columns=covercols).to_csv(HERE/'treatment_states.csv',index=False)
    pd.DataFrame(agealign).to_csv(HERE/'score_vs_cutoff_treatment.csv',index=False)
    return len(complete_ids)


def paired_outputs(g):
    summary=[];joint=[];txmatrix=[];missing=[];lmatrix=[];gradepairs=[]
    pairs=[(6,48),(6,12),(12,24),(24,36),(36,48)]
    configurations=[('primary',4,4,'cutoff'),('EM_2h',2,4,'cutoff'),('drug_1h',4,1,'cutoff'),('at_EM_time',4,4,'score')]
    for config,fresh,dwin,context in configurations:
        for start,end in pairs if config=='primary' else [(6,48)]:
            p=pair_data(g,start,end,fresh,dwin,context);paired=p[p.paired]
            meta={'analysis':config,'start_h':start,'end_h':end}
            for a in [False,True]:
                for b in [False,True]:missing.append({**meta,'start_available':a,'end_available':b,'n':int((p.start_present.eq(a)&p.end_present.eq(b)).sum()),'denominator':len(p)})
            z={**meta,'paired_stays':len(paired),'paired_subjects':paired.subject_id.nunique(),'initial_ceiling_n':int(paired.initial_ceiling.sum()),'rass_paired_n':int((paired.rass0.notna()&paired.rass1.notna()).sum())}
            z.update({c+'_n':int(paired.change.eq(c).sum()) for c in CHANGES});summary.append(z)
            for drug in DRUGS:
                for group in GROUPS:
                    t=paired[paired[drug+'_transition'].eq(group)]
                    r={**meta,'drug':drug,'record_transition':group,'stays':len(t),'subjects':t.subject_id.nunique(),'initial_ceiling_n':int(t.initial_ceiling.sum()),'initial_non_ceiling_n':int((~t.initial_ceiling).sum())}
                    r.update({c+'_n':int(t.change.eq(c).sum()) for c in CHANGES})
                    rr=t[t.rass0.notna()&t.rass1.notna()]
                    r['rass_both_n']=len(rr);r['rass0_median_paired']=rr.rass0.median();r['rass1_median_paired']=rr.rass1.median()
                    r['rass_grade_increased_n']=int((rr.rass1>rr.rass0).sum());r['rass_grade_decreased_n']=int((rr.rass1<rr.rass0).sum())
                    r['higher_fraction']=r['higher_n']/len(t) if len(t)>=20 else np.nan
                    r['higher_fraction_non_ceiling']=r['higher_n']/r['initial_non_ceiling_n'] if r['initial_non_ceiling_n']>=20 else np.nan
                    r['ci_low']=r['ci_high']=np.nan
                    joint.append(r)
                for subset,t in [('all_cohort',p),('EM_paired',paired)]:
                    if context=='score' and subset=='all_cohort':continue
                    for st0 in STATES:
                        for st1 in STATES:
                            txmatrix.append({**meta,'drug':drug,'subset':subset,'state0':st0,'state1':st1,'n':int((t[drug+'0'].eq(st0)&t[drug+'1'].eq(st1)).sum()),'denominator':len(t)})
            if config=='primary':
                for c in ['E','M']:
                    levels=range(1,5) if c=='E' else range(1,7)
                    for v0 in levels:
                        for v1 in levels:gradepairs.append({**meta,'component':c,'score0':v0,'score1':v1,'n':int((paired[c+'0'].eq(v0)&paired[c+'1'].eq(v1)).sum()),'denominator':len(paired)})
                for lang0 in ['ETT','numeric','other']:
                    for lang1 in ['ETT','numeric','other']:lmatrix.append({**meta,'origin0':lang0,'origin1':lang1,'n':int((paired.lang0.eq(lang0)&paired.lang1.eq(lang1)).sum()),'denominator':len(paired)})
    joint=pd.DataFrame(joint)
    # Primary endpoint, sedative recording groups: patient-cluster CI for grade-increase proportion.
    p=pair_data(g,6,48);p=p[p.paired]
    subj,idx=np.unique(p.subject_id,return_inverse=True)
    counts=np.zeros((len(subj),4));ups=counts.copy()
    for j,group in enumerate(GROUPS):
        m=p.sedative_transition.eq(group).to_numpy();np.add.at(counts[:,j],idx[m],1);np.add.at(ups[:,j],idx[m],p.loc[m,'change'].eq('higher').astype(int).to_numpy())
    rng=np.random.default_rng(20260913);est=np.empty((1000,4))
    for b in range(1000):
        ids=rng.integers(0,len(subj),len(subj));n=counts[ids].sum(axis=0)
        est[b]=np.divide(ups[ids].sum(axis=0),n,out=np.full(4,np.nan),where=n>0)
    for j,group in enumerate(GROUPS):
        mask=joint.analysis.eq('primary')&joint.start_h.eq(6)&joint.end_h.eq(48)&joint.drug.eq('sedative')&joint.record_transition.eq(group)
        assert mask.sum()==1
        if joint.loc[mask,'stays'].iloc[0]>=20:joint.loc[mask,['ci_low','ci_high']]=np.nanquantile(est[:,j],[.025,.975])
    for filename,rows in [('paired_response_summary.csv',summary),('paired_missingness.csv',missing),('treatment_transition_matrices.csv',txmatrix),('component_transition_matrices.csv',gradepairs),('language_transition_matrices.csv',lmatrix)]:pd.DataFrame(rows).to_csv(HERE/filename,index=False)
    joint.to_csv(HERE/'response_treatment_cochanges.csv',index=False)
    # Score vs cutoff transition reassignment for the same complete EM pair.
    alt=pair_data(g,6,48,context='score');mask=p.index
    compare={'paired_stays':len(p),'sedative_record_transition_reassigned':int((p.sedative_transition!=alt.loc[mask,'sedative_transition']).sum()),'sedative_full_state_pair_reassigned':int(((p.sedative0!=alt.loc[mask,'sedative0'])|(p.sedative1!=alt.loc[mask,'sedative1'])).sum())}
    return compare


def assert_aggregates():
    cov=pd.read_csv(HERE/'coverage_and_ages.csv');gr=pd.read_csv(HERE/'grade_distributions.csv')
    for _,t in gr.groupby(['subset','cut_h','component']):assert t.n.sum()==t.denominator.iloc[0]
    s=pd.read_csv(HERE/'paired_response_summary.csv');j=pd.read_csv(HERE/'response_treatment_cochanges.csv')
    assert s[[c+'_n' for c in CHANGES]].sum(axis=1).eq(s.paired_stays).all()
    assert j[[c+'_n' for c in CHANGES]].sum(axis=1).eq(j.stays).all()
    for _,t in pd.read_csv(HERE/'treatment_transition_matrices.csv').groupby(['analysis','start_h','end_h','drug','subset']):assert t.n.sum()==t.denominator.iloc[0]
    for _,t in pd.read_csv(HERE/'paired_missingness.csv').groupby(['analysis','start_h','end_h']):assert t.n.sum()==24894
    assert (j.higher_n<=j.initial_non_ceiling_n).all()
    return True


def plot_outputs():
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/gcs_mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    grades=pd.read_csv(HERE/'grade_distributions.csv')
    fig,axs=plt.subplots(1,2,figsize=(12,5.3),constrained_layout=True)
    for ax,comp,levels in zip(axs,['E','M'],[range(1,5),range(1,7)]):
        t=grades[(grades.subset=='all_cohort')&(grades.component==comp)].copy();bottom=np.zeros(5)
        colors=plt.cm.Blues(np.linspace(.85,.28,len(levels)))
        for level,color in zip(list(map(str,levels))+['missing'],list(colors)+['#d4d4d4']):
            z=t[t.grade.astype(str).eq(level)].set_index('cut_h').loc[CUTS]
            val=100*z.n.to_numpy()/z.denominator.to_numpy()
            ax.bar(np.arange(5),val,bottom=bottom,color=color,label=comp+level if level!='missing' else 'No current E/M')
            bottom+=val
        ax.set_xticks(range(5),[str(x) for x in CUTS]);ax.set_ylim(0,100);ax.set_xlabel('Cutoff after ICU admission (h)');ax.set_ylabel('Full cohort (%)')
        ax.set_title('Eye scores' if comp=='E' else 'Motor scores');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,frameon=False,fontsize=10)
    fig.suptitle('Recorded grades in fixed windows: 24,894 ICU stays\nLast current E/M within the preceding 4 h; missing stays retained',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(HERE/f'fixed_window_grades.{ext}',dpi=180)
    plt.close(fig)
    j=pd.read_csv(HERE/'response_treatment_cochanges.csv');t=j[(j.analysis=='primary')&j.start_h.eq(6)&j.end_h.eq(48)&j.drug.eq('sedative')].set_index('record_transition').loc[GROUPS]
    fig,ax=plt.subplots(figsize=(10.5,6.6),constrained_layout=True)
    left=np.zeros(4);colors=['#2e8b87','#b6b9bc','#ce775d','#b4a0ca'];labels=['Both non-decreasing; at least one higher','Both unchanged','Both non-increasing; at least one lower','One higher, one lower']
    for col,color,label in zip(CHANGES,colors,labels):
        v=100*t[col+'_n']/t.stays
        ax.barh(np.arange(4),v,left=left,color=color,label=label)
        for k,pct in enumerate(v):
            if pct>=4:ax.text(left[k]+pct/2,k,f'{pct:.1f}%',ha='center',va='center',fontsize=10)
        left+=v.to_numpy()
    ax.set_yticks(range(4),[f'{g.replace("_to_"," → ")}  (n={n:,})' for g,n in zip(GROUPS,t.stays)])
    ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Stays with current E/M at both 6 h and 48 h (%)')
    ax.set_title('6–48 h response-score changes by sedative-record transitions\nR: administration or interval record; U: unconfirmed / unknown\nRecord transitions do not establish treatment cessation or effects.',fontsize=12)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2,frameon=False,fontsize=9)
    for ext in ['png','pdf']:fig.savefig(HERE/f'response_record_cochanges.{ext}',dpi=180)
    plt.close(fig)


def main():
    g,facts=build_grid();facts['complete_EM_all_five']=make_summaries(g);facts['score_context_comparison']=paired_outputs(g)
    facts['aggregate_checks_passed']=assert_aggregates();facts['analysis_code_sha256']=sha(Path(__file__));facts['bootstrap']={'unit':'subject_id','replicates':1000,'seed':20260913,'interval':'percentile 95%; descriptive; primary sedative record groups only'}
    (HERE/'verification.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2,default=int))
    plot_outputs()
    print(pd.read_csv(HERE/'coverage_and_ages.csv').query("subset == 'all_cohort'")[['cut_h','stays','em_n','rass_n','em_age_h_median','em_age_h_p95']].to_string(index=False))
    print(pd.read_csv(HERE/'paired_response_summary.csv').to_string(index=False))
    print(pd.read_csv(HERE/'response_treatment_cochanges.csv').query("analysis == 'primary' and start_h==6 and end_h==48 and drug=='sedative'")[['record_transition','stays','higher_n','unchanged_n','lower_n','mixed_n','initial_ceiling_n','higher_fraction','ci_low','ci_high']].to_string(index=False))
    print(json.dumps({'all_five_complete':facts['complete_EM_all_five'],'score_context_comparison':facts['score_context_comparison']},indent=2))

if __name__=='__main__':main()
