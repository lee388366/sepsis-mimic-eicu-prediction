from pathlib import Path
import json,subprocess,io,hashlib
import pandas as pd
H=Path('revision_2026-09-21/resubmission_repair/eicu_sofa_rebuild')
D={t:[] for t in ['patient','lab','nursecharting','respiratorycharting','respiratorycare','intakeoutput','infusiondrug','medication','vitalperiodic','vitalaperiodic']};E={};n=900000;seq=0

def case(name,**expect):
 global n;n+=1;D['patient'].append(dict(patientunitstayid=n,uniquepid=str(n),hospitaldischargestatus='Alive',admissionweight=80));E[n]=(name,expect);return n

def add(t,id,**x):
 global seq;seq+=1;x['patientunitstayid']=id
 key={'lab':'labid','intakeoutput':'intakeoutputid','infusiondrug':'infusiondrugid','nursecharting':'nursingchartid','respiratorycare':'respcareid','respiratorycharting':'respchartid','medication':'medicationid'}.get(t)
 if key:x.setdefault(key,seq)
 D[t].append(x)
def lab(i,t,name,value,unit):add('lab',i,labresultoffset=t,labname=name,labresult=value,labmeasurenamesystem=unit,labmeasurenameinterface=unit)
def bg(i,t=120,fio=50,pao=40):lab(i,t,'paO2',pao,'mm Hg');lab(i,t,'FiO2',fio,'%')
def device(i,t,v):add('nursecharting',i,nursingchartoffset=t,nursingchartcelltypecat='Vital Signs',nursingchartcelltypevallabel='O2 Admin Device',nursingchartcelltypevalname='O2 Admin Device',nursingchartvalue=v)
def resp(i,t,label,v):add('respiratorycharting',i,respchartoffset=t,respchartvaluelabel=label,respchartvalue=v)
def vaso(i,rate,name='Norepinephrine (mcg/kg/min)',t=30,**extra):add('infusiondrug',i,infusionoffset=t,drugname=name,drugrate=str(rate),patientweight='80',**extra)
P='flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|'
def urine(i,t,v,label='Urine',**extra):add('intakeoutput',i,intakeoutputoffset=t,intakeoutputentryoffset=t,cellpath=P+label,celllabel=label,cellvaluenumeric=v,outputtotal=10000,**extra)
i=case('all_missing',sofa_max_48h=0,respiration=None,renal=None,cns=None)
i=case('zero_in_2880_out',coagulation=0);lab(i,-1,'platelets x 1000',1,'K/mcL');lab(i,0,'platelets x 1000',200,'K/mcL');lab(i,2880,'platelets x 1000',1,'K/mcL')
i=case('2879_in',coagulation=4);lab(i,2879,'platelets x 1000',1,'K/mcL')
for v,s in [(150,0),(100,1),(50,2),(20,3),(19,4)]:i=case('platelet_'+str(v),coagulation=s);lab(i,0,'platelets x 1000',v,'K/mcL')
for v,s in [(1.2,1),(2,2),(6,3),(12,4)]:i=case('bili_'+str(v),liver=s);lab(i,0,'total bilirubin',v,'mg/dL')
for v,s in [(1.2,1),(2,2),(3.5,3),(5,4)]:i=case('cr_'+str(v),renal=s);lab(i,0,'creatinine',v,'mg/dL')
for value in [0,16,6.5]:
 i=case('invalid_gcs_'+str(value),cns=None);add('nursecharting',i,nursingchartoffset=0,nursingchartcelltypecat='Scores',nursingchartcelltypevallabel='Glasgow coma score',nursingchartcelltypevalname='GCS Total',nursingchartvalue=str(value))
for label,name in [('Glasgow coma score','GCS Total'),('Score (Glasgow Coma Scale)','Value')]:
 i=case('gcs_'+name,cns=3);add('nursecharting',i,nursingchartoffset=0,nursingchartcelltypecat='Scores',nursingchartcelltypevallabel=label,nursingchartcelltypevalname=name,nursingchartvalue='6')
for state,off,expected in [('vent',120,4),('vent',121,2),('trach collar',120,2),('venti mask',120,2),('vent',-1,2),('vent',0,4)]:
 i=case('support_'+state+str(off),respiration=expected);bg(i);device(i,off,state)
i=case('support_stale',respiration=2);bg(i,t=300);device(i,0,'vent')
i=case('support_sensitivity',respiration=4,respiration_1h=2);bg(i);device(i,0,'vent')
i=case('off_truncates',respiration=2);bg(i);device(i,0,'vent');resp(i,100,'RT Vent On/Off','Off')
i=case('simultaneous_off',respiration=2);bg(i);device(i,120,'vent');resp(i,120,'RT Vent On/Off','Off')
i=case('airway_only',respiration=2);bg(i);add('respiratorycare',i,respcarestatusoffset=100,airwaytype='Tracheostomy',ventstartoffset=0,ventendoffset=0)
i=case('explicit_NIV_with_airway',respiration=2);bg(i);add('respiratorycare',i,respcarestatusoffset=100,airwaytype='Tracheostomy');resp(i,110,'Non-invasive Ventilation Mode','CPAP')
i=case('future_fio2',respiration=None);lab(i,100,'paO2',40,'mm Hg');lab(i,101,'FiO2',50,'%')
i=case('preadmit_fio2',respiration=None);lab(i,0,'paO2',40,'mm Hg');lab(i,-1,'FiO2',50,'%')
for f,s in [(.5,4),(50,4),(5,None),(150,None)]:i=case('fio2_'+str(f),respiration=s);bg(i,fio=f);device(i,120,'vent')
for pf,s in [(100,3),(200,2),(300,1),(400,0)]:i=case('pf_'+str(pf),respiration=s);bg(i,pao=pf/2);device(i,120,'vent')
for r,s in [(0,0),('.1',3),('.1001',4)]:i=case('nor_'+str(r),cardiovascular=s);vaso(i,r)
for r,s in [(5,2),(15,3),(15.01,4)]:i=case('dop_'+str(r),cardiovascular=s);vaso(i,r,'Dopamine (mcg/kg/min)')
i=case('dob_zero',cardiovascular=0);vaso(i,0,'Dobutamine (mcg/kg/min)')
i=case('dob_pos_ml_unknown_conc',cardiovascular=2);vaso(i,1,'Dobutamine (ml/hr)')
i=case('known_concentration',cardiovascular=3);vaso(i,30,'Norepinephrine (ml/hr)',drugamount='4',volumeoffluid='250')
i=case('missing_concentration',cardiovascular=None,unconverted_positive_vaso_n=1);vaso(i,30,'Norepinephrine (ml/hr)')
i=case('future_vaso',cardiovascular=None);vaso(i,1,t=2880)
i=case('pre_vaso',cardiovascular=None);vaso(i,1,t=-1)
i=case('single_urine',renal=None,eligible_uo_endpoints=0);urine(i,1440,100)
i=case('steady_urine',renal=0,uo_24hr_min=1440)
for t in range(0,1441,60):urine(i,t,60)
for amount,s in [(199,4),(200,3),(499,3),(500,0)]:i=case('uo_'+str(amount),renal=s,uo_24hr_min=amount);urine(i,0,99999);urine(i,1440,amount)
i=case('uo_count_excluded',renal=None);urine(i,0,1,'Urine Count');urine(i,1440,1,'Urine Count')
i=case('duplicate_urine_revision',renal=3,uo_24hr_min=300);urine(i,0,1);urine(i,1440,300);urine(i,1440,300)
i=case('independent_nephrostomies',renal=0,uo_24hr_min=600);urine(i,0,1);urine(i,1440,300,'Nephrostomy Left Output');urine(i,1440,300,'Nephrostomy Right Output')
i=case('inclusive_23h_endpoint',renal=0,uo_24hr_min=500);urine(i,0,1);urine(i,60,400);urine(i,1440,100)
i=case('extubation_stops_support',respiration=2);bg(i);device(i,0,'vent');add('respiratorycare',i,respcarestatusoffset=100,airwaytype='No Artificial Airway')
for typ,expected in [('valid',3),('future',None),('conflict',None),('expired',None)]:
 i=case('med_concentration_'+typ,cardiovascular=expected);vaso(i,30,'Norepinephrine (ml/hr)',t=100)
 add('medication',i,drugname='NOREPINEPHRINE 4 MG/250 ML NS',drugordercancelled='No',drugorderoffset=101 if typ=='future' else 0,drugstartoffset=0,drugstopoffset=100 if typ=='expired' else 200)
 if typ=='conflict':add('medication',i,drugname='NOREPINEPHRINE 8 MG in 250mL NS',drugordercancelled='No',drugorderoffset=0,drugstartoffset=0,drugstopoffset=200)
i=case('urine_foley_same_stream',renal=3,uo_24hr_min=300);urine(i,0,1);urine(i,1440,300);urine(i,1440,300,'Foley cath')
i=case('same_side_nephro_not_double',renal=3,uo_24hr_min=300);urine(i,0,1);urine(i,1440,300,'Nephrostomy Left Output');urine(i,1440,300,'Urinary Catheter Output: Nephrostomy L')
i=case('nephro_unsided_total_not_double',renal=0,uo_24hr_min=600);urine(i,0,1);urine(i,1440,300,'Nephrostomy Left Output');urine(i,1440,300,'Nephrostomy Right Output');urine(i,1440,600,'Nephrostomy')
i=case('generic_total_ambiguity_sensitivity',renal=3,renal_uo_conservative=4,sofa_max_48h=3,sofa_uo_conservative=4,uo_24hr_min=220);urine(i,0,1);urine(i,1440,100);urine(i,1440,60,'Nephrostomy Left Output');urine(i,1440,60,'Nephrostomy Right Output')
i=case('unknown_route_not_added',renal=3,uo_24hr_min=300,uo_unknown_route_rows=1);urine(i,0,1);urine(i,1440,300);urine(i,1440,300,'Urinary Catheter Output: Indwelling/Continuous Nep')
i=case('unknown_route_larger_max',renal=3,renal_uo_conservative=3,uo_24hr_min=250);urine(i,0,1);urine(i,1440,100);urine(i,1440,60,'Nephrostomy Left Output');urine(i,1440,60,'Nephrostomy Right Output');urine(i,1440,250,'Urinary Catheter Output: Indwelling/Continuous Nep')
for label in ['vent/cpap','ventilator/CPAP','V-CPAP']:
 for airway_on,expected in [(False,2),(True,4)]:
  i=case('mixed_cpap_'+label+str(airway_on),respiration=expected);bg(i);device(i,100,label)
  if airway_on:add('respiratorycare',i,respcarestatusoffset=90,airwaytype='Oral ETT')
for label in ['trach to vent','trach/vent','trach to ventilator','ventilator/trach','ventilatortrach','11-Mechanical Ventilator','ventilator/ac-25/peep10','ventilator/ac-26','ventilator/AC 16/TV 300/Peep+5']:
 i=case('explicit_device_alias_'+label,respiration=4);bg(i);device(i,100,label)
i=case('delivery_removed_terminates',respiration=2);bg(i);device(i,0,'vent');resp(i,100,'Oxygen Delivery Status','Removed')
for label in ['intubated','ETT','ETT oral','95','ambu bag','venti','AC15/350/60%/8','CPAP']:
 i=case('not_confirmed_device_'+label,respiration=2);bg(i);device(i,100,label)
s=(H/'eicu_sofa_strict48.sql').read_text();body=s[s.index('cohort AS MATERIALIZED ('):]
for t in D:body=body.replace('eicu_crd.'+t,'fake_'+t)
fake=[]
for t,rows in D.items():
 data=json.dumps(rows).replace("'","''")
 fake.append(f"fake_{t} AS (SELECT * FROM json_populate_recordset(NULL::eicu_crd.{t},'{data}'))")
ids=','.join('('+str(i)+')' for i in E)
sql='BEGIN READ ONLY; SET LOCAL statement_timeout=\'30s\';\nWITH '+','.join(fake)+',frozen_ids(patientunitstayid) AS (VALUES '+ids+'),'+body+'\nROLLBACK;'
p=subprocess.run(['psql','-X','-q','-h','localhost','-U','postgres','-d','eicu','--csv','-v','ON_ERROR_STOP=1','-f','-'],input=sql,text=True,capture_output=True,timeout=40)
if p.returncode:raise RuntimeError(p.stderr)
df=pd.read_csv(io.StringIO(p.stdout)).set_index('patientunitstayid');results=[]
for id,(name,ex) in E.items():
 observed={k:(None if pd.isna(df.loc[id,k]) else float(df.loc[id,k])) for k in ex}
 good=all(observed[k] is None if v is None else observed[k] is not None and abs(observed[k]-v)<1e-8 for k,v in ex.items())
 results.append(dict(test=name,expected=ex,observed=observed,passed=good))
output=dict(synthetic_only=True,sql_sha256=hashlib.sha256(s.encode()).hexdigest(),cases=len(results),passed=sum(x['passed'] for x in results),failed=[x for x in results if not x['passed']],all_results=results)
(H/'independent_synthetic_results.json').write_text(json.dumps(output,indent=2));print(json.dumps({k:output[k] for k in ['sql_sha256','cases','passed','failed']},indent=2))
