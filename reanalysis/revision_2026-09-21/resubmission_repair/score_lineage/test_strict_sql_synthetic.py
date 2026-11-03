from pathlib import Path
import subprocess,io,json
import pandas as pd
H=Path(__file__).resolve().parent
s=(H/'mimic_sofa_strict48.sql').read_text();body=s[s.index(', co AS MATERIALIZED'):]
fixture=r'''WITH cohort(stay_id,subject_id,hadm_id,intime) AS (
 VALUES (-1,-1,-1,TIMESTAMP '2020-01-01'),(-2,-2,-2,TIMESTAMP '2020-01-01'),(-3,-3,-3,TIMESTAMP '2020-01-01'),(-4,-4,-4,TIMESTAMP '2020-01-01')
), fake_chart AS (
 SELECT id AS stay_id,TIMESTAMP '2020-01-01'+h*INTERVAL '1 hour' AS charttime,itemid,v::double precision AS valuenum,txt::text AS value
 FROM(VALUES (-1,-1.0,223900,0.0,'No Response-ETT'),(-1,0.0,220739,2.0,'To pain'),(-4,-1.0,223835,100.0,'100')) q(id,h,itemid,v,txt)
), fake_lab AS (
 SELECT id AS subject_id,id AS hadm_id,TIMESTAMP '2020-01-01'+h*INTERVAL '1 hour' AS charttime,itemid,v::double precision AS valuenum,txt::text AS value,specimen_id
 FROM(VALUES
 (-1,0.0,51265,30.0,NULL,1),(-1,48.0,51265,10.0,NULL,2),(-1,1.0,50912,1.0,NULL,3),
 (-1,1.0,52033,NULL,'ART.',10),(-1,1.0,50821,56.0,NULL,10),(-1,1.0,50816,70.0,NULL,10),
 (-1,2.5,52033,NULL,'ART.',11),(-1,2.5,50821,105.0,NULL,11),(-1,2.5,50816,70.0,NULL,11),
 (-4,0.5,52033,NULL,'ART.',12),(-4,0.5,50821,70.0,NULL,12),(-4,1.0,50816,100.0,NULL,12)
 ) q(id,h,itemid,v,txt,specimen_id)
), fake_vent AS (
 SELECT -1 AS stay_id,TIMESTAMP '2020-01-01 02:00' AS starttime,TIMESTAMP '2020-01-01 03:00' AS endtime,'InvasiveVent'::text AS ventilation_status
), fake_vs AS (
 SELECT -1 AS stay_id,TIMESTAMP '2020-01-01' AS charttime,65.0::double precision AS mbp
 UNION ALL SELECT -1,TIMESTAMP '2020-01-03',20.0
), fake_uo AS (
 SELECT -2 AS stay_id,TIMESTAMP '2020-01-01'+h*INTERVAL '1 hour' AS charttime,60.0::double precision AS urineoutput FROM generate_series(1,47) h
 UNION ALL SELECT -1,TIMESTAMP '2020-01-02',30.0
 UNION ALL SELECT -3,TIMESTAMP '2020-01-01 01:00',30.0
 UNION ALL SELECT -3,TIMESTAMP '2020-01-02',30.0
), fake_nor AS (
 SELECT -1 AS stay_id,TIMESTAMP '2020-01-02 23:30' AS starttime,TIMESTAMP '2020-01-03 01:00' AS endtime,0.2::double precision AS vaso_rate
 UNION ALL SELECT -4,TIMESTAMP '2020-01-01 01:00',TIMESTAMP '2020-01-01 02:00',0.0
), fake_empty_vaso AS (
 SELECT NULL::integer AS stay_id,NULL::timestamp AS starttime,NULL::timestamp AS endtime,NULL::double precision AS vaso_rate WHERE FALSE
)
'''
for real,fake in {'mimiciv_icu.chartevents':'fake_chart','mimiciv_hosp.labevents':'fake_lab','mimiciv_derived.ventilation':'fake_vent','mimiciv_derived.vitalsign':'fake_vs','mimiciv_derived.urine_output':'fake_uo','mimiciv_derived.norepinephrine':'fake_nor','mimiciv_derived.epinephrine':'fake_empty_vaso','mimiciv_derived.dopamine':'fake_empty_vaso','mimiciv_derived.dobutamine':'fake_empty_vaso'}.items():body=body.replace(real,fake)
assert 'mimiciv_' not in body
sql='BEGIN READ ONLY;\n'+fixture+body+'\nROLLBACK;'
(H/'synthetic_strict48.sql').write_text(sql)
p=subprocess.run(['psql','-X','-q','-h','localhost','-U','postgres','-d','mimiciv31','--csv','-v','ON_ERROR_STOP=1','-f','-'],input=sql,capture_output=True,text=True,timeout=30)
if p.returncode:raise RuntimeError(p.stderr)
df=pd.read_csv(io.StringIO(p.stdout)).set_index('stay_id')
checks={
 'zero_included_48_excluded_platelets':bool(df.loc[-1,'coagulation']==3),
 'invasive_vent_at_bg_time':bool(df.loc[-1,'respiration']==3),
 'pre_admit_gcs_not_carried':bool(df.loc[-1,'cns']==1),
 'post_landmark_interval_clipped':bool(df.loc[-1,'cardiovascular']==4),
 'zero_drug_no_score3':bool(df.loc[-4,'cardiovascular']==0),
 'single_delayed_uo_not_24h':bool(df.loc[-1,'eligible_uo_endpoints']==0),
 'steady_uo_not_false_oliguria':bool(df.loc[-2,'renal']==0 and df.loc[-2,'uo_24hr_min']==1440),
 'sparse_interval_assumption_reported':bool(df.loc[-3,'renal']==4 and df.loc[-3,'uo_max_gap_h']==23 and df.loc[-3,'high_renal_uo_only']==1),
 'future_or_pre_admit_fio2_excluded':bool(df.loc[-4,'respiration_hours']==0),
 'all_boundary_checks_zero':bool((df[[c for c in df.columns if c.endswith('outside_n')]]==0).all().all())}
(H/'synthetic_strict48_results.json').write_text(json.dumps({'synthetic_only':True,'checks':checks,'all_passed':all(checks.values())},indent=2));print(json.dumps(checks,indent=2));assert all(checks.values())
