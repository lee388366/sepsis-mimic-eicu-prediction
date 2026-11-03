from pathlib import Path
import subprocess
H=Path(__file__).resolve().parent
sql=(H/'mimic_sofa_strict48.sql').read_text();cohort=sql[sql.index('WITH sepsis AS'):sql.index(', co AS MATERIALIZED')]
q1=cohort+''', bg_src AS (
SELECT c.stay_id,le.specimen_id,le.itemid,le.charttime,le.valuenum
FROM cohort c JOIN mimiciv_hosp.labevents le ON le.subject_id=c.subject_id
AND le.charttime>=c.intime AND le.charttime<c.intime+INTERVAL '48 hour'
WHERE le.itemid IN (52033,50821,50816) AND (le.hadm_id=c.hadm_id OR le.hadm_id IS NULL)
),groups AS (SELECT stay_id,specimen_id,itemid,COUNT(*) AS n,COUNT(DISTINCT charttime) AS time_n FROM bg_src GROUP BY stay_id,specimen_id,itemid)
SELECT COUNT(*) AS specimen_item_groups,COUNT(*) FILTER(WHERE n>1) AS repeated_groups,COUNT(*) FILTER(WHERE time_n>1) AS distinct_time_groups FROM groups;'''
q2=cohort+''', source AS (
SELECT c.stay_id,ie.itemid,ie.starttime,ie.endtime,ie.rate,ie.statusdescription,ie.rateuom,
LEAST(ie.endtime,c.intime+INTERVAL '48 hour')-GREATEST(ie.starttime,c.intime) AS overlap_duration
FROM cohort c JOIN mimiciv_icu.inputevents ie ON ie.stay_id=c.stay_id
AND ie.starttime<c.intime+INTERVAL '48 hour' AND ie.endtime>c.intime
WHERE ie.itemid IN (221289,221906,221662,221653) AND ie.rate>0 AND ie.endtime>ie.starttime)
SELECT COUNT(*) AS positive_overlap_records,COUNT(DISTINCT stay_id) AS exposed_stays,
COUNT(*) FILTER(WHERE endtime-starttime<INTERVAL '1 hour') AS original_duration_under_1h_records,
COUNT(*) FILTER(WHERE overlap_duration<INTERVAL '1 hour') AS overlap_under_1h_records,
COUNT(DISTINCT stay_id) FILTER(WHERE overlap_duration<INTERVAL '1 hour') AS stays_with_short_overlap,
COUNT(*) FILTER(WHERE statusdescription='Bolus') AS bolus_records,
COUNT(DISTINCT stay_id) FILTER(WHERE statusdescription='Bolus') AS stays_with_bolus
FROM source;'''
for name,q in [('mimic_bg_duplicate_time_qa',q1),('mimic_vaso_short_interval_qa',q2)]:
 wrapped="BEGIN READ ONLY; SET LOCAL statement_timeout='120s'; SET LOCAL enable_nestloop=off;\n"+q+'\nROLLBACK;'
 (H/(name+'.sql')).write_text(wrapped)
 p=subprocess.run(['psql','-X','-q','-h','localhost','-U','postgres','-d','mimiciv31','--csv','-v','ON_ERROR_STOP=1','-f','-'],input=wrapped,capture_output=True,text=True,timeout=130)
 (H/(name+'.txt')).write_text(p.stdout+p.stderr);print(name,p.returncode,p.stdout,p.stderr,flush=True)
 if p.returncode:raise RuntimeError(name+' failed')
