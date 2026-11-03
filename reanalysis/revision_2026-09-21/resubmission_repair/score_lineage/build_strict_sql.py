from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parents[2].parent
cohort=(P/'04b_extract_sofa.sql').read_text().split('-- SOFA 表 48h')[0]
cohort=cohort[cohort.index('WITH sepsis AS'):].rstrip().rstrip(',')
cohort=cohort.replace('SELECT stay_id FROM ranked WHERE rn = 1','SELECT stay_id, subject_id, hadm_id, intime FROM ranked WHERE rn = 1')
body=r'''
, co AS MATERIALIZED (
 SELECT c.*, h.hr, c.intime+h.hr*INTERVAL '1 hour' AS starttime,
 c.intime+(h.hr+1)*INTERVAL '1 hour' AS endtime
 FROM cohort c CROSS JOIN generate_series(0,47) AS h(hr)
), chart_src AS MATERIALIZED (
 SELECT c.stay_id, c.intime, ce.charttime, ce.itemid, ce.valuenum, ce.value
 FROM cohort c JOIN mimiciv_icu.chartevents ce ON ce.stay_id=c.stay_id
 AND ce.charttime>=c.intime AND ce.charttime<c.intime+INTERVAL '48 hour'
 WHERE ce.itemid IN (223835,223900,223901,220739)
), lab_src AS MATERIALIZED (
 SELECT c.stay_id,c.intime,le.charttime,le.itemid,le.valuenum,le.value,le.specimen_id
 FROM cohort c JOIN mimiciv_hosp.labevents le ON le.subject_id=c.subject_id
 AND le.charttime>=c.intime AND le.charttime<c.intime+INTERVAL '48 hour'
 WHERE le.itemid IN (52033,50821,50816,51265,50885,50912)
 AND (le.hadm_id=c.hadm_id OR (le.hadm_id IS NULL AND le.itemid IN (52033,50821,50816)))
), fio2 AS MATERIALIZED (
 SELECT stay_id,charttime,MAX(CASE WHEN valuenum>0.2 AND valuenum<=1 THEN valuenum*100
 WHEN valuenum>=20 AND valuenum<=100 THEN valuenum ELSE NULL END) AS fio2_ce
 FROM chart_src WHERE itemid=223835 AND valuenum>0 AND valuenum<=100
 GROUP BY stay_id,charttime
), bg_raw AS MATERIALIZED (
 SELECT stay_id,specimen_id,MAX(charttime) AS charttime,MIN(charttime) AS min_source_time,
 MAX(CASE WHEN itemid=52033 THEN value END) AS specimen,
 MAX(CASE WHEN itemid=50821 THEN valuenum END) AS po2,
 MAX(CASE WHEN itemid=50821 AND valuenum IS NOT NULL THEN charttime END) AS po2_time,
 MAX(CASE WHEN itemid=50816 AND ((valuenum>20 AND valuenum<=100) OR (valuenum>0.2 AND valuenum<=1)) THEN charttime END) AS fio2_lab_time,
 MAX(CASE WHEN itemid=50816 THEN CASE WHEN valuenum>20 AND valuenum<=100 THEN valuenum
 WHEN valuenum>0.2 AND valuenum<=1 THEN valuenum*100 END END) AS fio2_lab
 FROM lab_src WHERE itemid IN (52033,50821,50816)
 GROUP BY stay_id,specimen_id
), bg AS MATERIALIZED (
 SELECT stay_id,specimen_id,po2_time AS charttime,min_source_time,specimen,po2,fio2_lab_time,
 CASE WHEN fio2_lab_time<=po2_time THEN fio2_lab ELSE NULL END AS fio2_lab
 FROM bg_raw
), pf_pair AS MATERIALIZED (
 SELECT DISTINCT ON (b.stay_id,b.specimen_id) b.*,
 f.fio2_ce,f.charttime AS fio2_ce_time
 FROM bg b LEFT JOIN fio2 f ON b.fio2_lab IS NULL AND f.stay_id=b.stay_id
 AND f.charttime>=b.charttime-INTERVAL '4 hour' AND f.charttime<=b.charttime AND f.fio2_ce>0
 WHERE b.specimen='ART.' AND b.po2 IS NOT NULL
 ORDER BY b.stay_id,b.specimen_id,f.charttime DESC NULLS LAST
), pf AS MATERIALIZED (
 SELECT b.stay_id,b.charttime,b.min_source_time,
 COALESCE(b.fio2_lab,b.fio2_ce) AS fio2_used,
 CASE WHEN b.fio2_lab IS NOT NULL THEN b.fio2_lab_time ELSE b.fio2_ce_time END AS fio2_source_time,
 100.0*b.po2/NULLIF(COALESCE(b.fio2_lab,b.fio2_ce),0) AS pf,
 (v.stay_id IS NOT NULL) AS invasivevent
 FROM pf_pair b LEFT JOIN mimiciv_derived.ventilation v
 ON v.stay_id=b.stay_id AND v.ventilation_status='InvasiveVent'
 AND v.starttime<=b.charttime AND v.endtime>b.charttime
), gcs_base AS MATERIALIZED (
 SELECT stay_id,charttime,
 MAX(CASE WHEN itemid=223901 THEN valuenum END) AS motor,
 MAX(CASE WHEN itemid=223900 AND value='No Response-ETT' THEN 0
 WHEN itemid=223900 THEN valuenum END) AS verbal,
 MAX(CASE WHEN itemid=220739 THEN valuenum END) AS eyes
 FROM chart_src WHERE itemid IN (223900,223901,220739)
 GROUP BY stay_id,charttime
), gcs_lag AS (
 SELECT *,LAG(charttime) OVER w AS prev_time,LAG(motor) OVER w AS motor_prev,
 LAG(verbal) OVER w AS verbal_prev,LAG(eyes) OVER w AS eyes_prev
 FROM gcs_base WINDOW w AS(PARTITION BY stay_id ORDER BY charttime)
), gcs_prepare AS (
 SELECT stay_id,charttime,motor,verbal,eyes,
 CASE WHEN prev_time>charttime-INTERVAL '6 hour' THEN motor_prev END AS pm,
 CASE WHEN prev_time>charttime-INTERVAL '6 hour' THEN verbal_prev END AS pv,
 CASE WHEN prev_time>charttime-INTERVAL '6 hour' THEN eyes_prev END AS pe
 FROM gcs_lag
), gcs AS MATERIALIZED (
 SELECT *,CASE WHEN verbal=0 OR (verbal IS NULL AND pv=0) THEN 15
 WHEN pv=0 THEN COALESCE(motor,6)+COALESCE(verbal,5)+COALESCE(eyes,4)
 ELSE COALESCE(motor,pm,6)+COALESCE(verbal,pv,5)+COALESCE(eyes,pe,4) END AS gcs
 FROM gcs_prepare
), gcs_qa AS (
 SELECT stay_id,COUNT(*) AS n_gcs_rows,
 COUNT(*) FILTER(WHERE motor IS NULL OR verbal IS NULL OR eyes IS NULL) AS n_gcs_partial_current,
 COUNT(*) FILTER(WHERE NOT(COALESCE(verbal=0,FALSE) OR (verbal IS NULL AND COALESCE(pv=0,FALSE))) AND pv IS DISTINCT FROM 0 AND ((motor IS NULL AND pm IS NOT NULL) OR (verbal IS NULL AND pv IS NOT NULL) OR (eyes IS NULL AND pe IS NOT NULL))) AS n_gcs_with_carry,
 COUNT(*) FILTER(WHERE NOT(COALESCE(verbal=0,FALSE) OR (verbal IS NULL AND COALESCE(pv=0,FALSE))) AND ((pv=0 AND (motor IS NULL OR verbal IS NULL OR eyes IS NULL)) OR (pv IS DISTINCT FROM 0 AND ((motor IS NULL AND pm IS NULL) OR (verbal IS NULL AND pv IS NULL) OR (eyes IS NULL AND pe IS NULL))))) AS n_gcs_with_normal_default,
 COUNT(*) FILTER(WHERE verbal=0 OR (verbal IS NULL AND pv=0)) AS n_gcs_ett_override
 FROM gcs GROUP BY stay_id
), uo_base AS MATERIALIZED (
 SELECT c.stay_id,c.intime,u.charttime,u.urineoutput,
 LAG(u.charttime) OVER(PARTITION BY c.stay_id ORDER BY u.charttime) AS prev_time
 FROM cohort c JOIN mimiciv_derived.urine_output u ON u.stay_id=c.stay_id
 AND u.charttime>=c.intime AND u.charttime<c.intime+INTERVAL '48 hour'
), uo_intervals AS MATERIALIZED (
 -- First quantity has unknown collection start and is NOT assigned synthetic duration.
 SELECT *,EXTRACT(EPOCH FROM(charttime-prev_time))/3600.0 AS interval_h
 FROM uo_base WHERE prev_time IS NOT NULL AND prev_time>=intime AND charttime>prev_time
 AND urineoutput IS NOT NULL
), uo_rolling AS MATERIALIZED (
 -- Official record lookback and duration normalization; not proof of continuous monitoring.
 SELECT stay_id,intime,charttime,SUM(urineoutput) OVER w AS volume_ml,
 SUM(interval_h) OVER w AS coverage_h,COUNT(*) OVER w AS interval_count,
 MAX(interval_h) OVER w AS max_gap_h,MIN(prev_time) OVER w AS collection_start
 FROM uo_intervals
 WINDOW w AS(PARTITION BY stay_id ORDER BY charttime RANGE BETWEEN INTERVAL '23 hour' PRECEDING AND CURRENT ROW)
), uo_eligible AS MATERIALIZED (
 SELECT *,volume_ml/coverage_h*24.0 AS uo_24hr
 FROM uo_rolling WHERE charttime>=intime+INTERVAL '24 hour'
 AND coverage_h>=22 AND coverage_h<=30 AND collection_start>=intime
), labs AS MATERIALIZED (
 SELECT co.stay_id,co.hr,
 MIN(CASE WHEN l.itemid=51265 AND l.valuenum>0 THEN l.valuenum END) AS platelet_min,
 MAX(CASE WHEN l.itemid=50885 AND l.valuenum>0 THEN l.valuenum END) AS bilirubin_max,
 MAX(CASE WHEN l.itemid=50912 AND l.valuenum>0 AND l.valuenum<=150 THEN l.valuenum END) AS creatinine_max
 FROM co LEFT JOIN lab_src l ON l.stay_id=co.stay_id AND l.charttime>=co.starttime AND l.charttime<co.endtime
 GROUP BY co.stay_id,co.hr
), resp AS MATERIALIZED (
 SELECT co.stay_id,co.hr,
 MIN(p.pf) FILTER(WHERE p.invasivevent) AS pf_vent,
 MIN(p.pf) FILTER(WHERE NOT p.invasivevent) AS pf_novent,
 COUNT(p.pf) AS n_pf
 FROM co LEFT JOIN pf p ON p.stay_id=co.stay_id AND p.charttime>=co.starttime AND p.charttime<co.endtime
 GROUP BY co.stay_id,co.hr
), vs AS MATERIALIZED (
 SELECT co.stay_id,co.hr,MIN(v.mbp) AS meanbp_min
 FROM co LEFT JOIN mimiciv_derived.vitalsign v ON v.stay_id=co.stay_id
 AND v.charttime>=co.starttime AND v.charttime<co.endtime
 GROUP BY co.stay_id,co.hr
), neuro AS MATERIALIZED (
 SELECT co.stay_id,co.hr,MIN(g.gcs) AS gcs_min
 FROM co LEFT JOIN gcs g ON g.stay_id=co.stay_id AND g.charttime>=co.starttime AND g.charttime<co.endtime
 GROUP BY co.stay_id,co.hr
), urine AS MATERIALIZED (
 -- Keep the official SOFA hourly MAX of eligible normalized UO values.
 SELECT co.stay_id,co.hr,MAX(u.uo_24hr) AS uo_24hr
 FROM co LEFT JOIN uo_eligible u ON u.stay_id=co.stay_id AND u.charttime>=co.starttime AND u.charttime<co.endtime
 GROUP BY co.stay_id,co.hr
), vaso_src AS MATERIALIZED (
 SELECT 'epi'::text AS drug,stay_id,starttime,endtime,vaso_rate FROM mimiciv_derived.epinephrine
 UNION ALL SELECT 'nor',stay_id,starttime,endtime,vaso_rate FROM mimiciv_derived.norepinephrine
 UNION ALL SELECT 'dop',stay_id,starttime,endtime,vaso_rate FROM mimiciv_derived.dopamine
 UNION ALL SELECT 'dob',stay_id,starttime,endtime,vaso_rate FROM mimiciv_derived.dobutamine
), vaso AS MATERIALIZED (
 SELECT co.stay_id,co.hr,
 MAX(v.vaso_rate) FILTER(WHERE drug='epi') AS epi,
 MAX(v.vaso_rate) FILTER(WHERE drug='nor') AS nor,
 MAX(v.vaso_rate) FILTER(WHERE drug='dop') AS dop,
 MAX(v.vaso_rate) FILTER(WHERE drug='dob') AS dob
 FROM co LEFT JOIN vaso_src v ON v.stay_id=co.stay_id
 AND v.starttime<co.endtime AND v.endtime>co.starttime AND v.endtime>v.starttime AND v.vaso_rate>0
 GROUP BY co.stay_id,co.hr
), components AS MATERIALIZED (
 SELECT co.stay_id,co.hr,
 CASE WHEN r.pf_vent<100 THEN 4 WHEN r.pf_vent<200 THEN 3
 WHEN r.pf_vent<300 OR r.pf_novent<300 THEN 2 WHEN r.pf_vent<400 OR r.pf_novent<400 THEN 1
 WHEN COALESCE(r.pf_vent,r.pf_novent) IS NULL THEN NULL ELSE 0 END AS respiration,
 CASE WHEN l.platelet_min<20 THEN 4 WHEN l.platelet_min<50 THEN 3 WHEN l.platelet_min<100 THEN 2
 WHEN l.platelet_min<150 THEN 1 WHEN l.platelet_min IS NULL THEN NULL ELSE 0 END AS coagulation,
 CASE WHEN l.bilirubin_max>=12 THEN 4 WHEN l.bilirubin_max>=6 THEN 3 WHEN l.bilirubin_max>=2 THEN 2
 WHEN l.bilirubin_max>=1.2 THEN 1 WHEN l.bilirubin_max IS NULL THEN NULL ELSE 0 END AS liver,
 CASE WHEN v.dop>15 OR v.epi>0.1 OR v.nor>0.1 THEN 4
 WHEN v.dop>5 OR (v.epi>0 AND v.epi<=0.1) OR (v.nor>0 AND v.nor<=0.1) THEN 3
 WHEN v.dop>0 OR v.dob>0 THEN 2 WHEN vs.meanbp_min<70 THEN 1
 WHEN COALESCE(v.dop,v.epi,v.nor,v.dob,vs.meanbp_min) IS NULL THEN NULL ELSE 0 END AS cardiovascular,
 CASE WHEN n.gcs_min BETWEEN 13 AND 14 THEN 1 WHEN n.gcs_min BETWEEN 10 AND 12 THEN 2
 WHEN n.gcs_min BETWEEN 6 AND 9 THEN 3 WHEN n.gcs_min<6 THEN 4 WHEN n.gcs_min IS NULL THEN NULL ELSE 0 END AS cns,
 CASE WHEN l.creatinine_max>=5 OR u.uo_24hr<200 THEN 4
 WHEN l.creatinine_max>=3.5 OR u.uo_24hr<500 THEN 3
 WHEN l.creatinine_max>=2 THEN 2 WHEN l.creatinine_max>=1.2 THEN 1
 WHEN COALESCE(l.creatinine_max,u.uo_24hr) IS NULL THEN NULL ELSE 0 END AS renal,
 l.creatinine_max,u.uo_24hr
 FROM co JOIN labs l USING(stay_id,hr) JOIN resp r USING(stay_id,hr) JOIN vs USING(stay_id,hr)
 JOIN neuro n USING(stay_id,hr) JOIN urine u USING(stay_id,hr) JOIN vaso v USING(stay_id,hr)
), agg AS (
 SELECT stay_id,MAX(respiration) AS respiration,MAX(coagulation) AS coagulation,
 MAX(liver) AS liver,MAX(cardiovascular) AS cardiovascular,MAX(cns) AS cns,MAX(renal) AS renal,
 COUNT(respiration) AS respiration_hours,COUNT(coagulation) AS coagulation_hours,COUNT(liver) AS liver_hours,
 COUNT(cardiovascular) AS cardiovascular_hours,COUNT(cns) AS cns_hours,COUNT(renal) AS renal_hours,
 MAX(creatinine_max) AS creatinine_max,MIN(uo_24hr) AS uo_24hr_min
 FROM components GROUP BY stay_id
), uo_qa AS (
 SELECT c.stay_id,COUNT(u.charttime) AS eligible_uo_endpoints,
 MIN(u.coverage_h) AS uo_coverage_min,MAX(u.coverage_h) AS uo_coverage_max,
 MIN(u.interval_count) AS uo_min_interval_count,MAX(u.interval_count) AS uo_max_interval_count,
 MAX(u.max_gap_h) AS uo_max_gap_h
 FROM cohort c LEFT JOIN uo_eligible u USING(stay_id) GROUP BY c.stay_id
), boundary_events AS (
 SELECT x.stay_id,'chart' AS src,x.charttime AS t1,x.charttime AS t2 FROM chart_src x
 UNION ALL SELECT x.stay_id,'lab',x.charttime,x.charttime FROM lab_src x
 UNION ALL SELECT x.stay_id,'pf',x.fio2_source_time,x.charttime FROM pf x WHERE x.pf IS NOT NULL
 UNION ALL SELECT x.stay_id,'uo',x.collection_start,x.charttime FROM uo_eligible x
), boundary_qa AS (
 SELECT c.stay_id,
 COUNT(*) FILTER(WHERE x.src='chart' AND (x.t1<c.intime OR x.t2>=c.intime+INTERVAL '48 hour')) AS chart_outside_n,
 COUNT(*) FILTER(WHERE x.src='lab' AND (x.t1<c.intime OR x.t2>=c.intime+INTERVAL '48 hour')) AS lab_outside_n,
 COUNT(*) FILTER(WHERE x.src='pf' AND (x.t1<c.intime OR x.t1>x.t2 OR x.t2>=c.intime+INTERVAL '48 hour')) AS pf_pair_outside_n,
 COUNT(*) FILTER(WHERE x.src='uo' AND (x.t1<c.intime OR x.t2>=c.intime+INTERVAL '48 hour')) AS uo_outside_n
 FROM cohort c LEFT JOIN boundary_events x USING(stay_id) GROUP BY c.stay_id
)
SELECT a.stay_id,
 COALESCE(a.respiration,0) AS respiration,COALESCE(a.coagulation,0) AS coagulation,
 COALESCE(a.liver,0) AS liver,COALESCE(a.cardiovascular,0) AS cardiovascular,
 COALESCE(a.cns,0) AS cns,COALESCE(a.renal,0) AS renal,
 COALESCE(a.respiration,0)+COALESCE(a.coagulation,0)+COALESCE(a.liver,0)+COALESCE(a.cardiovascular,0)+COALESCE(a.cns,0)+COALESCE(a.renal,0) AS sofa_max_48h,
 a.respiration_hours,a.coagulation_hours,a.liver_hours,a.cardiovascular_hours,a.cns_hours,a.renal_hours,
 a.creatinine_max,a.uo_24hr_min,
 CASE WHEN a.renal>=3 AND COALESCE(a.creatinine_max,0)<3.5 AND a.uo_24hr_min<500 THEN 1 ELSE 0 END AS high_renal_uo_only,
 u.eligible_uo_endpoints,u.uo_coverage_min,u.uo_coverage_max,u.uo_min_interval_count,u.uo_max_interval_count,u.uo_max_gap_h,
 b.chart_outside_n,b.lab_outside_n,b.pf_pair_outside_n,b.uo_outside_n,
 COALESCE(g.n_gcs_rows,0) AS n_gcs_rows,COALESCE(g.n_gcs_partial_current,0) AS n_gcs_partial_current,
 COALESCE(g.n_gcs_with_carry,0) AS n_gcs_with_carry,COALESCE(g.n_gcs_with_normal_default,0) AS n_gcs_with_normal_default,
 COALESCE(g.n_gcs_ett_override,0) AS n_gcs_ett_override
FROM agg a JOIN uo_qa u USING(stay_id) JOIN boundary_qa b USING(stay_id) LEFT JOIN gcs_qa g USING(stay_id)
ORDER BY a.stay_id;
'''
(H/'mimic_sofa_strict48.sql').write_text('-- DRAFT: REVIEW REQUIRED BEFORE EXECUTION. See REPAIR_PLAN.md\n'+cohort+body)
print('Wrote draft query; NOT EXECUTED')
