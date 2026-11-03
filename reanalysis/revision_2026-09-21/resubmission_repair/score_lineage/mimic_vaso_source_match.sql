BEGIN READ ONLY; SET LOCAL statement_timeout='60s';
WITH d AS MATERIALIZED (
 SELECT 'epi'::text AS drug,stay_id,linkorderid,starttime,endtime,vaso_rate FROM mimiciv_derived.epinephrine
 UNION ALL SELECT 'nor',stay_id,linkorderid,starttime,endtime,vaso_rate FROM mimiciv_derived.norepinephrine
 UNION ALL SELECT 'dop',stay_id,linkorderid,starttime,endtime,vaso_rate FROM mimiciv_derived.dopamine
 UNION ALL SELECT 'dob',stay_id,linkorderid,starttime,endtime,vaso_rate FROM mimiciv_derived.dobutamine),
 r AS MATERIALIZED (
 SELECT CASE itemid WHEN 221289 THEN 'epi' WHEN 221906 THEN 'nor' WHEN 221662 THEN 'dop' WHEN 221653 THEN 'dob' END AS drug,
 stay_id,linkorderid,starttime,endtime,
 CASE WHEN itemid=221906 AND rateuom='mg/kg/min' AND patientweight=1 THEN rate
 WHEN itemid=221906 AND rateuom='mg/kg/min' THEN rate*1000.0 ELSE rate END AS vaso_rate
 FROM mimiciv_icu.inputevents WHERE itemid IN(221289,221906,221662,221653)),
 d_minus_r AS(SELECT * FROM d EXCEPT ALL SELECT * FROM r),r_minus_d AS(SELECT * FROM r EXCEPT ALL SELECT * FROM d)
 SELECT (SELECT COUNT(*) FROM d) AS derived_rows,(SELECT COUNT(*) FROM r) AS raw_rows,
 (SELECT COUNT(*) FROM d_minus_r) AS derived_not_raw,(SELECT COUNT(*) FROM r_minus_d) AS raw_not_derived;
ROLLBACK;