cohort AS MATERIALIZED (
 SELECT i.patientunitstayid,p.uniquepid::text,CASE WHEN p.hospitaldischargestatus='Expired' THEN 1 WHEN p.hospitaldischargestatus='Alive' THEN 0 END AS label,p.admissionweight
 FROM frozen_ids i JOIN eicu_crd.patient p USING(patientunitstayid)
), hours AS (
 SELECT c.patientunitstayid,h.hr FROM cohort c CROSS JOIN generate_series(0,47) h(hr)
), lab_src AS MATERIALIZED (
 SELECT l.* FROM eicu_crd.lab l JOIN cohort c USING(patientunitstayid)
 WHERE l.labresultoffset>=0 AND l.labresultoffset<2880 AND l.labname IN ('paO2','FiO2','creatinine','total bilirubin','platelets x 1000')
), resp_src AS MATERIALIZED (
 SELECT r.* FROM eicu_crd.respiratorycharting r JOIN cohort c USING(patientunitstayid)
 WHERE r.respchartoffset>=0 AND r.respchartoffset<2880
), nurse_src AS MATERIALIZED (
 SELECT n.* FROM eicu_crd.nursecharting n JOIN cohort c USING(patientunitstayid)
 WHERE n.nursingchartoffset>=0 AND n.nursingchartoffset<2880 AND (
 (n.nursingchartcelltypecat IN ('Scores','Other Vital Signs and Infusions') AND n.nursingchartcelltypevallabel IN ('Glasgow coma score','Score (Glasgow Coma Scale)'))
 OR (n.nursingchartcelltypecat='Vital Signs' AND n.nursingchartcelltypevallabel='O2 Admin Device' AND n.nursingchartcelltypevalname='O2 Admin Device'))
), fio2_raw AS (
 SELECT patientunitstayid,labresultoffset AS offset_min,1 AS priority,labresult AS val
 FROM lab_src WHERE labname='FiO2' AND labmeasurenamesystem='%' AND COALESCE(trim(labmeasurenameinterface),'') IN ('','%')
 UNION ALL
 SELECT patientunitstayid,respchartoffset,2,CASE WHEN trim(respchartvalue) ~ '^[+]?([0-9]+([.][0-9]*)?|[.][0-9]+)%?$' THEN rtrim(trim(respchartvalue),'%')::numeric END
 FROM resp_src WHERE lower(respchartvaluelabel) LIKE '%fio2%'
), fio2_valid AS (
 SELECT patientunitstayid,offset_min,priority,CASE WHEN val>=0.21 AND val<=1 THEN val*100 WHEN val>=21 AND val<=100 THEN val END AS fio2
 FROM fio2_raw
), fio2 AS MATERIALIZED (
 SELECT patientunitstayid,offset_min,priority,AVG(fio2) AS fio2 FROM fio2_valid WHERE fio2 IS NOT NULL GROUP BY 1,2,3
), airway AS MATERIALIZED (
 SELECT patientunitstayid,respcarestatusoffset AS offset_min,
 MIN(CASE WHEN lower(trim(airwaytype)) IN ('oral ett','nasal ett','tracheostomy') THEN 1 WHEN airwaytype='No Artificial Airway' THEN 0 ELSE -1 END) AS artificial
 FROM eicu_crd.respiratorycare JOIN cohort USING(patientunitstayid)
 WHERE respcarestatusoffset>=0 AND respcarestatusoffset<2880 AND COALESCE(trim(airwaytype),'')<>'' GROUP BY 1,2
), support_raw AS (
 SELECT patientunitstayid,nursingchartoffset AS offset_min,'device' AS kind,lower(trim(nursingchartvalue)) AS val FROM nurse_src WHERE nursingchartcelltypevallabel='O2 Admin Device'
 UNION ALL SELECT patientunitstayid,respchartoffset,'device',lower(trim(respchartvalue)) FROM resp_src WHERE respchartvaluelabel='Oxygen Delivery Method'
 UNION ALL SELECT patientunitstayid,respchartoffset,'switch',lower(trim(respchartvalue)) FROM resp_src WHERE respchartvaluelabel='RT Vent On/Off'
 UNION ALL SELECT patientunitstayid,respchartoffset,CASE WHEN respchartvaluelabel='Non-invasive Ventilation Mode' THEN 'niv_mode' ELSE 'mode' END,lower(trim(respchartvalue)) FROM resp_src WHERE respchartvaluelabel IN ('Mechanical Ventilator Mode','Ventilator Support Mode','Non-invasive Ventilation Mode')
 UNION ALL SELECT patientunitstayid,respchartoffset,'switch','off' FROM resp_src WHERE respchartvaluelabel='Oxygen Delivery Status' AND lower(trim(respchartvalue))='removed'
 UNION ALL SELECT patientunitstayid,offset_min,'airway','no artificial airway' FROM airway WHERE artificial=0
), support_events AS MATERIALIZED (
 SELECT patientunitstayid,offset_min,
 MIN(CASE WHEN kind IN ('niv_mode','airway') THEN 0 WHEN kind='device' AND val IN ('ventilator','mechanical ventilator','vent','vented','vent/ac','trach to vent','trach/vent','trach to ventilator','ventilator/trach','ventilatortrach','11-mechanical ventilator','ventilator/ac-25/peep10','ventilator/ac-26','ventilator/ac 16/tv 300/peep+5') THEN 1
 WHEN kind='device' AND val IN ('vent/cpap','ventilator/cpap','v-cpap') THEN 2
 WHEN kind='switch' AND val IN ('off','suspended') THEN 0
 WHEN kind='switch' AND val IN ('start','continued') THEN 2
 WHEN kind='mode' AND val IN ('ac/cmv','simv','cmv','spontaneous','cpap') THEN 2
 WHEN val ~ '(bipap|non.?invasive|nasal|cannula|canula|room.?air|trach.?collar|t-piece|mask|non-rebreather|high.?flow|hfnc)' OR val IN ('ra','r/a','r/air','nc','nch','vm','tc','hf','hhfnc','hflo','nrb','nimv','s/t') THEN 0
 ELSE -1 END) AS state
 FROM support_raw GROUP BY 1,2
), pf_pair AS MATERIALIZED (
 SELECT l.patientunitstayid,l.labid,l.labresultoffset AS offset_min,l.labresult AS pao2,
 f.fio2,f.offset_min AS fio2_offset,f.priority AS fio2_priority,
 s.state,s.offset_min AS support_offset,a.artificial,a.offset_min AS airway_offset
 FROM lab_src l
 LEFT JOIN LATERAL (SELECT f.* FROM fio2 f WHERE f.patientunitstayid=l.patientunitstayid AND f.offset_min<=l.labresultoffset AND f.offset_min>=l.labresultoffset-240 ORDER BY f.offset_min DESC,f.priority LIMIT 1) f ON true
 LEFT JOIN LATERAL (SELECT s.* FROM support_events s WHERE s.patientunitstayid=l.patientunitstayid AND s.offset_min<=l.labresultoffset AND s.offset_min>=l.labresultoffset-240 ORDER BY s.offset_min DESC LIMIT 1) s ON true
 LEFT JOIN LATERAL (SELECT a.* FROM airway a WHERE a.patientunitstayid=l.patientunitstayid AND a.offset_min<=l.labresultoffset AND a.offset_min>=l.labresultoffset-240 ORDER BY a.offset_min DESC LIMIT 1) a ON true
 WHERE l.labname='paO2' AND l.labmeasurenamesystem='mm Hg' AND l.labresult>0
), pf AS MATERIALIZED (
 SELECT *,pao2*100/fio2 AS pf,
 CASE WHEN state=1 OR (state=2 AND artificial=1) THEN true ELSE false END AS invasive,
 CASE WHEN (state=1 OR (state=2 AND artificial=1 AND offset_min-airway_offset<=60)) AND offset_min-support_offset<=60 THEN true ELSE false END AS invasive_1h,
 CASE WHEN state IS NULL OR state=-1 OR (state=2 AND COALESCE(artificial,-1)<>1) THEN 1 ELSE 0 END AS support_unknown
 FROM pf_pair WHERE fio2>0
), uo_whitelist(cellpath) AS (VALUES __UO_WHITELIST__),
uo_raw AS MATERIALIZED (
 SELECT i.* FROM eicu_crd.intakeoutput i JOIN cohort c USING(patientunitstayid)
 WHERE i.intakeoutputoffset>=0 AND i.intakeoutputoffset<2880 AND (i.cellpath IN (SELECT cellpath FROM uo_whitelist)
 OR (i.cellpath ILIKE 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|foley%' AND lower(i.cellpath) NOT LIKE '%pacu%' AND lower(i.cellpath) NOT LIKE '%or%' AND lower(i.cellpath) NOT LIKE '%ir%')
 OR i.cellpath LIKE 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Output%Urinary Catheter%'
 OR i.cellpath LIKE 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Output%Urethral Catheter%'
 OR i.cellpath LIKE 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Output (mL)%'
 OR i.cellpath LIKE 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Output%External Urethral%'
 OR i.cellpath LIKE 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urinary Catheter Output%')
 AND lower(i.cellpath) !~ '(count|unmeasured|mixed)'
), uo_dedup AS MATERIALIZED (
 SELECT DISTINCT ON (patientunitstayid,intakeoutputoffset,cellpath) * FROM uo_raw
 ORDER BY patientunitstayid,intakeoutputoffset,cellpath,intakeoutputentryoffset DESC NULLS LAST,intakeoutputid DESC
), uo_routes AS MATERIALIZED (
 SELECT *,CASE WHEN celllabel='Urinary Catheter Output: Indwelling/Continuous Nep' THEN 'unknown' WHEN lower(celllabel) LIKE '%nephro%' AND lower(celllabel) ~ '\m(left|l)\M' THEN 'nephro_left'
 WHEN lower(celllabel) LIKE '%nephro%' AND lower(celllabel) ~ '\m(right|r|rt)\M' THEN 'nephro_right'
 WHEN lower(celllabel) LIKE '%nephro%' THEN 'nephro_unsided' ELSE 'bladder' END AS route
 FROM uo_dedup WHERE cellvaluenumeric>=0
), uo_route_max AS MATERIALIZED (
 SELECT patientunitstayid,intakeoutputoffset AS offset_min,COUNT(*) AS source_rows,SUM(cellvaluenumeric) AS raw_sum,
 MAX(cellvaluenumeric) FILTER(WHERE route='bladder') AS bladder,
 MAX(cellvaluenumeric) FILTER(WHERE route='nephro_left') AS nephro_left,
 MAX(cellvaluenumeric) FILTER(WHERE route='nephro_right') AS nephro_right,
 MAX(cellvaluenumeric) FILTER(WHERE route='nephro_unsided') AS nephro_unsided,MAX(cellvaluenumeric) FILTER(WHERE route='unknown') AS unknown_route,COUNT(*) FILTER(WHERE route='unknown') AS unknown_route_rows
 FROM uo_routes GROUP BY 1,2
), uo_times AS MATERIALIZED (
 -- Generic Urine and urethral/foley detail may duplicate one bladder stream.
 -- Left/right nephrostomies are independent; unsided nephrostomy can duplicate their total.
 SELECT patientunitstayid,offset_min,source_rows,raw_sum,unknown_route_rows,
 GREATEST(COALESCE(bladder,0)+GREATEST(COALESCE(nephro_unsided,0),COALESCE(nephro_left,0)+COALESCE(nephro_right,0)),COALESCE(unknown_route,0)) AS volume,
 GREATEST(COALESCE(bladder,0),COALESCE(nephro_unsided,0),COALESCE(nephro_left,0)+COALESCE(nephro_right,0),COALESCE(unknown_route,0)) AS volume_conservative
 FROM uo_route_max
), uo_intervals AS MATERIALIZED (
 SELECT *,LAG(offset_min) OVER(PARTITION BY patientunitstayid ORDER BY offset_min) AS prior_offset FROM uo_times
), uo_windows AS MATERIALIZED (
 SELECT e.patientunitstayid,e.offset_min,SUM(u.volume) AS volume,SUM(u.volume_conservative) AS volume_conservative,SUM(u.offset_min-u.prior_offset)/60.0 AS coverage_h,
 MIN(u.prior_offset) AS collection_start,COUNT(*) AS interval_count,MAX(u.offset_min-u.prior_offset)/60.0 AS max_gap_h
 FROM uo_intervals e JOIN uo_intervals u ON u.patientunitstayid=e.patientunitstayid AND u.offset_min>=e.offset_min-1380 AND u.offset_min<=e.offset_min AND u.prior_offset IS NOT NULL
 WHERE e.offset_min>=1440 AND e.offset_min<2880 GROUP BY 1,2
), uo_eligible AS MATERIALIZED (
 SELECT *,volume*24/coverage_h AS uo_24hr,volume_conservative*24/coverage_h AS uo_24hr_conservative FROM uo_windows WHERE coverage_h BETWEEN 22 AND 30 AND collection_start>=0
), uo_hourly AS MATERIALIZED (
 SELECT patientunitstayid,offset_min/60 AS hr,MAX(uo_24hr) AS uo_24hr,MAX(uo_24hr_conservative) AS uo_24hr_conservative FROM uo_eligible GROUP BY 1,2
), infusion_raw AS MATERIALIZED (
 SELECT i.*,c.admissionweight,CASE WHEN lower(drugname) LIKE '%norepinephrine%' THEN 'nor' WHEN lower(drugname) LIKE '%epinephrine%' THEN 'epi' WHEN lower(drugname) LIKE '%dopamine%' THEN 'dop' WHEN lower(drugname) LIKE '%dobu%' THEN 'dob' END AS drug,
 CASE WHEN trim(drugrate) ~ '^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)$' THEN trim(drugrate)::numeric END AS rate,
 CASE WHEN trim(patientweight) ~ '^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)$' THEN trim(patientweight)::numeric END AS infusion_weight,
 CASE WHEN trim(drugamount) ~ '^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)$' THEN trim(drugamount)::numeric END AS amount_mg,
 CASE WHEN trim(volumeoffluid) ~ '^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)$' THEN trim(volumeoffluid)::numeric END AS fluid_ml
 FROM eicu_crd.infusiondrug i JOIN cohort c USING(patientunitstayid)
 WHERE infusionoffset>=0 AND infusionoffset<2880 AND (lower(drugname) LIKE '%epinephrine%' OR lower(drugname) LIKE '%dopamine%' OR lower(drugname) LIKE '%dobu%')
), med_concentration AS MATERIALIZED (
 -- Explicit infusion-bag products observed locally; no stock injection/vial concentration.
 SELECT m.patientunitstayid,m.drugorderoffset,m.drugstartoffset,m.drugstopoffset,
 CASE WHEN lower(m.drugname) LIKE '%norepi%' THEN 'nor' WHEN lower(m.drugname) LIKE '%epinephrine%' THEN 'epi' END AS drug,
 CASE WHEN lower(trim(m.drugname)) IN ('norepinephrine 8 mg in 250ml ns','250 ml plas cont : norepinephrine in d5w 8 mg/250 ml','epinephrine 8 mg in 250 ml ns') THEN 8.0/250
 WHEN lower(trim(m.drugname)) IN ('norepinephrine 4 mg/250 ml ns', 'norepinephrine 4 mg/250 ml ns infusion (std con''c) (repackage)','prmx norepinephrine 4 mg/250 ml ns drip') THEN 4.0/250 END AS mg_per_ml
 FROM eicu_crd.medication m JOIN cohort c USING(patientunitstayid)
 WHERE m.drugordercancelled='No' AND m.drugorderoffset>=0 AND m.drugorderoffset<2880 AND m.drugstartoffset<2880 AND m.drugstopoffset>0
 AND lower(trim(m.drugname)) IN ('norepinephrine 8 mg in 250ml ns','250 ml plas cont : norepinephrine in d5w 8 mg/250 ml','epinephrine 8 mg in 250 ml ns','norepinephrine 4 mg/250 ml ns','norepinephrine 4 mg/250 ml ns infusion (std con''c) (repackage)','prmx norepinephrine 4 mg/250 ml ns drip')
), infusion_concentration AS MATERIALIZED (
 SELECT i.*,CASE WHEN infusion_weight>0 THEN infusion_weight WHEN admissionweight>0 THEN admissionweight END AS weight_used,
 CASE WHEN amount_mg>0 AND fluid_ml>0 THEN amount_mg/fluid_ml WHEN m.n_concentrations=1 THEN m.mg_per_ml END AS mg_per_ml,
 CASE WHEN amount_mg>0 AND fluid_ml>0 THEN 'infusion' WHEN m.n_concentrations=1 THEN 'medication' ELSE NULL END AS concentration_source,
 COALESCE(m.n_concentrations,0) AS medication_concentration_count
 FROM infusion_raw i LEFT JOIN LATERAL (
 SELECT COUNT(DISTINCT mg_per_ml) AS n_concentrations,MIN(mg_per_ml) AS mg_per_ml FROM med_concentration m
 WHERE m.patientunitstayid=i.patientunitstayid AND m.drug=i.drug AND m.drugorderoffset<=i.infusionoffset AND m.drugstartoffset<=i.infusionoffset AND m.drugstopoffset>i.infusionoffset AND mg_per_ml>0
 ) m ON true
), vaso AS MATERIALIZED (
 SELECT *,CASE
 WHEN lower(drugname) LIKE '%(mcg/kg/min)%' THEN rate
 WHEN lower(drugname) LIKE '%(mg/kg/min)%' THEN rate*1000
 WHEN lower(drugname) LIKE '%(mcg/kg/hr)%' THEN rate/60
 WHEN lower(drugname) LIKE '%(nanograms/kg/min)%' THEN rate/1000
 WHEN lower(drugname) LIKE '%(mcg/min)%' AND weight_used>0 THEN rate/weight_used
 WHEN lower(drugname) LIKE '%(mcg/hr)%' AND weight_used>0 THEN rate/60/weight_used
 WHEN lower(drugname) LIKE '%(mg/hr)%' AND weight_used>0 THEN rate*1000/60/weight_used
 WHEN lower(drugname) LIKE '%(ml/hr)%' AND weight_used>0 AND mg_per_ml>0 THEN rate*mg_per_ml*1000/60/weight_used END AS mcgkgmin,
 (drug='dob' AND rate>0 AND lower(drugname) ~ '\((mcg/kg/min|mg/kg/min|mcg/kg/hr|mcg/min|mcg/hr|mg/hr|ml/hr)\)') AS dobu_positive
 FROM infusion_concentration WHERE drug IS NOT NULL AND rate>=0
), vaso_hourly AS MATERIALIZED (
 SELECT patientunitstayid,infusionoffset/60 AS hr,
 MAX(mcgkgmin) FILTER(WHERE drug='nor' AND mcgkgmin>0) AS nor,
 MAX(mcgkgmin) FILTER(WHERE drug='epi' AND mcgkgmin>0) AS epi,
 MAX(mcgkgmin) FILTER(WHERE drug='dop' AND mcgkgmin>0) AS dop,
 BOOL_OR(dobu_positive) AS dob,COUNT(*) FILTER(WHERE mcgkgmin IS NOT NULL OR dobu_positive) AS usable_n
 FROM vaso GROUP BY 1,2
), map_src AS MATERIALIZED (
 SELECT patientunitstayid,observationoffset AS offset_min,systemicmean AS map FROM eicu_crd.vitalperiodic JOIN cohort USING(patientunitstayid) WHERE observationoffset>=0 AND observationoffset<2880 AND systemicmean>0
 UNION ALL SELECT patientunitstayid,observationoffset,noninvasivemean FROM eicu_crd.vitalaperiodic JOIN cohort USING(patientunitstayid) WHERE observationoffset>=0 AND observationoffset<2880 AND noninvasivemean>0
), gcs AS MATERIALIZED (
 SELECT patientunitstayid,nursingchartoffset AS offset_min,gcs FROM (
 SELECT *,CASE WHEN trim(nursingchartvalue) ~ '^([0-9]+([.][0-9]*)?|[.][0-9]+)$' THEN trim(nursingchartvalue)::numeric END AS gcs FROM nurse_src
 WHERE (nursingchartcelltypevallabel='Glasgow coma score' AND nursingchartcelltypevalname='GCS Total') OR (nursingchartcelltypevallabel='Score (Glasgow Coma Scale)' AND nursingchartcelltypevalname='Value')
 ) x WHERE gcs>=3 AND gcs<=15 AND gcs=TRUNC(gcs)
), hourly AS MATERIALIZED (
 SELECT h.*,l.platelets,l.bilirubin,l.creatinine,p.pf_vent,p.pf_novent,p.pf_vent_1h,p.pf_novent_1h,v.nor,v.epi,v.dop,v.dob,v.usable_n,m.map,g.gcs,u.uo_24hr,u.uo_24hr_conservative
 FROM hours h
 LEFT JOIN (SELECT patientunitstayid,labresultoffset/60 AS hr,MIN(labresult) FILTER(WHERE labname='platelets x 1000' AND labresult>0 AND labmeasurenamesystem='K/mcL') AS platelets,MAX(labresult) FILTER(WHERE labname='total bilirubin' AND labresult>=0 AND labmeasurenamesystem='mg/dL') AS bilirubin,MAX(labresult) FILTER(WHERE labname='creatinine' AND labresult>0 AND labmeasurenamesystem='mg/dL') AS creatinine FROM lab_src GROUP BY 1,2) l USING(patientunitstayid,hr)
 LEFT JOIN (SELECT patientunitstayid,offset_min/60 AS hr,MIN(pf) FILTER(WHERE invasive) AS pf_vent,MIN(pf) FILTER(WHERE NOT invasive) AS pf_novent,MIN(pf) FILTER(WHERE invasive_1h) AS pf_vent_1h,MIN(pf) FILTER(WHERE NOT invasive_1h) AS pf_novent_1h FROM pf GROUP BY 1,2) p USING(patientunitstayid,hr)
 LEFT JOIN vaso_hourly v USING(patientunitstayid,hr)
 LEFT JOIN (SELECT patientunitstayid,offset_min/60 AS hr,MIN(map) AS map FROM map_src GROUP BY 1,2) m USING(patientunitstayid,hr)
 LEFT JOIN (SELECT patientunitstayid,offset_min/60 AS hr,MIN(gcs) AS gcs FROM gcs GROUP BY 1,2) g USING(patientunitstayid,hr)
 LEFT JOIN uo_hourly u USING(patientunitstayid,hr)
), components AS MATERIALIZED (
 SELECT *,
 CASE WHEN pf_vent<100 THEN 4 WHEN pf_vent<200 THEN 3 WHEN LEAST(pf_vent,pf_novent)<300 THEN 2 WHEN LEAST(pf_vent,pf_novent)<400 THEN 1 WHEN COALESCE(pf_vent,pf_novent) IS NOT NULL THEN 0 END AS respiration,
 CASE WHEN pf_vent_1h<100 THEN 4 WHEN pf_vent_1h<200 THEN 3 WHEN LEAST(pf_vent_1h,pf_novent_1h)<300 THEN 2 WHEN LEAST(pf_vent_1h,pf_novent_1h)<400 THEN 1 WHEN COALESCE(pf_vent_1h,pf_novent_1h) IS NOT NULL THEN 0 END AS respiration_1h,
 CASE WHEN platelets<20 THEN 4 WHEN platelets<50 THEN 3 WHEN platelets<100 THEN 2 WHEN platelets<150 THEN 1 WHEN platelets IS NOT NULL THEN 0 END AS coagulation,
 CASE WHEN bilirubin>=12 THEN 4 WHEN bilirubin>=6 THEN 3 WHEN bilirubin>=2 THEN 2 WHEN bilirubin>=1.2 THEN 1 WHEN bilirubin IS NOT NULL THEN 0 END AS liver,
 CASE WHEN dop>15 OR nor>0.1 OR epi>0.1 THEN 4 WHEN dop>5 OR nor>0 OR epi>0 THEN 3 WHEN dop>0 OR dob THEN 2 WHEN map<70 THEN 1 WHEN map IS NOT NULL OR usable_n>0 THEN 0 END AS cardiovascular,
 CASE WHEN gcs<6 THEN 4 WHEN gcs<10 THEN 3 WHEN gcs<13 THEN 2 WHEN gcs<15 THEN 1 WHEN gcs=15 THEN 0 END AS cns,
 CASE WHEN creatinine>=5 OR uo_24hr<200 THEN 4 WHEN creatinine>=3.5 OR uo_24hr<500 THEN 3 WHEN creatinine>=2 THEN 2 WHEN creatinine>=1.2 THEN 1 WHEN COALESCE(creatinine,uo_24hr) IS NOT NULL THEN 0 END AS renal,
 CASE WHEN creatinine>=5 OR uo_24hr_conservative<200 THEN 4 WHEN creatinine>=3.5 OR uo_24hr_conservative<500 THEN 3 WHEN creatinine>=2 THEN 2 WHEN creatinine>=1.2 THEN 1 WHEN COALESCE(creatinine,uo_24hr_conservative) IS NOT NULL THEN 0 END AS renal_uo_conservative
 FROM hourly
), agg AS (
 SELECT patientunitstayid,MAX(respiration) AS respiration,MAX(respiration_1h) AS respiration_1h,MAX(coagulation) AS coagulation,MAX(liver) AS liver,MAX(cardiovascular) AS cardiovascular,MAX(cns) AS cns,MAX(renal) AS renal,MAX(renal_uo_conservative) AS renal_uo_conservative,
 COUNT(respiration) AS respiration_hours,COUNT(coagulation) AS coagulation_hours,COUNT(liver) AS liver_hours,COUNT(cardiovascular) AS cardiovascular_hours,COUNT(cns) AS cns_hours,COUNT(renal) AS renal_hours,MAX(creatinine) AS creatinine_max,MIN(uo_24hr) AS uo_24hr_min,MIN(uo_24hr_conservative) AS uo_24hr_conservative_min
 FROM components GROUP BY patientunitstayid
), pf_qa AS (
 SELECT patientunitstayid,COUNT(*) AS paired_pao2_n,COUNT(*) FILTER(WHERE invasive) AS invasive_pao2_n,SUM(support_unknown) AS support_unknown_pao2_n,MAX(offset_min-fio2_offset) AS max_fio2_lag_min,MAX(offset_min-support_offset) AS max_support_lag_min,
 COUNT(*) FILTER(WHERE fio2_offset<0 OR fio2_offset>offset_min OR offset_min>=2880 OR offset_min<0) AS pf_pair_outside_n,
 COUNT(*) FILTER(WHERE support_offset<0 OR support_offset>offset_min OR airway_offset<0 OR airway_offset>offset_min OR support_offset>=2880 OR airway_offset>=2880) AS support_outside_n
 FROM pf GROUP BY 1
), uo_qa AS (
 SELECT u.patientunitstayid,COUNT(*) AS eligible_uo_endpoints,MIN(coverage_h) AS uo_coverage_min,MAX(coverage_h) AS uo_coverage_max,MIN(interval_count) AS uo_min_interval_count,MAX(interval_count) AS uo_max_interval_count,MAX(max_gap_h) AS uo_max_gap_h,
 COUNT(*) FILTER(WHERE collection_start<0 OR u.offset_min>=2880 OR u.offset_min<1440) AS uo_outside_n,
 COUNT(*) FILTER(WHERE u.uo_24hr=h.uo_24hr AND h.uo_24hr=a.uo_24hr_min AND interval_count=1) AS uo_single_interval_determining_endpoints
 FROM uo_eligible u JOIN uo_hourly h ON h.patientunitstayid=u.patientunitstayid AND h.hr=u.offset_min/60 JOIN agg a ON a.patientunitstayid=u.patientunitstayid GROUP BY u.patientunitstayid
), vaso_qa AS (
 SELECT patientunitstayid,COUNT(*) AS vaso_numeric_n,COUNT(*) FILTER(WHERE rate>0 AND mcgkgmin IS NULL AND NOT dobu_positive) AS unconverted_positive_vaso_n,
 COUNT(*) FILTER(WHERE mcgkgmin>0 AND concentration_source='medication' AND lower(drugname) LIKE '%(ml/hr)%') AS medication_converted_vaso_n,
 COUNT(*) FILTER(WHERE mcgkgmin>0 AND infusion_weight IS NULL AND weight_used>0 AND lower(drugname) !~ '\((mcg/kg/min|mg/kg/min|mcg/kg/hr|nanograms/kg/min)\)') AS admissionweight_used_n,
 COUNT(*) FILTER(WHERE infusionoffset<0 OR infusionoffset>=2880) AS vaso_outside_n,
 COUNT(*) FILTER(WHERE drug='epi' AND lower(drugname) LIKE '%(mg/kg/min)%' AND rate>0) AS epi_mgkgmin_positive_n,
 MAX(mcgkgmin) FILTER(WHERE drug='epi' AND lower(drugname) LIKE '%(mg/kg/min)%') AS epi_mgkgmin_max,
 MAX(mcgkgmin) FILTER(WHERE drug='epi' AND lower(drugname) NOT LIKE '%(mg/kg/min)%') AS epi_other_max,
 MAX(mcgkgmin) FILTER(WHERE drug='nor') AS nor_max,MAX(mcgkgmin) FILTER(WHERE drug='dop') AS dop_max
 FROM vaso GROUP BY 1
), source_boundary_rows AS (
 SELECT patientunitstayid,'lab' AS src,labresultoffset AS offset_min FROM lab_src
 UNION ALL SELECT patientunitstayid,'gcs',offset_min FROM gcs
 UNION ALL SELECT patientunitstayid,'map',offset_min FROM map_src
), source_boundaries AS (
 SELECT patientunitstayid,COUNT(*) FILTER(WHERE src='lab' AND (offset_min<0 OR offset_min>=2880)) AS lab_outside_n,
 COUNT(*) FILTER(WHERE src='gcs' AND (offset_min<0 OR offset_min>=2880)) AS gcs_outside_n,
 COUNT(*) FILTER(WHERE src='map' AND (offset_min<0 OR offset_min>=2880)) AS map_outside_n
 FROM source_boundary_rows GROUP BY 1
), uo_duplicates AS (
 SELECT patientunitstayid,COUNT(*) AS n,COUNT(*) FILTER(WHERE cellvaluenumeric<0) AS negative_n FROM uo_raw GROUP BY 1
), uo_dedup_counts AS (
 SELECT patientunitstayid,COUNT(*) AS n FROM uo_dedup GROUP BY 1
), uo_multichannel AS (
 SELECT patientunitstayid,COUNT(*) AS n FROM (
 SELECT patientunitstayid,intakeoutputoffset FROM uo_dedup WHERE cellvaluenumeric>=0 GROUP BY 1,2
 HAVING COUNT(*)>1 AND BOOL_OR(lower(celllabel)='urine') AND BOOL_OR(lower(celllabel)<>'urine')
 ) x GROUP BY 1
), boundaries AS (
 SELECT c.patientunitstayid,COALESCE(b.lab_outside_n,0) AS lab_outside_n,COALESCE(b.gcs_outside_n,0) AS gcs_outside_n,COALESCE(b.map_outside_n,0) AS map_outside_n,
 COALESCE(u.negative_n,0) AS negative_uo_excluded_n,COALESCE(v.n,0) AS nonnumeric_vaso_n,COALESCE(u.n-d.n,0) AS duplicate_uo_rows_removed,COALESCE(m.n,0) AS uo_generic_plus_channel_offsets,COALESCE(r.nonadditive_n,0) AS uo_nonadditive_offsets,COALESCE(r.ambiguous_n,0) AS uo_generic_nephro_ambiguous_offsets,COALESCE(r.unknown_n,0) AS uo_unknown_route_rows
 FROM cohort c LEFT JOIN source_boundaries b USING(patientunitstayid) LEFT JOIN uo_duplicates u USING(patientunitstayid) LEFT JOIN uo_dedup_counts d USING(patientunitstayid) LEFT JOIN uo_multichannel m USING(patientunitstayid)
 LEFT JOIN (SELECT patientunitstayid,COUNT(*) FILTER(WHERE rate IS NULL) AS n FROM infusion_raw GROUP BY 1) v USING(patientunitstayid)
 LEFT JOIN (SELECT patientunitstayid,COUNT(*) FILTER(WHERE raw_sum<>volume) AS nonadditive_n,COUNT(*) FILTER(WHERE volume<>volume_conservative) AS ambiguous_n,SUM(unknown_route_rows) AS unknown_n FROM uo_times GROUP BY 1) r USING(patientunitstayid)
)
SELECT c.patientunitstayid,c.uniquepid,c.label,a.respiration,a.coagulation,a.liver,a.cardiovascular,a.cns,a.renal,
 COALESCE(a.respiration,0)+COALESCE(a.coagulation,0)+COALESCE(a.liver,0)+COALESCE(a.cardiovascular,0)+COALESCE(a.cns,0)+COALESCE(a.renal,0) AS sofa_max_48h,
 a.renal_uo_conservative,
 COALESCE(a.respiration,0)+COALESCE(a.coagulation,0)+COALESCE(a.liver,0)+COALESCE(a.cardiovascular,0)+COALESCE(a.cns,0)+COALESCE(a.renal_uo_conservative,0) AS sofa_uo_conservative,
 a.respiration_1h,
 COALESCE(a.respiration_1h,0)+COALESCE(a.coagulation,0)+COALESCE(a.liver,0)+COALESCE(a.cardiovascular,0)+COALESCE(a.cns,0)+COALESCE(a.renal,0) AS sofa_support_1h,
 (a.respiration IS NOT NULL) AS respiration_observed,(a.coagulation IS NOT NULL) AS coagulation_observed,(a.liver IS NOT NULL) AS liver_observed,(a.cardiovascular IS NOT NULL) AS cardiovascular_observed,(a.cns IS NOT NULL) AS cns_observed,(a.renal IS NOT NULL) AS renal_observed,
 a.respiration_hours,a.coagulation_hours,a.liver_hours,a.cardiovascular_hours,a.cns_hours,a.renal_hours,
 a.creatinine_max,a.uo_24hr_min,a.uo_24hr_conservative_min,(a.renal>=3 AND COALESCE(a.creatinine_max,0)<3.5 AND a.uo_24hr_min<500) AS high_renal_uo_only,
 COALESCE(u.eligible_uo_endpoints,0) AS eligible_uo_endpoints,u.uo_coverage_min,u.uo_coverage_max,u.uo_min_interval_count,u.uo_max_interval_count,u.uo_max_gap_h,COALESCE(u.uo_single_interval_determining_endpoints,0) AS uo_single_interval_determining_endpoints,
 COALESCE(p.paired_pao2_n,0) AS paired_pao2_n,COALESCE(p.invasive_pao2_n,0) AS invasive_pao2_n,COALESCE(p.support_unknown_pao2_n,0) AS support_unknown_pao2_n,p.max_fio2_lag_min,p.max_support_lag_min,
 COALESCE(v.vaso_numeric_n,0) AS vaso_numeric_n,COALESCE(v.unconverted_positive_vaso_n,0) AS unconverted_positive_vaso_n,COALESCE(v.medication_converted_vaso_n,0) AS medication_converted_vaso_n,COALESCE(v.admissionweight_used_n,0) AS admissionweight_used_n,b.nonnumeric_vaso_n,b.negative_uo_excluded_n,b.duplicate_uo_rows_removed,b.uo_generic_plus_channel_offsets,b.uo_nonadditive_offsets,b.uo_generic_nephro_ambiguous_offsets,b.uo_unknown_route_rows,
 COALESCE(v.epi_mgkgmin_positive_n,0) AS epi_mgkgmin_positive_n,COALESCE(v.epi_mgkgmin_max>0.1 AND COALESCE(v.epi_other_max,0)<=0.1 AND COALESCE(v.nor_max,0)<=0.1 AND COALESCE(v.dop_max,0)<=15,false) AS cv4_due_epi_mgkgmin_only,
 b.lab_outside_n,COALESCE(p.pf_pair_outside_n,0) AS pf_pair_outside_n,COALESCE(u.uo_outside_n,0) AS uo_outside_n,COALESCE(v.vaso_outside_n,0) AS vaso_outside_n,b.gcs_outside_n,b.map_outside_n,COALESCE(p.support_outside_n,0) AS support_outside_n
FROM cohort c JOIN agg a USING(patientunitstayid) LEFT JOIN pf_qa p USING(patientunitstayid) LEFT JOIN uo_qa u USING(patientunitstayid) LEFT JOIN vaso_qa v USING(patientunitstayid) JOIN boundaries b USING(patientunitstayid)
ORDER BY c.patientunitstayid;
