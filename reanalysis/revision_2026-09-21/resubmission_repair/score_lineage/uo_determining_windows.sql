WITH sepsis AS (
    SELECT stay_id
    FROM mimiciv_derived.sepsis3
    WHERE sepsis3 IS TRUE
      AND suspected_infection_time IS NOT NULL
),

icu_stay_count AS (
    SELECT hadm_id, COUNT(*) AS n_icu_stays
    FROM mimiciv_icu.icustays
    GROUP BY hadm_id
),

base AS (
    SELECT
        ie.subject_id,
        ie.hadm_id,
        ie.stay_id,
        ie.intime,
        ie.outtime,
        p.anchor_age AS age,
        adm.hospital_expire_flag
    FROM mimiciv_icu.icustays ie
    INNER JOIN sepsis s ON ie.stay_id = s.stay_id
    INNER JOIN mimiciv_hosp.patients p ON ie.subject_id = p.subject_id
    INNER JOIN mimiciv_hosp.admissions adm ON ie.hadm_id = adm.hadm_id
    INNER JOIN icu_stay_count cnt ON ie.hadm_id = cnt.hadm_id
    WHERE p.anchor_age >= 18
      AND ie.outtime >= ie.intime + INTERVAL '48 hours'
      AND adm.hospital_expire_flag IS NOT NULL
      AND cnt.n_icu_stays <= 2
),

ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY hadm_id
            ORDER BY intime ASC, stay_id ASC
        ) AS rn
    FROM base
),

cohort AS (
    SELECT stay_id, subject_id, hadm_id, intime FROM ranked WHERE rn = 1
)
, uo_base AS MATERIALIZED (
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
), by_hour AS (
 SELECT stay_id,FLOOR(EXTRACT(EPOCH FROM(charttime-intime))/3600)::int AS hr,MAX(uo_24hr) AS hourly_uo
 FROM uo_eligible GROUP BY stay_id,hr
), worst AS (SELECT stay_id,MIN(hourly_uo) AS worst_uo FROM by_hour GROUP BY stay_id), determining AS (
 SELECT u.* FROM uo_eligible u JOIN by_hour h ON h.stay_id=u.stay_id
 AND h.hr=FLOOR(EXTRACT(EPOCH FROM(u.charttime-u.intime))/3600)::int AND h.hourly_uo=u.uo_24hr
 JOIN worst w ON w.stay_id=h.stay_id AND w.worst_uo=h.hourly_uo
)
SELECT stay_id,MIN(uo_24hr) AS determining_uo,
MIN(interval_count) AS determining_min_intervals,MAX(interval_count) AS determining_max_intervals,
MIN(max_gap_h) AS determining_min_maxgap,MAX(max_gap_h) AS determining_max_maxgap,
COUNT(*) AS determining_endpoints,
COUNT(*) FILTER(WHERE interval_count=1) AS determining_single_interval_endpoints
FROM determining GROUP BY stay_id ORDER BY stay_id;