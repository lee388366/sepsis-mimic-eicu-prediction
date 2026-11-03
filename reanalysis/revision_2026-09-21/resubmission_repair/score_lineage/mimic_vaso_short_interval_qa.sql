BEGIN READ ONLY; SET LOCAL statement_timeout='120s'; SET LOCAL enable_nestloop=off;
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
, source AS (
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
FROM source;
ROLLBACK;