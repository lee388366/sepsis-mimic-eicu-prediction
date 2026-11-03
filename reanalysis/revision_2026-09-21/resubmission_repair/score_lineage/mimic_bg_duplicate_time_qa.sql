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
, bg_src AS (
SELECT c.stay_id,le.specimen_id,le.itemid,le.charttime,le.valuenum
FROM cohort c JOIN mimiciv_hosp.labevents le ON le.subject_id=c.subject_id
AND le.charttime>=c.intime AND le.charttime<c.intime+INTERVAL '48 hour'
WHERE le.itemid IN (52033,50821,50816) AND (le.hadm_id=c.hadm_id OR le.hadm_id IS NULL)
),groups AS (SELECT stay_id,specimen_id,itemid,COUNT(*) AS n,COUNT(DISTINCT charttime) AS time_n FROM bg_src GROUP BY stay_id,specimen_id,itemid)
SELECT COUNT(*) AS specimen_item_groups,COUNT(*) FILTER(WHERE n>1) AS repeated_groups,COUNT(*) FILTER(WHERE time_n>1) AS distinct_time_groups FROM groups;
ROLLBACK;