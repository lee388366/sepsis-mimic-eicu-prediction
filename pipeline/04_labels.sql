-- =====================================================================
-- 04_labels.sql
-- 结局标签：in-hospital mortality —— MIMIC-IV 脓毒症队列
--
-- 输出：每行 = 一个 ICU stay，含结局标签。
-- 结局定义：hospital_expire_flag = 1 → 住院死亡；0 → 存活出院
--          （in-hospital mortality，与 Liu 论文正文一致，也与你第一版 SA-AKI 一致）
--
-- 队列与 01_sepsis_cohort.sql 完全一致：
--   Sepsis-3 ∩ 成人 ∩ ICU LOS>=48h ∩ 同次住院 ICU<=2 次
--
-- 注意：此文件只输出「唯一的 stay」（rn=1，同次住院仅取首次 ICU），
--       与训练时患者的划分保持一致。
--
-- 使用方法：
--   psql -h localhost -U postgres -d mimiciv31 -Atc \
--     "COPY (SELECT * FROM (...)) TO STDOUT WITH CSV HEADER" > sepsis_labels.csv
-- =====================================================================

WITH sepsis AS (
    SELECT stay_id, suspected_infection_time AS t_inf
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
        s.t_inf,
        p.anchor_age AS age,
        adm.hospital_expire_flag,
        EXTRACT(EPOCH FROM (ie.outtime - ie.intime)) / 3600.0 AS los_h
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
)

SELECT
    stay_id,
    subject_id,
    hadm_id,
    intime,
    outtime,
    t_inf,
    hospital_expire_flag AS label,   -- 结局：1=住院死亡，0=存活出院
    los_h
FROM ranked
WHERE rn = 1
ORDER BY stay_id;
