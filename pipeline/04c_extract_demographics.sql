-- =====================================================================
-- 04c_extract_demographics.sql
-- 人口学字段提取（年龄 / 性别 / 种族 / 年龄组）—— 按 stay_id 索引
--
-- 用途：
--   1. TRIPOD+AI 3c 条目（健康不平等）：按种族×性别×年龄组汇总死亡率；
--   2. 与 baseline / all_predictions 按 stay_id 合并做亚组分析。
--
-- 队列与 04_labels.sql 完全一致（24894 例）：
--   Sepsis-3 ∩ 成人(≥18) ∩ ICU LOS>=48h ∩ 同次住院 ICU<=2 次 ∩ 每住院取首次 ICU
--
-- 种族归并口径：官方标准六分类
--   White / Black / Asian / Hispanic / Other / Unknown-Refused
--   （MIMIC-IV admissions.race 为 33 个细粒度原值，见 CASE 映射）
--
-- 使用方法：
--   psql -h localhost -U postgres -d mimiciv31 -Atc \
--     "COPY (SELECT * FROM (...)) TO STDOUT WITH CSV HEADER" > demographics_by_stay.csv
-- =====================================================================

WITH sepsis AS (
    SELECT stay_id
    FROM mimiciv_derived.sepsis3
    WHERE sepsis3 IS TRUE AND suspected_infection_time IS NOT NULL
),

icu_stay_count AS (
    SELECT hadm_id, COUNT(*) AS n_icu_stays
    FROM mimiciv_icu.icustays
    GROUP BY hadm_id
),

cohort AS (
    SELECT
        ie.subject_id,
        ie.hadm_id,
        ie.stay_id,
        ie.intime,
        p.anchor_age AS age,
        p.gender,
        adm.hospital_expire_flag AS label,
        adm.race,
        ROW_NUMBER() OVER (
            PARTITION BY ie.hadm_id
            ORDER BY ie.intime ASC, ie.stay_id ASC
        ) AS rn
    FROM mimiciv_icu.icustays ie
    INNER JOIN sepsis s ON ie.stay_id = s.stay_id
    INNER JOIN mimiciv_hosp.patients p ON ie.subject_id = p.subject_id
    INNER JOIN mimiciv_hosp.admissions adm ON ie.hadm_id = adm.hadm_id
    INNER JOIN icu_stay_count cnt ON ie.hadm_id = cnt.hadm_id
    WHERE p.anchor_age >= 18
      AND ie.outtime >= ie.intime + INTERVAL '48 hours'
      AND adm.hospital_expire_flag IS NOT NULL
      AND cnt.n_icu_stays <= 2
)

SELECT
    stay_id,
    age,
    gender,
    CASE WHEN age < 65 THEN '<65' ELSE '>=65' END AS age_group,
    CASE
        WHEN race LIKE 'WHITE%' THEN 'White'
        WHEN race LIKE 'BLACK%' THEN 'Black'
        WHEN race LIKE 'ASIAN%' THEN 'Asian'
        WHEN race LIKE 'HISPANIC%' THEN 'Hispanic'
        WHEN race IN ('AMERICAN INDIAN/ALASKA NATIVE',
                      'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
                      'MULTIPLE RACE/ETHNICITY',
                      'OTHER', 'PORTUGUESE', 'SOUTH AMERICAN') THEN 'Other'
        WHEN race IN ('UNKNOWN', 'UNABLE TO OBTAIN',
                      'PATIENT DECLINED TO ANSWER') THEN 'Unknown/Refused'
        ELSE 'Other'
    END AS race_group,
    label
FROM cohort
WHERE rn = 1
ORDER BY stay_id;
