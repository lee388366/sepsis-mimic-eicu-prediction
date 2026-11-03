-- =====================================================================
-- 04b_extract_sofa.sql
-- SOFA（Sequential Organ Failure Assessment）核心对标评分 —— 48h 窗口提取
--
-- 目的：为"ML 模型 vs 标准临床评分"提供第二个对标工具（qSOFA 之后）。
--       复用 MIMIC-IV 官方验证过的 mimiciv_derived.sofa 表（各组分已按
--       Vincent 1998 算法计算），只做三件事：
--         1. 复刻 04_labels.sql 的队列口径（保证与项目 24894 例完全对齐）
--         2. 限制在 ICU 前 48 小时（hr 0-47，与 ML 特征完全同窗，防"偷看未来"）
--         3. 每个器官组分取 48h 窗口内最差值（max SOFA，标准做法）
--
-- 队列口径（与 01/02/04 完全一致）：
--   Sepsis-3 ∩ 成人(≥18) ∩ ICU LOS≥48h ∩ 同住院 ICU≤2次 ∩ rn=1（首次 ICU）
--
-- 输出列：
--   stay_id            唯一 ICU stay
--   respiration/coagulation/liver/cardiovascular/cns/renal
--                      各组分 48h 最差评分（0-4 或 0-4；缺失按 0）
--   sofa_max_48h       六组分之和（0-24）
--   pao2fio2ratio_min  （验证用）48h 最低 PaO2/FiO2 比值
--   platelet_min        （验证用）48h 最低血小板
--   bilirubin_max       （验证用）48h 最高胆红素
--   creatinine_max      （验证用）48h 最高肌酐
--   meanbp_min          （验证用）48h 最低平均动脉压
--   gcs_min             （验证用）48h 最低 GCS
--   uo_24hr             （验证用）任意 24h 窗尿量（取最大连续 24h 累计）
--
-- 使用方法（与 01/02/04 相同）：
--   psql -h localhost -U postgres -d mimiciv31 -Atc \
--     "COPY (SELECT * FROM (...)) TO STDOUT WITH CSV HEADER" > sofa_extract.csv
--   （或直接用 psql -f 本文件，末尾的 SELECT 即最终输出）
-- =====================================================================

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
    SELECT stay_id FROM ranked WHERE rn = 1
),

-- SOFA 表 48h 窗口内的逐行数据（hr 0-47）
sofa48 AS (
    SELECT
        s.stay_id,
        s.respiration,
        s.coagulation,
        s.liver,
        s.cardiovascular,
        s.cns,
        s.renal,
        s.pao2fio2ratio_novent,
        s.pao2fio2ratio_vent,
        s.platelet_min,
        s.bilirubin_max,
        s.creatinine_max,
        s.meanbp_min,
        s.gcs_min,
        s.uo_24hr
    FROM mimiciv_derived.sofa s
    INNER JOIN cohort c ON s.stay_id = c.stay_id
    WHERE s.hr < 48
),

-- 每 stay 聚合：组分取最差值，原始值取极值（用于后续验证/敏感性）
agg AS (
    SELECT
        stay_id,
        MAX(COALESCE(respiration, 0))   AS respiration,
        MAX(COALESCE(coagulation, 0))   AS coagulation,
        MAX(COALESCE(liver, 0))         AS liver,
        MAX(COALESCE(cardiovascular,0)) AS cardiovascular,
        MAX(COALESCE(cns, 0))           AS cns,
        MAX(COALESCE(renal, 0))         AS renal,
        MIN(pao2fio2ratio_novent)       AS pao2fio2_novent_min,
        MIN(pao2fio2ratio_vent)         AS pao2fio2_vent_min,
        MIN(platelet_min)               AS platelet_min,
        MAX(bilirubin_max)              AS bilirubin_max,
        MAX(creatinine_max)             AS creatinine_max,
        MIN(meanbp_min)                 AS meanbp_min,
        MIN(gcs_min)                    AS gcs_min,
        MAX(uo_24hr)                    AS uo_24hr_max
    FROM sofa48
    GROUP BY stay_id
),

-- 未出现在 sofa48 的患者（理论上不应存在，但保险起见补全）
all_stays AS (
    SELECT c.stay_id
    FROM cohort c
    LEFT JOIN agg a ON c.stay_id = a.stay_id
)

SELECT
    a.stay_id,
    COALESCE(a.respiration, 0)   AS respiration,
    COALESCE(a.coagulation, 0)   AS coagulation,
    COALESCE(a.liver, 0)         AS liver,
    COALESCE(a.cardiovascular, 0) AS cardiovascular,
    COALESCE(a.cns, 0)           AS cns,
    COALESCE(a.renal, 0)         AS renal,
    COALESCE(a.respiration,0) + COALESCE(a.coagulation,0) + COALESCE(a.liver,0)
      + COALESCE(a.cardiovascular,0) + COALESCE(a.cns,0) + COALESCE(a.renal,0)
                                 AS sofa_max_48h,
    a.pao2fio2_novent_min,
    a.pao2fio2_vent_min,
    a.platelet_min,
    a.bilirubin_max,
    a.creatinine_max,
    a.meanbp_min,
    a.gcs_min,
    a.uo_24hr_max
FROM all_stays s
LEFT JOIN agg a ON s.stay_id = a.stay_id
ORDER BY a.stay_id;
