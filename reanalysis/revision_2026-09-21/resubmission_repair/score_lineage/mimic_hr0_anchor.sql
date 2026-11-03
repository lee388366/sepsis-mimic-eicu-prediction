BEGIN READ ONLY; SET LOCAL statement_timeout='60s';
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
)
SELECT COUNT(*) AS n_hr0, MIN(EXTRACT(EPOCH FROM(s.endtime-ie.intime))/3600.0) AS hr0_end_minus_intime_min_h, MAX(EXTRACT(EPOCH FROM(s.endtime-ie.intime))/3600.0) AS hr0_end_minus_intime_max_h, percentile_cont(0.5) WITHIN GROUP(ORDER BY EXTRACT(EPOCH FROM(s.endtime-ie.intime))/3600.0) AS hr0_end_minus_intime_median_h, COUNT(*) FILTER(WHERE s.endtime=date_trunc('hour',it.intime_hr)+CASE WHEN it.intime_hr=date_trunc('hour',it.intime_hr) THEN INTERVAL '0 hour' ELSE INTERVAL '1 hour' END) AS matches_official_hr_anchor FROM mimiciv_derived.sofa s JOIN cohort c USING(stay_id) JOIN mimiciv_icu.icustays ie USING(stay_id) JOIN mimiciv_derived.icustay_times it USING(stay_id) WHERE s.hr=0;
ROLLBACK;