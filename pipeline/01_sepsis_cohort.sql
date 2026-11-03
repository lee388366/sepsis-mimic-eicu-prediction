-- =====================================================================
-- 01_sepsis_cohort.sql
-- 脓毒症（Sepsis-3）ICU 成人队列 —— MIMIC-IV
--
-- 目标：复刻 Liu et al. 2025 (MSTD-GNN, IEEE JBHI) 的队列构建逻辑，
--       仅将疾病从「心衰（HF）」替换为「脓毒症（Sepsis-3）」。
--
-- 设计决策（与 Liu 对齐）：
--   1. 疾病定义：mimiciv_derived.sepsis3 表 sepsis3 = TRUE
--   2. 成人：anchor_age >= 18
--   3. ICU LOS >= 48 小时（保证 48h 特征窗完整 + 对齐 Liu）
--   4. 同一次住院内 ICU 入住次数 > 2 次者排除（Liu：排除转科/多次入 ICU）
--   5. 多次住院视为独立样本
--   6. 结局：in-hospital mortality（hospital_expire_flag）
--   7. t0 = ICU 入室时间（intime）；特征窗 [intime, intime + 48h)
--
-- 输出列说明：
--   rn    = 同一 hadm_id 内按 intime 排序的序号；
--            如需「每次住院只取首次 ICU」，在最终 SELECT 外再套一层 WHERE rn = 1
--   los_h = ICU 停留时长（小时）
--   t_inf = 疑似感染时间（Sepsis-3），供后续特征窗 / eICU 代理对齐使用
--
-- 使用方法（Navicat 直接运行，或 psql）：
--   psql -h localhost -U postgres -d mimiciv31 -f 01_sepsis_cohort.sql
-- =====================================================================

WITH sepsis AS (
    -- Sepsis-3 定义的患者（保留疑似感染时间，供 eICU 外验代理对齐）
    SELECT
        stay_id,
        subject_id,
        suspected_infection_time AS t_inf
    FROM mimiciv_derived.sepsis3
    WHERE sepsis3 IS TRUE
      AND suspected_infection_time IS NOT NULL
),

-- 每次住院的 ICU 入住次数（排除转科 / 多次入 ICU，对齐 Liu）
icu_stay_count AS (
    SELECT
        hadm_id,
        COUNT(*) AS n_icu_stays
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
        ie.first_careunit,
        ie.last_careunit,
        s.t_inf,
        p.gender,
        p.anchor_age AS age,
        adm.admission_type,
        adm.hospital_expire_flag,
        EXTRACT(EPOCH FROM (ie.outtime - ie.intime)) / 3600.0 AS los_h
    FROM mimiciv_icu.icustays ie
    INNER JOIN sepsis s ON ie.stay_id = s.stay_id
    INNER JOIN mimiciv_hosp.patients p ON ie.subject_id = p.subject_id
    INNER JOIN mimiciv_hosp.admissions adm ON ie.hadm_id = adm.hadm_id
    INNER JOIN icu_stay_count cnt ON ie.hadm_id = cnt.hadm_id
    WHERE p.anchor_age >= 18
      AND ie.outtime >= ie.intime + INTERVAL '48 hours'   -- ICU LOS >= 48h
      AND adm.hospital_expire_flag IS NOT NULL
      AND cnt.n_icu_stays <= 2                            -- Liu：排除同次住院 >2 次 ICU
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
    subject_id,
    hadm_id,
    stay_id,
    intime,
    outtime,
    t_inf,
    age,
    gender,
    admission_type,
    first_careunit,
    last_careunit,
    hospital_expire_flag,   -- 结局：1 = 住院死亡，0 = 存活出院
    los_h,
    rn
FROM ranked
ORDER BY stay_id;
