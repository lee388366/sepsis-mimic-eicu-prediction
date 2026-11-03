-- =====================================================================
-- 02_extract_hourly.sql
-- 脓毒症队列小时级变量提取 —— MIMIC-IV
--
-- 对齐 Liu et al. 2025 (MSTD-GNN) 源码 hcv_reader.py 的 14 个 HCV 变量
-- + 4 个静态变量（SV），输出为宽表：每行 = stay_id × hour(0-47)。
--
-- HCV 变量（小时级，48h 窗）：
--   Heart Rate, SBP, DBP, MBP, Respiratory Rate, SpO2, Temperature,
--   Glucose, FiO2, pH, GCS total, GCS eyes, GCS motor, GCS verbal
--   （Capillary Refill 在 MIMIC-IV 无有效记录，已从特征集中排除）
-- SV 静态变量：Age, Gender, Weight, Height
--
-- 聚合规则：每小时内取均值（mean）；完全缺失的单元格留空，
-- 后续按 Liu 的规则处理（nearest 插值 / 均值填充 / 固定值填充）。
--
-- 使用方法：
--   psql -h localhost -U postgres -d mimiciv31 -Atc \
--     "COPY (SELECT * FROM (...)) TO STDOUT WITH CSV HEADER" > sepsis_hourly.csv
-- =====================================================================

WITH sepsis AS (
    -- Sepsis-3 定义的患者（与 01 一致）
    SELECT
        stay_id,
        suspected_infection_time AS t_inf
    FROM mimiciv_derived.sepsis3
    WHERE sepsis3 IS TRUE
      AND suspected_infection_time IS NOT NULL
),

icu_stay_count AS (
    -- 每次住院 ICU 入住次数（排除 >2 次，对齐 Liu）
    SELECT hadm_id, COUNT(*) AS n_icu_stays
    FROM mimiciv_icu.icustays
    GROUP BY hadm_id
),

cohort AS (
    -- 队列：脓毒症 ∩ 成人 ∩ ICU LOS>=48h（与 01_sepsis_cohort.sql 一致）
    -- rn：同一次住院内按 intime 排序的 ICU 序号，最外层 WHERE rn=1 只保留首次 ICU，
    --     与 04_labels.sql / 05 训练口径一致（同时避免 bg/chem 按 hadm 关联被复制到多次 ICU）
    SELECT
        ie.subject_id,
        ie.hadm_id,
        ie.stay_id,
        ie.intime,
        ie.outtime,
        s.t_inf,
        p.gender,
        p.anchor_age AS age,
        adm.hospital_expire_flag,
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
),

hours AS (
    -- 0-47 小时网格
    SELECT generate_series(0, 47) AS hr
),

-- 1) 生命体征（HR / SBP / DBP / MBP / RR / SpO2 / Temp / Glucose）
vital AS (
    SELECT
        v.stay_id,
        FLOOR(EXTRACT(EPOCH FROM (v.charttime - ie.intime)) / 3600)::int AS hr,
        AVG(v.heart_rate)   AS heart_rate,
        AVG(v.sbp)          AS sbp,
        AVG(v.dbp)          AS dbp,
        AVG(v.mbp)          AS mbp,
        AVG(v.resp_rate)    AS resp_rate,
        AVG(v.spo2)         AS spo2,
        AVG(v.temperature)  AS temperature,
        AVG(v.glucose)      AS glucose_vital
    FROM mimiciv_derived.vitalsign v
    INNER JOIN mimiciv_icu.icustays ie ON v.stay_id = ie.stay_id
    WHERE v.stay_id IN (SELECT stay_id FROM cohort)
      AND v.charttime >= ie.intime
      AND v.charttime < ie.intime + INTERVAL '48 hours'
    GROUP BY v.stay_id, hr
),

-- 2) 血气（pH / FiO2；bg 按 hadm_id 关联）
bg AS (
    SELECT
        ie.stay_id,
        FLOOR(EXTRACT(EPOCH FROM (b.charttime - ie.intime)) / 3600)::int AS hr,
        AVG(b.ph)                               AS ph,
        AVG(COALESCE(b.fio2, b.fio2_chartevents)) AS fio2,
        AVG(b.glucose)                          AS glucose_bg
    FROM mimiciv_derived.bg b
    INNER JOIN mimiciv_icu.icustays ie ON b.hadm_id = ie.hadm_id
    WHERE ie.stay_id IN (SELECT stay_id FROM cohort)
      AND b.charttime >= ie.intime
      AND b.charttime < ie.intime + INTERVAL '48 hours'
    GROUP BY ie.stay_id, hr
),

-- 3) 生化（补充 Glucose，用 COALESCE 保覆盖率；chemistry 按 hadm_id 关联）
chem AS (
    SELECT
        ie.stay_id,
        FLOOR(EXTRACT(EPOCH FROM (c.charttime - ie.intime)) / 3600)::int AS hr,
        AVG(c.glucose) AS glucose_chem
    FROM mimiciv_derived.chemistry c
    INNER JOIN mimiciv_icu.icustays ie ON c.hadm_id = ie.hadm_id
    WHERE ie.stay_id IN (SELECT stay_id FROM cohort)
      AND c.charttime >= ie.intime
      AND c.charttime < ie.intime + INTERVAL '48 hours'
    GROUP BY ie.stay_id, hr
),

-- 4) GCS（总分 / 眼 / 运动 / 语言）
gcs AS (
    SELECT
        g.stay_id,
        FLOOR(EXTRACT(EPOCH FROM (g.charttime - ie.intime)) / 3600)::int AS hr,
        AVG(g.gcs)         AS gcs,
        AVG(g.gcs_eyes)    AS gcs_eyes,
        AVG(g.gcs_motor)   AS gcs_motor,
        AVG(g.gcs_verbal)  AS gcs_verbal
    FROM mimiciv_derived.gcs g
    INNER JOIN mimiciv_icu.icustays ie ON g.stay_id = ie.stay_id
    WHERE g.stay_id IN (SELECT stay_id FROM cohort)
      AND g.charttime >= ie.intime
      AND g.charttime < ie.intime + INTERVAL '48 hours'
    GROUP BY g.stay_id, hr
),

-- 5) 静态：Weight / Height（SV 变量）
static_wt AS (
    SELECT stay_id, weight
    FROM mimiciv_derived.first_day_weight
),
static_ht AS (
    SELECT stay_id, height
    FROM mimiciv_derived.first_day_height
)

SELECT
    c.stay_id,
    c.subject_id,
    c.hadm_id,
    c.age,
    c.gender,
    h.hr,
    v.heart_rate,
    v.sbp,
    v.dbp,
    v.mbp,
    v.resp_rate,
    v.spo2,
    v.temperature,
    COALESCE(v.glucose_vital, b.glucose_bg, ch.glucose_chem) AS glucose,
    b.ph,
    b.fio2,
    g.gcs,
    g.gcs_eyes,
    g.gcs_motor,
    g.gcs_verbal,
    sw.weight,
    sh.height,
    c.t_inf,
    c.hospital_expire_flag
FROM cohort c
CROSS JOIN hours h
LEFT JOIN vital v    ON v.stay_id = c.stay_id AND v.hr = h.hr
LEFT JOIN bg b       ON b.stay_id = c.stay_id AND b.hr = h.hr
LEFT JOIN chem ch    ON ch.stay_id = c.stay_id AND ch.hr = h.hr
LEFT JOIN gcs g      ON g.stay_id = c.stay_id AND g.hr = h.hr
LEFT JOIN static_wt sw ON sw.stay_id = c.stay_id
LEFT JOIN static_ht sh ON sh.stay_id = c.stay_id
WHERE c.rn = 1
ORDER BY c.stay_id, h.hr;
