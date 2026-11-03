-- =====================================================================
-- mimic_08_airway_chartevents.sql —— 气道类型/通气方式（护理记录层）
--
-- 与 mimic_06（派生时段）、mimic_07（操作事件）互为补充：
-- 本层是护士在床旁登记的"当前气道与支持方式"文本，可用来定位
-- 插管/拔管的具体时刻，以及核查"某评分时点是否在支持中"。
--
-- itemid（已由 d_items 核验，均属 Respiratory / General 类别）：
--   223836 Airway Type                → 气道类型文本（如 Endotracheal tube / Tracheostomy）
--   227567 Airway Location             → 气道位置
--   227568 Airway Size (mm)            → 管径
--   226815 Airway problems             → 气道问题
--   223059 Intubation - Details        → 插管细节
--   226260 Mechanically Ventilated     → 机械通气勾选（Checkbox）
--   223848 Ventilator Type             → 呼吸机型号
--   223849 Ventilator Mode             → 通气模式
--   229314 Ventilator Mode (Hamilton)  → 通气模式（Hamilton 机型）
--   226732 O2 Delivery Device(s)       → 氧疗装置（含面罩/鼻导管/高流量等）
--
-- 样本：本脚本原样导出 value 文本，不做"是否通气"的判定；
--       判定逻辑留给分析阶段，并须与 mimic_06/07 交叉验证。
-- 数据库：mimiciv31（只读）
-- =====================================================================
WITH sepsis AS (
    SELECT stay_id FROM mimiciv_derived.sepsis3
    WHERE sepsis3 IS TRUE AND suspected_infection_time IS NOT NULL
),
icu_stay_count AS (
    SELECT hadm_id, COUNT(*) AS n_icu_stays FROM mimiciv_icu.icustays GROUP BY hadm_id
),
base AS (
    SELECT ie.subject_id, ie.hadm_id, ie.stay_id, ie.intime,
           ie.intime - INTERVAL '6 hours'  AS win_start,
           ie.intime + INTERVAL '48 hours' AS win_end
    FROM mimiciv_icu.icustays ie
    INNER JOIN sepsis s                     ON ie.stay_id   = s.stay_id
    INNER JOIN mimiciv_hosp.patients p      ON ie.subject_id = p.subject_id
    INNER JOIN mimiciv_hosp.admissions adm  ON ie.hadm_id   = adm.hadm_id
    INNER JOIN icu_stay_count cnt           ON ie.hadm_id   = cnt.hadm_id
    WHERE p.anchor_age >= 18
      AND ie.outtime >= ie.intime + INTERVAL '48 hours'
      AND adm.hospital_expire_flag IS NOT NULL
      AND cnt.n_icu_stays <= 2
),
ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime ASC, stay_id ASC) AS rn
    FROM base
),
cohort AS (SELECT * FROM ranked WHERE rn = 1)
SELECT
    ce.subject_id, ce.hadm_id, ce.stay_id,
    ce.charttime,
    ROUND((EXTRACT(EPOCH FROM (ce.charttime - c.intime)) / 3600.0)::numeric, 4) AS hours_from_intime,
    ce.itemid, di.label AS item_label, di.category AS item_category,
    ce.value AS value_raw, ce.valuenum, ce.valueuom,
    CASE
        WHEN ce.itemid = 223836 THEN 'airway_type'
        WHEN ce.itemid = 227567 THEN 'airway_location'
        WHEN ce.itemid = 227568 THEN 'airway_size'
        WHEN ce.itemid = 226815 THEN 'airway_problem'
        WHEN ce.itemid = 223059 THEN 'intubation_details'
        WHEN ce.itemid = 226260 THEN 'mechanically_ventilated_flag'
        WHEN ce.itemid = 223848 THEN 'ventilator_type'
        WHEN ce.itemid = 223849 THEN 'ventilator_mode'
        WHEN ce.itemid = 229314 THEN 'ventilator_mode_hamilton'
        WHEN ce.itemid = 226732 THEN 'o2_delivery_device'
    END AS airway_field,
    ce.storetime, ce.warning
FROM mimiciv_icu.chartevents ce
INNER JOIN cohort c ON ce.stay_id = c.stay_id
LEFT JOIN mimiciv_icu.d_items di ON ce.itemid = di.itemid
WHERE ce.itemid IN (223836, 227567, 227568, 226815, 223059,
                    226260, 223848, 223849, 229314, 226732)
  AND ce.charttime >= c.win_start
  AND ce.charttime <= c.win_end
ORDER BY ce.stay_id, ce.charttime, ce.itemid;
