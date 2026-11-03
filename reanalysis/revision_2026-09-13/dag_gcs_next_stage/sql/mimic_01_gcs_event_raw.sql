-- =====================================================================
-- mimic_01_gcs_event_raw.sql —— MIMIC-IV 原始床旁 GCS 事件（event-level）
--
-- 目的：拿到"未小时平均、未插补、未改写特殊编码"的原始记录，用于区分
--       真实意识状态、镇静/插管导致的不可评估、以及记录规则影响。
--
-- itemid（已由 mimiciv_icu.d_items 核验）：
--   220739 = GCS - Eye Opening      (Text)
--   223900 = GCS - Verbal Response  (Text)   ← ETT 标记所在
--   223901 = GCS - Motor Response   (Text)
--
-- 关键：MIMIC-IV 把"因气管插管无法评估语言"记录成 verbal 取值文本
--       'No Response-ETT'（value 文本；valuenum=1）。必须与真正的
--       'No Response' 区分——前者是不可评估，后者是意识无反应。
--       本脚本按原始文本保留 value_raw，score_mapped 是另加的映射列，
--       不覆盖、不替换原始 value/valuenum。
--
-- 时间窗：保留 [t0 - 6h, t0 + 48h] 的事件。
--   向前多取 6 小时用于检查窗口起点附近的记录惯性（派生层存在 6h 内补值）；
--   48h 为硬上界，与模型特征窗一致。
--
-- 注意：本脚本只读 chartevents（床旁原始层）。派生层（含 6h 补值、
--       总分置 15 规则）由 mimic_02 单独导出。两层对照才能识别
--       "哪一层发生了什么"，不能只凭派生层认定床旁意识状态。
--
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
    SELECT ie.subject_id, ie.hadm_id, ie.stay_id, ie.intime
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
    ce.itemid,
    di.label AS item_label,
    ce.value AS value_raw,
    ce.valuenum,
    ce.valueuom,
    ce.storetime,
    ce.warning,
    CASE WHEN ce.storetime IS NOT NULL
         THEN ROUND((EXTRACT(EPOCH FROM (ce.storetime - ce.charttime)) / 3600.0)::numeric, 4)
    END AS chart_to_store_lag_h,
    CASE
        WHEN ce.itemid = 220739 THEN 'eyes'
        WHEN ce.itemid = 223900 THEN 'verbal'
        WHEN ce.itemid = 223901 THEN 'motor'
    END AS component,
    CASE WHEN ce.itemid = 223900 AND ce.value = 'No Response-ETT' THEN 1 ELSE 0 END AS ett_flag,
    CASE WHEN ce.itemid = 223900 AND ce.value = 'No Response'     THEN 1 ELSE 0 END AS true_no_response_flag,
    CASE WHEN ce.value = 'No Response-ETT' THEN 1 ELSE 0 END AS any_ett_text_flag,
    CASE
        WHEN ce.itemid = 220739 AND ce.value = 'Spontaneously' THEN 4
        WHEN ce.itemid = 220739 AND ce.value = 'To Speech'     THEN 3
        WHEN ce.itemid = 220739 AND ce.value = 'To Pain'       THEN 2
        WHEN ce.itemid = 220739 AND ce.value = 'None'          THEN 1
        WHEN ce.itemid = 223900 AND ce.value = 'Oriented'                THEN 5
        WHEN ce.itemid = 223900 AND ce.value = 'Confused'                THEN 4
        WHEN ce.itemid = 223900 AND ce.value = 'Inappropriate Words'     THEN 3
        WHEN ce.itemid = 223900 AND ce.value = 'Incomprehensible sounds' THEN 2
        WHEN ce.itemid = 223900 AND ce.value = 'No Response-ETT'         THEN 0
        WHEN ce.itemid = 223900 AND ce.value = 'No Response'             THEN 1
        WHEN ce.itemid = 223901 AND ce.value = 'Obeys Commands'     THEN 6
        WHEN ce.itemid = 223901 AND ce.value = 'Localizes Pain'     THEN 5
        WHEN ce.itemid = 223901 AND ce.value = 'Flex-withdraws'     THEN 4
        WHEN ce.itemid = 223901 AND ce.value = 'Abnormal Flexion'   THEN 3
        WHEN ce.itemid = 223901 AND ce.value = 'Abnormal extension' THEN 2
        WHEN ce.itemid = 223901 AND ce.value = 'No response'        THEN 1
    END AS score_mapped
FROM mimiciv_icu.chartevents ce
INNER JOIN cohort c ON ce.stay_id = c.stay_id
LEFT JOIN mimiciv_icu.d_items di ON ce.itemid = di.itemid
WHERE ce.itemid IN (220739, 223900, 223901)
  AND ce.charttime >= c.intime - INTERVAL '6 hours'
  AND ce.charttime <= c.intime + INTERVAL '48 hours'
ORDER BY ce.stay_id, ce.charttime, ce.itemid;
