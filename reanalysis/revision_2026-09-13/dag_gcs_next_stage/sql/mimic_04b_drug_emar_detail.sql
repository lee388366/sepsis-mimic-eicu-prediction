-- =====================================================================
-- mimic_04b_drug_emar_detail.sql —— eMAR 给药量化明细（剂量/速率/途径）
--
-- emar 与 emar_detail 是一对多（parent_field_ordinal），直接 JOIN 会使
-- 给药事件行数膨胀。本脚本单独导出明细，由分析者按需用
-- (emar_id, emar_seq) 关联，不在数据准备阶段做任何聚合或合并。
--
-- 关键列（正是"实际给药 / 医嘱 / 停药"三分的量化侧证据）：
--   dose_given / dose_given_unit        → 实给剂量
--   dose_due   / dose_due_unit          → 医嘱应收剂量
--   complete_dose_not_given             → 是否整剂未给（与 event_txt='Not Given' 互证）
--   infusion_rate / infusion_rate_unit  → 输注速率
--   prior_infusion_rate                 → 速率变更前的速率（判断是否调整）
--   route, administration_type          → 给药途径与方式
--   infusion_complete, new_iv_bag_hung  → 输注完成/换袋（持续治疗边界）
--
-- 注：本表只有 subject_id，无 hadm_id，故经 emar 关联住院与窗口。
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
cohort AS (SELECT * FROM ranked WHERE rn = 1),
target_emar AS (
    SELECT e.subject_id, e.hadm_id, e.emar_id, e.emar_seq, e.pharmacy_id,
           e.charttime, e.medication, e.event_txt
    FROM mimiciv_hosp.emar e
    WHERE btrim(e.medication) ~* 'propofol|midazolam|versed|dexmedetomidine|precedex|lorazepam|ativan|pentobarbital|ketamine|etomidate|clonidine|methohexital|fentanyl|hydromorphone|dilaudid|morphine|oxycodone|remifentanil|sufentanil|methadone|rocuronium|cisatracurium|vecuronium|pancuronium|succinylcholine|atracurium'
)
SELECT
    c.stay_id, t.subject_id, t.hadm_id,
    t.emar_id, t.emar_seq, t.pharmacy_id,
    t.charttime, t.medication AS medication_raw, t.event_txt,
    ROUND((EXTRACT(EPOCH FROM (t.charttime - c.intime)) / 3600.0)::numeric, 4) AS chart_hours_from_intime,
    ed.parent_field_ordinal,
    ed.administration_type,
    ed.dose_due, ed.dose_due_unit,
    ed.dose_given, ed.dose_given_unit,
    ed.complete_dose_not_given,
    ed.will_remainder_of_dose_be_given,
    ed.product_amount_given, ed.product_unit, ed.product_description,
    ed.prior_infusion_rate,
    ed.infusion_rate, ed.infusion_rate_unit,
    ed.infusion_rate_adjustment, ed.infusion_rate_adjustment_amount,
    ed.route,
    ed.infusion_complete,
    ed.new_iv_bag_hung,
    ed.continued_infusion_in_other_location,
    ed.restart_interval,
    ed.barcode_type,
    ed.reason_for_no_barcode
FROM mimiciv_hosp.emar_detail ed
INNER JOIN target_emar t
        ON ed.emar_id = t.emar_id AND ed.emar_seq = t.emar_seq
INNER JOIN cohort c
        ON t.hadm_id = c.hadm_id
WHERE t.charttime >= c.win_start
  AND t.charttime <= c.win_end
ORDER BY c.stay_id, t.charttime, t.emar_id, ed.parent_field_ordinal;
