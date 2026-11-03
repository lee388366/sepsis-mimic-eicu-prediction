-- =====================================================================
-- mimic_04_drug_emar.sql —— 镇静/阿片/肌松：eMAR 给药执行记录
--
-- 这是三层里唯一能区分"实际给药"与"未给药"的数据层：
--   event_txt = 'Administered'                 → 实际已给药
--   event_txt = 'Administered Bolus from IV Drip' → 从输注中给推注（实际给药）
--   event_txt = 'Not Given'                    → 明确未给药（可证否）
--   event_txt = 'Started' / 'Stopped' / 'Stopped - Unscheduled' → 输注起止事件
--   event_txt = 'Confirmed' / 'Infusion Reconciliation' → 输液泵确认/核对
--   event_txt = 'Assessed'                     → 评估（非给药）
--   event_txt IS NULL                          → 原因未记录
-- 因此本脚本原样导出 event_txt，不在 SQL 层做"是否给药"的判断。
--
-- 药物名匹配：eMAR 的 medication 为自由文本，且大小写不固定
-- （如 'HYDROmorphone (Dilaudid)'、'LORazepam'、'OxycoDONE'），
-- 故用 ~* 大小写不敏感匹配；复合制剂以 '|' 连接（如
-- 'Bupivacaine 0.1%|HYDROmorphone (Dilaudid)'），保留原样不拆分。
-- 注意：medication 值存在尾部空格变体，本脚本用 btrim 统一后再匹配，
--       但导出列同时保留 medication_raw 与 medication_clean 以便审计。
--
-- 量化信息在 emar_detail（dose_given / infusion_rate），由 mimic_04b 单独导出，
-- 本脚本不 JOIN，避免一对多导致行数膨胀。
--
-- 窗口：[t0 - 6h, t0 + 48h]（与 GCS 事件一致）
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
drug AS (
    SELECT
        e.subject_id, e.hadm_id, e.emar_id, e.emar_seq, e.pharmacy_id, e.poe_id,
        e.charttime, e.scheduletime, e.storetime,
        e.medication AS medication_raw,
        btrim(e.medication) AS medication_clean,
        e.event_txt,
        CASE
            WHEN btrim(e.medication) ~* 'propofol'                 THEN 'propofol'
            WHEN btrim(e.medication) ~* 'midazolam|versed'         THEN 'midazolam'
            WHEN btrim(e.medication) ~* 'dexmedetomidine|precedex' THEN 'dexmedetomidine'
            WHEN btrim(e.medication) ~* 'lorazepam|ativan'         THEN 'lorazepam'
            WHEN btrim(e.medication) ~* 'pentobarbital'            THEN 'pentobarbital'
            WHEN btrim(e.medication) ~* 'ketamine'                 THEN 'ketamine'
            WHEN btrim(e.medication) ~* 'etomidate'                THEN 'etomidate'
            WHEN btrim(e.medication) ~* 'clonidine'                THEN 'clonidine'
            WHEN btrim(e.medication) ~* 'methohexital'             THEN 'methohexital'
            WHEN btrim(e.medication) ~* 'fentanyl'                 THEN 'fentanyl'
            WHEN btrim(e.medication) ~* 'hydromorphone|dilaudid'   THEN 'hydromorphone'
            WHEN btrim(e.medication) ~* 'morphine'                 THEN 'morphine'
            WHEN btrim(e.medication) ~* 'oxycodone'                THEN 'oxycodone'
            WHEN btrim(e.medication) ~* 'remifentanil'             THEN 'remifentanil'
            WHEN btrim(e.medication) ~* 'sufentanil'               THEN 'sufentanil'
            WHEN btrim(e.medication) ~* 'methadone'                THEN 'methadone'
            WHEN btrim(e.medication) ~* 'rocuronium'               THEN 'rocuronium'
            WHEN btrim(e.medication) ~* 'cisatracurium'            THEN 'cisatracurium'
            WHEN btrim(e.medication) ~* 'vecuronium'               THEN 'vecuronium'
            WHEN btrim(e.medication) ~* 'pancuronium'              THEN 'pancuronium'
            WHEN btrim(e.medication) ~* 'succinylcholine'          THEN 'succinylcholine'
            WHEN btrim(e.medication) ~* 'atracurium'               THEN 'atracurium'
        END AS drug_name
    FROM mimiciv_hosp.emar e
    WHERE btrim(e.medication) ~* 'propofol|midazolam|versed|dexmedetomidine|precedex|lorazepam|ativan|pentobarbital|ketamine|etomidate|clonidine|methohexital|fentanyl|hydromorphone|dilaudid|morphine|oxycodone|remifentanil|sufentanil|methadone|rocuronium|cisatracurium|vecuronium|pancuronium|succinylcholine|atracurium'
)
SELECT
    d.subject_id, d.hadm_id, d.emar_id, d.emar_seq, d.pharmacy_id, d.poe_id,
    d.charttime, d.scheduletime, d.storetime,
    ROUND((EXTRACT(EPOCH FROM (d.charttime - c.intime)) / 3600.0)::numeric, 4) AS chart_hours_from_intime,
    d.medication_raw, d.medication_clean,
    d.event_txt,
    CASE
        WHEN d.event_txt IN ('Administered', 'Administered Bolus from IV Drip') THEN 'given'
        WHEN d.event_txt = 'Not Given'                                          THEN 'not_given'
        WHEN d.event_txt IN ('Started', 'Stopped', 'Stopped - Unscheduled')      THEN 'infusion_event'
        WHEN d.event_txt IN ('Confirmed', 'Infusion Reconciliation')             THEN 'pump_confirm'
        WHEN d.event_txt = 'Assessed'                                            THEN 'assessed_not_dose'
        WHEN d.event_txt IS NULL                                                 THEN 'unknown_reason'
        ELSE 'other'
    END AS event_category,
    d.drug_name,
    CASE
        WHEN d.drug_name IN ('propofol','midazolam','dexmedetomidine','lorazepam',
                             'pentobarbital','ketamine','etomidate','clonidine','methohexital')
            THEN 'sedative_hypnotic'
        WHEN d.drug_name IN ('fentanyl','hydromorphone','morphine','oxycodone',
                             'remifentanil','sufentanil','methadone')
            THEN 'opioid'
        WHEN d.drug_name IN ('rocuronium','cisatracurium','vecuronium',
                             'pancuronium','succinylcholine','atracurium')
            THEN 'neuromuscular_blocker'
        ELSE 'unclassified'
    END AS drug_class,
    c.stay_id
FROM drug d
INNER JOIN cohort c ON d.hadm_id = c.hadm_id
WHERE d.charttime >= c.win_start
  AND d.charttime <= c.win_end
ORDER BY c.stay_id, d.charttime, d.drug_name;
