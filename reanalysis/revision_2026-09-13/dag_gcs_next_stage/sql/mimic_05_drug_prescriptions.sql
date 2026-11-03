-- =====================================================================
-- mimic_05_drug_prescriptions.sql —— 镇静/阿片/肌松：医嘱层（含起止时间）
--
-- 本层回答的是"医生开了什么、从何时到何时有效"，不等于"实际给了"。
-- 与 mimic_03（inputevents 实际记帐）和 mimic_04（eMAR 实际执行）三者
-- 交叉核对，才能把"未记录 / 未开医嘱 / 开了未给 / 实际给了"区分开。
--
-- 关键列：
--   starttime / stoptime  → 医嘱有效期（overlap 逻辑判断窗口内是否有效）
--   drug_type             → BASE / MAIN / ADDITIVE（区分主药与溶媒/添加成分）
--   dose_val_rx / dose_unit_rx, route, doses_per_24_hrs → 剂量与频次
--   prod_strength, form_rx → 制剂浓度（用于单位换算审计）
--
-- 药物匹配：prescriptions.drug 亦为自由文本，使用 ~* 大小写不敏感匹配。
-- 注意 stoptime 在过去常常远晚于实际停药，因此 stoptime 不可当停药时刻。
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
rx AS (
    SELECT
        p.subject_id, p.hadm_id, p.pharmacy_id, p.poe_id, p.poe_seq,
        p.starttime, p.stoptime, p.drug_type,
        p.drug AS drug_raw,
        btrim(p.drug) AS drug_clean,
        p.formulary_drug_cd, p.prod_strength, p.form_rx,
        p.dose_val_rx, p.dose_unit_rx,
        p.form_val_disp, p.form_unit_disp,
        p.doses_per_24_hrs, p.route,
        CASE
            WHEN btrim(p.drug) ~* 'propofol'                 THEN 'propofol'
            WHEN btrim(p.drug) ~* 'midazolam|versed'          THEN 'midazolam'
            WHEN btrim(p.drug) ~* 'dexmedetomidine|precedex'  THEN 'dexmedetomidine'
            WHEN btrim(p.drug) ~* 'lorazepam|ativan'          THEN 'lorazepam'
            WHEN btrim(p.drug) ~* 'pentobarbital'             THEN 'pentobarbital'
            WHEN btrim(p.drug) ~* 'ketamine'                  THEN 'ketamine'
            WHEN btrim(p.drug) ~* 'etomidate'                 THEN 'etomidate'
            WHEN btrim(p.drug) ~* 'clonidine'                 THEN 'clonidine'
            WHEN btrim(p.drug) ~* 'methohexital'              THEN 'methohexital'
            WHEN btrim(p.drug) ~* 'fentanyl'                  THEN 'fentanyl'
            WHEN btrim(p.drug) ~* 'hydromorphone|dilaudid'    THEN 'hydromorphone'
            WHEN btrim(p.drug) ~* 'morphine'                  THEN 'morphine'
            WHEN btrim(p.drug) ~* 'oxycodone'                 THEN 'oxycodone'
            WHEN btrim(p.drug) ~* 'remifentanil'              THEN 'remifentanil'
            WHEN btrim(p.drug) ~* 'sufentanil'                THEN 'sufentanil'
            WHEN btrim(p.drug) ~* 'methadone'                 THEN 'methadone'
            WHEN btrim(p.drug) ~* 'rocuronium'                THEN 'rocuronium'
            WHEN btrim(p.drug) ~* 'cisatracurium'             THEN 'cisatracurium'
            WHEN btrim(p.drug) ~* 'vecuronium'                THEN 'vecuronium'
            WHEN btrim(p.drug) ~* 'pancuronium'               THEN 'pancuronium'
            WHEN btrim(p.drug) ~* 'succinylcholine'           THEN 'succinylcholine'
            WHEN btrim(p.drug) ~* 'atracurium'                THEN 'atracurium'
        END AS drug_name
    FROM mimiciv_hosp.prescriptions p
    WHERE btrim(p.drug) ~* 'propofol|midazolam|versed|dexmedetomidine|precedex|lorazepam|ativan|pentobarbital|ketamine|etomidate|clonidine|methohexital|fentanyl|hydromorphone|dilaudid|morphine|oxycodone|remifentanil|sufentanil|methadone|rocuronium|cisatracurium|vecuronium|pancuronium|succinylcholine|atracurium'
)
SELECT
    c.stay_id, r.subject_id, r.hadm_id,
    r.pharmacy_id, r.poe_id, r.poe_seq,
    r.starttime, r.stoptime,
    ROUND((EXTRACT(EPOCH FROM (r.starttime - c.intime)) / 3600.0)::numeric, 4) AS start_hours_from_intime,
    CASE WHEN r.stoptime IS NOT NULL
         THEN ROUND((EXTRACT(EPOCH FROM (r.stoptime - c.intime)) / 3600.0)::numeric, 4)
    END AS stop_hours_from_intime,
    r.drug_raw, r.drug_clean,
    r.drug_name,
    CASE
        WHEN r.drug_name IN ('propofol','midazolam','dexmedetomidine','lorazepam',
                             'pentobarbital','ketamine','etomidate','clonidine','methohexital')
            THEN 'sedative_hypnotic'
        WHEN r.drug_name IN ('fentanyl','hydromorphone','morphine','oxycodone',
                             'remifentanil','sufentanil','methadone')
            THEN 'opioid'
        WHEN r.drug_name IN ('rocuronium','cisatracurium','vecuronium',
                             'pancuronium','succinylcholine','atracurium')
            THEN 'neuromuscular_blocker'
        ELSE 'unclassified'
    END AS drug_class,
    r.drug_type, r.formulary_drug_cd, r.prod_strength, r.form_rx,
    r.dose_val_rx, r.dose_unit_rx,
    r.form_val_disp, r.form_unit_disp,
    r.doses_per_24_hrs, r.route,
    CASE WHEN r.starttime < c.win_start THEN 1 ELSE 0 END AS started_before_window,
    CASE WHEN r.stoptime IS NULL THEN 'open_ended'
         WHEN r.stoptime < c.win_start THEN 'ended_before_window'
         ELSE 'overlaps_window' END AS window_relation
FROM rx r
INNER JOIN cohort c ON r.hadm_id = c.hadm_id
WHERE r.starttime <= c.win_end
  AND (r.stoptime IS NULL OR r.stoptime >= c.win_start)
ORDER BY c.stay_id, r.starttime, r.drug_name;
