-- =====================================================================
-- eicu_07_airway_treatment.sql —— eICU 治疗记录（通气/气道/神经肌肉阻滞）
--
-- treatment 表是 eICU 的"治疗清单"层，带 treatmentoffset（相对 ICU 入室的
-- 分钟数），因此可以给出"何时开始某类治疗"的时间点。这是与 eicu_05
-- （结构化通气时段）、eicu_06（呼吸治疗记录）互不重叠的第三个来源。
--
-- 已核验的相关 treatmentstring 前缀（全库枚举）：
--   pulmonary|ventilation and oxygenation|mechanical ventilation        (117,481)
--   ...|non-invasive ventilation / CPAP/PEEP therapy / ventilator weaning
--   ...|oxygen therapy (< 40% / 40% to 60% / > 60%)  及各级子项（nasal cannula、
--        face mask、non-rebreather、trach collar 等）
--   pulmonary|surgery / incision and drainage of thorax|tracheostomy   (5,453)
--   ...|reintubation                                                    (1,766)
--   pulmonary|medications|neuromuscular blocking agent|cisatracurium   (1,307)
--
-- 归组说明：
--   - 'oxygen therapy ...|trach collar' 提示带气管造口套管，但不等于机械通气。
--   - 只有明确出现 'mechanical ventilation' 字样才归为 invasive_mechanical；
--     'non-invasive ventilation' 与 'CPAP/PEEP therapy' 单独成组，
--     不可与有创通气合并统计。
--
-- 时间：treatmentoffset（分钟）；窗口 [-360, 2880]。
-- 数据库：eicu（只读）
-- =====================================================================
WITH sepsis_dx AS (
    SELECT DISTINCT patientunitstayid
    FROM eicu_crd.diagnosis
    WHERE diagnosispriority = 'Primary'
      AND diagnosisoffset <= 1440
      AND (
             diagnosisstring LIKE 'surgery|infections|septic shock%'
          OR diagnosisstring LIKE 'surgery|infections|sepsis|severe%'
          OR diagnosisstring LIKE 'surgery|infections|sepsis|sepsis with multi-organ%'
          OR diagnosisstring LIKE 'surgery|infections|sepsis%'
          OR diagnosisstring LIKE 'infectious diseases|systemic/other infections|septic shock%'
          OR diagnosisstring LIKE 'infectious diseases|systemic/other infections|sepsis|severe%'
          OR diagnosisstring LIKE 'infectious diseases|systemic/other infections|sepsis|sepsis with multi-organ%'
          OR diagnosisstring LIKE 'infectious diseases|systemic/other infections|sepsis%'
          OR diagnosisstring LIKE 'hematology|coagulation disorders|DIC syndrome|associated with sepsis/septic shock%'
          OR diagnosisstring LIKE 'cardiovascular|shock / hypotension|septic shock%'
          OR diagnosisstring LIKE 'cardiovascular|shock / hypotension|sepsis|severe%'
          OR diagnosisstring LIKE 'cardiovascular|shock / hypotension|sepsis|sepsis with multi-organ%'
          OR diagnosisstring LIKE 'cardiovascular|shock / hypotension|sepsis%'
      )
),
cohort AS (
    SELECT p.patientunitstayid, p.patienthealthsystemstayid, p.uniquepid, p.hospitalid
    FROM eicu_crd.patient p
    INNER JOIN sepsis_dx s ON p.patientunitstayid = s.patientunitstayid
    WHERE p.unitvisitnumber = 1
      AND p.unitdischargeoffset > 2880
      AND p.hospitaldischargestatus IN ('Alive', 'Expired')
      AND (
             (p.age ~ '^[0-9]+$' AND CAST(p.age AS INT) >= 18)
          OR p.age = '> 89'
      )
),
tx AS (
    SELECT
        t.patientunitstayid, t.treatmentid, t.treatmentoffset,
        t.treatmentstring, t.activeupondischarge,
        CASE
            WHEN t.treatmentstring ~* 'mechanical ventilation'
                 AND t.treatmentstring !~* 'non-invasive' THEN 'invasive_mechanical_ventilation'
            WHEN t.treatmentstring ~* 'non-invasive ventilation' THEN 'noninvasive_ventilation'
            WHEN t.treatmentstring ~* 'CPAP/PEEP'                THEN 'cpap_peep'
            WHEN t.treatmentstring ~* 'ventilator weaning'       THEN 'ventilator_weaning'
            WHEN t.treatmentstring ~* 'tracheostomy'             THEN 'tracheostomy'
            WHEN t.treatmentstring ~* 'reintubation'             THEN 'reintubation'
            WHEN t.treatmentstring ~* 'neuromuscular blocking'   THEN 'neuromuscular_blockade'
            WHEN t.treatmentstring ~* 'oxygen therapy'           THEN 'oxygen_therapy'
            WHEN t.treatmentstring ~* 'tracheal suctioning'      THEN 'tracheal_suctioning'
        END AS airway_treatment_group
    FROM eicu_crd.treatment t
    WHERE t.treatmentstring ~* 'mechanical ventilation|non-invasive ventilation|CPAP/PEEP|tracheostomy|reintubation|neuromuscular blocking|oxygen therapy|tracheal suctioning|ventilator weaning'
)
SELECT
    c.patientunitstayid, c.patienthealthsystemstayid, c.uniquepid, c.hospitalid,
    tx.treatmentid, tx.treatmentoffset,
    ROUND((tx.treatmentoffset / 60.0)::numeric, 4) AS hours_from_unit_admit,
    tx.treatmentstring,
    tx.airway_treatment_group,
    tx.activeupondischarge,
    CASE WHEN tx.treatmentoffset BETWEEN -360 AND 2880 THEN 1 ELSE 0 END AS in_window_0_48h,
    CASE WHEN tx.treatmentoffset < 0 THEN 1 ELSE 0 END AS before_unit_admit
FROM tx
INNER JOIN cohort c ON tx.patientunitstayid = c.patientunitstayid
WHERE tx.treatmentoffset BETWEEN -360 AND 2880
  AND tx.airway_treatment_group IS NOT NULL
ORDER BY tx.patientunitstayid, tx.treatmentoffset, tx.airway_treatment_group;
