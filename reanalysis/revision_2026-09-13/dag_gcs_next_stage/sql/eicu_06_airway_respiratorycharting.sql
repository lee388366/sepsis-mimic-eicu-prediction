-- =====================================================================
-- eicu_06_airway_respiratorycharting.sql —— eICU 呼吸治疗记录（气道/通气/氧疗）
--
-- 与 eicu_05（respiratorycare，结构化气道与通气时段）互补：
-- 本表是呼吸治疗师的逐条记录，能定位"插管/拔管/气管切开/氧疗方式变更"的时刻。
--
-- 【已核验：respiratorycharting 中不存在 GCS 字段】
--   对 respchartvaluelabel 做过含 'gcs|glasgow' 的穷尽检索，无匹配。
--   eICU 的 GCS 只在 nursecharting 中（见 eicu_01）。这一点与 MIMIC 不同，
--   必须记录，否则会误以为 eICU 有第二套 GCS 来源。
--
-- 纳入的 respchartvaluelabel（均已在全库枚举确认存在）：
--   respFlowCareData : 'Oxygen Delivery Method' / 'Oxygen Delivery Status'
--     | 'Endotracheal Tube Placement' / 'Endotracheal Position at Lip'
--     | 'Endotracheal Tube Moved to' / 'ET Tube Repositioned'
--     | 'Tracheostomy Type' / 'Trachestomy Tube Size'（原文拼写如此）
--     | 'Airway Attachments' / 'Ventilator Type'
--     | 'Mechanical Ventilator Mode' / 'Non-invasive Ventilation Mode'
--     | 'Ventilator Support Mode' / 'Bipap Delivery Mode'
--     | 'Ventilator Population' / 'Sedation Scoring Ventilation Depth'
--   respFlowSettings: 'FiO2' / 'PEEP' / 'PEEP/CPAP'
--   respFlowPtVentData: 'Mean Airway Pressure'
-- 注意：FiO2/PEEP 只能作为"在接受呼吸支持"的旁证，不可单独作为有创通气
--   的金标准（无创、高流量、甚至未通气患者也可能记录 FiO2）。
--
-- 时间：respchartoffset（分钟，距 ICU 入室）；窗口 [-360, 2880]。
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
)
SELECT
    rc.patientunitstayid, c.patienthealthsystemstayid, c.uniquepid, c.hospitalid,
    rc.respchartid,
    rc.respchartoffset, rc.respchartentryoffset,
    rc.respchartentryoffset - rc.respchartoffset AS entry_minus_chart_min,
    ROUND((rc.respchartoffset / 60.0)::numeric, 4) AS hours_from_unit_admit,
    rc.respcharttypecat,
    rc.respchartvaluelabel AS field_label,
    rc.respchartvalue AS value_raw,
    CASE
        WHEN rc.respchartvalue IS NULL THEN 'null'
        WHEN rc.respchartvalue = ''    THEN 'empty_string'
        ELSE 'present'
    END AS value_state,
    -- 字段归组（便于分析阶段按类别取可用信息）
    CASE
        WHEN rc.respchartvaluelabel IN ('Oxygen Delivery Method','Oxygen Delivery Status')
            THEN 'oxygen_delivery'
        WHEN rc.respchartvaluelabel IN ('Endotracheal Tube Placement',
                                        'Endotracheal Position at Lip',
                                        'Endotracheal Tube Moved to',
                                        'ET Tube Repositioned')
            THEN 'endotracheal_tube'
        WHEN rc.respchartvaluelabel IN ('Tracheostomy Type','Trachestomy Tube Size')
            THEN 'tracheostomy'
        WHEN rc.respchartvaluelabel IN ('Mechanical Ventilator Mode',
                                        'Non-invasive Ventilation Mode',
                                        'Ventilator Support Mode',
                                        'Bipap Delivery Mode',
                                        'Ventilator Type',
                                        'Ventilator Population',
                                        'Airway Attachments',
                                        'Sedation Scoring Ventilation Depth')
            THEN 'ventilator_mode'
        WHEN rc.respchartvaluelabel IN ('FiO2','FIO2 (%)','Set Fraction of Inspired Oxygen (FIO2)')
            THEN 'fio2'
        WHEN rc.respchartvaluelabel IN ('PEEP','PEEP/CPAP') THEN 'peep'
        WHEN rc.respchartvaluelabel = 'Mean Airway Pressure' THEN 'mean_airway_pressure'
    END AS field_group,
    -- 明确提示"有创支持"的文本（需分析阶段确认，不作金标准）
    CASE WHEN rc.respchartvalue ~* 'endotracheal|oral et|nasal et|ett' THEN 1 ELSE 0 END AS et_text_flag,
    CASE WHEN rc.respchartvalue ~* 'trach' THEN 1 ELSE 0 END AS trach_text_flag,
    CASE WHEN rc.respchartvalue ~* 'invasive|mechanical' THEN 1 ELSE 0 END AS invasive_text_flag,
    CASE WHEN rc.respchartoffset BETWEEN -360 AND 2880 THEN 1 ELSE 0 END AS in_window_0_48h
FROM eicu_crd.respiratorycharting rc
INNER JOIN cohort c ON rc.patientunitstayid = c.patientunitstayid
WHERE rc.respchartvaluelabel IN (
        'Oxygen Delivery Method', 'Oxygen Delivery Status',
        'Endotracheal Tube Placement', 'Endotracheal Position at Lip',
        'Endotracheal Tube Moved to', 'ET Tube Repositioned',
        'Tracheostomy Type', 'Trachestomy Tube Size',
        'Mechanical Ventilator Mode', 'Non-invasive Ventilation Mode',
        'Ventilator Support Mode', 'Bipap Delivery Mode',
        'Ventilator Type', 'Ventilator Population',
        'Airway Attachments', 'Sedation Scoring Ventilation Depth',
        'FiO2', 'FIO2 (%)', 'Set Fraction of Inspired Oxygen (FIO2)',
        'PEEP', 'PEEP/CPAP', 'Mean Airway Pressure'
      )
  AND rc.respchartoffset BETWEEN -360 AND 2880
ORDER BY rc.patientunitstayid, rc.respchartoffset, rc.respchartvaluelabel;
