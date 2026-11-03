-- =====================================================================
-- eicu_02_rass_delirium_sedation.sql —— eICU 镇静深度/谵妄/疼痛评估
--
-- 这是判断"GCS 低反映意识还是镇静"的辅助层。eICU 在此比 MIMIC 更丰富，
-- 但各医院使用习惯差异极大，因此全部原样导出、逐条带时间，不做汇总。
--
-- 已由 nursecharting 穷尽枚举得到的相关字段（label → name）：
--   Scores 类别：
--     'Sedation Scale/Score/Goal' → Sedation Scale / Sedation Score / Sedation Goal
--     'Delirium Scale/Score'      → Delirium Scale / Delirium Score
--     'Pain Score/Goal'           → Pain Score / Pain Goal
--   Other Vital Signs and Infusions 类别（自由文本 Value）：
--     'RASS'                          → Richmond 躁动-镇静评分（实测）
--     'SEDATION SCORE'                → 另一套镇静评分字段
--     'Symptoms of Delirium Present'
--     'Pain Assessment' / 'Pain Present'
--     'Patient s Comfort/Function (Pain) GOAL At Rest'
--
-- 注意：
--   - 'Sedation Goal' / 'Pain Goal' 是目标值，不是实测，混用会高估镇静。
--   - 同一名患者可能同时有 'Sedation Scale' 与 'SEDATION SCORE' 两套编码，
--     本脚本保留 label 原名，不做跨字段归一，避免把不同量表合并。
--   - Delirium Scale 的具体量表名称（CAM-ICU / ICDSC）记录在 value 文本中，
--     需在分析阶段按 value 取值再分组。
--
-- 时间：nursingchartoffset（分钟，距 ICU 入室）；窗口 [-360, 2880]。
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
    n.patientunitstayid,
    c.hospitalid,
    n.nursingchartid,
    n.nursingchartoffset,
    ROUND((n.nursingchartoffset / 60.0)::numeric, 4) AS hours_from_unit_admit,
    n.nursingchartcelltypecat,
    n.nursingchartcelltypevallabel AS field_label,
    n.nursingchartcelltypevalname  AS field_name,
    n.nursingchartvalue AS value_raw,
    CASE
        WHEN n.nursingchartvalue IS NULL      THEN 'null'
        WHEN n.nursingchartvalue = ''         THEN 'empty_string'
        WHEN n.nursingchartvalue ~ '^-?[0-9]+$' THEN 'numeric'
        ELSE 'text'
    END AS value_state,
    CASE WHEN n.nursingchartvalue ~ '^-?[0-9]+$'
         THEN n.nursingchartvalue::integer END AS value_numeric,
    CASE
        WHEN n.nursingchartcelltypevallabel = 'RASS'                       THEN 'rass_actual'
        WHEN n.nursingchartcelltypevallabel = 'SEDATION SCORE'             THEN 'sedation_score_free'
        WHEN n.nursingchartcelltypevallabel = 'Sedation Scale/Score/Goal'
             AND n.nursingchartcelltypevalname = 'Sedation Scale'          THEN 'sedation_scale'
        WHEN n.nursingchartcelltypevallabel = 'Sedation Scale/Score/Goal'
             AND n.nursingchartcelltypevalname = 'Sedation Score'          THEN 'sedation_score'
        WHEN n.nursingchartcelltypevallabel = 'Sedation Scale/Score/Goal'
             AND n.nursingchartcelltypevalname = 'Sedation Goal'           THEN 'sedation_goal'
        WHEN n.nursingchartcelltypevallabel = 'Delirium Scale/Score'
             AND n.nursingchartcelltypevalname = 'Delirium Scale'          THEN 'delirium_scale'
        WHEN n.nursingchartcelltypevallabel = 'Delirium Scale/Score'
             AND n.nursingchartcelltypevalname = 'Delirium Score'          THEN 'delirium_score'
        WHEN n.nursingchartcelltypevallabel = 'Symptoms of Delirium Present' THEN 'delirium_symptoms_present'
        WHEN n.nursingchartcelltypevallabel = 'Pain Score/Goal'
             AND n.nursingchartcelltypevalname = 'Pain Score'              THEN 'pain_score'
        WHEN n.nursingchartcelltypevallabel = 'Pain Score/Goal'
             AND n.nursingchartcelltypevalname = 'Pain Goal'               THEN 'pain_goal'
        WHEN n.nursingchartcelltypevallabel = 'Pain Assessment'            THEN 'pain_assessment'
        WHEN n.nursingchartcelltypevallabel = 'Pain Present'               THEN 'pain_present'
        WHEN n.nursingchartcelltypevallabel LIKE 'Patient s Comfort/Function (Pain) GOAL%'
            THEN 'pain_goal_comfort'
    END AS assessment_field,
    CASE WHEN n.nursingchartcelltypevallabel = 'RASS' THEN
        CASE
            WHEN n.nursingchartvalue ~ '^-?[0-9]+$' THEN n.nursingchartvalue::integer
            WHEN n.nursingchartvalue ILIKE '%combative%'             THEN  4
            WHEN n.nursingchartvalue ILIKE '%very agitated%'         THEN  3
            WHEN n.nursingchartvalue ILIKE '%agitated%'              THEN  2
            WHEN n.nursingchartvalue ILIKE '%restless%'              THEN  1
            WHEN n.nursingchartvalue ILIKE '%alert and calm%'        THEN  0
            WHEN n.nursingchartvalue ILIKE '%drowsy%'                THEN -1
            WHEN n.nursingchartvalue ILIKE '%light sedation%'        THEN -2
            WHEN n.nursingchartvalue ILIKE '%moderate sedation%'     THEN -3
            WHEN n.nursingchartvalue ILIKE '%deep sedation%'         THEN -4
            WHEN n.nursingchartvalue ILIKE '%unarousable%'           THEN -5
        END
    END AS rass_numeric,
    CASE WHEN n.nursingchartoffset BETWEEN -360 AND 2880 THEN 1 ELSE 0 END AS in_window_0_48h
FROM eicu_crd.nursecharting n
INNER JOIN cohort c ON n.patientunitstayid = c.patientunitstayid
WHERE (
        n.nursingchartcelltypevallabel IN (
            'RASS', 'SEDATION SCORE', 'Sedation Scale/Score/Goal',
            'Delirium Scale/Score', 'Symptoms of Delirium Present',
            'Pain Score/Goal', 'Pain Assessment', 'Pain Present'
        )
     OR n.nursingchartcelltypevallabel LIKE 'Patient s Comfort/Function (Pain) GOAL%'
      )
  AND n.nursingchartoffset BETWEEN -360 AND 2880
ORDER BY n.patientunitstayid, n.nursingchartoffset, n.nursingchartcelltypevallabel;
