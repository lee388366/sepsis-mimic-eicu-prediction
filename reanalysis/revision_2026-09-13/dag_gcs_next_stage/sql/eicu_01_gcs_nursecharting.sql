-- =====================================================================
-- eicu_01_gcs_nursecharting.sql —— eICU 原始 GCS 记录（nursecharting）
--
-- 两条彼此独立的 GCS 记录通道（都保留，不合并、不择优）：
--   1) nursingchartcelltypevallabel = 'Glasgow coma score'
--        nursingchartcelltypevalname ∈ {Eyes, Verbal, Motor, GCS Total}
--        → 分项与总分分别成行
--   2) nursingchartcelltypevallabel = 'Score (Glasgow Coma Scale)'
--        nursingchartcelltypevalname = 'Value'
--        → 另一个总分通道（编码习惯不同的医院使用）
--      两通道并存说明 eICU 各医院记录方式不一致，不能假定单一来源完整。
--
-- 【已核验的关键差异：eICU 没有 ETT 语言标记】
--   对全库 'Glasgow coma score' 的非数值取值做穷尽枚举，只有：
--       GCS Total = 'Unable to score due to medication'  78,564 次
--       ''（空字符串，各分项均有：Eyes 88,274 / Motor 88,156 / Verbal 107,082）
--   而 MIMIC 有 'No Response-ETT'（784,911 次）。即：
--     - eICU 只能识别"因药物无法评估"，不能识别"因气管插管无法评估语言"；
--     - MIMIC 反向：能识别 ETT，但没有"因药物无法评估"的专用标记。
--   因此"可评估性"在两库的可判定程度不同，绝不可用同一套规则套用。
--
-- 【空字符串与 NULL 的区分】eicu 中 '' 与 NULL 语义不同：'' 表示字段被
--   填过但无值（多与"未评估"相容），NULL 表示未记录。本脚本用
--   CASE 显式输出 value_state ∈ {numeric, text, empty_string, null}，
--   不在 SQL 层把 '' 当作缺失删除。
--
-- 时间：nursingchartoffset 为"距 ICU 入室的分钟数"（可负）。
--       窗口 [-360, 2880] 分钟 = [t0-6h, t0+48h]，含边界。
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
    c.patienthealthsystemstayid,
    c.uniquepid,
    c.hospitalid,
    n.nursingchartid,
    n.nursingchartoffset,
    n.nursingchartentryoffset,
    n.nursingchartentryoffset - n.nursingchartoffset AS entry_minus_chart_min,
    ROUND((n.nursingchartoffset / 60.0)::numeric, 4) AS hours_from_unit_admit,
    n.nursingchartcelltypecat,
    n.nursingchartcelltypevallabel AS field_label,
    n.nursingchartcelltypevalname  AS field_name,
    n.nursingchartvalue AS value_raw,
    CASE
        WHEN n.nursingchartvalue IS NULL          THEN 'null'
        WHEN n.nursingchartvalue = ''             THEN 'empty_string'
        WHEN n.nursingchartvalue ~ '^[0-9]+$'     THEN 'numeric'
        ELSE 'text'
    END AS value_state,
    CASE WHEN n.nursingchartvalue ~ '^[0-9]+$'
         THEN n.nursingchartvalue::integer END AS value_numeric,
    -- GCS 语义分类（不修改原始值；只能识别"因药物无法评估"）
    CASE
        WHEN n.nursingchartcelltypevallabel = 'Glasgow coma score'
             AND n.nursingchartcelltypevalname = 'GCS Total'
             AND n.nursingchartvalue = 'Unable to score due to medication'
            THEN 'unable_due_to_medication'
        WHEN n.nursingchartvalue = '' THEN 'blank'
        WHEN n.nursingchartvalue ~ '^[0-9]+$' THEN 'scored'
        ELSE 'other_text'
    END AS gcs_value_class,
    -- 该条记录是否与 48h 观察窗重叠
    CASE WHEN n.nursingchartoffset BETWEEN -360 AND 2880 THEN 1 ELSE 0 END AS in_window_0_48h,
    CASE WHEN n.nursingchartoffset < 0 THEN 1 ELSE 0 END AS before_unit_admit,
    CASE WHEN n.nursingchartoffset > 2880 THEN 1 ELSE 0 END AS after_48h
FROM eicu_crd.nursecharting n
INNER JOIN cohort c ON n.patientunitstayid = c.patientunitstayid
-- 注意：窗口条件必须整体括住 OR 的两侧。若写成
--   (A) OR (B AND window)  则 A 分支不会被窗口约束（曾出现该缺陷，已修正）。
WHERE (
        (
             n.nursingchartcelltypevallabel = 'Glasgow coma score'
         AND n.nursingchartcelltypevalname IN ('Eyes', 'Verbal', 'Motor', 'GCS Total')
        )
     OR (
             n.nursingchartcelltypevallabel = 'Score (Glasgow Coma Scale)'
         AND n.nursingchartcelltypevalname = 'Value'
        )
      )
  AND n.nursingchartoffset BETWEEN -360 AND 2880
ORDER BY n.patientunitstayid, n.nursingchartoffset, n.nursingchartcelltypevalname;
