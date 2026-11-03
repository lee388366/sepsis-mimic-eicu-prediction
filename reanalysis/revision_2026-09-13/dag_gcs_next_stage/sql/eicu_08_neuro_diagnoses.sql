-- =====================================================================
-- eicu_08_neuro_diagnoses.sql —— eICU 神经系统基础病与急性神经诊断
--
-- eICU 的 diagnosis 表是**就诊级**诊断，带 diagnosisoffset（相对 ICU 入室
-- 的分钟数），因此比 MIMIC 的 ICD 汇总多一个"何时被记录"的线索。
-- 同时提供 diagnosisstring（eICU 自有层级文本）与 icd9code（ICD-9 编码）。
--
-- 编码说明：eICU 只有 ICD-9（icd9code 可能带小数点，如 '348.3'）。
--   本脚本用 replace(icd9code,'.','') 归一后做前缀匹配。
--   归组规则与 MIMIC 侧 mimic_10 对齐（同一套 neuro_group 命名），
--   便于两库比较；但两库编码体系不同，不可视为等价测量。
--
--   delirium_encephalopathy : 293 / 348.3 / 348.5 / 348.1 / 780.0
--   cerebrovascular         : 430-438
--   intracranial_injury     : 800-804 / 850-854 / 959.01
--   dementia_degenerative   : 290 / 331 / 332 / 333.4
--   epilepsy_seizure        : 345 / 780.3
--   neuromuscular           : 358 / 359
--   spinal_neuropathy       : 353-357
--
-- 同时输出 diagnosisstring 以便按 eICU 自有体系二次归组（例如
-- 'neurologic|...' 前缀），两条路径的结果应在 QA 报告中对照。
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
dx AS (
    SELECT
        d.patientunitstayid, d.diagnosisid, d.diagnosisoffset,
        d.diagnosisstring, d.icd9code, d.diagnosispriority, d.activeupondischarge,
        replace(coalesce(d.icd9code,''), '.', '') AS icd9_norm,
        CASE
            WHEN replace(coalesce(d.icd9code,''),'.','') ~ '^(293|3483|3485|3481|7800)' 
                THEN 'delirium_encephalopathy'
            WHEN replace(coalesce(d.icd9code,''),'.','') ~ '^(430|431|432|433|434|435|436|437|438)'
                THEN 'cerebrovascular'
            WHEN replace(coalesce(d.icd9code,''),'.','') ~ '^(800|801|802|803|804|850|851|852|853|854|95901)'
                THEN 'intracranial_injury'
            WHEN replace(coalesce(d.icd9code,''),'.','') ~ '^(290|331|332|3334)'
                THEN 'dementia_degenerative'
            WHEN replace(coalesce(d.icd9code,''),'.','') ~ '^(345|7803)'
                THEN 'epilepsy_seizure'
            WHEN replace(coalesce(d.icd9code,''),'.','') ~ '^(358|359)'
                THEN 'neuromuscular'
            WHEN replace(coalesce(d.icd9code,''),'.','') ~ '^(353|354|355|356|357)'
                THEN 'spinal_neuropathy'
            -- ICD-9 缺失时，退回 diagnosisstring 文本路径
            WHEN d.diagnosisstring ~* 'delirium|encephalopath'
                THEN 'delirium_encephalopathy'
            WHEN d.diagnosisstring ~* 'stroke|intracerebral hemorrhage|subarachnoid|ischemic|embolic'
                THEN 'cerebrovascular'
            WHEN d.diagnosisstring ~* 'dementia|alzheimer|parkinson'
                THEN 'dementia_degenerative'
            WHEN d.diagnosisstring ~* 'seizure|epilep'
                THEN 'epilepsy_seizure'
            WHEN d.diagnosisstring ~* 'myasthenia|guillain|neuropathy|muscular dystrophy'
                THEN 'neuromuscular'
        END AS neuro_group,
        CASE WHEN d.icd9code IS NULL OR d.icd9code = '' 
             THEN 'diagnosisstring_only' ELSE 'icd9_available' END AS coding_source
    FROM eicu_crd.diagnosis d
)
SELECT
    c.patientunitstayid, c.patienthealthsystemstayid, c.uniquepid, c.hospitalid,
    dx.diagnosisid, dx.diagnosisoffset,
    ROUND((dx.diagnosisoffset / 60.0)::numeric, 4) AS hours_from_unit_admit,
    dx.diagnosisstring, dx.icd9code, dx.icd9_norm,
    dx.diagnosispriority, dx.activeupondischarge, dx.neuro_group, dx.coding_source,
    CASE WHEN dx.diagnosisoffset BETWEEN -360 AND 2880 THEN 1 ELSE 0 END AS in_window_0_48h,
    CASE WHEN dx.diagnosisoffset < 0 THEN 1 ELSE 0 END AS before_unit_admit,
    CASE WHEN dx.diagnosisoffset > 2880 THEN 1 ELSE 0 END AS after_48h
FROM dx
INNER JOIN cohort c ON dx.patientunitstayid = c.patientunitstayid
WHERE dx.neuro_group IS NOT NULL
ORDER BY c.patientunitstayid, dx.diagnosisoffset, dx.neuro_group;
