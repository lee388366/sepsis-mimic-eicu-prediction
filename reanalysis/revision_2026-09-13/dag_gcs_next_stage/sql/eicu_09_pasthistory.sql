-- =====================================================================
-- eicu_09_pasthistory.sql —— eICU 既往史（神经系统基础病）
--
-- eICU 的 pasthistory 是入院时采集的既往史文本，path 为层级结构
-- （如 'pastHistory|neurologic|...'），是识别"基线神经功能受损"的
-- 另一条线索，可补充 diagnosis 表（后者偏急性/在院诊断）。
--
-- 设计取舍：本脚本同时保留 path 与 value 全文，并按关键词给出
--   neuro_history_group，但明确标注该归组是文本匹配、可能漏检，
--   因此不设互斥、不丢弃未归组的神经相关行：
--   只要 path 或 value 命中神经系统关键词就保留（NOT NULL 条件仅限命中）。
--
-- 时间：pasthistoryoffset（分钟，距 ICU 入室）通常为负数（入院采集），
--       pasthistoryenteredoffset 为录入时刻。窗口不设 48h 限制——
--       既往史属于基线信息，晚录入也可能描述入院前状态；但输出中标注
--       是否落在窗口内，供分析者自行决定是否使用。
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
    ph.patientunitstayid, c.patienthealthsystemstayid, c.uniquepid, c.hospitalid,
    ph.pasthistoryid,
    ph.pasthistoryoffset, ph.pasthistoryenteredoffset,
    ROUND((ph.pasthistoryoffset / 60.0)::numeric, 4) AS hours_from_unit_admit,
    ph.pasthistorynotetype,
    ph.pasthistorypath,
    ph.pasthistoryvalue,
    ph.pasthistoryvaluetext,
    CASE
        WHEN ph.pasthistoryvalue ~* 'stroke|cva|cerebrovascular|intracranial hemorrhage|subarachnoid|tia'
            THEN 'cerebrovascular'
        WHEN ph.pasthistoryvalue ~* 'dementia|alzheimer|parkinson|cognitive|memory loss'
            THEN 'dementia_degenerative'
        WHEN ph.pasthistoryvalue ~* 'seizure|epilep'
            THEN 'epilepsy_seizure'
        WHEN ph.pasthistoryvalue ~* 'myasthenia|guillain|neuropathy|muscular dystrophy|als|multiple sclerosis'
            THEN 'neuromuscular'
        WHEN ph.pasthistoryvalue ~* 'traumatic brain|tbi|head injury|brain injury'
            THEN 'intracranial_injury'
        WHEN ph.pasthistoryvalue ~* 'developmental|mental retardation|down syndrome|cerebral palsy'
            THEN 'developmental'
        WHEN ph.pasthistoryvalue ~* 'encephalopath|delirium'
            THEN 'delirium_encephalopathy'
    END AS neuro_history_group,
    CASE WHEN ph.pasthistoryoffset BETWEEN -360 AND 2880 THEN 1 ELSE 0 END AS in_window_0_48h,
    CASE WHEN ph.pasthistoryoffset < 0 THEN 1 ELSE 0 END AS before_unit_admit
FROM eicu_crd.pasthistory ph
INNER JOIN cohort c ON ph.patientunitstayid = c.patientunitstayid
WHERE ph.pasthistorypath ~* 'neurolog|neuro|mental|psych'
   OR ph.pasthistoryvalue ~* 'stroke|cva|seizure|epilep|dementia|alzheimer|parkinson|myasthenia|guillain|neuropathy|brain injury|encephalopath|cerebral palsy|multiple sclerosis|mental retardation'
ORDER BY ph.patientunitstayid, ph.pasthistoryoffset, ph.pasthistorypath;
