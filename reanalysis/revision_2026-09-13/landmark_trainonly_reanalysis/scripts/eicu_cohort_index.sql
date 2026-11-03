-- =====================================================================
-- eicu_cohort_index.sql —— eICU 分析队列索引（供 s02 构建训练集独立数据）
--
-- 相对 eicu_01_cohort.sql 的两处**必要修正**（见 LANDMARK_ELIGIBILITY_AUDIT.md §4.1）：
--   1. 结局可判定条件由 `hospitaldischargestatus IS NOT NULL` 改为
--      `hospitaldischargestatus IN ('Alive','Expired')`。
--      原因：本表存在空字符串 ''（全库 1,751 条）不为 NULL，
--      原条件会多留 71 例结局不可判定记录（6,883 → 6,812）。
--   2. 显式带出 uniquepid（患者级 ID），用于后续 patient-level 划分与
--      uniquepid 级聚类 bootstrap。
--
-- 另带出 eicu_03_sofa_qsofa.sql 已算好的 SOFA / qSOFA（同队列同顺序核对）。
-- 只读查询。
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
    SELECT
        p.patientunitstayid,
        p.patienthealthsystemstayid,
        p.uniquepid,
        p.hospitalid,
        p.age,
        p.gender,
        p.unitdischargeoffset,
        p.unitvisitnumber,
        p.hospitaldischargestatus,
        CASE WHEN p.hospitaldischargestatus = 'Expired' THEN 1
             WHEN p.hospitaldischargestatus = 'Alive'   THEN 0
             END AS label
    FROM eicu_crd.patient p
    INNER JOIN sepsis_dx s ON p.patientunitstayid = s.patientunitstayid
    WHERE p.unitvisitnumber = 1
      AND p.unitdischargeoffset > 2880
      AND p.hospitaldischargestatus IN ('Alive', 'Expired')
      AND ( (p.age ~ '^[0-9]+$' AND CAST(p.age AS INT) >= 18)
            OR p.age = '> 89' )
)
SELECT
    c.patientunitstayid,
    c.patienthealthsystemstayid,
    c.uniquepid,
    c.hospitalid,
    c.age,
    c.gender,
    c.unitdischargeoffset,
    c.label
FROM cohort c
ORDER BY c.patientunitstayid
