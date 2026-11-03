-- =====================================================================
-- eicu_04b_extract_demographics.sql —— eICU 外验集人口学提取（按 patientunitstayid 索引）
-- 队列与 eicu_01_cohort.sql 完全一致；ethnicity 映射为 MIMIC 可比口径
-- 使用方法：
--   psql -h localhost -U postgres -d eicu -Atc \
--     "COPY (SELECT * FROM (...)) TO STDOUT WITH CSV HEADER" > eicu_demographics.csv
-- =====================================================================

WITH sepsis_dx AS (
    SELECT DISTINCT patientunitstayid
    FROM eicu_crd.diagnosis
    WHERE diagnosispriority = 'Primary' AND diagnosisoffset <= 1440
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
)

SELECT
    p.patientunitstayid,
    p.age,
    p.gender,
    p.ethnicity,
    CASE
        WHEN p.ethnicity = 'Caucasian' THEN 'White'
        WHEN p.ethnicity = 'African American' THEN 'Black'
        WHEN p.ethnicity = 'Hispanic' THEN 'Hispanic'
        WHEN p.ethnicity = 'Asian' THEN 'Asian'
        WHEN p.ethnicity IN ('Native American', 'Other/Unknown') THEN 'Other/Unknown'
        WHEN p.ethnicity IS NULL OR p.ethnicity = '' THEN 'Other/Unknown'
        ELSE 'Other/Unknown'
    END AS race_group
FROM eicu_crd.patient p
INNER JOIN sepsis_dx s ON p.patientunitstayid = s.patientunitstayid
WHERE p.unitvisitnumber = 1
  AND p.unitdischargeoffset > 2880
  AND p.hospitaldischargestatus IS NOT NULL
  AND (p.age ~ '^[0-9]+$' AND CAST(p.age AS INT) >= 18 OR p.age = '> 89')
ORDER BY p.patientunitstayid;
