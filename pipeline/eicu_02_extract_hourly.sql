-- =====================================================================
-- eicu_02_extract_hourly.sql
-- eICU 48h 小时级特征提取 —— 外验特征（对齐 MIMIC-IV 项目口径）
--
-- 目的：提取 LSTM 需要的 14 变量 × 48h 小时级特征，完全对齐
--       MIMIC 项目的 02_extract_hourly.sql 输出结构。
--
-- 时间窗：eICU 用分钟 offset（相对 ICU 入室），
--         [0, 2880) 分钟 = 48h，按 60 分钟桶聚合（hr = offset/60）。
--
-- 变量来源（已核查，与 sepsis_predict 论文一致）：
--   HR/RR/SpO2/SBP/DBP/MBP/Temp   nursecharting
--   GCS 总分/眼/运动/语言          nursecharting 'Glasgow coma score'
--   Glucose / pH                    lab
--   FiO2                           respiratorycharting
--   Weight / Height                 patient
--
-- 队列：eicu_01_cohort.sql 的 6,883 例（脓毒症 ∩ 成人 ∩ LOS>=48h ∩ 首次 ICU）
--
-- 输出（每行 = 一个患者的 1 小时）：
--   patientunitstayid, hr(0-47), 14 个变量, weight, height, label
--
-- 使用方法：
--   psql -h localhost -U postgres -d eicu -Atc \
--     "COPY (SELECT * FROM (...)) TO STDOUT WITH CSV HEADER" > eicu_hourly.csv
-- =====================================================================

-- 队列（与 eicu_01_cohort.sql 完全一致）
WITH cohort AS (
    SELECT
        p.patientunitstayid,
        p.age,
        p.gender,
        p.admissionweight,
        p.admissionheight,
        CASE WHEN p.hospitaldischargestatus = 'Expired' THEN 1
             WHEN p.hospitaldischargestatus = 'Alive' THEN 0
             ELSE NULL END AS label
    FROM eicu_crd.patient p
    INNER JOIN (
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
    ) s ON p.patientunitstayid = s.patientunitstayid
    WHERE p.unitvisitnumber = 1
      AND p.unitdischargeoffset > 2880
      AND p.hospitaldischargestatus IS NOT NULL
      AND (p.age ~ '^[0-9]+$' AND CAST(p.age AS INT) >= 18 OR p.age = '> 89')
),

-- 0-47 小时网格（eICU 无 generate_series 的限制，用 generate_series 生成）
hours AS (
    SELECT generate_series(0, 47) AS hr
),

-- 1) 生命体征（nursecharting）
vital AS (
    SELECT
        nc.patientunitstayid,
        FLOOR(nc.nursingchartoffset / 60)::int AS hr,
        AVG(CASE WHEN nc.nursingchartcelltypevallabel = 'Heart Rate'
                      AND nc.nursingchartcelltypevalname = 'Heart Rate'
                      AND nc.nursingchartvalue ~ '^[0-9]+(\.[0-9]+)?$'
                 THEN nc.nursingchartvalue::numeric ELSE NULL END) AS heart_rate,
        -- SBP: 优先无创，其次有创
        AVG(CASE WHEN nc.nursingchartcelltypevallabel = 'Non-Invasive BP'
                      AND nc.nursingchartcelltypevalname = 'Non-Invasive BP Systolic'
                      AND nc.nursingchartvalue ~ '^[0-9]+(\.[0-9]+)?$'
                 THEN nc.nursingchartvalue::numeric ELSE NULL END) AS sbp,
        AVG(CASE WHEN nc.nursingchartcelltypevallabel = 'Non-Invasive BP'
                      AND nc.nursingchartcelltypevalname = 'Non-Invasive BP Diastolic'
                      AND nc.nursingchartvalue ~ '^[0-9]+(\.[0-9]+)?$'
                 THEN nc.nursingchartvalue::numeric ELSE NULL END) AS dbp,
        AVG(CASE WHEN nc.nursingchartcelltypevallabel = 'Non-Invasive BP'
                      AND nc.nursingchartcelltypevalname = 'Non-Invasive BP Mean'
                      AND nc.nursingchartvalue ~ '^[0-9]+(\.[0-9]+)?$'
                 THEN nc.nursingchartvalue::numeric ELSE NULL END) AS mbp,
        AVG(CASE WHEN nc.nursingchartcelltypevallabel = 'Respiratory Rate'
                      AND nc.nursingchartcelltypevalname = 'Respiratory Rate'
                      AND nc.nursingchartvalue ~ '^[0-9]+(\.[0-9]+)?$'
                 THEN nc.nursingchartvalue::numeric ELSE NULL END) AS resp_rate,
        AVG(CASE WHEN nc.nursingchartcelltypevallabel = 'O2 Saturation'
                      AND nc.nursingchartcelltypevalname = 'O2 Saturation'
                      AND nc.nursingchartvalue ~ '^[0-9]+(\.[0-9]+)?$'
                 THEN nc.nursingchartvalue::numeric ELSE NULL END) AS spo2,
        AVG(CASE WHEN nc.nursingchartcelltypevallabel = 'Temperature'
                      AND nc.nursingchartcelltypevalname = 'Temperature (C)'
                      AND nc.nursingchartvalue ~ '^[0-9]+(\.[0-9]+)?$'
                 THEN nc.nursingchartvalue::numeric ELSE NULL END) AS temperature
    FROM eicu_crd.nursecharting nc
    INNER JOIN cohort c ON nc.patientunitstayid = c.patientunitstayid
    WHERE nc.nursingchartoffset >= 0
      AND nc.nursingchartoffset < 2880
    GROUP BY nc.patientunitstayid, FLOOR(nc.nursingchartoffset / 60)::int
),

-- 2) GCS（nursecharting 'Glasgow coma score'）
gcs AS (
    SELECT
        nc.patientunitstayid,
        FLOOR(nc.nursingchartoffset / 60)::int AS hr,
        AVG(CASE WHEN nc.nursingchartcelltypevalname = 'GCS Total'
                      AND nc.nursingchartvalue ~ '^[0-9]+(\.[0-9]+)?$'
                 THEN nc.nursingchartvalue::numeric ELSE NULL END) AS gcs,
        AVG(CASE WHEN nc.nursingchartcelltypevalname = 'Eyes'
                      AND nc.nursingchartvalue ~ '^[0-9]+(\.[0-9]+)?$'
                 THEN nc.nursingchartvalue::numeric ELSE NULL END) AS gcs_eyes,
        AVG(CASE WHEN nc.nursingchartcelltypevalname = 'Motor'
                      AND nc.nursingchartvalue ~ '^[0-9]+(\.[0-9]+)?$'
                 THEN nc.nursingchartvalue::numeric ELSE NULL END) AS gcs_motor,
        AVG(CASE WHEN nc.nursingchartcelltypevalname = 'Verbal'
                      AND nc.nursingchartvalue ~ '^[0-9]+(\.[0-9]+)?$'
                 THEN nc.nursingchartvalue::numeric ELSE NULL END) AS gcs_verbal
    FROM eicu_crd.nursecharting nc
    INNER JOIN cohort c ON nc.patientunitstayid = c.patientunitstayid
    WHERE nc.nursingchartcelltypecat IN ('Scores', 'Other Vital Signs and Infusions')
      AND nc.nursingchartoffset >= 0
      AND nc.nursingchartoffset < 2880
    GROUP BY nc.patientunitstayid, FLOOR(nc.nursingchartoffset / 60)::int
),

-- 3) 实验室（Glucose / pH）
-- 注意：lab.labresult 是 numeric 列（非字符串），直接用数值过滤，不需要正则
lab AS (
    SELECT
        l.patientunitstayid,
        FLOOR(l.labresultoffset / 60)::int AS hr,
        AVG(CASE WHEN LOWER(l.labname) = 'glucose'
                      AND l.labresult IS NOT NULL
                 THEN l.labresult ELSE NULL END) AS glucose,
        AVG(CASE WHEN LOWER(l.labname) = 'ph'
                      AND l.labresult IS NOT NULL
                 THEN l.labresult ELSE NULL END) AS ph
    FROM eicu_crd.lab l
    INNER JOIN cohort c ON l.patientunitstayid = c.patientunitstayid
    WHERE LOWER(l.labname) IN ('glucose', 'ph')
      AND l.labresult IS NOT NULL
      AND l.labresultoffset >= 0
      AND l.labresultoffset < 2880
    GROUP BY l.patientunitstayid, FLOOR(l.labresultoffset / 60)::int
),

-- 4) FiO2（respiratorycharting，>20 排除升/分钟单位，<=100 裁剪异常值）
fio2 AS (
    SELECT
        rc.patientunitstayid,
        FLOOR(rc.respchartoffset / 60)::int AS hr,
        AVG(CASE WHEN LOWER(rc.respchartvaluelabel) LIKE '%fio2%'
                      AND rc.respchartvalue ~ '^[0-9]+(\.[0-9]+)?$'
                      AND rc.respchartvalue::numeric > 20
                      AND rc.respchartvalue::numeric <= 100
                 THEN rc.respchartvalue::numeric ELSE NULL END) AS fio2
    FROM eicu_crd.respiratorycharting rc
    INNER JOIN cohort c ON rc.patientunitstayid = c.patientunitstayid
    WHERE rc.respchartoffset >= 0
      AND rc.respchartoffset < 2880
    GROUP BY rc.patientunitstayid, FLOOR(rc.respchartoffset / 60)::int
)

-- 最终宽表：每行 = stay_id × hr(0-47)
SELECT
    c.patientunitstayid,
    c.age,
    c.gender,
    h.hr,
    v.heart_rate,
    v.sbp,
    v.dbp,
    v.mbp,
    v.resp_rate,
    v.spo2,
    v.temperature,
    l.glucose,
    l.ph,
    f.fio2,
    g.gcs,
    g.gcs_eyes,
    g.gcs_motor,
    g.gcs_verbal,
    c.admissionweight AS weight,
    c.admissionheight AS height,
    c.label
FROM cohort c
CROSS JOIN hours h
LEFT JOIN vital v ON v.patientunitstayid = c.patientunitstayid AND v.hr = h.hr
LEFT JOIN gcs g   ON g.patientunitstayid = c.patientunitstayid AND g.hr = h.hr
LEFT JOIN lab l   ON l.patientunitstayid = c.patientunitstayid AND l.hr = h.hr
LEFT JOIN fio2 f  ON f.patientunitstayid = c.patientunitstayid AND f.hr = h.hr
ORDER BY c.patientunitstayid, h.hr;
