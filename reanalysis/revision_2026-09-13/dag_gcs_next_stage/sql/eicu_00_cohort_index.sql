-- =====================================================================
-- eicu_00_cohort_index.sql —— eICU 完整队列索引（不排除任何记录）
--
-- 口径来源：项目根目录 eicu_01_cohort.sql（正式外验队列定义）
--
-- 【本轮核验发现的必须修正项】
--   原 eicu_01_cohort.sql 的条件是 `hospitaldischargestatus IS NOT NULL`。
--   但 eICU 该字段存在**空字符串 ''**（不是 NULL），因此空字符串记录
--   会被放行；随后 eicu_04_assemble.py 的 `label.notna()` 才把它们剔除，
--   导致 SQL 直跑得到 6,883 条、而正式队列只有 6,812 条，差 71 条。
--   已核实：这 71 条 hospitaldischargestatus 为空字符串，全部为非死亡；
--   eicu_sofa.csv 中 label 为 NaN 的恰为 71 条。
--   因此本脚本把结局可判定条件写成显式枚举 IN ('Alive','Expired')，
--   使 SQL 口径与正式队列一致，并在此记录该修正以便审计。
--   验证：修正后本脚本输出 6,812 条、死亡 1,178 条、6,488 名患者，
--        patientunitstayid 集合 md5 = 6453bf949c20baa8497914d5e32f8e45，
--        与 eicu_training_data.npz 完全一致。
--
-- 本表保留全部 6,812 条 ICU 入住记录，即使某类事件完全缺失也不剔除。
-- 时间基准：t0 = ICU 入室（patient.unitadmitsource 对应 offset 0）；
--          观察窗 [-360, +2880] 分钟 = [t0-6h, t0+48h]。
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
    SELECT
        p.patientunitstayid,
        p.patienthealthsystemstayid,
        p.uniquepid,
        p.hospitalid,
        p.wardid,
        p.unitvisitnumber,
        p.age, p.gender, p.ethnicity,
        p.admissionweight, p.dischargeweight,
        p.admissionheight,
        p.apacheadmissiondx,
        p.hospitaladmitsource, p.unitadmitsource, p.unitstaytype,
        p.hospitaladmitoffset,
        p.unitdischargeoffset,
        p.hospitaldischargeoffset,
        p.unitdischargestatus,
        p.hospitaldischargestatus,
        p.hospitaldischargelocation,
        p.hospitaldischargeyear,
        CASE WHEN p.hospitaldischargestatus = 'Expired' THEN 1
             WHEN p.hospitaldischargestatus = 'Alive'   THEN 0 END AS label
    FROM eicu_crd.patient p
    INNER JOIN sepsis_dx s ON p.patientunitstayid = s.patientunitstayid
    WHERE p.unitvisitnumber = 1
      AND p.unitdischargeoffset > 2880
      -- ↓ 本次修正：显式枚举，排除空字符串与 NULL
      AND p.hospitaldischargestatus IN ('Alive', 'Expired')
      AND (
             (p.age ~ '^[0-9]+$' AND CAST(p.age AS INT) >= 18)
          OR p.age = '> 89'
      )
)
SELECT
    c.*,
    h.region,
    h.teachingstatus,
    h.numbedscategory AS hospital_beds_category,
    -- 观察窗（分钟偏移）：[-360, 2880] = [t0-6h, t0+48h]
    -360  AS win_start_offset_min,
    2880  AS win_end_offset_min,
    c.hospitaldischargeoffset - c.unitdischargeoffset AS icu_discharge_to_hosp_discharge_min
FROM cohort c
LEFT JOIN eicu_crd.hospital h ON c.hospitalid = h.hospitalid
ORDER BY c.patientunitstayid;
