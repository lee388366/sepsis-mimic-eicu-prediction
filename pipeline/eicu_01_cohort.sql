-- =====================================================================
-- eicu_01_cohort.sql
-- eICU 脓毒症队列构建 —— 外验队列（对齐 MIMIC-IV 项目口径）
--
-- 目的：为"MIMIC-IV 训练 → eICU 外验"构建脓毒症外验队列。
--       对齐 MIMIC 项目（01_sepsis_cohort.sql）的口径：
--         脓毒症 ∩ 成人(>=18) ∩ ICU LOS>=48h ∩ 首次 ICU ∩ 结局可判定
--
-- 与 MIMIC 的差异（必须声明，论文 Methods/Limitations）：
--   - eICU 无官方 Sepsis-3 表，脓毒症用 diagnosis 表诊断字符串识别
--     （参考 sepsis_predict 论文清单：'sepsis'/'septic shock'/'severe sepsis'
--      等 Primary 诊断，24h 内记录）
--   - SOFA>=2 验证可选（依赖 eicu_03_sofa_qsofa.sql，见文件末尾注释）
--
-- 队列定义（eICU 版本）：
--   1. 脓毒症：diagnosis 表诊断字符串含 sepsis/septic shock（Primary, offset<=1440）
--   2. 成人：age 为数字且 >=18，或 '> 89'（视为 89+ 成人）
--   3. ICU LOS >= 48h：unitdischargeoffset > 2880（分钟）
--   4. 首次 ICU：unitvisitnumber = 1
--   5. 结局可判定：hospitaldischargestatus IS NOT NULL
--
-- 输出列：
--   patientunitstayid  唯一 ICU stay
--   patienthealthsystemstayid  住院级 ID
--   age                （'> 89' 保留原样，供 Python 处理）
--   gender
--   admissionweight / admissionheight
--   unitdischargeoffset  ICU 停留时长（分钟）
--   hospitaldischargestatus  结局（Expired=1）
--
-- 使用方法：
--   psql -h localhost -U postgres -d eicu -f eicu_01_cohort.sql
--   或导出：psql -h localhost -U postgres -d eicu -Atc \
--     "COPY (SELECT * FROM (...)) TO STDOUT WITH CSV HEADER" > eicu_cohort.csv
-- =====================================================================

WITH sepsis_dx AS (
    -- 脓毒症诊断：参考 sepsis_predict 论文的字符串清单
    SELECT DISTINCT patientunitstayid
    FROM eicu_crd.diagnosis
    WHERE diagnosispriority = 'Primary'
      AND diagnosisoffset <= 1440          -- 入 ICU 24h 内
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

base AS (
    SELECT
        p.patientunitstayid,
        p.patienthealthsystemstayid,
        p.age,
        p.gender,
        p.admissionweight,
        p.admissionheight,
        p.unitvisitnumber,
        p.unitdischargeoffset,
        p.hospitaldischargestatus,
        p.hospitaldischargeoffset,
        p.hospitalid
    FROM eicu_crd.patient p
    INNER JOIN sepsis_dx s ON p.patientunitstayid = s.patientunitstayid
)

-- 最终队列
SELECT
    patientunitstayid,
    patienthealthsystemstayid,
    age,
    gender,
    admissionweight,
    admissionheight,
    unitdischargeoffset,
    hospitaldischargestatus,
    CASE WHEN hospitaldischargestatus = 'Expired' THEN 1
         WHEN hospitaldischargestatus = 'Alive' THEN 0
         ELSE NULL END AS label,
    hospitalid
FROM base
WHERE unitvisitnumber = 1                      -- 首次 ICU
  AND unitdischargeoffset > 2880               -- LOS >= 48h (2880 分钟)
  AND hospitaldischargestatus IS NOT NULL      -- 结局可判定
  AND (
         age ~ '^[0-9]+$' AND CAST(age AS INT) >= 18   -- 数字年龄 >=18
      OR age = '> 89'                                   -- 89+ 视为成人
  )
ORDER BY patientunitstayid;

-- =====================================================================
-- 注意：SOFA>=2 验证（Sepsis-3 器官衰竭标准）依赖 eicu_03_sofa_qsofa.sql。
-- 阶段 3 完成后，如需加入 SOFA>=2 条件，在上方加：
--   INNER JOIN eicu_sofa s2 ON s2.patientunitstayid = base.patientunitstayid
--   WHERE ... AND s2.sofatotal >= 2
-- 但注意：qSOFA/SOFA 作为"对标评分"，通常要求模型输入变量与评分变量
-- 在同窗计算。若 SOFA>=2 作为队列条件，则 SOFA 同时是队列定义与对标，
-- 需在论文中说明这一循环。建议默认不加 SOFA 到队列条件，保持对标独立性。
-- =====================================================================
