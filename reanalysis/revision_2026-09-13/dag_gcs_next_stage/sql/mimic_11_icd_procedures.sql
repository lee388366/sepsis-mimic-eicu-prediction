-- =====================================================================
-- mimic_11_icd_procedures.sql —— 呼吸支持相关操作（ICD 手术/操作编码）
--
-- 作用：作为 mimic_06/07/08 之外的第四个独立来源，用于交叉验证
--       "是否接受过机械通气/气道操作"。ICD 编码为住院级汇总，
--       只有日期（chartdate）而无精确时刻，不能替代床旁时间。
--
-- 编码规则（icd_code 在 MIMIC 中不带小数点）：
--   ICD-9  : 9670/9671/9672 = 持续机械通气 <96h / 96-<168h / >=168h
--            9601/9602/9603/9604 = 气管插管相关
--            9604 = 插入气管内导管
--   ICD-10 : 5A1935Z / 5A1945Z / 5A1955Z = 机械通气 <24h / 24-96h / >96h
--            0BH17EZ = 经口气管内插管
--            0B110F4 / 0B113F4 等 = 气道相关操作
-- 本脚本同时输出 icd_version 与原始编码，归组可审计。
--
-- 数据库：mimiciv31（只读）
-- =====================================================================
WITH sepsis AS (
    SELECT stay_id FROM mimiciv_derived.sepsis3
    WHERE sepsis3 IS TRUE AND suspected_infection_time IS NOT NULL
),
icu_stay_count AS (
    SELECT hadm_id, COUNT(*) AS n_icu_stays FROM mimiciv_icu.icustays GROUP BY hadm_id
),
base AS (
    SELECT ie.subject_id, ie.hadm_id, ie.stay_id, ie.intime
    FROM mimiciv_icu.icustays ie
    INNER JOIN sepsis s                     ON ie.stay_id   = s.stay_id
    INNER JOIN mimiciv_hosp.patients p      ON ie.subject_id = p.subject_id
    INNER JOIN mimiciv_hosp.admissions adm  ON ie.hadm_id   = adm.hadm_id
    INNER JOIN icu_stay_count cnt           ON ie.hadm_id   = cnt.hadm_id
    WHERE p.anchor_age >= 18
      AND ie.outtime >= ie.intime + INTERVAL '48 hours'
      AND adm.hospital_expire_flag IS NOT NULL
      AND cnt.n_icu_stays <= 2
),
ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime ASC, stay_id ASC) AS rn
    FROM base
),
cohort AS (SELECT * FROM ranked WHERE rn = 1),
px AS (
    SELECT
        pr.subject_id, pr.hadm_id, pr.seq_num, pr.chartdate,
        pr.icd_code, pr.icd_version, dp.long_title,
        CASE
            WHEN pr.icd_version = 9 AND pr.icd_code IN ('9670','9671','9672') THEN 'mechanical_ventilation'
            WHEN pr.icd_version = 9 AND pr.icd_code IN ('9601','9602','9603','9604') THEN 'endotracheal_intubation'
            WHEN pr.icd_version = 9 AND pr.icd_code IN ('311','3121','3129') THEN 'tracheostomy'
            WHEN pr.icd_version = 10 AND pr.icd_code IN ('5A1935Z','5A1945Z','5A1955Z') THEN 'mechanical_ventilation'
            WHEN pr.icd_version = 10 AND pr.icd_code = '0BH17EZ' THEN 'endotracheal_intubation'
            WHEN pr.icd_version = 10 AND pr.icd_code IN ('0B110F4','0B113F4','0B114F4','0B21XZ9') THEN 'tracheostomy'
        END AS airway_procedure_group
    FROM mimiciv_hosp.procedures_icd pr
    LEFT JOIN mimiciv_hosp.d_icd_procedures dp
           ON pr.icd_code = dp.icd_code AND pr.icd_version = dp.icd_version
)
SELECT
    c.stay_id, px.subject_id, px.hadm_id,
    px.seq_num, px.chartdate, px.icd_version, px.icd_code, px.long_title,
    px.airway_procedure_group
FROM px
INNER JOIN cohort c ON px.hadm_id = c.hadm_id
WHERE px.airway_procedure_group IS NOT NULL
ORDER BY c.stay_id, px.chartdate, px.seq_num;
