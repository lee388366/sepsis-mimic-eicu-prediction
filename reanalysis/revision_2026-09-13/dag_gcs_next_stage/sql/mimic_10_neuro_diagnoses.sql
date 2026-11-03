-- =====================================================================
-- mimic_10_neuro_diagnoses.sql —— 神经系统基础病与急性神经诊断
--
-- 用途（DAG 中的 U 与部分 N 的背景约束）：
--   区分"GCS 低是既有神经系统疾病造成的基线状态"与"急性获得性意识改变"。
--   没有这一层，低 GCS 会被误读为单纯的急性脑损伤。
--
-- 编码规则：MIMIC 的 icd_code 不带小数点。本脚本对 ICD-9 与 ICD-10 分别
--   建立前缀规则，并输出自定义的 neuro_group（可审计、可调整）：
--     delirium_encephalopathy : ICD-10 F05/F06/G93.1/G93.4 ; ICD-9 293.x/348.3/348.5/780.0x
--     cerebrovascular         : ICD-10 I60-I69/G45/G46 ; ICD-9 430-438
--     intracranial_injury     : ICD-10 S06 ; ICD-9 800-804/850-854/959.01
--     dementia_degenerative   : ICD-10 G10-G31/F00-F03 ; ICD-9 290/331/332/333.4
--     epilepsy_seizure        : ICD-10 G40/G41/R56 ; ICD-9 345/780.3x
--     neuromuscular           : ICD-10 G70-G73 ; ICD-9 358/359
--     spinal_neuropathy       : ICD-10 G50-G64 ; ICD-9 353-357
--   一位患者可同时落入多组（本脚本不做互斥归组，保留原始行）。
--
-- 同时输出 diagnosis_priority 与 seq_num（若在表中），但本表无时间戳——
-- ICD 诊断是住院级总结，不能确定"何时发生"，须在使用时明确这一限制。
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
dx AS (
    SELECT
        d.subject_id, d.hadm_id, d.seq_num, d.icd_code, d.icd_version,
        dd.long_title,
        CASE
            -- ---------- ICD-10 ----------
            WHEN d.icd_version = 10 AND (
                    d.icd_code LIKE 'F05%' OR d.icd_code LIKE 'F06%'
                 OR d.icd_code LIKE 'G931%' OR d.icd_code LIKE 'G934%'
                 OR d.icd_code LIKE 'G92%'
                ) THEN 'delirium_encephalopathy'
            WHEN d.icd_version = 10 AND (
                    d.icd_code ~ '^I6[0-9]' OR d.icd_code LIKE 'G45%' OR d.icd_code LIKE 'G46%'
                 OR d.icd_code LIKE 'I60%' OR d.icd_code LIKE 'I61%' OR d.icd_code LIKE 'I62%'
                 OR d.icd_code LIKE 'I63%' OR d.icd_code LIKE 'I64%' OR d.icd_code LIKE 'I65%'
                 OR d.icd_code LIKE 'I66%' OR d.icd_code LIKE 'I67%' OR d.icd_code LIKE 'I68%'
                 OR d.icd_code LIKE 'I69%'
                ) THEN 'cerebrovascular'
            WHEN d.icd_version = 10 AND d.icd_code LIKE 'S06%' THEN 'intracranial_injury'
            WHEN d.icd_version = 10 AND (
                    d.icd_code ~ '^G1[0-9]' OR d.icd_code ~ '^G2[0-9]' OR d.icd_code ~ '^G3[01]'
                 OR d.icd_code LIKE 'F00%' OR d.icd_code LIKE 'F01%'
                 OR d.icd_code LIKE 'F02%' OR d.icd_code LIKE 'F03%'
                ) THEN 'dementia_degenerative'
            WHEN d.icd_version = 10 AND (
                    d.icd_code LIKE 'G40%' OR d.icd_code LIKE 'G41%' OR d.icd_code LIKE 'R56%'
                ) THEN 'epilepsy_seizure'
            WHEN d.icd_version = 10 AND (d.icd_code ~ '^G7[0-3]') THEN 'neuromuscular'
            WHEN d.icd_version = 10 AND (d.icd_code ~ '^G5[0-9]' OR d.icd_code ~ '^G6[0-4]')
                THEN 'spinal_neuropathy'
            -- ---------- ICD-9 ----------
            WHEN d.icd_version = 9 AND (
                    d.icd_code LIKE '293%' OR d.icd_code LIKE '3483%' OR d.icd_code LIKE '3485%'
                 OR d.icd_code LIKE '7800%' OR d.icd_code LIKE '3481%'
                ) THEN 'delirium_encephalopathy'
            WHEN d.icd_version = 9 AND (
                    d.icd_code ~ '^43[0-8]' OR d.icd_code LIKE '430%' OR d.icd_code LIKE '431%'
                 OR d.icd_code LIKE '432%' OR d.icd_code LIKE '433%' OR d.icd_code LIKE '434%'
                 OR d.icd_code LIKE '435%' OR d.icd_code LIKE '436%' OR d.icd_code LIKE '437%'
                 OR d.icd_code LIKE '438%'
                ) THEN 'cerebrovascular'
            WHEN d.icd_version = 9 AND (
                    d.icd_code LIKE '800%' OR d.icd_code LIKE '801%' OR d.icd_code LIKE '802%'
                 OR d.icd_code LIKE '803%' OR d.icd_code LIKE '804%' OR d.icd_code LIKE '850%'
                 OR d.icd_code LIKE '851%' OR d.icd_code LIKE '852%' OR d.icd_code LIKE '853%'
                 OR d.icd_code LIKE '854%' OR d.icd_code LIKE '95901%'
                ) THEN 'intracranial_injury'
            WHEN d.icd_version = 9 AND (
                    d.icd_code LIKE '290%' OR d.icd_code LIKE '331%'
                 OR d.icd_code LIKE '332%' OR d.icd_code LIKE '3334%'
                ) THEN 'dementia_degenerative'
            WHEN d.icd_version = 9 AND (
                    d.icd_code LIKE '345%' OR d.icd_code LIKE '7803%'
                ) THEN 'epilepsy_seizure'
            WHEN d.icd_version = 9 AND (
                    d.icd_code LIKE '358%' OR d.icd_code LIKE '359%'
                ) THEN 'neuromuscular'
            WHEN d.icd_version = 9 AND (
                    d.icd_code LIKE '353%' OR d.icd_code LIKE '354%' OR d.icd_code LIKE '355%'
                 OR d.icd_code LIKE '356%' OR d.icd_code LIKE '357%'
                ) THEN 'spinal_neuropathy'
        END AS neuro_group
    FROM mimiciv_hosp.diagnoses_icd d
    LEFT JOIN mimiciv_hosp.d_icd_diagnoses dd
           ON d.icd_code = dd.icd_code AND d.icd_version = dd.icd_version
)
SELECT
    c.stay_id, dx.subject_id, dx.hadm_id,
    dx.seq_num, dx.icd_version, dx.icd_code, dx.long_title, dx.neuro_group
FROM dx
INNER JOIN cohort c ON dx.hadm_id = c.hadm_id
WHERE dx.neuro_group IS NOT NULL
ORDER BY c.stay_id, dx.seq_num;
