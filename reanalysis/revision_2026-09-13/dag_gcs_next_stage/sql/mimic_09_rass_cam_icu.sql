-- =====================================================================
-- mimic_09_rass_cam_icu.sql —— 镇静深度与谵妄评估（辅助解释变量）
--
-- 这一层是判断"GCS 低到底反映意识还是镇静"的关键辅助信息：
--   RASS 直接刻画镇静深度，可与同刻 GCS 对照；
--   CAM-ICU 评估谵妄，是"意识改变"的另一个独立测量通道；
--   Riker-SAS 是 RASS 之外的镇静量表（不同 ICU 习惯不同）。
--
-- itemid（已由 d_items 核验，类别均为 Pain/Sedation）：
--   228096 Richmond-RAS Scale          → 实际 RASS 评分（-5 ~ +4）
--   228299 Goal Richmond-RAS Scale     → RASS 目标值（医嘱性目标，非实测）
--   228302 CAM-ICU RASS LOC            → CAM-ICU 中引用的 RASS/意识水平
--   228332 Delirium assessment         → 谵妄评估结论
--   228334 CAM-ICU Altered LOC         → 特征1：意识状态急性改变/波动
--   228301/228336/229325 CAM-ICU Inattention   → 特征2：注意力障碍
--   228303/228335/229324 CAM-ICU Disorganized thinking → 特征4：思维混乱
--   228300/228337/229326 CAM-ICU MS Change     → 精神状态变化
--   223753 Riker-SAS Scale             → Riker 镇静-躁动量表
--   228688 Delirium（MD Progress Note）→ 医生记录的谵妄（文本，非量表）
--   228710 Sedation goal（MD 记录）     → 镇静目标（文本）
--
-- 注意：Goal Richmond-RAS Scale 是"目标"，不可当实测镇静深度使用。
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
    SELECT ie.subject_id, ie.hadm_id, ie.stay_id, ie.intime,
           ie.intime - INTERVAL '6 hours'  AS win_start,
           ie.intime + INTERVAL '48 hours' AS win_end
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
cohort AS (SELECT * FROM ranked WHERE rn = 1)
SELECT
    ce.subject_id, ce.hadm_id, ce.stay_id,
    ce.charttime,
    ROUND((EXTRACT(EPOCH FROM (ce.charttime - c.intime)) / 3600.0)::numeric, 4) AS hours_from_intime,
    ce.itemid, di.label AS item_label,
    ce.value AS value_raw,
    ce.valuenum,
    ce.valueuom,
    CASE
        WHEN ce.itemid = 228096 THEN 'rass_actual'
        WHEN ce.itemid = 228299 THEN 'rass_goal'
        WHEN ce.itemid = 228302 THEN 'cam_icu_rass_loc'
        WHEN ce.itemid = 228332 THEN 'delirium_assessment'
        WHEN ce.itemid = 228334 THEN 'cam_icu_altered_loc'
        WHEN ce.itemid IN (228301, 228336, 229325) THEN 'cam_icu_inattention'
        WHEN ce.itemid IN (228303, 228335, 229324) THEN 'cam_icu_disorganized_thinking'
        WHEN ce.itemid IN (228300, 228337, 229326) THEN 'cam_icu_ms_change'
        WHEN ce.itemid = 223753 THEN 'riker_sas'
        WHEN ce.itemid = 228688 THEN 'delirium_md_note'
        WHEN ce.itemid = 228710 THEN 'sedation_goal_md'
    END AS assessment_field,
    -- RASS 文本转数值（如 'Alert and calm'=0, '-3'=-3）；仅做映射，不改原始值
    CASE WHEN ce.itemid = 228096 THEN
        CASE
            WHEN ce.value ~ '^-?[0-9]+$'            THEN ce.value::numeric
            WHEN ce.value ILIKE '%combative%'        THEN  4
            WHEN ce.value ILIKE '%very agitated%'    THEN  3
            WHEN ce.value ILIKE '%agitated%'         THEN  2
            WHEN ce.value ILIKE '%restless%'         THEN  1
            WHEN ce.value ILIKE '%alert and calm%'   THEN  0
            WHEN ce.value ILIKE '%drowsy%'           THEN -1
            WHEN ce.value ILIKE '%light sedation%'   THEN -2
            WHEN ce.value ILIKE '%moderate sedation%' THEN -3
            WHEN ce.value ILIKE '%deep sedation%'    THEN -4
            WHEN ce.value ILIKE '%unarousable%'      THEN -5
        END
    END AS rass_numeric,
    ce.storetime
FROM mimiciv_icu.chartevents ce
INNER JOIN cohort c ON ce.stay_id = c.stay_id
LEFT JOIN mimiciv_icu.d_items di ON ce.itemid = di.itemid
WHERE ce.itemid IN (228096, 228299, 228302, 228332, 228334,
                    228301, 228336, 229325,
                    228303, 228335, 229324,
                    228300, 228337, 229326,
                    223753, 228688, 228710)
  AND ce.charttime >= c.win_start
  AND ce.charttime <= c.win_end
ORDER BY ce.stay_id, ce.charttime, ce.itemid;
