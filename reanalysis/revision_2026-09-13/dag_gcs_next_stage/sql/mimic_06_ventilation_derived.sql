-- =====================================================================
-- mimic_06_ventilation_derived.sql —— 派生呼吸支持时段（含窗口外延续）
--
-- 数据来源：mimiciv_derived.ventilation（MIT-LCP 派生表）
-- 取值（已核验全库分布）：
--   InvasiveVent | NonInvasiveVent | HFNC | SupplementalOxygen | Tracheostomy | None
--
-- 用途：判断"某次 GCS 评分时点是否处于有创/无创通气中"。
-- 关键点：绝不可以用 FiO2 是否存在、或 GCS 语言分项为 0 来金标准式地
--         认定机械通气（脚本内不产生此类派生）。
-- 窗口关系：overlaps_window 用重叠逻辑，窗口外的 starttime/endtime 原样保留，
--         以便区分"窗口开始时已在通气中"与"窗口内新插管"。
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
    c.subject_id, c.hadm_id, c.stay_id,
    v.starttime, v.endtime, v.ventilation_status,
    ROUND((EXTRACT(EPOCH FROM (v.starttime - c.intime)) / 3600.0)::numeric, 4) AS start_hours_from_intime,
    ROUND((EXTRACT(EPOCH FROM (v.endtime   - c.intime)) / 3600.0)::numeric, 4) AS end_hours_from_intime,
    ROUND((EXTRACT(EPOCH FROM (v.endtime - v.starttime)) / 3600.0)::numeric, 4) AS duration_h,
    CASE WHEN v.starttime <  c.win_start THEN 1 ELSE 0 END AS started_before_window,
    CASE WHEN v.starttime <= c.win_end AND v.endtime >= c.win_start THEN 1 ELSE 0 END AS overlaps_window,
    CASE WHEN v.starttime >= c.intime AND v.starttime <= c.win_end THEN 1 ELSE 0 END AS started_in_0_48h
FROM mimiciv_derived.ventilation v
INNER JOIN cohort c ON v.stay_id = c.stay_id
WHERE v.starttime <= c.win_end
  AND v.endtime   >= c.win_start
ORDER BY c.stay_id, v.starttime;
