-- =====================================================================
-- mimic_07_airway_procedures.sql —— 气道操作与呼吸支持（procedureevents）
--
-- 这是"床旁真实操作"的记录层，比 chartevents 的气道字段更接近事件事实：
--   224385 Intubation                            → 插管
--   227194 Extubation                            → 拔管（计划）
--   225477 Unplanned Extubation (non-patient initiated)  → 非计划拔管
--   225468 Unplanned Extubation (patient-initiated)      → 自行拔管
--   226237 Open Tracheostomy                     → 外科气管切开
--   225448 Percutaneous Tracheostomy             → 经皮气管切开
--   225792 Invasive Ventilation                  → 有创通气
--   225794 Non-invasive Ventilation              → 无创通气
--
-- 注：procedureevents 中 endtime 可能为 NULL（过程未记录结束），
--     一律用 COALESCE(endtime, starttime) 处理窗口重叠，并保留原始 NULL。
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
    pe.subject_id, pe.hadm_id, pe.stay_id,
    pe.starttime, pe.endtime,
    ROUND((EXTRACT(EPOCH FROM (pe.starttime - c.intime)) / 3600.0)::numeric, 4) AS start_hours_from_intime,
    CASE WHEN pe.endtime IS NOT NULL
         THEN ROUND((EXTRACT(EPOCH FROM (pe.endtime - c.intime)) / 3600.0)::numeric, 4)
    END AS end_hours_from_intime,
    ROUND((EXTRACT(EPOCH FROM (COALESCE(pe.endtime, pe.starttime) - pe.starttime)) / 3600.0)::numeric, 4) AS duration_h,
    pe.itemid, di.label AS procedure_label, di.category AS procedure_category,
    CASE
        WHEN pe.itemid = 224385 THEN 'intubation'
        WHEN pe.itemid IN (227194, 225477, 225468) THEN 'extubation'
        WHEN pe.itemid IN (226237, 225448) THEN 'tracheostomy'
        WHEN pe.itemid = 225792 THEN 'invasive_ventilation'
        WHEN pe.itemid = 225794 THEN 'noninvasive_ventilation'
    END AS airway_event_type,
    pe.location, pe.locationcategory,
    pe.orderid, pe.linkorderid,
    pe.value, pe.valueuom,
    pe.ordercategoryname, pe.statusdescription,
    CASE WHEN pe.starttime <  c.win_start THEN 1 ELSE 0 END AS started_before_window,
    CASE WHEN pe.starttime <= c.win_end
              AND COALESCE(pe.endtime, pe.starttime) >= c.win_start THEN 1 ELSE 0 END AS overlaps_window,
    CASE WHEN pe.starttime >= c.intime AND pe.starttime <= c.win_end THEN 1 ELSE 0 END AS started_in_0_48h
FROM mimiciv_icu.procedureevents pe
INNER JOIN cohort c ON pe.stay_id = c.stay_id
LEFT JOIN mimiciv_icu.d_items di ON pe.itemid = di.itemid
WHERE pe.itemid IN (224385, 227194, 225477, 225468, 226237, 225448, 225792, 225794)
  AND pe.starttime <= c.win_end
  AND COALESCE(pe.endtime, pe.starttime) >= c.win_start
ORDER BY c.stay_id, pe.starttime, pe.itemid;
