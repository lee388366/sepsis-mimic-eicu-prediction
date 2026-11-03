-- =====================================================================
-- mimic_03_drug_inputevents.sql —— 镇静/阿片/肌松：ICU 实际给药记录
--
-- 数据层说明（三层必须分开，不可互相替代）：
--   本脚本 = inputevents：ICU 内实际输注/推注的记帐记录（含速率与总量）
--   mimic_04 = emar：全院给药执行记录，event_txt 明确区分
--              Administered / Not Given / Started / Stopped / Confirmed
--   mimic_05 = prescriptions：医嘱层，只有医嘱起止时间
--   → "未记录用药" ≠ "未用药"：只有 emar 的 'Not Given' 才能证否给药。
--
-- itemid 已由 mimiciv_icu.d_items 穷尽核查（见 qa/drug_itemid_census）。
-- 药物类别（class）由脚本按 itemid 显式标注，不使用模糊文本匹配，
-- 避免把 "Conscious Sedation" 这类文本字段误当药品。
--
-- 窗口：与 GCS 事件一致 [t0 - 6h, t0 + 48h]；采用"重叠"而非"包含"逻辑，
--       跨越窗口边界的持续输注一律保留（否则无法判断评分时点是否在治疗中）。
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
    ie.subject_id, ie.hadm_id, ie.stay_id,
    ie.starttime, ie.endtime,
    ROUND((EXTRACT(EPOCH FROM (ie.starttime - c.intime)) / 3600.0)::numeric, 4) AS start_hours_from_intime,
    ROUND((EXTRACT(EPOCH FROM (ie.endtime   - c.intime)) / 3600.0)::numeric, 4) AS end_hours_from_intime,
    ie.itemid, di.label AS drug_label, di.unitname AS item_unit,
    CASE
        WHEN ie.itemid IN (222168, 221668, 221385, 225150, 229420, 225156, 221712)
            THEN 'sedative_hypnotic'
        WHEN ie.itemid IN (221744, 225942, 225972, 221833, 225154, 227520)
            THEN 'opioid'
        WHEN ie.itemid IN (221555, 229233, 222062)
            THEN 'neuromuscular_blocker'
    END AS drug_class,
    ie.amount, ie.amountuom,
    ie.rate,   ie.rateuom,
    CASE WHEN ie.rateuom IS NULL THEN NULL
         WHEN ie.rateuom ~* '(min|hour|hr)' THEN 'infusion'
         ELSE 'other_rate' END AS administration_mode,
    ie.statusdescription,
    ie.orderid, ie.linkorderid,
    ie.ordercategoryname,
    ie.patientweight,
    ie.totalamount, ie.totalamountuom,
    ie.isopenbag, ie.continueinnextdept,
    -- 是否在窗口起点之前就已开始（用于判断窗口内的"治疗中"状态）
    CASE WHEN ie.starttime < c.win_start THEN 1 ELSE 0 END AS started_before_window,
    -- 该次给药是否与 48h 观察窗真正重叠
    CASE WHEN ie.starttime <= c.win_end AND ie.endtime >= c.win_start THEN 1 ELSE 0 END AS overlaps_window
FROM mimiciv_icu.inputevents ie
INNER JOIN cohort c ON ie.stay_id = c.stay_id
LEFT JOIN mimiciv_icu.d_items di ON ie.itemid = di.itemid
WHERE ie.itemid IN (
        222168, 221668, 221385, 225150, 229420, 225156, 221712,   -- 镇静/催眠
        221744, 225942, 225972, 221833, 225154, 227520,           -- 阿片
        221555, 229233, 222062                                    -- 肌松
      )
  AND ie.starttime <= c.win_end
  AND ie.endtime   >= c.win_start
ORDER BY ie.stay_id, ie.starttime, ie.itemid;
