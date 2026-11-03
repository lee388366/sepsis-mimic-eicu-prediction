-- =====================================================================
-- eicu_05_airway_respiratorycare.sql —— eICU 气道类型与通气时段
--
-- 这是 eICU 里最接近"人工气道状态 + 通气起止"的一张表。
--
-- airwaytype 取值（已穷尽枚举全库）：
--   '' (空字符串，770,432) | 'Oral ETT' (61,942) | 'No Artificial Airway' (24,085)
--   | 'Tracheostomy' (8,468) | 'Nasal ETT' (224) | 'Other' (205)
--   | 'Double-Lumen Tube' (16) | 'Cricothyrotomy' (9)
--   → '' 与 'No Artificial Airway' 必须区分：前者是未记录，后者是明确"无人工气道"。
--      若把 '' 当作"无人工气道"，会把人道气道患者误判为未插管。
--
-- ventstartoffset / ventendoffset        → 本次通气时段
-- priorventstartoffset / priorventendoffset → 本次之前的上一次通气时段
--   这两组字段使 eICU 可以部分重建通气区间，但仍无"拔管时刻"的直接字段。
--
-- 时间：respcarestatusoffset 为该状态记录时刻（分钟）；窗口 [-360, 2880]。
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
    SELECT p.patientunitstayid, p.patienthealthsystemstayid, p.uniquepid, p.hospitalid
    FROM eicu_crd.patient p
    INNER JOIN sepsis_dx s ON p.patientunitstayid = s.patientunitstayid
    WHERE p.unitvisitnumber = 1
      AND p.unitdischargeoffset > 2880
      AND p.hospitaldischargestatus IN ('Alive', 'Expired')
      AND (
             (p.age ~ '^[0-9]+$' AND CAST(p.age AS INT) >= 18)
          OR p.age = '> 89'
      )
)
SELECT
    r.patientunitstayid, c.patienthealthsystemstayid, c.uniquepid, c.hospitalid,
    r.respcareid,
    r.respcarestatusoffset, r.currenthistoryseqnum,
    ROUND((r.respcarestatusoffset / 60.0)::numeric, 4) AS status_hours_from_unit_admit,
    r.airwaytype,
    -- 未记录 vs 明确无人工气道：必须显式区分
    CASE
        WHEN r.airwaytype IS NULL OR r.airwaytype = '' THEN 'not_recorded'
        WHEN r.airwaytype = 'No Artificial Airway'      THEN 'no_artificial_airway'
        WHEN r.airwaytype IN ('Oral ETT','Nasal ETT')   THEN 'endotracheal_tube'
        WHEN r.airwaytype = 'Tracheostomy'              THEN 'tracheostomy'
        WHEN r.airwaytype = 'Cricothyrotomy'            THEN 'cricothyrotomy'
        WHEN r.airwaytype = 'Double-Lumen Tube'         THEN 'double_lumen_tube'
        ELSE 'other'
    END AS airway_state,
    CASE WHEN r.airwaytype IN ('Oral ETT','Nasal ETT','Tracheostomy',
                               'Cricothyrotomy','Double-Lumen Tube')
         THEN 1 ELSE 0 END AS artificial_airway_flag,
    r.airwaysize, r.airwayposition, r.cuffpressure,
    r.ventstartoffset, r.ventendoffset,
    r.priorventstartoffset, r.priorventendoffset,
    -- ===============================================================
    -- 【本轮核验发现的关键语义陷阱】
    -- respiratorycare 的 vent* / priorvent* 字段是"零填充"而非 NULL：
    --   全库 865,381 行中 ventendoffset 为 0 的有 865,224 行（99.98%），
    --   仅 157 行为正值；ventstartoffset 为 0 的有 685,205 行（79.2%）。
    -- 因此：
    --   ventendoffset = 0 的意思是"没有结束时间"，**不是"在入室时刻结束"**。
    --   ventstartoffset = 0 的意思是"没有开始时间"，**不是"在入室时刻开始"**。
    -- 若按 `IS NOT NULL` 当作有效区间，会得到大量 (0,0) 的假区间，
    -- 并把"仅记录了开始时刻"误判成"通气零时长"。本脚本显式区分状态，
    -- 不再输出笼统的 vent_duration_h。
    -- ===============================================================
    CASE
        WHEN r.ventstartoffset IS NULL THEN 'null'
        WHEN r.ventstartoffset = 0    THEN 'zero_placeholder'
        WHEN r.ventstartoffset < 0    THEN 'negative_started_before_unit_admit'
        ELSE 'positive_started_in_icu'
    END AS ventstart_state,
    CASE
        WHEN r.ventendoffset IS NULL THEN 'null'
        WHEN r.ventendoffset = 0    THEN 'zero_placeholder'
        ELSE 'positive'
    END AS ventend_state,
    -- 只有两端都是真实正值且前后有序，才构成可用区间（全库仅 157 行有正 end）
    CASE WHEN r.ventstartoffset > 0 AND r.ventendoffset > r.ventstartoffset
         THEN 1 ELSE 0 END AS vent_interval_valid,
    -- 明确信号：入室前已在通气中
    CASE WHEN r.ventstartoffset < 0 THEN 1 ELSE 0 END AS vent_before_unit_admit,
    -- 明确信号：入室后才开始（时间已知）
    CASE WHEN r.ventstartoffset > 0 THEN 1 ELSE 0 END AS vent_started_in_icu,
    -- 既往通气史（priorventstartoffset 非 0）
    CASE WHEN r.priorventstartoffset IS NOT NULL AND r.priorventstartoffset <> 0
         THEN 1 ELSE 0 END AS has_prior_vent_episode,
    -- 该行是否给出任何通气线索（用于构造"有证据"而非"有完整区间"）
    CASE WHEN (r.ventstartoffset IS NOT NULL AND r.ventstartoffset <> 0)
              OR (r.priorventstartoffset IS NOT NULL AND r.priorventstartoffset <> 0)
         THEN 1 ELSE 0 END AS any_vent_signal,
    CASE WHEN r.ventstartoffset > 0 AND r.ventendoffset > r.ventstartoffset
         THEN ROUND(((r.ventendoffset - r.ventstartoffset) / 60.0)::numeric, 4) END AS vent_duration_h,
    -- 通气时段与 48h 观察窗是否重叠：仅在区间有效时才可判定
    CASE WHEN r.ventstartoffset > 0 AND r.ventendoffset > r.ventstartoffset
              AND r.ventstartoffset <= 2880 AND r.ventendoffset >= -360
         THEN 1 ELSE 0 END AS vent_overlaps_window,
    r.cpaplimit, r.peeplimit, r.meanairwaypreslimit,
    r.lowironoxlimit, r.highironoxlimit
FROM eicu_crd.respiratorycare r
INNER JOIN cohort c ON r.patientunitstayid = c.patientunitstayid
WHERE r.respcarestatusoffset BETWEEN -360 AND 2880
ORDER BY r.patientunitstayid, r.respcarestatusoffset, r.respcareid;
