-- =====================================================================
-- mimic_12_vitals_timeseries.sql —— 精确观察时刻的生理变量（MAP / RR / 体温）
--
-- 【为什么必须单独提取，而不复用 sepsis_hourly.csv】
--   既有交付的 sepsis_hourly.csv 是"小时均值"宽表：
--     hr = FLOOR((charttime - intime)/3600)，每格是该小时内所有记录的 AVG。
--   本阶段底表要求"按 GCS 评分时点对齐，且禁止使用未来记录"。
--   若用小时均值，一个落在 [h, h+1) 内的评分时点会用到同小时内更晚的记录，
--   即小时内未来信息。因此这里保留**逐条精确 charttime**，由分析侧按
--   "同一时刻或既往"自行对齐，不做任何小时聚合。
--
-- 变量与来源严格沿用原研究 02_extract_hourly.sql 的定义，保证同源可比：
--   meanbp      ← mimiciv_derived.vitalsign.mbp
--   resp_rate   ← mimiciv_derived.vitalsign.resp_rate
--   temperature ← mimiciv_derived.vitalsign.temperature
-- （原研究另外的 glucose 不在本次要求内，未提取。）
--
-- 单位：来自上游派生表约定，非本脚本推断，见 unit_* 常量列：
--   mbp         = mmHg
--   resp_rate   = insp/min（每分钟呼吸次数）
--   temperature = °C（上游 vitalsign 已将华氏度换算为摄氏度）
--   ⚠️ 上游派生表不保留原始 valueuom，故无法逐条回溯原始单位字符串；
--      此为已知缺口，已记入 GAPS.md。
--
-- 缺失：保留 NULL，不填补、不插补、不删异常值。
-- 时间窗：[t0 - 6h, t0 + 48h]（向前 6h 供"既往对齐"使用）。
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
    v.charttime,
    ROUND((EXTRACT(EPOCH FROM (v.charttime - c.intime)) / 3600.0)::numeric, 4) AS hours_from_intime,
    v.mbp,
    'mmHg'::text       AS unit_mbp,
    v.resp_rate,
    'insp/min'::text   AS unit_resp_rate,
    v.temperature,
    'degC'::text       AS unit_temperature,
    -- 该条记录是否提供至少一个目标变量（避免全空行）
    CASE WHEN v.mbp IS NOT NULL OR v.resp_rate IS NOT NULL
              OR v.temperature IS NOT NULL THEN 1 ELSE 0 END AS has_target_variable,
    CASE WHEN v.charttime >= c.intime AND v.charttime < c.win_end
         THEN 1 ELSE 0 END AS in_window_0_48h,
    CASE WHEN v.charttime < c.intime THEN 1 ELSE 0 END AS before_intime
FROM mimiciv_derived.vitalsign v
INNER JOIN cohort c ON v.stay_id = c.stay_id
WHERE v.charttime >= c.win_start
  AND v.charttime <= c.win_end
  AND (v.mbp IS NOT NULL OR v.resp_rate IS NOT NULL OR v.temperature IS NOT NULL)
ORDER BY v.stay_id, v.charttime;
