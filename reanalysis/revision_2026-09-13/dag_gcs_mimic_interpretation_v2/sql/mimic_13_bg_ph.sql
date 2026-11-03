-- =====================================================================
-- mimic_13_bg_ph.sql —— 精确观察时刻的动脉血 pH
--
-- 来源：mimiciv_derived.bg.ph（与原研究 02_extract_hourly.sql 的 b.ph 同源）。
--
-- 单位：pH 为无量纲标度（无物理单位），upstream 派生表不加 uom；
--       unit_ph 列为显式标注，不代表原始记录中一定存在单位字段。
--
-- 关联方式：与原研究一致，按 hadm_id 关联（血气属住院级化验），
--   再经 icustays 映射到 stay_id。⚠️ 已知限制：同一次住院若有多次 ICU
--   入住，血气记录会归属到该住院的首次 ICU（队列只取 rn=1），
--   因此不能断言血气一定发生在该 stay 的物理时段内；此限制已记入 GAPS.md，
--   不做"时段内/外"的静默过滤，全部原样保留并给出相对 t0 的时间。
--
-- 时间窗：[t0 - 6h, t0 + 48h]。
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
           ie.intime + INTERVAL '48 hours' AS win_end,
           cnt.n_icu_stays
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
    b.charttime,
    ROUND((EXTRACT(EPOCH FROM (b.charttime - c.intime)) / 3600.0)::numeric, 4) AS hours_from_intime,
    b.ph,
    'pH_unit'::text AS unit_ph,
    -- 是否落在该 stay 的物理时段与本窗内（供分析侧判断，不用于过滤）
    CASE WHEN b.charttime >= c.intime AND b.charttime < c.win_end
         THEN 1 ELSE 0 END AS in_window_0_48h,
    CASE WHEN b.charttime < c.intime THEN 1 ELSE 0 END AS before_intime,
    c.n_icu_stays
FROM mimiciv_derived.bg b
INNER JOIN cohort c ON b.hadm_id = c.hadm_id
WHERE b.ph IS NOT NULL
  AND b.charttime >= c.win_start
  AND b.charttime <= c.win_end
ORDER BY c.stay_id, b.charttime;
