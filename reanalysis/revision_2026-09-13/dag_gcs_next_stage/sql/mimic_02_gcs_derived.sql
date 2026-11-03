-- =====================================================================
-- mimic_02_gcs_derived.sql —— MIMIC-IV 派生 GCS 层（mimiciv_derived.gcs）
--
-- 目的：与 mimic_01（床旁原始层）对照，识别派生规则对 GCS 做了什么：
--         - 6 小时内前一记录的补值（carry-forward）
--         - 语言分项为 ETT 时的编码方式
--         - 是否存在总分被置为 15 的情形
--         - gcs_unable 的含义与边界
--
-- 【已核验事实】对全库 mimiciv_derived.gcs 做 gcs_unable × gcs_verbal 交叉表：
--     gcs_unable=1 且 gcs_verbal=0 : 784,911 行
--     gcs_unable=0 且 gcs_verbal=0 :   3,526 行
--   而 chartevents 中 value='No Response-ETT' 的原始事件恰为 784,911 条。
--   即：gcs_unable=1 与"曾记录过 ETT 语言不可评估"精确对应；
--   且派生层把 ETT 的 verbal 编码为 0（而非 1）。
--   → 小时表中 verbal=0 不可读作"语言无反应"，须结合 unable 标记解释。
--   另有 3,526 行 unable=0 但 verbal=0，需在分析中单独甄别（见 QA 报告）。
--
-- 【重要缺口】本地未保存 mimiciv_derived 的安装版 SQL。因此本脚本只导出
--   派生结果本身，不对"当时运行的是哪个版本 gcs.sql"作任何断言。
--   上游现行 gcs.sql（MIT-LCP）仅作背景参照，不能当作本地执行证明。
--
-- 时间窗：与 mimic_01 一致，[t0 - 6h, t0 + 48h]，便于逐事件对齐两层。
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
cohort AS (SELECT * FROM ranked WHERE rn = 1)
SELECT
    g.subject_id, c.hadm_id, g.stay_id,
    g.charttime,
    ROUND((EXTRACT(EPOCH FROM (g.charttime - c.intime)) / 3600.0)::numeric, 4) AS hours_from_intime,
    g.gcs            AS gcs_total,
    g.gcs_eyes,
    g.gcs_verbal,
    g.gcs_motor,
    g.gcs_unable,
    -- 总分与三分项之和的差；两者不等说明不是普通加法（派生规则/异步取值所致）
    CASE WHEN g.gcs IS NOT NULL AND g.gcs_eyes IS NOT NULL
              AND g.gcs_verbal IS NOT NULL AND g.gcs_motor IS NOT NULL
         THEN g.gcs - (g.gcs_eyes + g.gcs_verbal + g.gcs_motor)
    END AS total_minus_sum,
    CASE WHEN g.gcs = 15 AND g.gcs_eyes IS NOT NULL AND g.gcs_verbal IS NOT NULL
              AND g.gcs_motor IS NOT NULL
              AND (g.gcs_eyes + g.gcs_verbal + g.gcs_motor) < 15
         THEN 1 ELSE 0 END AS total15_with_sum_below15,
    CASE WHEN g.gcs_verbal = 0 THEN 1 ELSE 0 END AS verbal_zero_flag,
    CASE WHEN g.gcs_unable = 1 THEN 1 ELSE 0 END AS unable_flag
FROM mimiciv_derived.gcs g
INNER JOIN cohort c ON g.stay_id = c.stay_id
WHERE g.charttime >= c.intime - INTERVAL '6 hours'
  AND g.charttime <= c.intime + INTERVAL '48 hours'
ORDER BY g.stay_id, g.charttime;
