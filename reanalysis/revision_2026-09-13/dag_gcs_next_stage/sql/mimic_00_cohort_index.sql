-- =====================================================================
-- mimic_00_cohort_index.sql —— MIMIC-IV 完整队列索引（不排除任何记录）
--
-- 口径来源：项目根目录 01_sepsis_cohort.sql（作者已冻结的队列定义）
-- 唯一新增条件：rn = 1 —— 即"同一次住院取首次 ICU 入住"
--   依据（已核验）：01_sepsis_cohort.sql 注释写明
--     "如需「每次住院只取首次 ICU」，在最终 SELECT 外再套一层 WHERE rn = 1"
--   而正式队列 = 24,894 条 = distinct(hadm_id) = rn=1 行数，三者相等，
--   且与项目 sepsis_labels.csv 的 stay_id 集合 md5 完全一致：
--     af316e5c70a1a26b00c448e5e20970aa
--   因此 rn=1 是正式口径，不是本脚本自行附加的筛选。
--
-- 本表保留全部 24,894 条入住记录，即使某类事件完全缺失也不剔除，
-- 保证后续覆盖率统计的分母恒为完整队列。
--
-- 时间基准：t0 = icustays.intime（ICU 入室）；观察窗 [t0, t0+48h)
-- 数据库：mimiciv31（只读，未建表、未写入任何数据）
-- 执行：见 ../run_extract.sh（本文件只含一条 SELECT，不含 \copy）
-- =====================================================================
WITH sepsis AS (
    SELECT stay_id, subject_id, suspected_infection_time AS t_inf
    FROM mimiciv_derived.sepsis3
    WHERE sepsis3 IS TRUE AND suspected_infection_time IS NOT NULL
),
icu_stay_count AS (
    SELECT hadm_id, COUNT(*) AS n_icu_stays FROM mimiciv_icu.icustays GROUP BY hadm_id
),
base AS (
    SELECT
        ie.subject_id, ie.hadm_id, ie.stay_id,
        ie.intime, ie.outtime, ie.first_careunit, ie.last_careunit,
        s.t_inf,
        p.gender, p.anchor_age AS age, p.dod,
        adm.admittime, adm.dischtime, adm.deathtime,
        adm.admission_type, adm.insurance,
        adm.hospital_expire_flag,
        cnt.n_icu_stays,
        EXTRACT(EPOCH FROM (ie.outtime - ie.intime)) / 3600.0 AS los_h,
        EXTRACT(EPOCH FROM (s.t_inf - ie.intime)) / 3600.0  AS t_inf_minus_intime_h
    FROM mimiciv_icu.icustays ie
    INNER JOIN sepsis s                      ON ie.stay_id   = s.stay_id
    INNER JOIN mimiciv_hosp.patients p       ON ie.subject_id = p.subject_id
    INNER JOIN mimiciv_hosp.admissions adm   ON ie.hadm_id   = adm.hadm_id
    INNER JOIN icu_stay_count cnt            ON ie.hadm_id   = cnt.hadm_id
    WHERE p.anchor_age >= 18
      AND ie.outtime >= ie.intime + INTERVAL '48 hours'
      AND adm.hospital_expire_flag IS NOT NULL
      AND cnt.n_icu_stays <= 2
),
ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime ASC, stay_id ASC) AS rn
    FROM base
)
SELECT
    subject_id, hadm_id, stay_id,
    intime, outtime,
    intime + INTERVAL '48 hours' AS window_end,
    los_h, n_icu_stays, rn,
    t_inf, t_inf_minus_intime_h,
    CASE WHEN t_inf <= intime + INTERVAL '48 hours' THEN 1 ELSE 0 END AS infection_within_window,
    age, gender, admission_type, insurance,
    first_careunit, last_careunit,
    admittime, dischtime, deathtime, dod,
    hospital_expire_flag AS label,
    EXTRACT(EPOCH FROM (dischtime - intime)) / 3600.0 AS dischtime_minus_intime_h
FROM ranked
WHERE rn = 1;
