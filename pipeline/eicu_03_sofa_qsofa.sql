-- =====================================================================
-- eicu_03_sofa_qsofa.sql
-- eICU SOFA + qSOFA + 结局 —— 外验评分（对齐 MIMIC 项目口径）
--
-- 目的：计算外验所需的 SOFA/qSOFA 评分 + 结局标签，完全对齐
--       MIMIC 项目的 04b_extract_sofa.sql（SOFA）和 10_qsofa_baseline.py（qSOFA）。
--
-- 时间窗：eICU 用分钟 offset，[0, 2880) = ICU 前 48h（与特征同窗）
--
-- SOFA（Vincent 1996，复用 sepsis_predict 论文的算法）：
--   6 组分各 0-4 分，48h 窗内取最差值：
--     respiration: PaO2/FiO2 比（或 SpO2/FiO2 代理），用 mechvent 判断
--     coagulation: 血小板（'platelets x 1000'，单位 k/μL）
--     liver:       total bilirubin
--     cardiovascular: MAP <70 或升压药（dopamine/norepinephrine/dobutamine）
--     cns:         GCS
--     renal:       creatinine + 尿量（任意 24h 滚动窗最小累计，200/500 mL 阈值）
--
-- 2026-08-27 修复：①心血管多巴胺分支 dopa>0 AND <=5（原 <=5 使无药者恒得 2 分）；
--   ②pao2_min 改 MIN（原用 MAX 低估呼吸 SOFA）；③尿量改 24h 滚动窗最小累计
--   （原 48h 总和稀释阈值），且只取体积类尿液记录（MIT-LCP pivoted_uo 白名单 +
--   泌尿系引流；剔除 Mixed Urine/Stool 与计数行）；④FiO2 裁剪 21-100（原保留 9010 异常值）。
--
-- qSOFA（Singer 2016）：RR>=22 / SBP<=100 / GCS<15，6h 和 24h 两窗
--
-- 队列：eicu_01_cohort.sql 的 6,883 例
--
-- 输出：patientunitstayid + 各组分 + sofa_max_48h + qsofa6/qsofa24 + label
-- =====================================================================

-- 队列（与 eicu_01_cohort.sql 完全一致）
WITH cohort AS (
    SELECT
        p.patientunitstayid,
        CASE WHEN p.hospitaldischargestatus = 'Expired' THEN 1
             WHEN p.hospitaldischargestatus = 'Alive' THEN 0
             ELSE NULL END AS label
    FROM eicu_crd.patient p
    INNER JOIN (
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
    ) s ON p.patientunitstayid = s.patientunitstayid
    WHERE p.unitvisitnumber = 1
      AND p.unitdischargeoffset > 2880
      AND p.hospitaldischargestatus IS NOT NULL
      AND (p.age ~ '^[0-9]+$' AND CAST(p.age AS INT) >= 18 OR p.age = '> 89')
),

-- ============ 组分 1：心血管（MAP + 升压药）============
-- MAP：vitalperiodic.systemicmean（有创）优先，vitalaperiodic.noninvasivemean（无创）兜底
map_data AS (
    SELECT
        c.patientunitstayid,
        MIN(vp.systemicmean) AS map_min
    FROM cohort c
    LEFT JOIN eicu_crd.vitalperiodic vp
        ON c.patientunitstayid = vp.patientunitstayid
        AND vp.observationoffset >= 0 AND vp.observationoffset < 2880
        AND vp.systemicmean IS NOT NULL
    GROUP BY c.patientunitstayid
),
map_nibp AS (
    SELECT
        c.patientunitstayid,
        MIN(va.noninvasivemean) AS map_min
    FROM cohort c
    LEFT JOIN eicu_crd.vitalaperiodic va
        ON c.patientunitstayid = va.patientunitstayid
        AND va.observationoffset >= 0 AND va.observationoffset < 2880
        AND va.noninvasivemean IS NOT NULL
    GROUP BY c.patientunitstayid
),
-- 升压药（infusiondrug）
vaso AS (
    SELECT patientunitstayid,
        MAX(CASE WHEN LOWER(drugname) LIKE '%dopamine%' THEN
              CASE WHEN LOWER(drugname) LIKE '%(ml/hr)%' THEN ROUND(CAST(drugrate AS numeric)/3, 3)
                   WHEN LOWER(drugname) LIKE '%(mcg/kg/min)%' THEN CAST(drugrate AS numeric)
                   ELSE NULL END ELSE NULL END) AS dopa,
        MAX(CASE WHEN LOWER(drugname) LIKE '%norepinephrine%' THEN
              CASE WHEN LOWER(drugname) LIKE '%(ml/hr)%' THEN ROUND(CAST(drugrate AS numeric)/300, 3)
                   WHEN LOWER(drugname) LIKE '%(mcg/min)%' THEN
                     ROUND(CAST(drugrate AS numeric)/NULLIF(NULLIF(patientweight, ''), '0')::numeric, 3)
                   WHEN LOWER(drugname) LIKE '%(mcg/kg/min)%' THEN CAST(drugrate AS numeric)
                   ELSE NULL END ELSE NULL END) AS norepi,
        MAX(CASE WHEN LOWER(drugname) LIKE '%dobutamine%' THEN 1 ELSE NULL END) AS dobu
    FROM eicu_crd.infusiondrug
    WHERE infusionoffset >= -120 AND infusionoffset < 2880
      AND drugrate ~ '^[-]?[0-9]+(\.[0-9]+)?$'
      AND drugrate <> '' AND drugrate <> '.'
    GROUP BY patientunitstayid
),
sofa_cv AS (
    SELECT
        c.patientunitstayid,
        (CASE WHEN COALESCE(v.dopa,0) > 15 OR COALESCE(v.norepi,0) > 0.1 THEN 4
              WHEN COALESCE(v.dopa,0) > 5 OR (COALESCE(v.norepi,0) > 0 AND COALESCE(v.norepi,0) <= 0.1) THEN 3
              WHEN (COALESCE(v.dopa,0) > 0 AND COALESCE(v.dopa,0) <= 5) OR COALESCE(v.dobu,0) > 0 THEN 2
              WHEN COALESCE(m1.map_min, m2.map_min, 999) < 70 THEN 1
              ELSE 0 END) AS sofacv
    FROM cohort c
    LEFT JOIN map_data m1 ON c.patientunitstayid = m1.patientunitstayid
    LEFT JOIN map_nibp m2 ON c.patientunitstayid = m2.patientunitstayid
    LEFT JOIN vaso v ON c.patientunitstayid = v.patientunitstayid
),

-- ============ 组分 2：呼吸（PaO2/FiO2 或 SpO2/FiO2）============
-- FiO2（respiratorycharting，>20 排除升/分钟，<=100 裁剪异常值）
fio2_data AS (
    SELECT patientunitstayid,
        MAX(CASE WHEN LOWER(respchartvaluelabel) LIKE '%fio2%'
                      AND respchartvalue ~ '^[0-9]+(\.[0-9]+)?$'
                      AND respchartvalue::numeric > 20
                      AND respchartvalue::numeric <= 100
                 THEN respchartvalue::numeric ELSE NULL END) AS fio2_max
    FROM eicu_crd.respiratorycharting
    WHERE respchartoffset >= 0 AND respchartoffset < 2880
    GROUP BY patientunitstayid
),
-- paO2 / SpO2
pao2_data AS (
    SELECT patientunitstayid,
        MIN(CASE WHEN LOWER(labname) = 'pao2' AND labresult IS NOT NULL
                 THEN labresult ELSE NULL END) AS pao2_min
    FROM eicu_crd.lab
    WHERE labresultoffset >= 0 AND labresultoffset < 2880
    GROUP BY patientunitstayid
),
spo2_data AS (
    SELECT patientunitstayid,
        MIN(sao2) AS sao2_min
    FROM eicu_crd.vitalperiodic
    WHERE observationoffset >= 0 AND observationoffset < 2880
      AND sao2 IS NOT NULL
    GROUP BY patientunitstayid
),
-- 机械通气（respiratorycharting 提示 ventilator/vent/bipap/cpap 等）
mechvent AS (
    SELECT patientunitstayid, 1 AS mechvent
    FROM eicu_crd.respiratorycharting
    WHERE respchartoffset >= 0 AND respchartoffset < 2880
      AND (LOWER(respchartvalue) LIKE '%ventilator%'
           OR LOWER(respchartvalue) LIKE '%vent%'
           OR LOWER(respchartvalue) LIKE '%bipap%'
           OR LOWER(respchartvalue) LIKE '%cpap%'
           OR LOWER(respchartvalue) LIKE '%drager%'
           OR LOWER(respchartvalue) LIKE '%servo%'
           OR LOWER(respchartvalue) LIKE '%peep%'
           OR LOWER(respchartvalue) LIKE '%840%')
    GROUP BY patientunitstayid
),
resp_comp AS (
    SELECT
        c.patientunitstayid,
        f.fio2_max,
        p.pao2_min,
        s.sao2_min,
        mv.mechvent,
        -- 计算 pf 比（无 FiO2 时默认 21 = 吸空气）
        (COALESCE(p.pao2_min, 100) / COALESCE(NULLIF(f.fio2_max, 0), 21)) AS pf,
        (COALESCE(s.sao2_min, 100) / COALESCE(NULLIF(f.fio2_max, 0), 21)) AS sf
    FROM cohort c
    LEFT JOIN fio2_data f ON c.patientunitstayid = f.patientunitstayid
    LEFT JOIN pao2_data p ON c.patientunitstayid = p.patientunitstayid
    LEFT JOIN spo2_data s ON c.patientunitstayid = s.patientunitstayid
    LEFT JOIN mechvent mv ON c.patientunitstayid = mv.patientunitstayid
),
sofa_resp AS (
    SELECT patientunitstayid,
        (CASE WHEN pf < 1 OR sf < 0.67 THEN 4
              WHEN pf BETWEEN 1 AND 2 OR sf BETWEEN 0.67 AND 1.41 THEN 3
              WHEN pf BETWEEN 2 AND 3 OR sf BETWEEN 1.42 AND 2.2 THEN 2
              WHEN pf BETWEEN 3 AND 4 OR sf BETWEEN 2.21 AND 3.01 THEN 1
              ELSE 0 END) AS sofaresp
    FROM resp_comp
),

-- ============ 组分 3：凝血（血小板）============
coag_data AS (
    SELECT patientunitstayid,
        MIN(labresult) AS plt_min
    FROM eicu_crd.lab
    WHERE LOWER(labname) = 'platelets x 1000'
      AND labresult IS NOT NULL
      AND labresultoffset >= 0 AND labresultoffset < 2880
    GROUP BY patientunitstayid
),
sofa_coag AS (
    SELECT
        c.patientunitstayid,
        (CASE WHEN cd.plt_min < 20 THEN 4
              WHEN cd.plt_min < 50 THEN 3
              WHEN cd.plt_min < 100 THEN 2
              WHEN cd.plt_min < 150 THEN 1
              ELSE 0 END) AS sofacoa
    FROM cohort c
    LEFT JOIN coag_data cd ON c.patientunitstayid = cd.patientunitstayid
),

-- ============ 组分 4：肝（总胆红素）============
bili_data AS (
    SELECT patientunitstayid,
        MAX(labresult) AS bili_max
    FROM eicu_crd.lab
    WHERE LOWER(labname) = 'total bilirubin'
      AND labresult IS NOT NULL
      AND labresultoffset >= 0 AND labresultoffset < 2880
    GROUP BY patientunitstayid
),
sofa_liver AS (
    SELECT
        c.patientunitstayid,
        (CASE WHEN bd.bili_max > 12 THEN 4
              WHEN bd.bili_max > 6 THEN 3
              WHEN bd.bili_max > 2 THEN 2
              WHEN bd.bili_max > 1.2 THEN 1
              ELSE 0 END) AS sofaliv
    FROM cohort c
    LEFT JOIN bili_data bd ON c.patientunitstayid = bd.patientunitstayid
),

-- ============ 组分 5：CNS（GCS）============
gcs_data AS (
    SELECT patientunitstayid,
        MIN(CASE WHEN nursingchartcelltypevalname = 'GCS Total'
                      AND nursingchartvalue ~ '^[0-9]+(\.[0-9]+)?$'
                 THEN nursingchartvalue::numeric ELSE NULL END) AS gcs_min
    FROM eicu_crd.nursecharting
    WHERE nursingchartcelltypevalname = 'GCS Total'
      AND nursingchartoffset >= 0 AND nursingchartoffset < 2880
    GROUP BY patientunitstayid
),
sofa_cns AS (
    SELECT
        c.patientunitstayid,
        (CASE WHEN gd.gcs_min = 15 THEN 0
              WHEN gd.gcs_min >= 13 THEN 1
              WHEN gd.gcs_min >= 10 THEN 2
              WHEN gd.gcs_min >= 6 THEN 3
              WHEN gd.gcs_min >= 3 THEN 4
              ELSE 0 END) AS sofacns
    FROM cohort c
    LEFT JOIN gcs_data gd ON c.patientunitstayid = gd.patientunitstayid
),

-- ============ 组分 6：肾（肌酐 + 尿量）============
creat_data AS (
    SELECT patientunitstayid,
        MAX(labresult) AS creat_max
    FROM eicu_crd.lab
    WHERE LOWER(labname) = 'creatinine'
      AND labresult IS NOT NULL
      AND labresultoffset >= 0 AND labresultoffset < 2880
    GROUP BY patientunitstayid
),
-- 尿量：先按小时分桶，再用窗口函数取任意 24h 滚动窗内的最小累计尿量
--（Vincent 1996 的 200/500 mL 为 24h 阈值；对齐 MIMIC sofa.uo_24hr 口径。
--  48h 总和不适用 —— 会把 24h 判据稀释成几乎不触发）
-- 行筛选：只取体积类尿液记录（对齐 MIT-LCP eicu-code pivoted_uo 白名单，
--   并纳入泌尿系引流 nephrostomy/urostomy/ureteral stent/ileal conduit）；
--   按保守口径剔除 Mixed Urine/Stool 及计数/累计类行（Urine Count/Occurrence、
--   Number of Incontinent Voids、TOTAL OUT、24HR TOTAL、variance 等）。
uo_hourly AS (
    SELECT patientunitstayid,
        FLOOR(intakeoutputoffset / 60)::int AS hr,
        SUM(outputtotal) AS uo_hr
    FROM eicu_crd.intakeoutput
    WHERE intakeoutputoffset >= 0 AND intakeoutputoffset < 2880
      AND outputtotal IS NOT NULL
      AND cellpath LIKE 'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|%'
      AND (
              cellpath ILIKE '%urine%'
           OR cellpath ILIKE '%foley%'
           OR cellpath ILIKE '%straight cath%'
           OR cellpath ILIKE '%straight catheter%'
           OR cellpath ILIKE '%condom%'
           OR cellpath ILIKE '%indwelling%'
           OR cellpath ILIKE '%catheter%'
           OR cellpath ILIKE '%void%'
           OR cellpath ILIKE '%nephrostom%'
           OR cellpath ILIKE '%ureter%'
           OR cellpath ILIKE '%urostom%'
           OR cellpath ILIKE '%ileal conduit%'
           OR cellpath ILIKE '%suprapubic%'
      )
      AND cellpath NOT ILIKE '%urine count%'
      AND cellpath NOT ILIKE '%urine occurrence%'
      AND cellpath NOT ILIKE '%number of incontinent voids%'
      AND cellpath NOT ILIKE '%mixed urine/stool%'
      AND cellpath NOT ILIKE '%total out%'
      AND cellpath NOT ILIKE '%total output%'
      AND cellpath NOT ILIKE '%8 hr total%'
      AND cellpath NOT ILIKE '%variance%'
    GROUP BY patientunitstayid, FLOOR(intakeoutputoffset / 60)::int
),
uo_rolling AS (
    SELECT patientunitstayid, hr,
        SUM(uo_hr) OVER (
            PARTITION BY patientunitstayid
            ORDER BY hr
            RANGE BETWEEN 23 PRECEDING AND CURRENT ROW
        ) AS uo_24h
    FROM uo_hourly
),
uo_data AS (
    SELECT patientunitstayid,
        MIN(uo_24h) AS uo_min_24h
    FROM uo_rolling
    GROUP BY patientunitstayid
),
sofa_renal AS (
    SELECT
        c.patientunitstayid,
        (CASE WHEN COALESCE(uo.uo_min_24h, 99999) < 200 OR COALESCE(cd.creat_max, 0) > 5 THEN 4
              WHEN COALESCE(uo.uo_min_24h, 99999) < 500 OR COALESCE(cd.creat_max, 0) > 3.5 THEN 3
              WHEN COALESCE(cd.creat_max, 0) BETWEEN 2 AND 3.5 THEN 2
              WHEN COALESCE(cd.creat_max, 0) BETWEEN 1.2 AND 2 THEN 1
              ELSE 0 END) AS sofarenal
    FROM cohort c
    LEFT JOIN creat_data cd ON c.patientunitstayid = cd.patientunitstayid
    LEFT JOIN uo_data uo ON c.patientunitstayid = uo.patientunitstayid
),

-- ============ qSOFA（6h/24h 窗，any-time）============
qsofa_vitals AS (
    SELECT
        nc.patientunitstayid,
        nc.nursingchartoffset,
        MAX(CASE WHEN nc.nursingchartcelltypevallabel = 'Respiratory Rate'
                      AND nc.nursingchartcelltypevalname = 'Respiratory Rate'
                      AND nc.nursingchartvalue ~ '^[0-9]+(\.[0-9]+)?$'
                 THEN nc.nursingchartvalue::numeric ELSE NULL END) AS rr,
        MAX(CASE WHEN nc.nursingchartcelltypevallabel = 'Non-Invasive BP'
                      AND nc.nursingchartcelltypevalname = 'Non-Invasive BP Systolic'
                      AND nc.nursingchartvalue ~ '^[0-9]+(\.[0-9]+)?$'
                 THEN nc.nursingchartvalue::numeric ELSE NULL END) AS sbp,
        MAX(CASE WHEN nc.nursingchartcelltypevalname = 'GCS Total'
                      AND nc.nursingchartvalue ~ '^[0-9]+(\.[0-9]+)?$'
                 THEN nc.nursingchartvalue::numeric ELSE NULL END) AS gcs
    FROM eicu_crd.nursecharting nc
    INNER JOIN cohort c ON nc.patientunitstayid = c.patientunitstayid
    WHERE nc.nursingchartoffset >= 0 AND nc.nursingchartoffset < 2880
    GROUP BY nc.patientunitstayid, nc.nursingchartoffset
),
qsofa_calc AS (
    SELECT
        patientunitstayid,
        MAX(CASE WHEN nursingchartoffset < 360 AND rr >= 22 THEN 1 ELSE 0 END)
          + MAX(CASE WHEN nursingchartoffset < 360 AND sbp <= 100 THEN 1 ELSE 0 END)
          + MAX(CASE WHEN nursingchartoffset < 360 AND gcs < 15 THEN 1 ELSE 0 END) AS qsofa6,
        MAX(CASE WHEN nursingchartoffset < 1440 AND rr >= 22 THEN 1 ELSE 0 END)
          + MAX(CASE WHEN nursingchartoffset < 1440 AND sbp <= 100 THEN 1 ELSE 0 END)
          + MAX(CASE WHEN nursingchartoffset < 1440 AND gcs < 15 THEN 1 ELSE 0 END) AS qsofa24
    FROM qsofa_vitals
    GROUP BY patientunitstayid
)

-- ============ 汇总输出 ============
SELECT
    c.patientunitstayid,
    COALESCE(rc.sofaresp, 0) AS respiration,
    COALESCE(cg.sofacoa, 0)  AS coagulation,
    COALESCE(cl.sofaliv, 0)  AS liver,
    COALESCE(cv.sofacv, 0)   AS cardiovascular,
    COALESCE(cn.sofacns, 0)  AS cns,
    COALESCE(rn.sofarenal, 0) AS renal,
    COALESCE(rc.sofaresp,0) + COALESCE(cg.sofacoa,0) + COALESCE(cl.sofaliv,0)
      + COALESCE(cv.sofacv,0) + COALESCE(cn.sofacns,0) + COALESCE(rn.sofarenal,0) AS sofa_max_48h,
    COALESCE(q.qsofa6, 0)  AS qsofa6,
    COALESCE(q.qsofa24, 0) AS qsofa24,
    c.label
FROM cohort c
LEFT JOIN sofa_resp rc ON c.patientunitstayid = rc.patientunitstayid
LEFT JOIN sofa_coag cg ON c.patientunitstayid = cg.patientunitstayid
LEFT JOIN sofa_liver cl ON c.patientunitstayid = cl.patientunitstayid
LEFT JOIN sofa_cv cv   ON c.patientunitstayid = cv.patientunitstayid
LEFT JOIN sofa_cns cn  ON c.patientunitstayid = cn.patientunitstayid
LEFT JOIN sofa_renal rn ON c.patientunitstayid = rn.patientunitstayid
LEFT JOIN qsofa_calc q ON c.patientunitstayid = q.patientunitstayid
ORDER BY c.patientunitstayid;
