-- =====================================================================
-- eicu_04_sedation_medication.sql —— eICU 镇静/阿片/肌松：医嘱与给药记录
--
-- 【这是 eICU 里唯一能区分"医嘱 / 开始 / 停止 / 取消"的数据层】
--   drugorderoffset       → 医嘱开具时间
--   drugstartoffset       → 实际开始时间
--   drugstopoffset        → 停止时间
--   drugordercancelled    → 医嘱是否被取消（'True'/'False'）
--   dosage / routeadmin / frequency / prn / loadingdose / drugivadmixture
--   gtc (general treatment code)
-- 因此本脚本同时输出上述四类时间与取消标志，交由分析阶段判断
-- "开了但取消（未给）"与"开了并执行"的区别。注意本表是医嘱/给药流水，
-- 不保证等同于床旁已输注剂量（后者见 eicu_03）。
--
-- 重要边界：
--   - 各字段可为 NULL；NULL 表示未记录，不等于"未给药"。
--   - dosage 为文本（如 '5 mg'），单位与数值混写，未在 SQL 层拆分。
--   - 存在 PRN（按需）医嘱，prn='True' 时"是否有实际给药"无法仅凭本表确定。
--
-- 时间：上述 offset 均为分钟，距 ICU 入室；窗口 [-360, 2880]。
--       采用重叠逻辑：只要任一相关时间落入窗口即保留。
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
),
drug AS (
    SELECT
        m.patientunitstayid, m.medicationid,
        m.drugname, m.dosage, m.routeadmin, m.frequency,
        m.loadingdose, m.prn, m.drugivadmixture, m.gtc,
        m.drugorderoffset, m.drugstartoffset, m.drugstopoffset,
        m.drugordercancelled,
        CASE
            WHEN m.drugname ~* 'propofol|diprivan'        THEN 'propofol'
            WHEN m.drugname ~* 'midazolam|versed'         THEN 'midazolam'
            WHEN m.drugname ~* 'dexmedetomidine|dexmetetomidine|precedex' THEN 'dexmedetomidine'
            WHEN m.drugname ~* 'lorazepam|ativan'         THEN 'lorazepam'
            WHEN m.drugname ~* 'pentobarbital'            THEN 'pentobarbital'
            WHEN m.drugname ~* 'ketamine'                 THEN 'ketamine'
            WHEN m.drugname ~* 'etomidate'                THEN 'etomidate'
            WHEN m.drugname ~* 'clonidine'                THEN 'clonidine'
            WHEN m.drugname ~* 'fentanyl'                 THEN 'fentanyl'
            WHEN m.drugname ~* 'hydromorphone|dilaudid'   THEN 'hydromorphone'
            WHEN m.drugname ~* 'morphine'                 THEN 'morphine'
            WHEN m.drugname ~* 'oxycodone'                THEN 'oxycodone'
            WHEN m.drugname ~* 'remifentanil'             THEN 'remifentanil'
            WHEN m.drugname ~* 'sufentanil'               THEN 'sufentanil'
            WHEN m.drugname ~* 'methadone'                THEN 'methadone'
            WHEN m.drugname ~* 'rocuronium'               THEN 'rocuronium'
            WHEN m.drugname ~* 'cisatracurium'            THEN 'cisatracurium'
            WHEN m.drugname ~* 'vecuronium'               THEN 'vecuronium'
            WHEN m.drugname ~* 'pancuronium'              THEN 'pancuronium'
            WHEN m.drugname ~* 'succinylcholine'          THEN 'succinylcholine'
            WHEN m.drugname ~* 'atracurium'               THEN 'atracurium'
        END AS drug_name,
        -- 从 drugname 尾部括号提取单位（若数据自带）
        NULLIF(substring(m.drugname from '\(([^()]*)\)\s*$'), '') AS unit_in_name
    FROM eicu_crd.medication m
    WHERE m.drugname ~* 'propofol|diprivan|midazolam|versed|dexmedetomidine|dexmetetomidine|precedex|lorazepam|ativan|pentobarbital|ketamine|etomidate|clonidine|fentanyl|hydromorphone|dilaudid|morphine|oxycodone|remifentanil|sufentanil|methadone|rocuronium|cisatracurium|vecuronium|pancuronium|succinylcholine|atracurium'
)
SELECT
    c.patientunitstayid, c.patienthealthsystemstayid, c.uniquepid, c.hospitalid,
    d.medicationid,
    d.drugname, d.drug_name, d.unit_in_name,
    CASE
        WHEN d.drug_name IN ('propofol','midazolam','dexmedetomidine','lorazepam',
                             'pentobarbital','ketamine','etomidate','clonidine')
            THEN 'sedative_hypnotic'
        WHEN d.drug_name IN ('fentanyl','hydromorphone','morphine','oxycodone',
                             'remifentanil','sufentanil','methadone') THEN 'opioid'
        WHEN d.drug_name IN ('rocuronium','cisatracurium','vecuronium',
                             'pancuronium','succinylcholine','atracurium')
            THEN 'neuromuscular_blocker'
        ELSE 'unclassified'
    END AS drug_class,
    d.drugorderoffset, d.drugstartoffset, d.drugstopoffset,
    ROUND((d.drugorderoffset / 60.0)::numeric, 4)  AS order_hours_from_unit_admit,
    CASE WHEN d.drugstartoffset IS NOT NULL
         THEN ROUND((d.drugstartoffset / 60.0)::numeric, 4) END AS start_hours_from_unit_admit,
    CASE WHEN d.drugstopoffset IS NOT NULL
         THEN ROUND((d.drugstopoffset / 60.0)::numeric, 4) END AS stop_hours_from_unit_admit,
    CASE WHEN d.drugstartoffset IS NOT NULL AND d.drugstopoffset IS NOT NULL
         THEN ROUND(((d.drugstopoffset - d.drugstartoffset) / 60.0)::numeric, 4) END AS duration_h,
    d.drugordercancelled,
    d.dosage, d.routeadmin, d.frequency, d.prn,
    d.loadingdose, d.drugivadmixture, d.gtc,
    -- 医嘱→开始→停止 三者是否齐全，用于识别"只有医嘱无执行"的情形
    CASE
        WHEN d.drugstartoffset IS NULL AND d.drugstopoffset IS NULL THEN 'order_only_no_start_recorded'
        WHEN d.drugstartoffset IS NOT NULL AND d.drugstopoffset IS NULL THEN 'started_no_stop_recorded'
        WHEN d.drugstartoffset IS NOT NULL AND d.drugstopoffset IS NOT NULL THEN 'start_and_stop_recorded'
        ELSE 'other'
    END AS lifecycle_completeness,
    CASE WHEN d.drugordercancelled ~* '^(yes|true)$' THEN 1 ELSE 0 END AS order_cancelled_flag,
    CASE WHEN d.drugorderoffset BETWEEN -360 AND 2880 THEN 1 ELSE 0 END AS order_in_window,
    CASE WHEN d.drugstartoffset BETWEEN -360 AND 2880 THEN 1 ELSE 0 END AS start_in_window,
    CASE WHEN d.drugstopoffset  BETWEEN -360 AND 2880 THEN 1 ELSE 0 END AS stop_in_window
FROM drug d
INNER JOIN cohort c ON d.patientunitstayid = c.patientunitstayid
WHERE (d.drugorderoffset  BETWEEN -360 AND 2880
    OR d.drugstartoffset  BETWEEN -360 AND 2880
    OR d.drugstopoffset   BETWEEN -360 AND 2880
    -- 跨越窗口边界的持续医嘱：开始时已在窗内之前、结束时仍在窗内之后
    OR (d.drugstartoffset < -360  AND (d.drugstopoffset IS NULL OR d.drugstopoffset > 2880))
      )
ORDER BY d.patientunitstayid, d.drugorderoffset, d.drug_name;
