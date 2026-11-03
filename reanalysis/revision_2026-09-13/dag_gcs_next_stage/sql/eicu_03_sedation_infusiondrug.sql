-- =====================================================================
-- eicu_03_sedation_infusiondrug.sql —— eICU 镇静/阿片/肌松：输注记录
--
-- 【eICU 与 MIMIC 的结构性差异，必须声明】
--   infusiondrug 每条记录代表"某药物在某一时刻的输注设定"（infusionoffset），
--   表中**没有结束时间字段**：输注何时停止只能由后续记录或护理记录推断。
--   因此"持续治疗区间"无法像 MIMIC 的 inputevents（有 starttime/endtime）
--   那样直接得到，只能给出"首次出现/最后一次出现/记录条数"。
--   本脚本原样导出全部记录与速率单位，不做区间重建。
--
-- drugname 把药物名与速率单位写在一起（如 'Propofol (mcg/kg/min)'、
--   'Fentanyl (ml/hr)'、'Cisatracurium IVF Infused (ml/hr)'），
--   因此：① 匹配用 ~*（大小写不敏感）；② 单位从名称尾部括号解析，
--   不另设单位列。已核验数据中存在拼写变体 'Dexmetetomidine (Precedex)'
--   （少一个 d），靠 'precedex' 分支兜住；'Diprivan' 为丙泊酚商品名，已纳入。
--
-- 时间：infusionoffset（分钟，距 ICU 入室）；窗口 [-360, 2880]。
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
        i.patientunitstayid, i.infusiondrugid, i.infusionoffset,
        i.drugname, i.drugrate, i.infusionrate, i.drugamount,
        i.volumeoffluid, i.patientweight,
        CASE
            WHEN i.drugname ~* 'propofol|diprivan'        THEN 'propofol'
            WHEN i.drugname ~* 'midazolam|versed'         THEN 'midazolam'
            WHEN i.drugname ~* 'dexmedetomidine|dexmetetomidine|precedex' THEN 'dexmedetomidine'
            WHEN i.drugname ~* 'lorazepam|ativan'         THEN 'lorazepam'
            WHEN i.drugname ~* 'pentobarbital'            THEN 'pentobarbital'
            WHEN i.drugname ~* 'ketamine'                 THEN 'ketamine'
            WHEN i.drugname ~* 'etomidate'                THEN 'etomidate'
            WHEN i.drugname ~* 'clonidine'                THEN 'clonidine'
            WHEN i.drugname ~* 'fentanyl'                 THEN 'fentanyl'
            WHEN i.drugname ~* 'hydromorphone|dilaudid'   THEN 'hydromorphone'
            WHEN i.drugname ~* 'morphine'                 THEN 'morphine'
            WHEN i.drugname ~* 'oxycodone'                THEN 'oxycodone'
            WHEN i.drugname ~* 'remifentanil'             THEN 'remifentanil'
            WHEN i.drugname ~* 'sufentanil'               THEN 'sufentanil'
            WHEN i.drugname ~* 'methadone'                THEN 'methadone'
            WHEN i.drugname ~* 'rocuronium'               THEN 'rocuronium'
            WHEN i.drugname ~* 'cisatracurium'            THEN 'cisatracurium'
            WHEN i.drugname ~* 'vecuronium'               THEN 'vecuronium'
            WHEN i.drugname ~* 'pancuronium'              THEN 'pancuronium'
            WHEN i.drugname ~* 'succinylcholine'          THEN 'succinylcholine'
            WHEN i.drugname ~* 'atracurium'               THEN 'atracurium'
        END AS drug_name,
        -- 从 drugname 尾部括号提取速率单位（数据自带，非本脚本推断）
        NULLIF(substring(i.drugname from '\(([^()]*)\)\s*$'), '') AS rate_unit_in_name
    FROM eicu_crd.infusiondrug i
    WHERE i.drugname ~* 'propofol|diprivan|midazolam|versed|dexmedetomidine|dexmetetomidine|precedex|lorazepam|ativan|pentobarbital|ketamine|etomidate|clonidine|fentanyl|hydromorphone|dilaudid|morphine|oxycodone|remifentanil|sufentanil|methadone|rocuronium|cisatracurium|vecuronium|pancuronium|succinylcholine|atracurium'
)
SELECT
    c.patientunitstayid, c.patienthealthsystemstayid, c.uniquepid, c.hospitalid,
    d.infusiondrugid, d.infusionoffset,
    ROUND((d.infusionoffset / 60.0)::numeric, 4) AS hours_from_unit_admit,
    d.drugname, d.drug_name, d.rate_unit_in_name,
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
    d.drugrate, d.infusionrate, d.drugamount, d.volumeoffluid, d.patientweight,
    CASE WHEN d.infusionoffset BETWEEN -360 AND 2880 THEN 1 ELSE 0 END AS in_window_0_48h,
    CASE WHEN d.infusionoffset < 0 THEN 1 ELSE 0 END AS started_before_unit_admit
FROM drug d
INNER JOIN cohort c ON d.patientunitstayid = c.patientunitstayid
WHERE d.infusionoffset BETWEEN -360 AND 2880
ORDER BY d.patientunitstayid, d.infusionoffset, d.drug_name;
