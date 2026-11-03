from pathlib import Path
import subprocess
H=Path(__file__).resolve().parent;P=H.parents[2].parent
cohort=(P/'04b_extract_sofa.sql').read_text().split('-- SOFA 表 48h')[0].rstrip().rstrip(',')
cs=['respiration','coagulation','liver','cardiovascular','cns','renal']
variants={'old':'s.hr<48','end_before_landmark':'s.hr<48 AND s.endtime<=ie.intime+INTERVAL \'48 hour\'','contained_first48':'s.starttime>=ie.intime AND s.endtime<=ie.intime+INTERVAL \'48 hour\''}
fields=[]
for variant,pred in variants.items():
 vals=['COALESCE(MAX(s.'+c+') FILTER(WHERE '+pred+'),0)' for c in cs]
 fields.append('+'.join(vals)+' AS '+variant)
 fields.append('COUNT(*) FILTER(WHERE '+pred+') AS n_'+variant)
q=cohort+', scored AS (SELECT c.stay_id,'+','.join(fields)+''' FROM cohort c JOIN mimiciv_icu.icustays ie USING(stay_id) LEFT JOIN mimiciv_derived.sofa s USING(stay_id) GROUP BY c.stay_id) SELECT COUNT(*) AS n_stays, COUNT(*) FILTER(WHERE old<>end_before_landmark) AS changed_removing_postlandmark_hours, COUNT(*) FILTER(WHERE old<>contained_first48) AS changed_using_contained_hours, COUNT(*) FILTER(WHERE n_end_before_landmark=0) AS no_eligible_original_rows_before_landmark, COUNT(*) FILTER(WHERE n_contained_first48=0) AS no_fully_contained_rows, MIN(old-end_before_landmark) AS min_original_minus_upper_bound,MAX(old-end_before_landmark) AS max_original_minus_upper_bound, AVG(old-end_before_landmark) AS mean_original_minus_upper_bound FROM scored;'''
sql="BEGIN READ ONLY;SET LOCAL statement_timeout='60s';\n"+q+'\nROLLBACK;'
f=H/'mimic_score_boundary_diagnostic.sql';f.write_text(sql)
p=subprocess.run(['psql','-X','-h','localhost','-U','postgres','-d','mimiciv31','--csv','-f',str(f)],capture_output=True,text=True,timeout=75)
(H/'mimic_score_boundary_diagnostic.txt').write_text(p.stdout+p.stderr);print(p.returncode,p.stdout,p.stderr)
