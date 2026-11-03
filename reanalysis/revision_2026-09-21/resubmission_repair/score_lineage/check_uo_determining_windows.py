from pathlib import Path
import subprocess,pandas as pd,json,io
H=Path(__file__).resolve().parent;s=(H/'mimic_sofa_strict48.sql').read_text()
cohort=s[s.index('WITH sepsis AS'):s.index(', co AS MATERIALIZED')]
uo=s[s.index(', uo_base AS MATERIALIZED'):s.index(', labs AS MATERIALIZED')]
q=cohort+uo+''', by_hour AS (
 SELECT stay_id,FLOOR(EXTRACT(EPOCH FROM(charttime-intime))/3600)::int AS hr,MAX(uo_24hr) AS hourly_uo
 FROM uo_eligible GROUP BY stay_id,hr
), worst AS (SELECT stay_id,MIN(hourly_uo) AS worst_uo FROM by_hour GROUP BY stay_id), determining AS (
 SELECT u.* FROM uo_eligible u JOIN by_hour h ON h.stay_id=u.stay_id
 AND h.hr=FLOOR(EXTRACT(EPOCH FROM(u.charttime-u.intime))/3600)::int AND h.hourly_uo=u.uo_24hr
 JOIN worst w ON w.stay_id=h.stay_id AND w.worst_uo=h.hourly_uo
)
SELECT stay_id,MIN(uo_24hr) AS determining_uo,
MIN(interval_count) AS determining_min_intervals,MAX(interval_count) AS determining_max_intervals,
MIN(max_gap_h) AS determining_min_maxgap,MAX(max_gap_h) AS determining_max_maxgap,
COUNT(*) AS determining_endpoints,
COUNT(*) FILTER(WHERE interval_count=1) AS determining_single_interval_endpoints
FROM determining GROUP BY stay_id ORDER BY stay_id;'''
(H/'uo_determining_windows.sql').write_text(q)
wrapped="BEGIN READ ONLY; SET LOCAL statement_timeout='180s'; SET LOCAL enable_nestloop=off; SET LOCAL jit=off;\nCOPY ("+q.rstrip().rstrip(';')+") TO STDOUT WITH (FORMAT CSV,HEADER TRUE);\nROLLBACK;"
p=subprocess.run(['psql','-X','-q','-h','localhost','-U','postgres','-d','mimiciv31','-v','ON_ERROR_STOP=1','-f','-'],input=wrapped,text=True,capture_output=True,timeout=190)
(H/'uo_determining_stderr.txt').write_text(p.stderr)
if p.returncode:raise RuntimeError(p.stderr)
f=H/'data/uo_determining_windows.csv';f.write_text(p.stdout)
d=pd.read_csv(io.StringIO(p.stdout)).set_index('stay_id');scores=pd.read_csv(H/'data/mimic_sofa_strict48.csv').set_index('stay_id');j=scores.join(d)
qa={'score_uo_equal_diagnostic':bool((j.loc[j.eligible_uo_endpoints>0,'uo_24hr_min'].sub(j.loc[j.eligible_uo_endpoints>0,'determining_uo']).abs()<1e-7).all()),'eligible_stays':len(d),'low_uo_below500_stays':int((j.determining_uo<500).sum()),'high_renal_uo_only_n':int(j.high_renal_uo_only.sum())}
for tag,mask in [('all_eligible',j.eligible_uo_endpoints>0),('high_renal_uo_only',j.high_renal_uo_only==1)]:
 z=j[mask];qa[tag]={'n':len(z),'determining_at_least_one_single_interval_n':int((z.determining_min_intervals==1).sum()),'determining_only_single_intervals_n':int((z.determining_max_intervals==1).sum()),'determining_min_interval_count_summary':z.determining_min_intervals.describe().to_dict(),'determining_maxgap_summary':z.determining_max_maxgap.describe().to_dict(),'maxgap_ge6h_n':int((z.determining_max_maxgap>=6).sum()),'maxgap_ge12h_n':int((z.determining_max_maxgap>=12).sum()),'maxgap_ge22h_n':int((z.determining_max_maxgap>=22).sum())}
qa['interpretation']='These are estimated collection intervals, not proof of continuous monitoring. A single interval uses two measured timepoints. Threshold bins are diagnostics only, not new scoring exclusions.'
(H/'uo_determining_qa.json').write_text(json.dumps(qa,indent=2));print(json.dumps(qa,indent=2));assert qa['score_uo_equal_diagnostic']
