from pathlib import Path
import subprocess
H=Path(__file__).resolve().parent;s=(H/'eicu_sofa_strict48.sql').read_text();base=s.rsplit('\nSELECT c.patientunitstayid',1)[0]
queries={
'uo_duplicate_diagnostic':'''SELECT (SELECT COUNT(*) FROM uo_raw) AS raw_n,(SELECT COUNT(*) FROM uo_dedup) AS dedup_n,(SELECT SUM(n) FROM uo_multichannel) AS generic_plus_channel_offsets,(SELECT COUNT(*) FROM uo_multichannel) AS generic_plus_channel_stays,(SELECT SUM(negative_n) FROM uo_duplicates) AS negative_rows,(SELECT MAX(volume) FROM uo_times) AS largest_time_volume''',
'epi_mg_unit_diagnostic':'''SELECT COUNT(*) AS n,COUNT(DISTINCT patientunitstayid) AS stays,MIN(rate) AS min_rate,MAX(rate) AS max_rate,percentile_cont(.5) WITHIN GROUP(ORDER BY rate) AS median_rate,MIN(amount_mg) AS min_drugamount,MAX(amount_mg) AS max_drugamount,percentile_cont(.5) WITHIN GROUP(ORDER BY amount_mg) AS median_drugamount,MIN(fluid_ml) AS min_fluid_ml,MAX(fluid_ml) AS max_fluid_ml FROM infusion_raw WHERE drug='epi' AND lower(drugname) LIKE '%(mg/kg/min)%' '''
}
for name,q in queries.items():
 sql="BEGIN READ ONLY;SET LOCAL statement_timeout='240s';SET LOCAL jit=off;\n"+base+'\n'+q+';ROLLBACK;'
 (H/(name+'.sql')).write_text(sql)
 p=subprocess.run(['psql','-X','-q','-h','localhost','-U','postgres','-d','eicu','--csv','-v','ON_ERROR_STOP=1','-f','-'],input=sql,capture_output=True,text=True,timeout=250)
 (H/(name+'.csv')).write_text(p.stdout);(H/(name+'.stderr')).write_text(p.stderr);print(name,p.returncode,p.stdout,flush=True)
