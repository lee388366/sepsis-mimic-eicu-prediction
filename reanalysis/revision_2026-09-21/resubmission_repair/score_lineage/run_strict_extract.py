from pathlib import Path
import subprocess,hashlib,json,datetime,time
H=Path(__file__).resolve().parent
sqlpath=H/'mimic_sofa_strict48.sql';sql=sqlpath.read_text().strip().rstrip(';')
outdir=H/'data';outdir.mkdir(exist_ok=True)
out=outdir/'mimic_sofa_strict48.csv';partial=out.with_suffix('.csv.partial')
if out.exists():raise RuntimeError('Refusing to overwrite an existing completed corrected score artifact')
transaction="BEGIN READ ONLY;\nSET LOCAL statement_timeout='20min';\nSET LOCAL work_mem='128MB';\nSET LOCAL enable_nestloop=off;\nSET LOCAL jit=off;\nCOPY (\n"+sql+"\n) TO STDOUT WITH (FORMAT CSV, HEADER TRUE);\nROLLBACK;\n"
(H/'strict48_execution.sql').write_text(transaction)
start=datetime.datetime.now().isoformat();t=time.monotonic();print('START strict score extraction; read-only; no model training',start,flush=True)
with partial.open('w') as f:
 p=subprocess.run(['psql','-X','-q','-h','localhost','-U','postgres','-d','mimiciv31','-v','ON_ERROR_STOP=1','-f','-'],input=transaction,text=True,stdout=f,stderr=subprocess.PIPE,timeout=1260)
(H/'strict48_execution_stderr.txt').write_text(p.stderr)
manifest={'start':start,'elapsed_s':time.monotonic()-t,'returncode':p.returncode,'query_sha256':hashlib.sha256(sqlpath.read_bytes()).hexdigest(),'transaction_read_only':True,'raw_database_mutation':False,'trained_models':False,'output_path':str(out)}
if p.returncode==0:
 partial.rename(out);manifest['output_sha256']=hashlib.sha256(out.read_bytes()).hexdigest();manifest['output_bytes']=out.stat().st_size
(H/'strict48_execution_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)
if p.returncode:raise RuntimeError('Extraction failed; inspect stderr locally; partial retained')
