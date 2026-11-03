from pathlib import Path
import subprocess,hashlib,json,datetime,time,sys
H=Path(__file__).resolve().parent;p=H/'eicu_sofa_strict48.sql';actual=hashlib.sha256(p.read_bytes()).hexdigest()
if len(sys.argv)!=2 or sys.argv[1]!=actual:raise RuntimeError('Pass the independently reviewed current SQL SHA256; no implicit approval')
outdir=H/'data';outdir.mkdir(exist_ok=True);out=outdir/'eicu_sofa_strict48.csv';partial=out.with_suffix('.csv.partial')
if out.exists():raise RuntimeError('Refusing overwrite of completed score CSV')
transaction="BEGIN READ ONLY;\nSET LOCAL statement_timeout='30min';SET LOCAL work_mem='128MB';SET LOCAL jit=off;\nCOPY (\n"+p.read_text().strip().rstrip(';')+"\n) TO STDOUT WITH (FORMAT CSV,HEADER TRUE);\nROLLBACK;"
(H/'execution.sql').write_text(transaction);start=datetime.datetime.now().isoformat();t=time.monotonic();print('START READ ONLY extraction',start,flush=True)
with partial.open('w') as f:
 proc=subprocess.run(['psql','-X','-q','-h','localhost','-U','postgres','-d','eicu','-v','ON_ERROR_STOP=1','-f','-'],input=transaction,text=True,stdout=f,stderr=subprocess.PIPE,timeout=1860)
(H/'execution_stderr.txt').write_text(proc.stderr)
man={'start':start,'elapsed_seconds':time.monotonic()-t,'returncode':proc.returncode,'query_sha256':actual,'reviewed_sha256_argument':sys.argv[1],'transaction_read_only':True,'trained_models':False,'frozen_cohort':True,'output_path':str(out)}
if proc.returncode==0:
 partial.rename(out);man['output_sha256']=hashlib.sha256(out.read_bytes()).hexdigest();man['output_bytes']=out.stat().st_size
(H/'execution_manifest.json').write_text(json.dumps(man,indent=2));print(json.dumps(man,indent=2),flush=True)
if proc.returncode:raise RuntimeError('Read-only extraction failed; see isolated stderr')
