from pathlib import Path
import urllib.request,json,hashlib,numpy as np,subprocess
H=Path(__file__).resolve().parent;ROOT=H.parents[2];R=ROOT/'revision_2026-09-13/landmark_trainonly_reanalysis'
z=np.load(R/'data/trainonly_eicu.npz',allow_pickle=False)
print('frozen_schema',[(k,z[k].shape,str(z[k].dtype)) for k in z.files])
ids=z['stay_id'];assert len(ids)==6812 and len(set(ids))==6812
(H/'cohort_cte.sql').write_text('WITH cohort AS MATERIALIZED (SELECT patientunitstayid FROM (VALUES '+','.join('('+str(int(i))+')' for i in ids)+') v(patientunitstayid))\n')
files=[R/'data/trainonly_eicu.npz',R/'data/preds_A.npz',ROOT.parent/'eicu_sofa.csv']
manifest={'frozen_files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'n':len(ids),'deaths':int(z['y'].sum()),'sources':{}}
for name in ['pivoted-uo','pivoted-gcs','pivoted-o2','pivoted-infusion']:
 url=f'https://raw.githubusercontent.com/MIT-LCP/eicu-code/main/concepts/pivoted/{name}.sql'
 try:
  raw=urllib.request.urlopen(url,timeout=30).read();p=H/'sources'/(name+'.sql');p.write_bytes(raw);manifest['sources'][name]={'url':url,'sha256':hashlib.sha256(raw).hexdigest()}
 except Exception as e:manifest['sources'][name]={'error':str(e)}
(H/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
