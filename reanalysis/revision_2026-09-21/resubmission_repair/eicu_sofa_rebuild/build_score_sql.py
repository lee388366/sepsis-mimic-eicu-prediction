from pathlib import Path
import re,hashlib,json
H=Path(__file__).resolve().parent
# Source-derived urinary volume labels; explicitly exclude count/unmeasured and non-urine mixtures.
s=(H/'sources/pivoted-uo.sql').read_text();paths=set(re.findall(r"'(flowsheet\|Flowsheet Cell Labels\|I&O\|Output \(ml\)\|(?:[^']|'')*)'",s))
paths={x.replace("''", "'") for x in paths if '%' not in x and not re.search(r'count|unmeasured|mixed|pacu',x,re.I)}
for label in ['Nephrostomy Left Output','Nephrostomy Right Output','Nephrostomy','Right Nephrostomy','Output (mL)-Nephrostomy Right 10 Fr.','Output (ml)-Nephrostomy Right','Output (ml)-Nephrostomy Left','Output (mL)-Nephrostomy Left 10 Fr.','SN Urine Output(ml)']:
 paths.add('flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|'+label)
(H/'urine_volume_whitelist.json').write_text(json.dumps(sorted(paths),indent=2))
whitelist=','.join("('"+x.replace("'","''")+"')" for x in sorted(paths))
cte=(H/'cohort_cte.sql').read_text().strip().replace('WITH cohort AS MATERIALIZED','WITH frozen_ids AS MATERIALIZED')
body=(H/'score_body.sql').read_text().replace('__UO_WHITELIST__',whitelist)
(H/'eicu_sofa_strict48.sql').write_text('-- Isolated SELECT only. Cohort fixed from trainonly_eicu.npz; see REPAIR_PLAN.md.\n'+cte+',\n'+body)
print('SQL_SHA256',hashlib.sha256((H/'eicu_sofa_strict48.sql').read_bytes()).hexdigest())
