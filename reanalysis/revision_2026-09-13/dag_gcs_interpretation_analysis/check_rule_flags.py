from pathlib import Path
import pandas as pd,json
p=Path(__file__).resolve().parent
s=p.parent/'dag_gcs_mimic_interpretation/base/mimic_gcs_timebase.csv'
d=pd.read_csv(s,usecols=lambda c:c.startswith('gcs_'))
orig=d[['gcs_eyes_origin','gcs_motor_origin','gcs_verbal_origin']]
plain=d.gcs_total_rule.eq('plain_sum');prev=d.gcs_total_rule.eq('prev_ett_recompute')
expected_default=(plain & orig.eq('missing').any(axis=1))|(prev & ~orig.eq('measured_current').all(axis=1))
expected_carry=plain&orig.apply(lambda x:x.str.startswith('carried')).any(axis=1)
bad=d.gcs_total_recorded.eq(15)&d.gcs_total_minus_component_sum.gt(0)
f={'total15_component_sum_lower':int(bad.sum()),'by_rule':d.loc[bad].gcs_total_rule.value_counts().to_dict(),'default_flag_mismatch':int(d.gcs_total_used_default_for_missing.ne(expected_default.astype(int)).sum()),'carry_flag_mismatch':int(d.gcs_total_used_carryforward.ne(expected_carry.astype(int)).sum()),'prev_recompute_total15_sum_lower':int((bad&prev).sum()),'those_expected_default':int((bad&prev&expected_default).sum())}
(p/'rule_flag_verification.json').write_text(json.dumps(f,indent=2));print(json.dumps(f,indent=2))
