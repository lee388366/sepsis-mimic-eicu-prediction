"""Additional descriptive five-cutoff pattern summary; no group/test selection."""
from pathlib import Path
import json, numpy as np, pandas as pd
from analyze_trajectories import build_grid, plot_outputs, HERE, CUTS, sha

g,facts=build_grid()
e=g.pivot(index='stay_id',columns='cut_h',values='E').reindex(columns=CUTS)
m=g.pivot(index='stay_id',columns='cut_h',values='M').reindex(columns=CUTS)
valid=e.notna().all(axis=1)&m.notna().all(axis=1)
e=e.loc[valid];m=m.loc[valid]
de=np.diff(e.to_numpy(),axis=1);dm=np.diff(m.to_numpy(),axis=1)
unchanged=(de==0).all(axis=1)&(dm==0).all(axis=1)
up=(de>=0).all(axis=1)&(dm>=0).all(axis=1)&~unchanged
down=(de<=0).all(axis=1)&(dm<=0).all(axis=1)&~unchanged
pattern=np.select([unchanged,up,down],['all_unchanged','both_non_decreasing','both_non_increasing'],default='fluctuating_or_discordant')
rows=[{'pattern':p,'n':int((pattern==p).sum()),'denominator':len(e)} for p in ['all_unchanged','both_non_decreasing','both_non_increasing','fluctuating_or_discordant']]
pd.DataFrame(rows).to_csv(HERE/'all_five_trajectory_patterns.csv',index=False)
Ereverse=(de>0).any(axis=1)&(de<0).any(axis=1)
Mreverse=(dm>0).any(axis=1)&(dm<0).any(axis=1)
endup=(e[48]>=e[6])&(m[48]>=m[6])&((e[48]>e[6])|(m[48]>m[6]))
jointlower=(de<=0)&(dm<=0)&((de<0)|(dm<0))
out={'all_five_EM_n':len(e),'any_E_or_M_direction_reversal_n':int((Ereverse|Mreverse).sum()),'endpoint_joint_higher_n':int(endup.sum()),'endpoint_higher_with_intermediate_joint_lower_n':int((endup.to_numpy()&jointlower.any(axis=1)).sum()),'comment':'Supplementary descriptive trajectory pattern summary after primary run; endpoint change does not imply monotonic course.'}
(HERE/'supplementary_trajectory_facts.json').write_text(json.dumps(out,indent=2))
plot_outputs()
v=json.loads((HERE/'verification.json').read_text());v['analysis_code_sha256']=sha(HERE/'analyze_trajectories.py');v['supplement_code_sha256']=sha(Path(__file__));(HERE/'verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2))
print(pd.DataFrame(rows).to_string(index=False));print(json.dumps(out,indent=2))
