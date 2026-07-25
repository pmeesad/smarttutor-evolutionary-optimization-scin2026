from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from smarttutor_revision.benchmark import generate_benchmark
from smarttutor_revision.scenarios import uncertainty_scenarios
from smarttutor_revision.simulator import SmartTutorSimulator
from smarttutor_revision.search_space import CONFIG_COLUMNS,SearchSpace

OUT=ROOT/'outputs_v4'
sel=pd.concat([pd.read_csv(OUT/'scenarios'/f'scenario_{i:02d}'/'selected.csv') for i in range(1,65)],ignore_index=True)
sel.to_csv(OUT/'scenario_selected_profiles_v4.csv',index=False)
# Evaluate each scenario-selected profile on the four held-out benchmarks under the same scenario.
scenarios=uncertainty_scenarios(64); rows=[]
for i,(scenario,(_,r)) in enumerate(zip(scenarios,sel.iterrows()),1):
    cfg=r[CONFIG_COLUMNS].to_numpy(float)
    for b in range(9,13):
        ev=SmartTutorSimulator(generate_benchmark(2026072400+b),scenario).evaluate_one(cfg)
        rows.append({'scenario_id':i,'scenario_name':scenario.name,'block_id':b,**{c:r[c] for c in CONFIG_COLUMNS},'p95_latency_s':ev.p95_latency_s,'mean_quality':ev.mean_quality,'mean_token_work':ev.mean_token_work})
held=pd.DataFrame(rows); held.to_csv(OUT/'scenario_selected_heldout_v4.csv',index=False)
summary=held.groupby(['scenario_id','scenario_name'],as_index=False).agg(heldout_p95_mean=('p95_latency_s','mean'),heldout_quality_mean=('mean_quality','mean'),heldout_token_work_mean=('mean_token_work','mean'))
summary=sel.merge(summary,on=['scenario_id','scenario_name'])
summary.to_csv(OUT/'scenario_level_summary_v4.csv',index=False)
# Grid-normalized medoid of the 64 selected parameter vectors.
space=SearchSpace(); x=space.encode_midpoint(sel[CONFIG_COLUMNS].to_numpy(float))
d=((x[:,None,:]-x[None,:,:])**2).sum(axis=2)**0.5
mid=int(np.argmin(d.sum(axis=1))); med=sel.iloc[mid]
pd.DataFrame([{'medoid_scenario_id':int(med.scenario_id),**{c:med[c] for c in CONFIG_COLUMNS},'definition':'observed selected profile minimizing total Euclidean distance in midpoint-encoded grid coordinates'}]).to_csv(OUT/'scenario_medoid_profile_v4.csv',index=False)
medrows=[]; cfg=med[CONFIG_COLUMNS].to_numpy(float)
for i,scenario in enumerate(scenarios,1):
    for b in range(9,13):
        ev=SmartTutorSimulator(generate_benchmark(2026072400+b),scenario).evaluate_one(cfg)
        medrows.append({'scenario_id':i,'scenario_name':scenario.name,'block_id':b,**{c:med[c] for c in CONFIG_COLUMNS},'p95_latency_s':ev.p95_latency_s,'mean_quality':ev.mean_quality,'mean_token_work':ev.mean_token_work})
pd.DataFrame(medrows).to_csv(OUT/'scenario_medoid_heldout_256_v4.csv',index=False)
# Parameter stability.
st=[]
for c in CONFIG_COLUMNS:
    vc=sel[c].value_counts().sort_values(ascending=False)
    st.append({'control':c,'minimum':sel[c].min(),'maximum':sel[c].max(),'median':sel[c].median(),'distinct_values':sel[c].nunique(),'most_common_value':vc.index[0],'most_common_count':int(vc.iloc[0])})
pd.DataFrame(st).to_csv(OUT/'scenario_parameter_stability_v4.csv',index=False)
# Scenario-level marginal rank associations: n=64, not 256.
inputs=['weight_recall','weight_chunk','weight_overlap','weight_rerank','weight_context','distraction_weight','ret_token','generation_tokens_per_s','queue_coefficient']
assoc=[]
for a in inputs:
    for y in ['heldout_p95_mean','heldout_quality_mean','heldout_token_work_mean']:
        rho,p=spearmanr(summary[a],summary[y])
        assoc.append({'input':a,'outcome':y,'n_scenarios':64,'spearman_rho':rho,'p_value_uncorrected':p,'interpretation':'marginal rank association, not causal dominance'})
pd.DataFrame(assoc).to_csv(OUT/'scenario_spearman_v4.csv',index=False)
print('scenario aggregation complete; medoid', {c:med[c] for c in CONFIG_COLUMNS})
