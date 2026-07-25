from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from smarttutor_revision.benchmark import generate_benchmark
from smarttutor_revision.scenarios import uncertainty_scenarios
from smarttutor_revision.simulator import SmartTutorSimulator
from smarttutor_revision.search_space import CONFIG_COLUMNS
from smarttutor_revision.pareto import Normalization,nondominated_mask,normalize


def run(scenario_id:int):
    out=ROOT/'outputs_v4'/'scenarios'/f'scenario_{scenario_id:02d}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'selected.csv').exists(): print('skip',scenario_id); return
    scenario=uncertainty_scenarios(64)[scenario_id-1]
    cat=pd.read_csv(ROOT/'outputs_v4'/'training_candidate_catalog_v4.csv.gz')
    cfg=cat[CONFIG_COLUMNS].to_numpy(float)
    p95=[]; q=[]; tw=[]
    for block_id in range(1,9):
        sim=SmartTutorSimulator(generate_benchmark(2026072400+block_id),scenario)
        ev=sim.evaluate_batch(cfg)
        p95.append(ev.p95_latency_s.to_numpy()); q.append(ev.mean_quality.to_numpy()); tw.append(ev.mean_token_work.to_numpy())
    res=cat[CONFIG_COLUMNS].copy()
    res['training_p95']=np.mean(p95,axis=0); res['training_qw']=np.mean(q,axis=0); res['training_token_work']=np.mean(tw,axis=0)
    obj=np.column_stack([res.training_p95,1-res.training_qw]); mask=nondominated_mask(obj); front=res.loc[mask].copy().reset_index(drop=True)
    z=normalize(np.column_stack([front.training_p95,1-front.training_qw]),Normalization())
    front['_distance']=np.sqrt(np.sum(z*z,axis=1))
    sel=front.sort_values(['_distance','training_p95','training_token_work','context_tokens','top_k','chunk_tokens','overlap','rerank_depth'],kind='mergesort').iloc[0]
    row={'scenario_id':scenario_id,'scenario_name':scenario.name,'chunk_form':scenario.chunk_form,**{k:v for k,v in scenario.to_dict().items() if k not in {'name','chunk_form'}},**{c:sel[c] for c in CONFIG_COLUMNS},'training_p95':sel.training_p95,'training_qw':sel.training_qw,'training_token_work':sel.training_token_work,'catalog_size':len(res),'front_size':len(front),'selection_rule':'minimum normalized ideal distance on fixed 20,334-configuration adaptive catalog'}
    pd.DataFrame([row]).to_csv(out/'selected.csv',index=False)
    front.drop(columns=['_distance']).to_csv(out/'front.csv.gz',index=False,compression='gzip')
    print('done scenario',scenario_id,flush=True)

if __name__=='__main__': run(int(sys.argv[1]))
