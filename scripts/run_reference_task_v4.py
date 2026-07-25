from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from smarttutor_revision.benchmark import generate_benchmark
from smarttutor_revision.scenarios import SimulatorScenario
from smarttutor_revision.simulator import SmartTutorSimulator
from smarttutor_revision.search_space import SearchSpace,CONFIG_COLUMNS
from smarttutor_revision.pareto import nondominated_mask


def thin_front(frame:pd.DataFrame,n=500)->pd.DataFrame:
    f=frame.sort_values('p95_latency_s').reset_index(drop=True)
    if len(f)>n:
        idx=np.unique(np.rint(np.linspace(0,len(f)-1,n)).astype(int))
        f=f.iloc[idx].reset_index(drop=True)
    return f

def run(block_id:int):
    out=ROOT/'outputs_v4'/'references'/f'block_{block_id:02d}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'independent_sample_250k.csv.gz').exists() and (out/'independent_front_250k.csv').exists():
        print(f'skip reference block={block_id}',flush=True); return
    bseed=2026072400+block_id
    sim=SmartTutorSimulator(generate_benchmark(bseed),SimulatorScenario())
    space=SearchSpace()
    sampler=qmc.Sobol(d=5,scramble=True,seed=121000+block_id,bits=30,optimization=None)
    points=sampler.random_base2(m=19)  # 524,288 LMS+shift-scrambled points
    decoded=space.decode(points)
    frame=space.dataframe(decoded)
    frame=frame.drop_duplicates(CONFIG_COLUMNS,keep='first').head(250000).reset_index(drop=True)
    ev=sim.evaluate_batch(frame[CONFIG_COLUMNS].to_numpy())
    sample=pd.concat([frame,ev],axis=1)
    sample['block_id']=block_id; sample['benchmark_seed']=bseed
    sample.to_csv(out/'independent_sample_250k.csv.gz',index=False,compression='gzip')
    for n in (100000,250000):
        sub=sample.iloc[:n].copy()
        obj=sub[['p95_latency_s','quality_loss']].to_numpy()
        full=sub.loc[nondominated_mask(obj)].sort_values('p95_latency_s').reset_index(drop=True)
        full.to_csv(out/f'independent_front_{n//1000}k_full.csv',index=False)
        thin_front(full,500).to_csv(out/f'independent_front_{n//1000}k.csv',index=False)
    counts=(generate_benchmark(bseed).groupby('query_class').size().rename('n').reset_index())
    counts['block_id']=block_id; counts['benchmark_seed']=bseed
    counts.to_csv(out/'class_counts.csv',index=False)
    print(f'done reference block={block_id}, unique={len(sample)}',flush=True)

if __name__=='__main__':
    if len(sys.argv)!=2: raise SystemExit('usage: run_reference_task_v4.py BLOCK_ID')
    run(int(sys.argv[1]))
