from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from smarttutor_revision.benchmark import generate_benchmark
from smarttutor_revision.scenarios import SimulatorScenario
from smarttutor_revision.simulator import SmartTutorSimulator
from smarttutor_revision.search_space import SearchSpace
from smarttutor_revision.pareto import Normalization
from smarttutor_revision.algorithms import shared_lhs_initial,run_method

METHODS=("D-NSGA-II","D-SPEA2","Sobol","Random")

def run(block_id:int, seed_id:int):
    out=ROOT/'outputs_v4'/'runs'/f'block_{block_id:02d}'/f'seed_{seed_id:02d}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.csv').exists() and (out/'fronts.csv.gz').exists() and (out/'candidates.csv.gz').exists():
        print(f'skip block={block_id} seed={seed_id}',flush=True); return
    bseed=2026072400+block_id
    benchmark=generate_benchmark(bseed)
    simulator=SmartTutorSimulator(benchmark,SimulatorScenario())
    space=SearchSpace(); norm=Normalization()
    # Matched design/seed blocks: all methods receive the same initial design and
    # seed label within each benchmark x optimizer-seed cell.
    design_seed=810000+block_id*100+seed_id
    algorithm_seed=910000+block_id*100+seed_id
    init,init_x,init_proposals=shared_lhs_initial(space,seed=design_seed,size=60)
    results=[]
    for method in METHODS:
        r=run_method(method,simulator=simulator,space=space,norm=norm,
                     block_id=block_id,benchmark_seed=bseed,algorithm_seed=algorithm_seed,
                     initial_configs=init,initial_x=init_x,budget=3000,reference_front=None)
        r.summary['optimizer_seed_id']=seed_id
        r.summary['initial_design_seed']=design_seed
        r.summary['initial_design_proposals']=init_proposals
        r.candidates['optimizer_seed_id']=seed_id
        r.front['optimizer_seed_id']=seed_id
        r.convergence['optimizer_seed_id']=seed_id
        results.append(r)
    pd.DataFrame([r.summary for r in results]).to_csv(out/'summary.csv',index=False)
    pd.concat([r.candidates for r in results],ignore_index=True).to_csv(out/'candidates.csv.gz',index=False,compression='gzip')
    pd.concat([r.front.assign(method=r.method,block_id=block_id,benchmark_seed=bseed) for r in results],ignore_index=True).to_csv(out/'fronts.csv.gz',index=False,compression='gzip')
    pd.concat([r.convergence for r in results],ignore_index=True).to_csv(out/'convergence.csv',index=False)
    print(f'done block={block_id} seed={seed_id}',flush=True)

if __name__=='__main__':
    if len(sys.argv)!=3: raise SystemExit('usage: run_seed_task_v4.py BLOCK_ID SEED_ID')
    run(int(sys.argv[1]),int(sys.argv[2]))
