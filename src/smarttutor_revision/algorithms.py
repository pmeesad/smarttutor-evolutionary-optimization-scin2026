from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from scipy.stats import qmc

from .search_space import SearchSpace, CONFIG_COLUMNS
from .simulator import SmartTutorSimulator
from .pareto import (
    Normalization,
    fast_nondominated_sort,
    crowding_distance,
    rank_and_crowding,
    nondominated_mask,
    hypervolume_2d,
    ideal_point_index,
    igd_plus,
    additive_epsilon,
)


@dataclass
class SearchResult:
    method: str
    block_id: int
    benchmark_seed: int
    algorithm_seed: int
    summary: dict
    candidates: pd.DataFrame
    front: pd.DataFrame
    convergence: pd.DataFrame


def shared_lhs_initial(space: SearchSpace, seed: int, size: int = 60) -> tuple[np.ndarray,np.ndarray,int]:
    sampler=qmc.LatinHypercube(d=space.dimensions,scramble=True,seed=seed)
    selected=[]; selected_x=[]; seen=set(); proposals=0
    while len(selected)<size:
        x=sampler.random(max(2*size,128))
        decoded=space.decode(x)
        for row,xx in zip(decoded,x):
            proposals+=1; key=space.key(row)
            if key in seen: continue
            seen.add(key); selected.append(row); selected_x.append(xx)
            if len(selected)>=size: break
    return np.asarray(selected,float),np.asarray(selected_x,float),proposals


def _evaluate_records(
    configs: np.ndarray,
    normalized: np.ndarray,
    simulator: SmartTutorSimulator,
    method: str,
    block_id: int,
    benchmark_seed: int,
    algorithm_seed: int,
    generation: int,
    sequence_start: int,
) -> pd.DataFrame:
    ev=simulator.evaluate_batch(configs)
    frame=pd.DataFrame(configs,columns=CONFIG_COLUMNS)
    frame=frame.astype({"top_k":int,"chunk_tokens":int,"overlap":float,"rerank_depth":int,"context_tokens":int})
    frame=pd.concat([frame,ev],axis=1)
    frame["method"]=method; frame["block_id"]=block_id
    frame["benchmark_seed"]=benchmark_seed; frame["algorithm_seed"]=algorithm_seed
    frame["generation"]=generation
    frame["evaluation_sequence"]=np.arange(sequence_start,sequence_start+len(frame))
    for j in range(5): frame[f"x{j+1}"]=normalized[:,j]
    return frame


def _summary(
    method: str,
    block_id: int,
    benchmark_seed: int,
    algorithm_seed: int,
    candidates: pd.DataFrame,
    norm: Normalization,
    raw_proposals: int,
    reference_front: np.ndarray|None,
    checkpoints: tuple[int,...],
) -> SearchResult:
    objectives=candidates[["p95_latency_s","quality_loss"]].to_numpy()
    mask=nondominated_mask(objectives)
    front=candidates.loc[mask].copy().sort_values("p95_latency_s").reset_index(drop=True)
    fobj=front[["p95_latency_s","quality_loss"]].to_numpy()
    idx=ideal_point_index(fobj,norm)
    compromise=front.iloc[idx]
    summary={
        "method":method,"block_id":block_id,"benchmark_seed":benchmark_seed,"algorithm_seed":algorithm_seed,
        "unique_evaluations":int(len(candidates)),"raw_proposals":int(raw_proposals),
        "unique_ratio":float(len(candidates)/raw_proposals),"duplicate_rate":float(1-len(candidates)/raw_proposals),
        "front_size":int(len(front)),"hypervolume":hypervolume_2d(fobj,norm),
        "compromise_p95_s":float(compromise.p95_latency_s),"compromise_quality":float(compromise.mean_quality),
        "compromise_top_k":int(compromise.top_k),"compromise_chunk":int(compromise.chunk_tokens),
        "compromise_overlap":float(compromise.overlap),"compromise_rerank":int(compromise.rerank_depth),
        "compromise_context":int(compromise.context_tokens),
    }
    if reference_front is not None:
        summary["igd_plus"]=igd_plus(fobj,reference_front,norm)
        summary["epsilon_additive"]=additive_epsilon(fobj,reference_front,norm)
    conv=[]
    for cp in checkpoints:
        sub=candidates[candidates.evaluation_sequence<=cp]
        obj=sub[["p95_latency_s","quality_loss"]].to_numpy()
        m=nondominated_mask(obj); fo=obj[m]
        row={"method":method,"block_id":block_id,"benchmark_seed":benchmark_seed,"evaluations":cp,"hypervolume":hypervolume_2d(fo,norm)}
        if reference_front is not None:
            row["igd_plus"]=igd_plus(fo,reference_front,norm)
        conv.append(row)
    return SearchResult(method,block_id,benchmark_seed,algorithm_seed,summary,candidates,front,pd.DataFrame(conv))


def _sbx(rng,a,b,prob=0.9,eta=15.0):
    if rng.random()>prob: return a.copy(),b.copy()
    u=rng.random(len(a)); beta=np.where(u<=0.5,(2*u)**(1/(eta+1)),(1/(2*(1-u)))**(1/(eta+1)))
    c1=0.5*((1+beta)*a+(1-beta)*b); c2=0.5*((1-beta)*a+(1+beta)*b)
    return np.clip(c1,0,np.nextafter(1.0,0.0)),np.clip(c2,0,np.nextafter(1.0,0.0))


def _mutate(rng,x,prob=0.2,eta=20.0):
    y=x.copy()
    for j in range(len(y)):
        if rng.random()>prob: continue
        u=rng.random()
        if u<0.5:
            delta=(2*u+(1-2*u)*(1-y[j])**(eta+1))**(1/(eta+1))-1
        else:
            delta=1-(2*(1-u)+2*(u-0.5)*y[j]**(eta+1))**(1/(eta+1))
        y[j]=np.clip(y[j]+delta,0,np.nextafter(1.0,0.0))
    return y


def _tournament_indices(rng, rank, crowd):
    i,j=rng.integers(0,len(rank),2)
    if rank[i]<rank[j] or (rank[i]==rank[j] and crowd[i]>crowd[j]): return i
    return j


def _generate_unique_offspring(
    *, rng, pop, obj, space, global_seen, target, method, spea_fitness=None, nsga_rank=None, nsga_crowd=None
):
    children=[]; child_x=[]; raw=0; attempts=0
    while len(children)<target and attempts<target*30:
        attempts+=1
        if method in {"NSGA-II", "D-NSGA-II"}:
            i=_tournament_indices(rng,nsga_rank,nsga_crowd); j=_tournament_indices(rng,nsga_rank,nsga_crowd)
            p1=pop[i]; p2=pop[j]
        else:
            # SPEA2: lower fitness is better.
            i,j=rng.integers(0,len(pop),2); p1=pop[i] if spea_fitness[i]<=spea_fitness[j] else pop[j]
            i,j=rng.integers(0,len(pop),2); p2=pop[i] if spea_fitness[i]<=spea_fitness[j] else pop[j]
        for child in _sbx(rng,p1,p2):
            raw+=1
            child=_mutate(rng,child)
            decoded=space.decode(child)[0]; key=space.key(decoded)
            if key in global_seen: continue
            global_seen.add(key); children.append(decoded); child_x.append(child)
            if len(children)>=target: break
    while len(children)<target:
        raw+=1
        x=rng.random(5); decoded=space.decode(x)[0]; key=space.key(decoded)
        if key in global_seen: continue
        global_seen.add(key); children.append(decoded); child_x.append(x)
    return np.asarray(children,float),np.asarray(child_x,float),raw


def run_nsga2(
    simulator,space,norm,block_id,benchmark_seed,algorithm_seed,initial_configs,initial_x,
    budget=3000,population_size=60,reference_front=None,checkpoints=(60,300,600,1200,1800,2400,3000),
):
    rng=np.random.default_rng(algorithm_seed)
    seen=set(space.keys(initial_configs)); raw_proposals=len(initial_configs)
    records=[_evaluate_records(initial_configs,initial_x,simulator,"D-NSGA-II",block_id,benchmark_seed,algorithm_seed,0,1)]
    pop=initial_x.copy(); pop_eval=records[0][["p95_latency_s","quality_loss"]].copy()
    completed=len(initial_configs); generation=0
    while completed<budget:
        generation+=1; target=min(population_size,budget-completed)
        nsga_rank,nsga_crowd,_=rank_and_crowding(pop_eval.to_numpy())
        offspring,offspring_x,raw=_generate_unique_offspring(rng=rng,pop=pop,obj=pop_eval.to_numpy(),space=space,global_seen=seen,target=target,method="NSGA-II",nsga_rank=nsga_rank,nsga_crowd=nsga_crowd)
        raw_proposals+=raw
        rec=_evaluate_records(offspring,offspring_x,simulator,"D-NSGA-II",block_id,benchmark_seed,algorithm_seed,generation,completed+1)
        records.append(rec); completed+=target
        combined_x=np.vstack([pop,offspring_x])
        combined_eval=pd.concat([pop_eval,rec[["p95_latency_s","quality_loss"]]],ignore_index=True)
        selected=[]
        for f in fast_nondominated_sort(combined_eval.to_numpy()):
            if len(selected)+len(f)<=population_size:
                selected.extend(f.tolist())
            else:
                d=crowding_distance(combined_eval.to_numpy(),f); order=np.argsort(-d,kind="mergesort")
                selected.extend(f[order[:population_size-len(selected)]].tolist()); break
        selected=np.asarray(selected,int); pop=combined_x[selected]; pop_eval=combined_eval.iloc[selected].reset_index(drop=True)
    candidates=pd.concat(records,ignore_index=True)
    return _summary("D-NSGA-II",block_id,benchmark_seed,algorithm_seed,candidates,norm,raw_proposals,reference_front,checkpoints)


def _spea_fitness(objectives: np.ndarray):
    v=np.asarray(objectives,float); n=len(v)
    strength=np.zeros(n,float)
    dominates_lists=[]
    for i in range(n):
        dom=np.all(v[i]<=v,axis=1)&np.any(v[i]<v,axis=1); dom[i]=False
        dominates_lists.append(dom); strength[i]=np.sum(dom)
    raw=np.zeros(n,float)
    for i in range(n):
        dominators=np.all(v<=v[i],axis=1)&np.any(v<v[i],axis=1); dominators[i]=False
        raw[i]=np.sum(strength[dominators])
    ranges=np.ptp(v,axis=0); ranges[ranges==0]=1
    z=(v-v.min(axis=0))/ranges
    dist=np.linalg.norm(z[:,None,:]-z[None,:,:],axis=2); np.fill_diagonal(dist,np.inf)
    kth=max(1,int(math.sqrt(n))); sigma=np.partition(dist,kth-1,axis=1)[:,kth-1]
    density=1/(sigma+2)
    return raw+density,raw,dist


def _spea_select(objectives: np.ndarray,size:int):
    fit,raw,dist=_spea_fitness(objectives)
    selected=list(np.flatnonzero(raw<1.0))
    if len(selected)<size:
        remaining=[i for i in np.argsort(fit,kind="mergesort") if i not in set(selected)]
        selected.extend(remaining[:size-len(selected)])
    elif len(selected)>size:
        selected=list(selected)
        while len(selected)>size:
            sub=dist[np.ix_(selected,selected)].copy(); np.fill_diagonal(sub,np.inf)
            sorted_d=np.sort(sub,axis=1)
            # Remove lexicographically smallest distance vector (most crowded).
            remove_local=min(range(len(selected)),key=lambda i: tuple(sorted_d[i]))
            selected.pop(remove_local)
    selected=np.asarray(selected,int)
    selected_fit=fit[selected]
    return selected,selected_fit


def run_spea2(
    simulator,space,norm,block_id,benchmark_seed,algorithm_seed,initial_configs,initial_x,
    budget=3000,population_size=60,reference_front=None,checkpoints=(60,300,600,1200,1800,2400,3000),
):
    rng=np.random.default_rng(algorithm_seed+99173)
    seen=set(space.keys(initial_configs)); raw_proposals=len(initial_configs)
    records=[_evaluate_records(initial_configs,initial_x,simulator,"D-SPEA2",block_id,benchmark_seed,algorithm_seed,0,1)]
    pop=initial_x.copy(); pop_obj=records[0][["p95_latency_s","quality_loss"]].to_numpy()
    archive_x=np.empty((0,5)); archive_obj=np.empty((0,2)); completed=len(initial_configs); generation=0
    while completed<budget:
        union_x=np.vstack([pop,archive_x]) if len(archive_x) else pop
        union_obj=np.vstack([pop_obj,archive_obj]) if len(archive_obj) else pop_obj
        selected,selected_fit=_spea_select(union_obj,population_size)
        archive_x=union_x[selected]; archive_obj=union_obj[selected]
        generation+=1; target=min(population_size,budget-completed)
        offspring,offspring_x,raw=_generate_unique_offspring(rng=rng,pop=archive_x,obj=archive_obj,space=space,global_seen=seen,target=target,method="SPEA2",spea_fitness=selected_fit)
        raw_proposals+=raw
        rec=_evaluate_records(offspring,offspring_x,simulator,"D-SPEA2",block_id,benchmark_seed,algorithm_seed,generation,completed+1)
        records.append(rec); completed+=target
        pop=offspring_x; pop_obj=rec[["p95_latency_s","quality_loss"]].to_numpy()
    candidates=pd.concat(records,ignore_index=True)
    return _summary("D-SPEA2",block_id,benchmark_seed,algorithm_seed,candidates,norm,raw_proposals,reference_front,checkpoints)


def run_random(
    simulator,space,norm,block_id,benchmark_seed,algorithm_seed,initial_configs,initial_x,
    budget=3000,reference_front=None,checkpoints=(60,300,600,1200,1800,2400,3000),
):
    rng=np.random.default_rng(algorithm_seed+1777)
    configs=list(initial_configs); xs=list(initial_x); seen=set(space.keys(initial_configs)); raw=len(initial_configs)
    while len(configs)<budget:
        x=rng.random((max(256,budget-len(configs)),5)); dec=space.decode(x)
        for row,xx in zip(dec,x):
            raw+=1; key=space.key(row)
            if key in seen: continue
            seen.add(key); configs.append(row); xs.append(xx)
            if len(configs)>=budget: break
    candidates=_evaluate_records(np.asarray(configs),np.asarray(xs),simulator,"Random",block_id,benchmark_seed,algorithm_seed,0,1)
    return _summary("Random",block_id,benchmark_seed,algorithm_seed,candidates,norm,raw,reference_front,checkpoints)


def run_sobol(
    simulator,space,norm,block_id,benchmark_seed,algorithm_seed,initial_configs,initial_x,
    budget=3000,reference_front=None,checkpoints=(60,300,600,1200,1800,2400,3000),
):
    configs=list(initial_configs); xs=list(initial_x); seen=set(space.keys(initial_configs)); raw=len(initial_configs)
    m=13
    while len(configs)<budget:
        sampler=qmc.Sobol(d=5, scramble=True, seed=algorithm_seed+31337+m, bits=30, optimization=None)
        points=sampler.random_base2(m=m)
        dec=space.decode(points)
        for row,xx in zip(dec,points):
            raw+=1; key=space.key(row)
            if key in seen: continue
            seen.add(key); configs.append(row); xs.append(xx)
            if len(configs)>=budget: break
        m+=1
    candidates=_evaluate_records(np.asarray(configs),np.asarray(xs),simulator,"Sobol",block_id,benchmark_seed,algorithm_seed,0,1)
    return _summary("Sobol",block_id,benchmark_seed,algorithm_seed,candidates,norm,raw,reference_front,checkpoints)


def run_method(method: str, **kwargs) -> SearchResult:
    funcs={"NSGA-II":run_nsga2,"D-NSGA-II":run_nsga2,"SPEA2":run_spea2,"D-SPEA2":run_spea2,"Random":run_random,"Sobol":run_sobol}
    return funcs[method](**kwargs)
