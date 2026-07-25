from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

from .pareto import Normalization, hypervolume_2d, nondominated_mask, igd_plus, additive_epsilon, ideal_point_index


def bootstrap_mean_ci(values, confidence=0.95, resamples=10000, seed=20260725):
    x=np.asarray(values,float); rng=np.random.default_rng(seed)
    means=np.mean(rng.choice(x,(resamples,len(x)),replace=True),axis=1)
    alpha=(1-confidence)/2
    return float(np.quantile(means,alpha)),float(np.quantile(means,1-alpha))


def exact_sign_flip_pvalue(differences, alternative="greater"):
    d=np.asarray(differences,float); d=d[np.abs(d)>1e-15]
    observed=float(np.mean(d)); n=len(d)
    stats=[]
    for signs in itertools.product([-1.0,1.0],repeat=n):
        stats.append(float(np.mean(d*np.asarray(signs))))
    stats=np.asarray(stats)
    if alternative=="greater": return float(np.mean(stats>=observed-1e-15))
    if alternative=="less": return float(np.mean(stats<=observed+1e-15))
    return float(np.mean(np.abs(stats)>=abs(observed)-1e-15))


def holm_adjust(pvalues: list[float]) -> list[float]:
    p=np.asarray(pvalues,float); order=np.argsort(p); out=np.empty(len(p)); running=0
    m=len(p)
    for rank,idx in enumerate(order):
        adjusted=min(1.0,(m-rank)*p[idx]); running=max(running,adjusted); out[idx]=running
    return out.tolist()


def paired_comparisons(summary: pd.DataFrame, reference_method="D-NSGA-II") -> pd.DataFrame:
    rows=[]
    others=[m for m in summary.method.unique() if m!=reference_method]
    raw=[]
    for metric,direction in [("hypervolume","greater"),("igd_plus","less"),("epsilon_additive","less")]:
        pivot=summary.pivot(index="block_id",columns="method",values=metric)
        for method in others:
            if direction=="greater": diff=pivot[reference_method]-pivot[method]
            else: diff=pivot[method]-pivot[reference_method]
            stat,p=wilcoxon(diff,alternative="greater",zero_method="wilcox",method="auto")
            ci=bootstrap_mean_ci(diff.to_numpy())
            sign_p=exact_sign_flip_pvalue(diff.to_numpy(),"greater")
            rows.append({
                "metric":metric,"reference":reference_method,"comparator":method,
                "mean_improvement":float(diff.mean()),"ci95_low":ci[0],"ci95_high":ci[1],
                "wilcoxon_W":float(stat),"wilcoxon_p_one_sided":float(p),
                "exact_sign_flip_p":sign_p,"wins":int(np.sum(diff>0)),"ties":int(np.sum(np.isclose(diff,0))),"blocks":len(diff),
            }); raw.append(float(p))
    adj=holm_adjust(raw)
    for row,val in zip(rows,adj): row["holm_p"]=val
    return pd.DataFrame(rows)


def algorithm_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for method,g in run_summary.groupby("method",sort=False):
        row={"method":method,"blocks":len(g)}
        for metric in ["hypervolume","igd_plus","epsilon_additive","compromise_p95_s","compromise_quality","unique_ratio","duplicate_rate","front_size"]:
            values=g[metric].to_numpy(float); lo,hi=bootstrap_mean_ci(values)
            row[f"{metric}_mean"]=float(np.mean(values)); row[f"{metric}_sd"]=float(np.std(values,ddof=1)); row[f"{metric}_ci_low"]=lo; row[f"{metric}_ci_high"]=hi
            row[f"{metric}_median"]=float(np.median(values))
        rows.append(row)
    return pd.DataFrame(rows)


def build_reference_front(sample: pd.DataFrame, candidates: list[pd.DataFrame]) -> pd.DataFrame:
    combined=pd.concat([sample]+candidates,ignore_index=True)
    combined=combined.drop_duplicates(["top_k","chunk_tokens","overlap","rerank_depth","context_tokens"])
    obj=combined[["p95_latency_s","quality_loss"]].to_numpy()
    front=combined.loc[nondominated_mask(obj)].sort_values("p95_latency_s").reset_index(drop=True)
    # The source archive is high-resolution; a deterministic 500-point thinning
    # keeps IGD+/epsilon computation tractable without changing the front envelope.
    if len(front)>500:
        idx=np.unique(np.rint(np.linspace(0,len(front)-1,500)).astype(int))
        front=front.iloc[idx].reset_index(drop=True)
    return front


def recalc_reference_metrics(result, reference_front: pd.DataFrame, norm: Normalization):
    ref=reference_front[["p95_latency_s","quality_loss"]].to_numpy()
    fobj=result.front[["p95_latency_s","quality_loss"]].to_numpy()
    result.summary["igd_plus"]=igd_plus(fobj,ref,norm)
    result.summary["epsilon_additive"]=additive_epsilon(fobj,ref,norm)
    for i,row in result.convergence.iterrows():
        cp=int(row.evaluations); sub=result.candidates[result.candidates.evaluation_sequence<=cp]
        obj=sub[["p95_latency_s","quality_loss"]].to_numpy(); fo=obj[nondominated_mask(obj)]
        result.convergence.loc[i,"igd_plus"]=igd_plus(fo,ref,norm)
        result.convergence.loc[i,"epsilon_additive"]=additive_epsilon(fo,ref,norm)
    return result


def normalization_sensitivity(results, normalizations: dict[str,Normalization]) -> pd.DataFrame:
    rows=[]
    for name,norm in normalizations.items():
        for result in results:
            obj=result.front[["p95_latency_s","quality_loss"]].to_numpy()
            rows.append({"scheme":name,"block_id":result.block_id,"method":result.method,"hypervolume":hypervolume_2d(obj,norm)})
    return pd.DataFrame(rows)


def select_consensus_profile(fronts: list[pd.DataFrame], simulators, norm: Normalization):
    candidates=pd.concat(fronts,ignore_index=True).drop_duplicates(["top_k","chunk_tokens","overlap","rerank_depth","context_tokens"])
    cfg=candidates[["top_k","chunk_tokens","overlap","rerank_depth","context_tokens"]].to_numpy()
    metrics=[]
    for sim in simulators:
        metrics.append(sim.evaluate_batch(cfg)[["p95_latency_s","quality_loss"]].to_numpy())
    mean_obj=np.mean(np.stack(metrics,axis=0),axis=0)
    idx=ideal_point_index(mean_obj,norm)
    row=candidates.iloc[idx].copy()
    row["training_mean_p95_s"]=mean_obj[idx,0]; row["training_mean_quality"]=1-mean_obj[idx,1]
    return row,cfg[idx]


def spearman_sensitivity(scenario_frame: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    merged=scenario_frame.merge(selections,on="name",how="inner")
    inputs=[c for c in scenario_frame.columns if c not in {"name","chunk_form"}]
    outputs=["top_k","chunk_tokens","overlap","rerank_depth","context_tokens","p95_latency_s","mean_quality"]
    rows=[]
    for x in inputs:
        for y in outputs:
            rho,p=spearmanr(merged[x],merged[y])
            rows.append({"input":x,"output":y,"spearman_rho":float(rho),"p_value":float(p)})
    return pd.DataFrame(rows)
