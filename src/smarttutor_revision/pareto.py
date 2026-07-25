from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Normalization:
    latency_min: float = 1.5
    latency_max: float = 6.0
    loss_min: float = 0.0
    loss_max: float = 0.60
    ref_x: float = 1.02
    ref_y: float = 1.02


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def nondominated_mask(objectives: np.ndarray) -> np.ndarray:
    values = np.asarray(objectives, dtype=float)
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(values[:,0], kind="mergesort")
    mask_sorted = np.zeros(len(values), dtype=bool)
    best_y = np.inf
    # Keep ties with exact same objective only once later; all weakly efficient here.
    for pos, idx in enumerate(order):
        y = values[idx,1]
        if y < best_y - 1e-15:
            mask_sorted[pos] = True
            best_y = y
    mask = np.zeros(len(values), dtype=bool)
    mask[order[mask_sorted]] = True
    return mask


def fast_nondominated_sort(objectives: np.ndarray) -> list[np.ndarray]:
    v = np.asarray(objectives, dtype=float)
    n = len(v)
    domination_count = np.zeros(n, dtype=int)
    dominated: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(p+1,n):
            if dominates(v[p],v[q]):
                dominated[p].append(q); domination_count[q]+=1
            elif dominates(v[q],v[p]):
                dominated[q].append(p); domination_count[p]+=1
    for i in range(n):
        if domination_count[i]==0:
            fronts[0].append(i)
    level=0
    while fronts[level]:
        nxt=[]
        for p in fronts[level]:
            for q in dominated[p]:
                domination_count[q]-=1
                if domination_count[q]==0:
                    nxt.append(q)
        level+=1; fronts.append(nxt)
    return [np.asarray(f,dtype=int) for f in fronts[:-1]]


def crowding_distance(objectives: np.ndarray, front: np.ndarray) -> np.ndarray:
    values=np.asarray(objectives,float); front=np.asarray(front,int)
    d=np.zeros(len(front),float)
    if len(front)<=2:
        d[:]=np.inf; return d
    for j in range(values.shape[1]):
        order=np.argsort(values[front,j],kind="mergesort")
        vals=values[front[order],j]
        d[order[0]]=d[order[-1]]=np.inf
        span=vals[-1]-vals[0]
        if span>0:
            d[order[1:-1]] += (vals[2:]-vals[:-2])/span
    return d


def rank_and_crowding(objectives: np.ndarray):
    fronts=fast_nondominated_sort(objectives)
    rank=np.empty(len(objectives),int); crowd=np.zeros(len(objectives),float)
    for r,f in enumerate(fronts):
        rank[f]=r; crowd[f]=crowding_distance(objectives,f)
    return rank,crowd,fronts


def normalize(objectives: np.ndarray, norm: Normalization) -> np.ndarray:
    v=np.asarray(objectives,float).copy()
    v[:,0]=(v[:,0]-norm.latency_min)/(norm.latency_max-norm.latency_min)
    v[:,1]=(v[:,1]-norm.loss_min)/(norm.loss_max-norm.loss_min)
    return v


def hypervolume_2d(objectives: np.ndarray, norm: Normalization) -> float:
    z=normalize(objectives,norm)
    z=z[(z[:,0]<norm.ref_x)&(z[:,1]<norm.ref_y)]
    if len(z)==0: return 0.0
    z=z[nondominated_mask(z)]
    z=z[np.argsort(z[:,0],kind="mergesort")]
    area=0.0; prev_y=norm.ref_y
    for x,y in z:
        if y<prev_y:
            area+=(norm.ref_x-x)*(prev_y-y); prev_y=y
    return float(area)


def ideal_point_index(objectives: np.ndarray, norm: Normalization, weights=(1.0,1.0)) -> int:
    z=normalize(objectives,norm)
    w=np.asarray(weights,float)
    return int(np.argmin(np.sqrt(np.sum((z*w)**2,axis=1))))


def constrained_index(objectives: np.ndarray, *, min_quality: float|None=None, max_latency: float|None=None) -> int:
    v=np.asarray(objectives,float)
    quality=1.0-v[:,1]
    eligible=np.ones(len(v),bool)
    if min_quality is not None: eligible &= quality>=min_quality
    if max_latency is not None: eligible &= v[:,0]<=max_latency
    idx=np.flatnonzero(eligible)
    if len(idx)==0:
        # fall back to closest violation in normalized raw units
        violation=np.zeros(len(v))
        if min_quality is not None: violation+=np.maximum(0,min_quality-quality)
        if max_latency is not None: violation+=np.maximum(0,v[:,0]-max_latency)
        return int(np.argmin(violation))
    if min_quality is not None and max_latency is None:
        return int(idx[np.argmin(v[idx,0])])
    if max_latency is not None and min_quality is None:
        return int(idx[np.argmin(v[idx,1])])
    # both: choose minimum quality loss then latency
    order=np.lexsort((v[idx,0],v[idx,1])); return int(idx[order[0]])


def igd_plus(approx: np.ndarray, reference: np.ndarray, norm: Normalization) -> float:
    a=normalize(np.asarray(approx,float),norm); r=normalize(np.asarray(reference,float),norm)
    # IGD+ distance from each reference point to approximation in minimization.
    distances=[]
    for point in r:
        diff=np.maximum(a-point,0.0)
        distances.append(float(np.min(np.linalg.norm(diff,axis=1))))
    return float(np.mean(distances))


def additive_epsilon(approx: np.ndarray, reference: np.ndarray, norm: Normalization) -> float:
    a=normalize(np.asarray(approx,float),norm); r=normalize(np.asarray(reference,float),norm)
    eps=[]
    for rp in r:
        eps.append(float(np.min(np.max(a-rp,axis=1))))
    return float(np.max(eps))
