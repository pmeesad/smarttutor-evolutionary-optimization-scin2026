from __future__ import annotations
from pathlib import Path
import itertools,json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT=Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0,str(ROOT/'src'))
from smarttutor_revision.benchmark import generate_benchmark,CLASSES
from smarttutor_revision.scenarios import SimulatorScenario
from smarttutor_revision.simulator import SmartTutorSimulator
from smarttutor_revision.search_space import CONFIG_COLUMNS,SearchSpace
from smarttutor_revision.pareto import Normalization,nondominated_mask,igd_plus,additive_epsilon,normalize,hypervolume_2d

OUT=ROOT/'outputs_v4'; norm=Normalization(); METHODS=['D-NSGA-II','D-SPEA2','Sobol','Random']
ADAPTIVE=['D-NSGA-II','D-SPEA2']

def thin_front(frame,n=500):
    f=frame.sort_values('p95_latency_s').drop_duplicates(['p95_latency_s','quality_loss']).reset_index(drop=True)
    if len(f)>n:
        idx=np.unique(np.rint(np.linspace(0,len(f)-1,n)).astype(int)); f=f.iloc[idx].reset_index(drop=True)
    return f

def exact_sign_flip(diff,alternative='greater'):
    d=np.asarray(diff,float); d=d[np.abs(d)>1e-15]; obs=d.mean(); n=len(d)
    vals=[]
    for signs in itertools.product([-1.,1.],repeat=n): vals.append(np.mean(d*np.asarray(signs)))
    vals=np.asarray(vals)
    if alternative=='greater': return float(np.mean(vals>=obs-1e-15))
    if alternative=='less': return float(np.mean(vals<=obs+1e-15))
    return float(np.mean(np.abs(vals)>=abs(obs)-1e-15))

def holm(ps):
    p=np.asarray(ps,float); order=np.argsort(p); out=np.empty(len(p)); running=0.; m=len(p)
    for rank,idx in enumerate(order):
        running=max(running,min(1.,(m-rank)*p[idx])); out[idx]=running
    return out

def hierarchical_bootstrap(diff_df,resamples=10000,seed=20260725):
    """Two-stage bootstrap: resample benchmark blocks, then seeds within block."""
    rng=np.random.default_rng(seed)
    blocks=np.sort(diff_df.block_id.unique())
    arrays=[diff_df.loc[diff_df.block_id==b,'difference'].to_numpy(float) for b in blocks]
    # The experiment has a balanced 12 x 10 block-by-seed layout.
    mat=np.vstack(arrays)
    block_idx=rng.integers(0,len(blocks),size=(resamples,len(blocks)))
    seed_idx=rng.integers(0,mat.shape[1],size=(resamples,len(blocks),mat.shape[1]))
    selected=mat[block_idx[:,:,None],seed_idx]
    means=selected.mean(axis=(1,2))
    return np.quantile(means,[.025,.975])

def load_runs():
    summaries=[]; fronts=[]; conv=[]
    for b in range(1,13):
      for s in range(1,11):
        p=OUT/'runs'/f'block_{b:02d}'/f'seed_{s:02d}'
        summaries.append(pd.read_csv(p/'summary.csv'))
        fronts.append(pd.read_csv(p/'fronts.csv.gz'))
        conv.append(pd.read_csv(p/'convergence.csv'))
    return pd.concat(summaries,ignore_index=True),pd.concat(fronts,ignore_index=True),pd.concat(conv,ignore_index=True)

def reference_metrics(summary,fronts):
    rows=[]; refspec=[]
    for b in range(1,13):
        rp=OUT/'references'/f'block_{b:02d}'
        ref100=pd.read_csv(rp/'independent_front_100k.csv')
        ref250=pd.read_csv(rp/'independent_front_250k.csv')
        sample=pd.read_csv(rp/'independent_sample_250k.csv.gz')
        refspec.append({'block_id':b,'independent_sample_n':len(sample),'front_100k_n':len(pd.read_csv(rp/'independent_front_100k_full.csv')),'front_250k_n':len(pd.read_csv(rp/'independent_front_250k_full.csv'))})
        # method-specific leave-one-method-out references use the independent sample
        # plus nondominated points from all competing methods except the evaluated method.
        lomo={}
        for method in METHODS:
            other=fronts[(fronts.block_id==b)&(fronts.method!=method)]
            combined=pd.concat([sample,other],ignore_index=True)
            combined=combined.drop_duplicates(CONFIG_COLUMNS)
            obj=combined[['p95_latency_s','quality_loss']].to_numpy()
            lomo[method]=thin_front(combined.loc[nondominated_mask(obj)],500)
        for _,sr in summary[summary.block_id==b].iterrows():
            method=sr.method; sid=int(sr.optimizer_seed_id)
            f=fronts[(fronts.block_id==b)&(fronts.optimizer_seed_id==sid)&(fronts.method==method)]
            obj=f[['p95_latency_s','quality_loss']].to_numpy()
            row=sr.to_dict()
            for name,ref in [('ind100',ref100),('ind250',ref250),('lomo250',lomo[method])]:
                ro=ref[['p95_latency_s','quality_loss']].to_numpy()
                row[f'igd_plus_{name}']=igd_plus(obj,ro,norm)
                row[f'epsilon_{name}']=additive_epsilon(obj,ro,norm)
            rows.append(row)
    out=pd.DataFrame(rows)
    out.to_csv(OUT/'algorithm_runs_multiseed_v4.csv',index=False)
    pd.DataFrame(refspec).to_csv(OUT/'reference_front_spec_v4.csv',index=False)
    return out

def summarize(run):
    metrics=['hypervolume','igd_plus_ind250','epsilon_ind250','igd_plus_ind100','epsilon_ind100','igd_plus_lomo250','epsilon_lomo250','front_size','raw_proposals','duplicate_rate','compromise_p95_s','compromise_quality']
    rows=[]
    for m,g in run.groupby('method',sort=False):
        row={'method':m,'cells':len(g),'benchmarks':g.block_id.nunique(),'optimizer_seeds_per_benchmark':g.optimizer_seed_id.nunique()}
        for metric in metrics:
            x=g[metric].to_numpy(float); row[f'{metric}_mean']=x.mean(); row[f'{metric}_sd_all_cells']=x.std(ddof=1)
            block_means=g.groupby('block_id')[metric].mean(); row[f'{metric}_sd_between_blocks']=block_means.std(ddof=1)
            within=g.groupby('block_id')[metric].std(ddof=1); row[f'{metric}_mean_within_block_seed_sd']=within.mean()
            lo,hi=hierarchical_bootstrap(pd.DataFrame({'block_id':g.block_id,'difference':x}),seed=20260725+len(rows)); row[f'{metric}_hci_low']=lo; row[f'{metric}_hci_high']=hi
        rows.append(row)
    df=pd.DataFrame(rows); df.to_csv(OUT/'algorithm_summary_multiseed_v4.csv',index=False); return df

def confirmatory(run):
    tests=[]; rawp=[]
    for adaptive in ADAPTIVE:
      for comp in ['Sobol','Random']:
        for metric,direction in [('hypervolume','greater'),('igd_plus_ind250','less'),('epsilon_ind250','less')]:
            p=run.pivot_table(index=['block_id','optimizer_seed_id'],columns='method',values=metric)
            if direction=='greater': d=p[adaptive]-p[comp]
            else: d=p[comp]-p[adaptive]
            dd=d.rename('difference').reset_index()
            block=dd.groupby('block_id').difference.mean()
            W,pw=wilcoxon(block,alternative='greater',zero_method='wilcox',method='auto')
            lo,hi=hierarchical_bootstrap(dd,seed=20260725+len(tests))
            ps=exact_sign_flip(block.to_numpy(),'greater')
            tests.append({'adaptive':adaptive,'comparator':comp,'metric':metric,'mean_improvement':dd.difference.mean(),'hierarchical_ci_low':lo,'hierarchical_ci_high':hi,'cell_wins':int((dd.difference>0).sum()),'cells':len(dd),'block_wins':int((block>0).sum()),'blocks':len(block),'wilcoxon_W_block_means':W,'wilcoxon_p_one_sided':pw,'exact_sign_flip_p_4096':ps})
            rawp.append(pw)
    adj=holm(rawp)
    for r,a in zip(tests,adj): r['holm_p_12_tests']=a
    df=pd.DataFrame(tests); df.to_csv(OUT/'confirmatory_tests_multiseed_v4.csv',index=False)
    # Exploratory adaptive comparison, same block-mean framework.
    ex=[]
    for metric,direction in [('hypervolume','greater'),('igd_plus_ind250','less'),('epsilon_ind250','less')]:
        p=run.pivot_table(index=['block_id','optimizer_seed_id'],columns='method',values=metric)
        d=(p['D-NSGA-II']-p['D-SPEA2']) if direction=='greater' else (p['D-SPEA2']-p['D-NSGA-II'])
        dd=d.rename('difference').reset_index(); block=dd.groupby('block_id').difference.mean()
        W,pw=wilcoxon(block,alternative='two-sided',zero_method='wilcox',method='auto'); lo,hi=hierarchical_bootstrap(dd,seed=20260800+len(ex))
        ex.append({'metric':metric,'mean_difference_favoring_D_NS':dd.difference.mean(),'hierarchical_ci_low':lo,'hierarchical_ci_high':hi,'cell_wins':int((dd.difference>0).sum()),'cells':len(dd),'block_wins':int((block>0).sum()),'blocks':len(block),'wilcoxon_p_two_sided':pw,'sign_flip_p_two_sided':exact_sign_flip(block.to_numpy(),'two-sided')})
    pd.DataFrame(ex).to_csv(OUT/'exploratory_adaptive_multiseed_v4.csv',index=False)
    return df

def evaluate_catalog(configs,benchmarks,scenario=SimulatorScenario(),include_class=True,batch_size=96):
    cfg=np.asarray(configs,float); n=len(cfg)
    p95=[]; q=[]; tw=[]; class_q={s:[] for s in CLASSES}
    for b in benchmarks:
        sim=SmartTutorSimulator(b,scenario)
        ev=sim.evaluate_batch(cfg,batch_size=batch_size)
        p95.append(ev.p95_latency_s.to_numpy()); q.append(ev.mean_quality.to_numpy()); tw.append(ev.mean_token_work.to_numpy())
        if include_class:
            block={s:np.empty(n,float) for s in CLASSES}
            for start in range(0,n,batch_size):
                part=cfg[start:start+batch_size]; _,qual,_=sim._request_matrices(part)
                for s in CLASSES:
                    mask=(sim.query_class==s); block[s][start:start+len(part)]=qual[:,mask].mean(axis=1)
            for s in CLASSES: class_q[s].append(block[s])
    out=pd.DataFrame(cfg,columns=CONFIG_COLUMNS)
    out['training_p95']=np.mean(p95,axis=0); out['training_qw']=np.mean(q,axis=0); out['training_token_work']=np.mean(tw,axis=0)
    if include_class:
        for s in CLASSES: out[f'training_q_{s}']=np.mean(class_q[s],axis=0)
        out['training_qmacro']=out[[f'training_q_{s}' for s in CLASSES]].mean(axis=1)
        out['training_qworst']=out[[f'training_q_{s}' for s in CLASSES]].min(axis=1)
    return out

def tie_select(df,eligible,primary):
    e=df.loc[eligible].copy()
    if len(e)==0: raise ValueError('No eligible configuration')
    if primary=='ideal':
        obj=np.column_stack([e.training_p95.to_numpy(),1-e.training_qw.to_numpy()]); z=normalize(obj,norm); e['_primary']=np.sqrt(np.sum(z*z,axis=1))
        asc=['_primary','training_p95','training_qworst','training_token_work','context_tokens','top_k','chunk_tokens','overlap','rerank_depth']; flags=[True,True,False,True,True,True,True,True,True]
    elif primary=='latency':
        asc=['training_p95','training_qworst','training_token_work','context_tokens','top_k','chunk_tokens','overlap','rerank_depth']; flags=[True,False,True,True,True,True,True,True]
    elif primary=='quality':
        asc=['training_qw','training_qworst','training_token_work','training_p95','context_tokens','top_k','chunk_tokens','overlap','rerank_depth']; flags=[False,False,True,True,True,True,True,True,True]
    else: raise ValueError(primary)
    return e.sort_values(asc,ascending=flags,kind='mergesort').iloc[0]

def decision_rules(fronts):
    cat=fronts[(fronts.block_id<=8)&(fronts.method.isin(ADAPTIVE))][CONFIG_COLUMNS].drop_duplicates().reset_index(drop=True)
    benchmarks=[generate_benchmark(2026072400+i) for i in range(1,9)]
    train=evaluate_catalog(cat.to_numpy(),benchmarks,include_class=True)
    # Keep central training nondominated catalog for scenario sensitivity.
    obj=np.column_stack([train.training_p95,1-train.training_qw]); train['central_nondominated']=nondominated_mask(obj)
    train.to_csv(OUT/'training_candidate_catalog_v4.csv.gz',index=False,compression='gzip')
    rules=[]
    r=tie_select(train,np.ones(len(train),bool),'ideal'); rules.append(('ideal_point',r))
    r=tie_select(train,train.training_qw>=.88,'latency'); rules.append(('weighted_quality_floor_0.88',r))
    r=tie_select(train,train.training_p95<=3.25,'quality'); rules.append(('latency_ceiling_3.25',r))
    r=tie_select(train,train.training_qworst>=.81,'latency'); rules.append(('class_protected_0.81',r))
    rule_rows=[]; held=[]; class_rows=[]
    held_bench=[generate_benchmark(2026072400+i) for i in range(9,13)]
    for name,r in rules:
        cfg=r[CONFIG_COLUMNS].to_numpy(float)
        rule_rows.append({'rule':name,**{c:r[c] for c in CONFIG_COLUMNS},'training_p95':r.training_p95,'training_qw':r.training_qw,'training_qmacro':r.training_qmacro,'training_qworst':r.training_qworst,'training_token_work':r.training_token_work,'candidate_pool':'unique union of D-NSGA-II and D-SPEA2 fronts, blocks 1-8, 10 matched seeds'})
        for bi,b in enumerate(held_bench,9):
            sim=SmartTutorSimulator(b,SimulatorScenario()); ev=sim.evaluate_one(cfg); cm=sim.class_metrics(cfg)
            qmap=dict(zip(cm.query_class,cm.quality)); pmap=dict(zip(cm.query_class,cm.p95_latency_s))
            held.append({'rule':name,'block_id':bi,**{c:v for c,v in zip(CONFIG_COLUMNS,cfg)},'p95_latency_s':ev.p95_latency_s,'weighted_quality':ev.mean_quality,'macro_quality':np.mean(list(qmap.values())),'worst_class_quality':min(qmap.values()),'mean_token_work':ev.mean_token_work})
            for s in CLASSES: class_rows.append({'rule':name,'block_id':bi,'query_class':s,'n':int((b.query_class==s).sum()),'quality':qmap[s],'p95_latency_s':pmap[s]})
    # Heuristic diagnostic
    heuristic=np.array([5,512,.15,8,4096],float)
    for bi,b in enumerate(held_bench,9):
        sim=SmartTutorSimulator(b,SimulatorScenario()); cm=sim.class_metrics(heuristic)
        for _,rr in cm.iterrows(): class_rows.append({'rule':'predeclared_heuristic','block_id':bi,'query_class':rr.query_class,'n':int(rr.n),'quality':rr.quality,'p95_latency_s':rr.p95_latency_s})
    pd.DataFrame(rule_rows).to_csv(OUT/'decision_rules_training_v4.csv',index=False)
    h=pd.DataFrame(held); h.to_csv(OUT/'decision_rules_heldout_blocks_v4.csv',index=False)
    summ=[]
    for name,g in h.groupby('rule',sort=False):
        row={'rule':name}
        for c in ['p95_latency_s','weighted_quality','macro_quality','worst_class_quality','mean_token_work']:
            row[f'{c}_mean']=g[c].mean(); row[f'{c}_min']=g[c].min(); row[f'{c}_max']=g[c].max()
        summ.append(row)
    pd.DataFrame(summ).to_csv(OUT/'decision_rules_heldout_summary_v4.csv',index=False)
    pd.DataFrame(class_rows).to_csv(OUT/'heldout_class_metrics_v4.csv',index=False)
    pd.concat([pd.read_csv(OUT/'references'/f'block_{b:02d}'/'class_counts.csv') for b in range(1,13)],ignore_index=True).to_csv(OUT/'benchmark_class_counts_v4.csv',index=False)
    return train,pd.DataFrame(rule_rows),pd.DataFrame(held),pd.DataFrame(class_rows)

def metadata(run,train):
    import scipy,numpy,pandas
    meta={'study_version':'v4-full-revision','benchmark_blocks':12,'requests_per_block':1200,'exact_class_counts':{'factual':420,'procedural':360,'conceptual':240,'troubleshooting':180},'optimizer_seeds_per_block':10,'matched_seed_cells':120,'methods':METHODS,'unique_evaluation_budget':3000,'initial_design_size':60,'independent_reference_unique':250000,'independent_reference_sobol_generated':524288,'scipy_version':scipy.__version__,'numpy_version':numpy.__version__,'pandas_version':pandas.__version__,'sobol_implementation':{'class':'scipy.stats.qmc.Sobol','scramble':True,'strategy_documented_by_scipy':'LMS+digital shift','bits':30,'optimization':None,'random_base2_m_search':13,'random_base2_m_reference':19},'search_space_size':SearchSpace().size,'decision_candidate_catalog_n':len(train),'normalization':norm.__dict__,'hypervolume_formula':'2-D minimization HV after affine normalization; no clipping; points at or beyond reference rectangle excluded; exact duplicate objectives removed by nondominated scan'}
    (OUT/'experiment_metadata_v4.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__':
    summary,fronts,conv=load_runs(); summary.to_csv(OUT/'algorithm_run_base_v4.csv',index=False); fronts.to_csv(OUT/'all_fronts_multiseed_v4.csv.gz',index=False,compression='gzip'); conv.to_csv(OUT/'convergence_multiseed_v4.csv',index=False)
    run=reference_metrics(summary,fronts); summarize(run); confirmatory(run); train,*_=decision_rules(fronts); metadata(run,train)
    print('analysis v4 complete',flush=True)
