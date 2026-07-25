from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs_v4'; FIG=ROOT/'figures_v4'; FIG.mkdir(exist_ok=True)
# Figure 1: reuse audited architecture image.
import shutil
shutil.copy(ROOT/'figures'/'figure1_architecture.png',FIG/'figure1_architecture.png')
# Figure 2: independent-only reference and one representative matched seed cell.
b=7; sid=1
ref=pd.read_csv(OUT/'references'/f'block_{b:02d}'/'independent_front_250k_full.csv')
front=pd.read_csv(OUT/'runs'/f'block_{b:02d}'/f'seed_{sid:02d}'/'fronts.csv.gz')
plt.figure(figsize=(7.2,4.6))
plt.plot(ref.p95_latency_s,1-ref.quality_loss,label='Independent 250k reference front',linewidth=2)
markers={'D-NSGA-II':'o','D-SPEA2':'s','Sobol':'^','Random':'x'}
for m,g in front.groupby('method',sort=False):
    plt.scatter(g.p95_latency_s,g.mean_quality,s=16,marker=markers[m],label=m,alpha=.75)
plt.xlabel('P95 simulated latency (s)'); plt.ylabel('Mean surrogate quality'); plt.title('Representative Pareto approximation (block 7, seed 1)'); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG/'figure2_pareto_independent.png',dpi=300); plt.close()
# Figure 3: seed-averaged block trajectories, then CI across 12 blocks.
cv=pd.read_csv(OUT/'convergence_multiseed_v4.csv')
bm=cv.groupby(['method','block_id','evaluations'],as_index=False).hypervolume.mean()
plt.figure(figsize=(7.2,4.5))
for m,g in bm.groupby('method',sort=False):
    agg=g.groupby('evaluations').hypervolume.agg(['mean','std','count']).reset_index(); ci=1.96*agg['std']/np.sqrt(agg['count'])
    plt.plot(agg.evaluations,agg['mean'],marker='o',label=m); plt.fill_between(agg.evaluations,agg['mean']-ci,agg['mean']+ci,alpha=.14)
plt.xlabel('Unique feasible configurations evaluated'); plt.ylabel('Normalized hypervolume'); plt.title('Convergence across 12 benchmarks and 10 matched seeds'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'figure3_convergence_multiseed.png',dpi=300); plt.close()
# Figure 4: grid-normalized control distributions and scenario-level held-out outcomes.
sel=pd.read_csv(OUT/'scenario_selected_profiles_v4.csv'); summ=pd.read_csv(OUT/'scenario_level_summary_v4.csv')
# Normalize by full grid endpoints/indexes; context now included.
grids={'top_k':np.arange(2,13),'chunk_tokens':np.arange(128,1025,64),'overlap':np.round(np.arange(.05,.301,.01),2),'rerank_depth':np.arange(0,21),'context_tokens':np.arange(1024,8193,256)}
vals=[]
for c,grid in grids.items():
    mp={round(float(v),6):i/(len(grid)-1) for i,v in enumerate(grid)}
    vals.append([mp[round(float(v),6)] for v in sel[c]])
fig,axes=plt.subplots(1,2,figsize=(9.2,4.3))
axes[0].boxplot(vals,tick_labels=['top-k','chunk','overlap','rerank','context'],showfliers=True)
axes[0].set_ylabel('Grid-normalized selected value'); axes[0].set_ylim(-.05,1.05); axes[0].set_title('Profiles selected across 64 scenarios')
axes[1].scatter(summ.heldout_p95_mean,summ.heldout_quality_mean,s=25,alpha=.75)
axes[1].set_xlabel('Held-out mean P95 latency (s)'); axes[1].set_ylabel('Held-out mean surrogate quality'); axes[1].set_title('Scenario-specific profiles (n=64)')
fig.tight_layout(); fig.savefig(FIG/'figure4_scenario_catalog.png',dpi=300); plt.close(fig)
print(FIG)
