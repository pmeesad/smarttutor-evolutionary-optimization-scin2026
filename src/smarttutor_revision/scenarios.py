from __future__ import annotations

from dataclasses import dataclass, replace, asdict
import numpy as np
import pandas as pd
from scipy.stats import qmc


@dataclass(frozen=True)
class SimulatorScenario:
    name: str = "central"
    chunk_form: str = "log_gaussian"  # log_gaussian, plateau, piecewise
    quality_intercept: float = -1.15
    weight_recall: float = 1.55
    weight_chunk: float = 0.72
    weight_overlap: float = 0.50
    weight_rerank: float = 0.78
    weight_context: float = 1.20
    weight_interaction: float = 0.42
    distraction_weight: float = 0.62
    distraction_threshold: float = 1.50
    distraction_power: float = 1.35
    ret_base: float = 0.034
    ret_candidate: float = 0.0038
    ret_token: float = 0.0000105
    rerank_base: float = 0.010
    rerank_token: float = 0.0000140
    prompt_base: float = 0.038
    prompt_token: float = 0.000052
    ttft_base: float = 0.085
    ttft_sqrt_token: float = 0.00095
    generation_tokens_per_s: float = 260.0
    context_slowdown: float = 0.52
    concurrency_slowdown: float = 0.12
    queue_coefficient: float = 0.018
    queue_exponent: float = 1.28
    capacity_scale: float = 8.0
    network_base: float = 0.025
    system_prompt_tokens: float = 300.0

    def to_dict(self) -> dict:
        return asdict(self)


def named_scenarios() -> list[SimulatorScenario]:
    base = SimulatorScenario()
    return [
        base,
        replace(base, name="retrieval_fast", ret_base=base.ret_base*0.8, ret_candidate=base.ret_candidate*0.8, ret_token=base.ret_token*0.8, rerank_token=base.rerank_token*0.8),
        replace(base, name="retrieval_slow", ret_base=base.ret_base*1.2, ret_candidate=base.ret_candidate*1.2, ret_token=base.ret_token*1.2, rerank_token=base.rerank_token*1.2),
        replace(base, name="generation_fast", generation_tokens_per_s=base.generation_tokens_per_s*1.25, prompt_token=base.prompt_token*0.8, ttft_sqrt_token=base.ttft_sqrt_token*0.8),
        replace(base, name="generation_slow", generation_tokens_per_s=base.generation_tokens_per_s*0.75, prompt_token=base.prompt_token*1.2, ttft_sqrt_token=base.ttft_sqrt_token*1.2),
        replace(base, name="queue_heavy", queue_coefficient=base.queue_coefficient*1.3, capacity_scale=base.capacity_scale*0.8, concurrency_slowdown=base.concurrency_slowdown*1.3),
        replace(base, name="queue_light", queue_coefficient=base.queue_coefficient*0.7, capacity_scale=base.capacity_scale*1.2, concurrency_slowdown=base.concurrency_slowdown*0.7),
        replace(base, name="quality_plateau", chunk_form="plateau", distraction_weight=base.distraction_weight*0.85),
        replace(base, name="quality_piecewise", chunk_form="piecewise", distraction_weight=base.distraction_weight*1.2, distraction_threshold=1.35),
    ]


def uncertainty_scenarios(n: int = 64, seed: int = 2026072501) -> list[SimulatorScenario]:
    """Latin-hypercube coefficient ensemble spanning ±30% and three forms."""
    sampler = qmc.LatinHypercube(d=9, scramble=True, seed=seed)
    unit = sampler.random(n)
    multipliers = 0.70 + 0.60 * unit
    base = SimulatorScenario()
    forms = ["log_gaussian", "plateau", "piecewise"]
    scenarios: list[SimulatorScenario] = []
    for i, row in enumerate(multipliers):
        scenarios.append(
            replace(
                base,
                name=f"uncertainty_{i+1:02d}",
                chunk_form=forms[i % len(forms)],
                weight_recall=base.weight_recall*row[0],
                weight_chunk=base.weight_chunk*row[1],
                weight_overlap=base.weight_overlap*row[2],
                weight_rerank=base.weight_rerank*row[3],
                weight_context=base.weight_context*row[4],
                distraction_weight=base.distraction_weight*row[5],
                ret_token=base.ret_token*row[6],
                generation_tokens_per_s=base.generation_tokens_per_s/row[7],
                queue_coefficient=base.queue_coefficient*row[8],
            )
        )
    return scenarios


def scenarios_frame(scenarios: list[SimulatorScenario]) -> pd.DataFrame:
    return pd.DataFrame([s.to_dict() for s in scenarios])
