from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .scenarios import SimulatorScenario


@dataclass(frozen=True)
class Evaluation:
    p95_latency_s: float
    mean_quality: float
    quality_loss: float
    mean_token_work: float


class SmartTutorSimulator:
    """Request-interactive deterministic surrogate for a RAG tutoring service."""

    def __init__(self, benchmark: pd.DataFrame, scenario: SimulatorScenario):
        self.benchmark = benchmark.reset_index(drop=True).copy()
        self.scenario = scenario
        for column in benchmark.columns:
            if column not in {"query_id", "query_class"}:
                setattr(self, column, benchmark[column].to_numpy(dtype=float))
        self.query_class = benchmark["query_class"].to_numpy(dtype=object)
        self.n_requests = len(benchmark)

    def evaluate_batch(self, configurations: np.ndarray, batch_size: int = 96) -> pd.DataFrame:
        cfg = np.asarray(configurations, dtype=float)
        if cfg.ndim == 1:
            cfg = cfg.reshape(1, -1)
        outputs = []
        for start in range(0, len(cfg), batch_size):
            block = cfg[start:start+batch_size]
            latency, quality, terms = self._request_matrices(block)
            outputs.append(
                pd.DataFrame(
                    {
                        "p95_latency_s": np.quantile(latency, 0.95, axis=1),
                        "mean_quality": np.mean(quality, axis=1),
                        "quality_loss": 1.0 - np.mean(quality, axis=1),
                        "mean_token_work": np.mean(terms["token_work"], axis=1),
                    }
                )
            )
        return pd.concat(outputs, ignore_index=True)

    def evaluate_one(self, configuration: np.ndarray) -> Evaluation:
        row = self.evaluate_batch(np.asarray(configuration).reshape(1, 5)).iloc[0]
        return Evaluation(**{k: float(row[k]) for k in Evaluation.__annotations__})

    def request_level(self, configuration: np.ndarray) -> pd.DataFrame:
        cfg = np.asarray(configuration, dtype=float).reshape(1,5)
        latency, quality, terms = self._request_matrices(cfg)
        out = self.benchmark.copy()
        for name, matrix in terms.items():
            out[name] = matrix[0]
        out["latency_s"] = latency[0]
        out["quality"] = quality[0]
        return out

    def class_metrics(self, configuration: np.ndarray) -> pd.DataFrame:
        request = self.request_level(configuration)
        return (
            request.groupby("query_class", sort=False)
            .agg(
                n=("query_class", "size"),
                quality=("quality", "mean"),
                p95_latency_s=("latency_s", lambda x: float(np.quantile(x, 0.95))),
                mean_output_tokens=("output_tokens", "mean"),
                mean_context_tokens=("effective_context", "mean"),
            )
            .reset_index()
        )

    def _chunk_suitability(self, c: np.ndarray, span: np.ndarray) -> np.ndarray:
        form = self.scenario.chunk_form
        ratio = c / np.maximum(span, 1.0)
        if form == "log_gaussian":
            return np.exp(-(np.log(ratio) / 0.72) ** 2)
        if form == "plateau":
            lower = 1.0 / (1.0 + np.exp(-(ratio - 0.58) / 0.13))
            upper = 1.0 / (1.0 + np.exp((ratio - 2.15) / 0.30))
            return np.clip(lower * upper * 1.25, 0.0, 1.0)
        if form == "piecewise":
            rising = np.minimum(1.0, ratio / 0.95)
            falling = np.where(ratio <= 1.55, 1.0, np.exp(-0.38 * (ratio - 1.55)))
            return np.clip(rising * falling, 0.0, 1.0)
        raise ValueError(f"Unknown chunk form: {form}")

    def _request_matrices(self, configurations: np.ndarray):
        s = self.scenario
        cfg = np.asarray(configurations, dtype=float)
        k = cfg[:,0:1]
        c = cfg[:,1:2]
        overlap = cfg[:,2:3]
        rerank = cfg[:,3:4]
        budget = cfg[:,4:5]

        complexity = self.complexity.reshape(1,-1)
        q_tokens = self.query_tokens.reshape(1,-1)
        out_tokens = self.output_tokens.reshape(1,-1)
        required = self.required_context.reshape(1,-1)
        span = self.evidence_span.reshape(1,-1)
        retrievability = self.retrievability.reshape(1,-1)
        index_difficulty = self.index_difficulty.reshape(1,-1)
        boundary_risk = self.boundary_risk.reshape(1,-1)
        density = self.evidence_density.reshape(1,-1)
        concurrency = self.concurrency.reshape(1,-1)
        burst = self.arrival_burst.reshape(1,-1)
        jitter = self.service_jitter.reshape(1,-1)

        candidate_pool = np.where(rerank > 0, np.maximum(k, rerank), k)
        nominal_evidence = k * c * (1.0-overlap)
        effective_context = np.minimum(
            budget,
            nominal_evidence * density * (0.78 + 0.22*retrievability),
        )
        effective_context = np.broadcast_to(effective_context, (len(cfg), self.n_requests))

        chunk_fit = self._chunk_suitability(c, span)
        boundary_coverage = 1.0 - np.exp(-overlap*c / np.maximum(45.0 + 115.0*boundary_risk, 1.0))
        duplication_penalty = np.exp(-1.45 * overlap * (1.0 + 0.18*k))
        overlap_fit = boundary_coverage * duplication_penalty

        retrieval_recall = 1.0 - np.exp(
            -candidate_pool
            * retrievability
            * (0.45 + 0.55*chunk_fit)
            / (2.0 + 1.45*complexity)
        )
        rerank_gain = np.where(
            rerank > 0,
            (1.0 - np.exp(-rerank/(3.2 + 1.8*complexity)))
            * retrieval_recall
            * (0.45 + 0.55*(1.0-retrievability)),
            0.0,
        )
        context_sufficiency = 1.0 - np.exp(-effective_context/(0.82*required))
        excess_ratio = effective_context / np.maximum(required, 1.0)
        distraction = np.maximum(0.0, excess_ratio/s.distraction_threshold - 1.0) ** s.distraction_power
        interaction = retrieval_recall * context_sufficiency

        linear_quality = (
            s.quality_intercept
            + s.weight_recall*retrieval_recall
            + s.weight_chunk*chunk_fit
            + s.weight_overlap*overlap_fit
            + s.weight_rerank*rerank_gain
            + s.weight_context*context_sufficiency
            + s.weight_interaction*interaction
            - s.distraction_weight*distraction
            - 0.11*(complexity-1.0)
        )
        quality = 1.0/(1.0+np.exp(-linear_quality))

        t_embed = 0.018 + 0.00048*q_tokens
        t_retrieval = (
            s.ret_base
            + s.ret_candidate*candidate_pool*(1.0+0.85*index_difficulty)
            + s.ret_token*candidate_pool*c*(1.0+0.85*overlap)*(0.55+0.45*(1.0-retrievability))
        )
        t_rerank = np.where(
            rerank > 0,
            s.rerank_base
            + s.rerank_token*rerank*(q_tokens+c)*(1.0+0.022*np.maximum(rerank-1.0,0.0)),
            0.0,
        )
        prompt_tokens = effective_context + q_tokens + s.system_prompt_tokens
        t_prompt = s.prompt_base + s.prompt_token*prompt_tokens
        t_ttft = s.ttft_base + s.ttft_sqrt_token*np.sqrt(prompt_tokens)
        throughput = s.generation_tokens_per_s / (
            1.0
            + s.context_slowdown*(effective_context/4096.0)
            + s.concurrency_slowdown*((np.maximum(concurrency-1.0,0.0)/8.0)**0.72)
        )
        t_generation = out_tokens / np.maximum(throughput, 1.0)
        queue_driver = t_rerank + t_prompt + t_ttft + t_generation
        t_queue = (
            s.queue_coefficient
            * ((np.maximum(concurrency-1.0,0.0)/s.capacity_scale)**s.queue_exponent)
            * (queue_driver**1.08)
            * burst
        )
        t_network = s.network_base*(1.0+0.07*index_difficulty)
        latency = (t_embed+t_retrieval+t_rerank+t_prompt+t_ttft+t_generation+t_queue+t_network)*jitter

        token_work = candidate_pool*c*(1.0+0.85*overlap) + rerank*(q_tokens+c) + prompt_tokens + out_tokens
        terms = {
            "candidate_pool": np.broadcast_to(candidate_pool, quality.shape),
            "effective_context": effective_context,
            "retrieval_recall": retrieval_recall,
            "chunk_suitability": np.broadcast_to(chunk_fit, quality.shape),
            "overlap_suitability": np.broadcast_to(overlap_fit, quality.shape),
            "rerank_gain": rerank_gain,
            "context_sufficiency": context_sufficiency,
            "distraction": distraction,
            "embedding_time_s": np.broadcast_to(t_embed, quality.shape),
            "retrieval_time_s": t_retrieval,
            "rerank_time_s": t_rerank,
            "prompt_time_s": t_prompt,
            "ttft_s": t_ttft,
            "generation_time_s": t_generation,
            "queue_time_s": t_queue,
            "token_work": token_work,
        }
        return latency, quality, terms
