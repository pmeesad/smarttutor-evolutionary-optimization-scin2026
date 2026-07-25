from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES = ("factual", "procedural", "conceptual", "troubleshooting")
CLASS_PROBS = np.asarray([0.35, 0.30, 0.20, 0.15], dtype=float)
COMPLEXITY = {
    "factual": 0.80,
    "procedural": 1.00,
    "conceptual": 1.25,
    "troubleshooting": 1.55,
}
QUERY_TOKEN_MEDIANS = {
    "factual": 18.0,
    "procedural": 30.0,
    "conceptual": 44.0,
    "troubleshooting": 58.0,
}
OUTPUT_TOKEN_MEDIANS = {
    "factual": 110.0,
    "procedural": 210.0,
    "conceptual": 300.0,
    "troubleshooting": 410.0,
}
CONTEXT_MEDIANS = {
    "factual": 760.0,
    "procedural": 1450.0,
    "conceptual": 2250.0,
    "troubleshooting": 3250.0,
}
EVIDENCE_SPAN_MEDIANS = {
    "factual": 210.0,
    "procedural": 330.0,
    "conceptual": 480.0,
    "troubleshooting": 610.0,
}
RETRIEVABILITY_BASE = {
    "factual": 0.86,
    "procedural": 0.80,
    "conceptual": 0.74,
    "troubleshooting": 0.67,
}
INDEX_DIFFICULTY_BASE = {
    "factual": 0.28,
    "procedural": 0.38,
    "conceptual": 0.52,
    "troubleshooting": 0.64,
}
BOUNDARY_RISK_BASE = {
    "factual": 0.22,
    "procedural": 0.36,
    "conceptual": 0.52,
    "troubleshooting": 0.66,
}


def _class_map(values: np.ndarray, mapping: dict[str, float]) -> np.ndarray:
    return np.asarray([mapping[str(v)] for v in values], dtype=float)


def generate_benchmark(seed: int, n_queries: int = 1200) -> pd.DataFrame:
    """Generate one benchmark realization with correlated request characteristics.

    A latent request-difficulty variable induces correlations among retrievability,
    required context, output length, index difficulty, and evidence-boundary risk.
    The benchmark is frozen after generation, making candidate evaluation
    deterministic within a block while allowing block-level replication by seed.
    """
    rng = np.random.default_rng(seed)
    # Use exact class counts so each 1,200-request realization has the declared
    # 35/30/20/15 percent composition.  The class labels are then shuffled.
    raw_counts = CLASS_PROBS * n_queries
    counts = np.floor(raw_counts).astype(int)
    remainder = int(n_queries - counts.sum())
    if remainder:
        order = np.argsort(-(raw_counts - counts), kind="mergesort")
        counts[order[:remainder]] += 1
    query_class = np.concatenate([
        np.repeat(np.asarray([name], dtype=object), int(count))
        for name, count in zip(CLASSES, counts)
    ])
    rng.shuffle(query_class)
    complexity = _class_map(query_class, COMPLEXITY)
    latent = rng.normal(0.0, 1.0, size=n_queries)

    query_tokens = np.exp(
        np.log(_class_map(query_class, QUERY_TOKEN_MEDIANS))
        + 0.23 * latent
        + rng.normal(0.0, 0.19, size=n_queries)
    )
    query_tokens = np.clip(query_tokens, 6, 180)

    output_tokens = np.exp(
        np.log(_class_map(query_class, OUTPUT_TOKEN_MEDIANS))
        + 0.25 * latent
        + rng.normal(0.0, 0.24, size=n_queries)
    )
    output_tokens = np.clip(output_tokens, 35, 1100)

    required_context = np.exp(
        np.log(_class_map(query_class, CONTEXT_MEDIANS))
        + 0.30 * latent
        + rng.normal(0.0, 0.25, size=n_queries)
    )
    required_context = np.clip(required_context, 350, 7200)

    evidence_span = np.exp(
        np.log(_class_map(query_class, EVIDENCE_SPAN_MEDIANS))
        + 0.18 * latent
        + rng.normal(0.0, 0.22, size=n_queries)
    )
    evidence_span = np.clip(evidence_span, 80, 1000)

    retrievability = (
        _class_map(query_class, RETRIEVABILITY_BASE)
        - 0.055 * latent
        + rng.normal(0.0, 0.055, size=n_queries)
    )
    retrievability = np.clip(retrievability, 0.38, 0.97)

    index_difficulty = (
        _class_map(query_class, INDEX_DIFFICULTY_BASE)
        + 0.075 * latent
        + rng.normal(0.0, 0.06, size=n_queries)
    )
    index_difficulty = np.clip(index_difficulty, 0.08, 0.95)

    boundary_risk = (
        _class_map(query_class, BOUNDARY_RISK_BASE)
        + 0.07 * latent
        + rng.normal(0.0, 0.055, size=n_queries)
    )
    boundary_risk = np.clip(boundary_risk, 0.05, 0.95)

    # Evidence density describes how much of a nominal retrieved chunk is useful.
    evidence_density = np.clip(
        0.78 - 0.13 * index_difficulty + 0.07 * retrievability + rng.normal(0, 0.045, n_queries),
        0.42,
        0.96,
    )

    concurrency = rng.choice(
        np.asarray([1, 4, 8, 16, 32], dtype=int),
        size=n_queries,
        p=np.asarray([0.15, 0.25, 0.25, 0.20, 0.15]),
    )
    arrival_burst = np.exp(rng.normal(0.0, 0.16 + 0.04 * (concurrency >= 16), n_queries))
    service_jitter = np.exp(rng.normal(0.0, 0.035, n_queries))

    frame = pd.DataFrame(
        {
            "query_id": [f"Q{seed}_{i+1:04d}" for i in range(n_queries)],
            "benchmark_seed": seed,
            "query_class": query_class,
            "complexity": complexity,
            "query_tokens": query_tokens,
            "output_tokens": output_tokens,
            "required_context": required_context,
            "evidence_span": evidence_span,
            "retrievability": retrievability,
            "index_difficulty": index_difficulty,
            "boundary_risk": boundary_risk,
            "evidence_density": evidence_density,
            "concurrency": concurrency,
            "arrival_burst": arrival_burst,
            "service_jitter": service_jitter,
        }
    )
    return frame
