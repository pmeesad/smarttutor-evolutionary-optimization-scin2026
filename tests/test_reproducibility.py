from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from smarttutor_revision.benchmark import generate_benchmark
from smarttutor_revision.scenarios import SimulatorScenario
from smarttutor_revision.search_space import SearchSpace
from smarttutor_revision.simulator import SmartTutorSimulator


def test_exact_class_counts() -> None:
    frame = generate_benchmark(2026072401)
    assert frame.groupby("query_class").size().to_dict() == {
        "conceptual": 240,
        "factual": 420,
        "procedural": 360,
        "troubleshooting": 180,
    }


def test_feasible_decoder_and_grid_size() -> None:
    space = SearchSpace()
    assert space.size == 1_866_150
    rng = np.random.default_rng(42)
    decoded = space.decode(rng.random((20_000, 5)))
    assert all(space.is_feasible(row) for row in decoded)
    assert not np.any((decoded[:, 3] > 0) & (decoded[:, 3] < decoded[:, 0]))


def test_simulator_is_deterministic_within_block() -> None:
    simulator = SmartTutorSimulator(generate_benchmark(2026072401), SimulatorScenario())
    config = np.asarray([3, 384, 0.22, 13, 1024], dtype=float)
    first = simulator.evaluate_one(config)
    second = simulator.evaluate_one(config)
    assert first == second
    assert 0.0 < first.mean_quality < 1.0
    assert first.p95_latency_s > 0.0


def test_sobol_configuration_is_reproducible() -> None:
    a = qmc.Sobol(d=5, scramble=True, seed=1234, bits=30, optimization=None).random_base2(13)
    b = qmc.Sobol(d=5, scramble=True, seed=1234, bits=30, optimization=None).random_base2(13)
    assert np.array_equal(a, b)
    assert a.shape == (8192, 5)
