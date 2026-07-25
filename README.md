# SmartTutor Full-Revision Simulation Benchmark

This release is the complete code-and-data evidence package for the full
revision of the SCIN-2026 manuscript:

**Evolutionary Multi-Objective Search for Retrieval-Augmented Tutor
Configuration: A Reproducible SmartTutor Simulation Benchmark**

## Scope and evidence boundary

The package contains a synthetic, author-designed simulation benchmark. It does
not contain learner records, production prompts, SmartTutor telemetry, or human
answer judgments. The numerical findings support algorithm comparison only
within the declared benchmark and surrogate-model assumptions.

## Full-revision design

- 12 independent benchmark realizations, each with 1,200 requests.
- Exact class composition per block: 420 factual, 360 procedural, 240
  conceptual, and 180 troubleshooting requests.
- 10 matched optimizer seeds per benchmark.
- Four methods: D-NSGA-II, D-SPEA2, LMS+shift-scrambled Sobol, and random
  search.
- 3,000 unique feasible configurations per method and matched cell.
- 120 matched benchmark-seed cells per method.
- Independent reference designs with 250,000 unique feasible configurations per
  benchmark; 100,000-point and leave-one-method-out sensitivity checks.
- 64 fixed-catalog simulator scenarios.
- Explicit decision rules, class-protected selection, hierarchical bootstrap,
  Wilcoxon tests, exact sign-flip tests, and Holm correction.

## Key directories

- `src/smarttutor_revision/`: simulator, search space, algorithms, Pareto
  indicators, benchmark generator, and scenario definitions.
- `scripts/`: task-level and end-to-end reproduction scripts.
- `outputs_v4/runs/`: all candidate, front, convergence, and summary evidence
  for 12 x 10 matched cells.
- `outputs_v4/references/`: independent 100k and 250k samples/fronts.
- `outputs_v4/scenarios/`: all 64 scenario-specific fronts and selected profiles.
- `outputs_v4/*.csv`: aggregated statistical, decision-rule, class, scenario,
  and metadata outputs used in the manuscript.
- `figures/`: manuscript figures generated from the archived outputs.
- `tests/`: core reproducibility and feasibility tests.

## Environment

Python 3.11 or later is required. The archived environment uses:

- NumPy 2.3.5
- pandas 2.2.3
- SciPy 1.17.0
- Matplotlib 3.10.8

Install:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Verify the archived release

The full outputs are already included. Run the lightweight checks:

```bash
pytest -q
sha256sum -c MANIFEST.sha256
```

## Reproduce all experiments

```bash
bash scripts/reproduce_all.sh
```

This is computationally intensive. Existing files are skipped by the task
scripts, which permits interrupted runs to resume.

## Sobol implementation

The comparator and independent reference designs use
`scipy.stats.qmc.Sobol` with `scramble=True`, `bits=30`,
`optimization=None`, and `random_base2`. SciPy documents the scramble as a
left linear matrix scramble followed by a digital random shift (LMS+shift).

## Licenses

- Code: MIT License (`LICENSE`).
- Synthetic data and documentation: CC BY 4.0 (`DATA_LICENSE`).

## Archival metadata

Before public release, replace the placeholders in `CITATION.cff` with the
GitHub repository URL and the version-specific Zenodo DOI.
