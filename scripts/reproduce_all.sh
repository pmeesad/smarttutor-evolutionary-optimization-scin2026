#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# 12 benchmark realizations x 10 matched optimizer seeds.
for block in $(seq 1 12); do
  for seed in $(seq 1 10); do
    python scripts/run_seed_task_v4.py "$block" "$seed"
  done
done

# Independent 100k/250k reference fronts.
for block in $(seq 1 12); do
  python scripts/run_reference_task_v4.py "$block"
done

python scripts/analyze_full_revision_v4.py

# Fixed-catalog scenario analysis.
for scenario in $(seq 1 64); do
  python scripts/run_scenario_catalog_task_v4.py "$scenario"
done
python scripts/aggregate_scenarios_v4.py
python scripts/make_figures_v4.py

python -m pytest -q
