# Reproducibility Protocol

## Experimental blocks

Benchmark seeds are `2026072401` through `2026072412`. Optimizer seed IDs are
1 through 10. Within each benchmark-seed cell, all methods share the same
60-point Latin-hypercube initial design and seed label.

## Unique-information budget

Every method evaluates exactly 3,000 distinct feasible decoded configurations.
Duplicate evolutionary proposals are rejected and regenerated, so D-NSGA-II
and D-SPEA2 are novelty-filtered variants. Raw proposal counts and runtime are
not controlled.

## Reference fronts

For each benchmark, SciPy generates `2^19` LMS+shift-scrambled Sobol points.
After mixed-discrete decoding and deduplication, the first 250,000 unique
feasible configurations form the independent sample. Nondominated fronts from
the first 100,000 and all 250,000 configurations support sensitivity analyses.
Leave-one-method-out reference fronts combine the independent sample with
competitor candidates while excluding the method being evaluated.

## Statistical inference

- Seed results are averaged within each of the 12 benchmark blocks before
  Wilcoxon and sign-flip inference.
- Exact sign-flip tests enumerate all `2^12 = 4096` assignments.
- Hierarchical percentile-bootstrap intervals resample benchmark blocks and
  then optimizer seeds within selected blocks.
- Confirmatory Holm correction spans 12 adaptive-versus-non-adaptive tests.

## Decision rules

The candidate set is the union of distinct configurations appearing on any
D-NSGA-II or D-SPEA2 nondominated set from training blocks 1-8 across all ten
seeds. Every candidate is reevaluated on all eight training blocks. The
manuscript defines the ideal-point, quality-floor, latency-ceiling, and
class-protected rules and deterministic tie breaks.

## Known limitation

The simulator has not been calibrated against SmartTutor telemetry or expert
answer judgments. Production validation requires staged configuration tests and
rank-correlation analysis between simulated and observed outcomes.
