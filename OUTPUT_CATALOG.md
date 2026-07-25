# Output Catalog

## Aggregated manuscript evidence

- `algorithm_summary_multiseed_v4.csv`: method-level means across 120 matched cells.
- `confirmatory_tests_multiseed_v4.csv`: 12 Holm-corrected confirmatory comparisons.
- `exploratory_adaptive_multiseed_v4.csv`: D-NSGA-II versus D-SPEA2 comparisons.
- `hypervolume_sensitivity_multiseed_v4.csv`: three normalization schemes.
- `reference_metric_sensitivity_v4.csv`: independent-100k, independent-250k,
  and leave-one-method-out IGD+/epsilon results.
- `decision_rules_training_v4.csv`: selected rules and training metrics.
- `decision_rules_heldout_summary_v4.csv`: held-out decision-rule performance.
- `heldout_class_metrics_v4.csv`: query-class quality and latency evidence.
- `scenario_parameter_stability_v4.csv`: selected-control ranges and frequencies.
- `scenario_spearman_v4.csv`: 64-scenario marginal rank associations.
- `experiment_metadata_v4.json`: versions, seeds, Sobol parameters, grid size,
  normalization, and experiment counts.

## Raw and intermediate evidence

- `runs/block_XX/seed_YY/candidates.csv.gz`: every evaluated candidate.
- `runs/block_XX/seed_YY/fronts.csv.gz`: accumulated nondominated sets.
- `runs/block_XX/seed_YY/summary.csv`: cell-level metrics and proposal counts.
- `references/block_XX/independent_sample_250k.csv.gz`: independent samples.
- `scenarios/scenario_XX/front.csv.gz`: fixed-catalog scenario fronts.
