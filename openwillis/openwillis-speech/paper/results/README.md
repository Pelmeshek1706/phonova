# Paper Result Tables And Source Data

This folder contains public aggregate/source-data artifacts for the current paper supplement.

## Manuscript Tables

`paper/results/tables/` contains CSV versions of manuscript Tables 1-21. The index is `table_manifest.csv`, which records the linked table number, row count, column count, and whether a file contains participant rows. All entries are aggregate or feature-level summaries; none contain raw transcripts or participant-level feature rows.

## Feature And Model Source Data

`paper/results/source_data/` contains companion files for the paper feature and model-result contract:

- `paper_82_feature_inventory.csv`: the fixed 82-feature paper inventory.
- `paper_82_feature_to_phonova_mapping.csv`: paper feature names mapped to Phonova summary columns.
- `cohort_label_split_summary.csv`: aggregate split and label counts used in the manuscript.
- `model_metric_bootstrap_ci_summary.csv`: bootstrap confidence-interval source values for the model tables.
- `language_specific_performance_summary.csv`: EN/UA performance summaries for the nested feature tiers.
- `proxy_timing_sensitivity_exact.csv`, `structural_timing_sensitivity_exact_public.csv`, `leakage_outcomes_exact.csv`, and `transcript_condition_probe_source.csv`: sensitivity and error-analysis source tables.

The full participant-level E-DAIC feature matrices and prediction rows are not included here because they are governed by the E-DAIC data-use terms. The public substitutes are the complete feature inventory, the feature-level consistency audit, checksums for frozen restricted artifacts, and the aggregate model/source tables.

## Figure Source

`paper/results/figure_source/` contains source CSVs for the paper figures:

- `figure01_signal_retention_source.csv`
- `figure02_and_figure04_feature_importance_all_coefficients.csv`
- `figure02_feature_importance_top15_coefficients.csv`
- `figure03_sentiment_ablation_source.csv`

The current manuscript reports PR-AUC and AUROC values in tables, but does not include ROC or PR curve panels. The metric source values are therefore in the model and language-specific performance CSVs rather than curve-coordinate files.

## Checksums

`paper/results/result_file_checksums.csv` records SHA-256 checksums for all files under `paper/results/`.
