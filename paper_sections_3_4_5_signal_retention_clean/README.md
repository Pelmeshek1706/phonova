# Clean Package

This folder is a reduced deliverable package for the Sections 3-5 notebook.

It retains only:

- compact EN and UA inputs
- the consistency table
- the self-contained notebook
- the Section 0 baseline sweep script
- the Section 0 baseline sweep result files used by the notebook

## Layout

- `data/inputs/`
  - `summary_en_3_4_2026_expanded_b0_union_features.csv`
  - `summary_ukr_3_4_2026_expanded_b0_union_features.csv`
  - `consistency_en_ua_all_plus_sentiment_hist.csv`
- `scripts/`
  - `run_3_4_2026_expanded_stable05_full_vader_model_family_sweep.py`
- `notebooks/`
  - `sections_3_4_5_en_to_ua_signal_retention.ipynb`
- `results/baseline_sweep/`
  - `best_by_family_target.csv`
  - `best_by_target.csv`
  - `run_info.json`

## Notes

- The notebook is now self-contained and does not import local helper scripts.
- Section 0 reads the baseline sweep result files from `results/baseline_sweep/`.
