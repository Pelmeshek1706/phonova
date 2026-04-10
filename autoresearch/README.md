# AiRest Autoresearch Results

This directory contains the final tabular classification experiment for `train.py`.

## Final Setup

The final model uses the cleaned full feature set:

- All available features except:
  - `Participant`
  - `split`
  - `Depression_label`
  - `PTSD_label`
  - `gender`
- Removed redundant columns:
  - `mean_pre_turn_pause`
  - `mean_turn_length_minutes`
  - `mean_turn_length_words`
  - `turn_to_turn_tangeniality_mean`
  - `turn_to_previous_speaker_turn_similarity_mean`
  - `first_order_sentence_tangeniality_mean`
  - `second_order_sentence_tangeniality_mean`
  - `semantic_perplexity_mean`
  - `semantic_perplexity_5_mean`
  - `semantic_perplexity_11_mean`
  - `semantic_perplexity_15_mean`
  - `semantic_perplexity_11_var`
  - `semantic_perplexity_15_var`

The remaining feature set contains 83 numeric features.

## Preprocessing

All branches use median imputation for missing values.

Target-specific preprocessing:

- `Depression_label`
  - `rf_bal`: median imputation only
  - `logreg_en_quantile`: median imputation + `QuantileTransformer(output_distribution="normal")`
  - `raw_hgbdeep`: median imputation only

- `PTSD_label`
  - `raw_bnb_standard`: median imputation + `StandardScaler()`
  - `lsvc_cal_standard`: median imputation + `StandardScaler()` + calibrated `LinearSVC`
  - `raw_hgb3`: median imputation only

## Final Dev Results

### Depression_label

- Selected model: `blend_rf_bal_logreg_en_quantile_raw_hgbdeep_20_30_50`
- Dev PR AUC: `0.453796`
- Dev F1: `0.486486`
- Dev ROC AUC: `0.693182`
- Threshold: `0.176062`

### PTSD_label

- Selected model: `blend_raw_bnb_standard_lsvc_cal_standard_raw_hgb3_38_22_40`
- Dev PR AUC: `0.663086`
- Dev F1: `0.611111`
- Dev ROC AUC: `0.752640`
- Threshold: `0.500000`

## Final Test Results

One final test evaluation was run with `--include-test`.

### Depression_label

- Test PR AUC: `0.638230`
- Test F1: `0.521739`
- Test ROC AUC: `0.725490`

### PTSD_label

- Test PR AUC: `0.694103`
- Test F1: `0.578947`
- Test ROC AUC: `0.761905`

## Notes

- Test data was used only once for the final evaluation.
- `results.tsv` contains the full experiment history for this directory.
- `results.json` contains the latest selected models and metrics.
