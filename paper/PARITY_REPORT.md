# Paper Parity Verification

Status: **passed**

This report verifies paper scripts against the frozen paper artifacts
without re-running transformer sentiment inference.

## Selected Features
- EN: 275 rows, 82 features, 6 metadata columns, cell differences = 0
- UK: 275 rows, 82 features, 6 metadata columns, cell differences = 0

## Model Metrics
- Depression_label:
  - n_features: actual 82, reference 82, abs diff 0
  - threshold: actual 0.5320324848487864, reference 0.5320324848487864, abs diff 0
  - test_pr_auc: actual 0.7290024732412276, reference 0.7290024732412276, abs diff 0
  - test_f1_macro: actual 0.7888386123680242, reference 0.7888386123680242, abs diff 0
  - test_f1_binary: actual 0.7058823529411765, reference 0.7058823529411765, abs diff 0
  - test_accuracy: actual 0.8214285714285714, reference 0.8214285714285714, abs diff 0
  - test_roc_auc: actual 0.7873303167420815, reference 0.7873303167420815, abs diff 0
- PTSD_label:
  - n_features: actual 82, reference 82, abs diff 0
  - threshold: actual 0.5720423927420021, reference 0.5720423927420021, abs diff 0
  - test_pr_auc: actual 0.7439755267542281, reference 0.7439755267542281, abs diff 0
  - test_f1_macro: actual 0.7213930348258707, reference 0.7213930348258707, abs diff 0
  - test_f1_binary: actual 0.6666666666666666, reference 0.6666666666666666, abs diff 0
  - test_accuracy: actual 0.7321428571428571, reference 0.7321428571428571, abs diff 0
  - test_roc_auc: actual 0.8068027210884353, reference 0.8068027210884353, abs diff 0

## Predictions
- Depression_label: 56 rows, y_true diffs = 0, y_pred diffs = 0, max probability abs diff = 4.77e-09
- PTSD_label: 56 rows, y_true diffs = 0, y_pred diffs = 0, max probability abs diff = 2.84e-09

## Extraction Spot Check
- Participant 300 structural/timing extraction from the cleaned role-labeled
  Whisper-like JSON matched the frozen paper row for the checked non-sentiment
  columns; max absolute difference = 3.77e-15.
