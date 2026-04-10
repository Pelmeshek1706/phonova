# program_legacy.md

Read this file fully before making any changes.

## Scope

You are optimizing the single-dataset training workflow defined in `train_legacy.py`.

You may edit only:
- `train_legacy.py`

You may also create or update runtime artifacts only as part of normal experiment execution:
- `results.tsv`
- files inside `artifacts/`

Do not edit unrelated files.
Do not create alternative dataset files.
Do not add a second dataset source.

---

## Dataset policy

This workflow uses **exactly one dataset file**:
- `merged_labels.csv`

All experiments must run on the same merged dataset passed through:
- `--dataset-path /path/to/merged_labels.csv`

Do not search over multiple datasets.
Do not compare `B0`, `L`, `B0+L`, or any other dataset variants.
The only allowed source of data is the single merged CSV file.

---

## Fixed feature policy

Use **all available features** from `merged_labels.csv` except the fixed excluded and redundant columns below.

Always exclude:
- `Participant`
- `split`
- `Depression_label`
- `PTSD_label`
- `gender`

Always remove these redundant columns if present:
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

After this fixed cleanup, the model should train on the full remaining feature set.
Do not perform arbitrary feature subset search.
Do not switch to top-k selected features.
Do not manually keep only a small subset of features.

The expected cleaned feature set contains **83 numeric features**.
If the actual count differs, log a warning and continue only if the difference is explainable by the dataset contents.

---

## Main objective

Optimize classification performance for:
- `Depression_label`
- `PTSD_label`

Primary metric:
- **dev PR AUC**

Secondary metric:
- **dev F1 score**

Selection rule:
- Prefer higher **dev PR AUC** first.
- Use **dev F1** as a secondary criterion or tie-breaker.
- Do not prefer a worse dev PR AUC just because dev F1 is slightly higher, unless that run is explicitly marked as exploratory and justified.

---

## Critical evaluation policy

### During search / experimentation

Use only:
- train -> dev

This means:
- train on `train`
- evaluate on `dev`
- compare configs only on `dev`

Do **not** use `test` for:
- model selection
- threshold tuning
- preprocessing choice
- hyperparameter search
- deciding whether a change is good or bad

### Final validation only

Use `test` only when you are confident that the current configuration is a serious final candidate.

This means:
- train on `train+dev`
- evaluate once on `test`
- log the result clearly as final validation

If the final test result is disappointing, do **not** optimize against test.
Go back to experimentation on train/dev.

---

## Workflow rules

For each target label:
1. Inspect current `train_legacy.py` behavior.
2. Choose one meaningful experimental change.
3. Run in `mode=experiment`.
4. Log the run to `results.tsv`.
5. Compare against previous runs using dev PR AUC first and dev F1 second.
6. Continue iterating.
7. Only when the configuration looks strong and stable, run one final validation in `mode=final`.

Do not stop after one iteration.
Do not repeatedly rerun nearly identical settings without a reason.
Keep the search systematic.
Prefer small coherent changes over chaotic rewrites.

---

## Allowed search directions

You should actively explore:
- model family
- model hyperparameters
- preprocessing strategies
- imputation strategy
- numeric scaling on/off
- class weighting
- linear-model tuning on/off
- tree-model tuning on/off
- CV settings
- XGBoost usage when available

You must **not** explore:
- multiple datasets
- arbitrary feature subsets
- top-k feature selection branches
- hand-picked tiny feature groups

You must respect the fixed feature policy above.

---

## Logging requirements

Every training run must append structured results to `results.tsv`.
The logged data must be sufficient to reconstruct the experiment history and plot metric progression.

At minimum, each row must document:
- `run_id`
- timestamp
- target label
- dataset path
- mode (`experiment` or `final`)
- train split(s)
- eval split
- model
- model variant
- hyperparameters
- preprocessing configuration
- feature count
- removed fixed columns
- PR AUC
- F1
- short notes / experiment tag

---

## Plotting requirement

The workflow must support plotting metrics from `results.tsv`.
The graph must be based on logged runs, not manual notes.

At minimum, support plotting:
- PR AUC vs logged row order
- F1 vs logged row order

It must be possible to filter plots by:
- target label
- eval split (`dev` or `test`)
- mode (`experiment` or `final`)

---

## End-of-search reporting

At the end, summarize:
- best dev configuration for `Depression_label`
- best dev configuration for `PTSD_label`
- best model for each target
- best preprocessing setup for each target
- whether final test validation was run
- if final test validation was run, report final test PR AUC and F1
- confirm that test was not used during iterative search
- confirm that the fixed feature policy was preserved

---

## Command examples

Use the same merged dataset in every example below.

### 1) Experiment on dev with a broad default search for PTSD

```bash
python train_legacy.py train \
  --dataset-path /path/to/merged_labels.csv \
  --target-col PTSD_label \
  --mode experiment \
  --experiment-tag ptsd_default_search \
  --notes "broad search on single merged dataset" \
  --results-tsv artifacts/results.tsv \
  --save-dir artifacts \
  --plot-after-run
```

### 2) Experiment on dev with only linear models for PTSD

```bash
python train_legacy.py train \
  --dataset-path /path/to/merged_labels.csv \
  --target-col PTSD_label \
  --mode experiment \
  --models logreg sgd \
  --logreg-c-grid 0.01,0.1,1,10 \
  --sgd-alpha-grid 0.000001,0.00001,0.0001,0.001 \
  --imputer-strategy median \
  --scale-numeric \
  --experiment-tag ptsd_linear_search \
  --notes "linear models only" \
  --results-tsv artifacts/results.tsv \
  --save-dir artifacts \
  --plot-after-run
```

### 3) Experiment on dev with tree-based models for PTSD

```bash
python train_legacy.py train \
  --dataset-path /path/to/merged_labels.csv \
  --target-col PTSD_label \
  --mode experiment \
  --models tree rf xgb \
  --tree-max-depth-grid 3,5,8,None \
  --rf-n-estimators-grid 200,500,800 \
  --rf-max-depth-grid 5,10,None \
  --xgb-n-estimators-grid 200,500 \
  --xgb-max-depth-grid 3,6,10 \
  --experiment-tag ptsd_tree_search \
  --notes "tree-heavy configuration" \
  --results-tsv artifacts/results.tsv \
  --save-dir artifacts \
  --plot-after-run
```

### 4) Experiment on dev with no numeric scaling for Depression

```bash
python train_legacy.py train \
  --dataset-path /path/to/merged_labels.csv \
  --target-col Depression_label \
  --mode experiment \
  --models logreg sgd svm rf \
  --no-scale-numeric \
  --imputer-strategy mean \
  --experiment-tag depression_no_scaling \
  --notes "check whether scaling hurts or helps" \
  --results-tsv artifacts/results.tsv \
  --save-dir artifacts \
  --plot-after-run
```

### 5) Experiment on dev with no class weighting for Depression

```bash
python train_legacy.py train \
  --dataset-path /path/to/merged_labels.csv \
  --target-col Depression_label \
  --mode experiment \
  --class-weight none \
  --models logreg svm rf \
  --experiment-tag depression_no_class_weight \
  --notes "compare against balanced weighting" \
  --results-tsv artifacts/results.tsv \
  --save-dir artifacts \
  --plot-after-run
```

### 6) Final validation on test for PTSD

Run this only after the dev result is clearly strong and stable.

```bash
python train_legacy.py train \
  --dataset-path /path/to/merged_labels.csv \
  --target-col PTSD_label \
  --mode final \
  --models logreg rf xgb \
  --experiment-tag ptsd_final_candidate \
  --notes "final validation on test" \
  --results-tsv artifacts/results.tsv \
  --save-dir artifacts \
  --plot-after-run
```

### 7) Final validation on test for Depression

```bash
python train_legacy.py train \
  --dataset-path /path/to/merged_labels.csv \
  --target-col Depression_label \
  --mode final \
  --models logreg svm rf \
  --experiment-tag depression_final_candidate \
  --notes "final validation on test" \
  --results-tsv artifacts/results.tsv \
  --save-dir artifacts \
  --plot-after-run
```

### 8) Plot PTSD dev progression from results.tsv

```bash
python train_legacy.py plot-results \
  --results-tsv artifacts/results.tsv \
  --out-path artifacts/metrics_progression_ptsd_dev.svg \
  --target PTSD_label \
  --eval-split dev
```

### 9) Plot Depression dev progression from results.tsv

```bash
python train_legacy.py plot-results \
  --results-tsv artifacts/results.tsv \
  --out-path artifacts/metrics_progression_depression_dev.svg \
  --target Depression_label \
  --eval-split dev
```

### 10) Plot final test progression for PTSD

```bash
python train_legacy.py plot-results \
  --results-tsv artifacts/results.tsv \
  --out-path artifacts/metrics_progression_ptsd_test.svg \
  --target PTSD_label \
  --eval-split test \
  --mode final
```

---

## Final principle

Use one merged dataset.
Use the fixed feature policy.
Experiment on `dev`.
Validate on `test` only when confident.
If not confident, keep experimenting.
