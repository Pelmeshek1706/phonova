# program_legacy.md

Read this file fully before making any changes.

## Scope

You are optimizing the legacy training workflow defined in `train_legacy.py`.

You may edit only:
- `train_legacy.py`

You may also create or update runtime artifacts only as part of normal experiment execution:
- `results.tsv`
- files inside `artifacts/`

Do not edit unrelated files.
Do not edit other training scripts.

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
- feature-selection choice
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
- dataset variant choice: `B0`, `L`, `B0+L`
- model family
- model hyperparameters
- preprocessing strategies
- class balancing settings
- importance-based selected-feature branch settings
- `importance_top_k`
- `importance_corr_thr`
- tree tuning settings
- threshold choices when appropriate
- whether importance-based selected variants help or hurt
- whether demographics should be dropped or kept, but only when consistent with the experiment goal and leakage policy

You must respect leakage-safe behavior already implemented in the code.

---

## Logging requirements

Every training run must append structured results to `results.tsv`.
The logged data should be sufficient to reconstruct the experiment history and plot metric progression.

At minimum, each run should document:
- `run_id`
- timestamp
- target label
- dataset path
- dataset variant if known
- mode (`experiment` or `final`)
- train split(s)
- eval split
- model
- model variant
- hyperparameters
- preprocessing notes
- dev/test PR AUC
- dev/test F1
- short description of what changed
- whether the run is exploratory, best-so-far, or final validation

---

## Plotting requirement

The workflow must support plotting metrics from `results.tsv`.
The graph should be based on logged runs, not manual notes.

At the end of a search phase, or after selected runs, generate plots showing metric progression across runs.
Prefer plotting at least:
- PR AUC vs run order
- F1 vs run order

It should be possible to filter plots by:
- target label
- eval split (`dev` or `test`)

---

## When a configuration is good enough to validate on test

Only run final validation when all of the following are true:
- the dev result is among the best seen so far
- the configuration is stable and sensible
- the code looks like a realistic final candidate
- there is a clear reason to believe this configuration is worth validating

Do not use test as a frequent checkpoint.
Test is a confirmation gate, not a search tool.

---

## End-of-search reporting

At the end, summarize:
- best dev configuration for `Depression_label`
- best dev configuration for `PTSD_label`
- best dataset variant for each target
- best model for each target
- best threshold if used
- whether selected-feature variants helped
- whether final test validation was run
- if final test validation was run, report final test PR AUC and F1
- confirm that test was not used during iterative search

---

## Command examples

### 1) Build legacy datasets

```bash
python train_legacy.py build-datasets \
  --detailed-labels-path /path/to/detailed_labels.csv \
  --data-dir /path/to/result_openwillis_dir \
  --out-dir /path/to/output_dir \
  --language eng \
  --iteration 3
```

### 2) Experiment on dev for PTSD

Use this during search.
This is the default search-style run.

```bash
python train_legacy.py train \
  --dataset-path /path/to/dataset_B0_plus_L_gemma_eng_full3.csv \
  --target-col PTSD_label \
  --mode experiment \
  --importance-top-k 5 \
  --importance-corr-thr 0.90 \
  --experiment-tag b0l_ptsd_search \
  --notes "baseline with selected branch" \
  --results-tsv artifacts/results.tsv \
  --save-dir artifacts \
  --plot-after-run
```

### 3) Experiment on dev for Depression

```bash
python train_legacy.py train \
  --dataset-path /path/to/dataset_B0_plus_L_gemma_eng_full3.csv \
  --target-col Depression_label \
  --mode experiment \
  --importance-top-k 5 \
  --importance-corr-thr 0.90 \
  --experiment-tag b0l_depression_search \
  --notes "depression dev search" \
  --results-tsv artifacts/results.tsv \
  --save-dir artifacts \
  --plot-after-run
```

### 4) Final validation on test for PTSD

Run this only after you are confident in the candidate.

```bash
python train_legacy.py train \
  --dataset-path /path/to/dataset_B0_plus_L_gemma_eng_full3.csv \
  --target-col PTSD_label \
  --mode final \
  --experiment-tag final_candidate \
  --notes "final validation on test" \
  --results-tsv artifacts/results.tsv \
  --save-dir artifacts \
  --plot-after-run
```

### 5) Final validation on test for Depression

```bash
python train_legacy.py train \
  --dataset-path /path/to/dataset_B0_plus_L_gemma_eng_full3.csv \
  --target-col Depression_label \
  --mode final \
  --experiment-tag final_candidate \
  --notes "final validation on test" \
  --results-tsv artifacts/results.tsv \
  --save-dir artifacts \
  --plot-after-run
```

### 6) Plot PTSD dev progression from results.tsv

```bash
python train_legacy.py plot-results \
  --results-tsv artifacts/results.tsv \
  --out-path artifacts/metrics_progression_ptsd_dev.svg \
  --target PTSD_label \
  --eval-split dev
```

### 7) Plot Depression dev progression from results.tsv

```bash
python train_legacy.py plot-results \
  --results-tsv artifacts/results.tsv \
  --out-path artifacts/metrics_progression_depression_dev.svg \
  --target Depression_label \
  --eval-split dev
```

### 8) Plot PTSD test progression from results.tsv

```bash
python train_legacy.py plot-results \
  --results-tsv artifacts/results.tsv \
  --out-path artifacts/metrics_progression_ptsd_test.svg \
  --target PTSD_label \
  --eval-split test
```

---

## Final principle

Experiment on `dev`.
Validate on `test` only when confident.
If not confident, keep experimenting.
