# AiRest Autoresearch Program

You are an autonomous coding research agent working in a small repo inspired by `karpathy/autoresearch`.

Your job is to improve the tabular clinical classifier in `train.py` on `merged_features.csv`.

## Objective

Increase `results.json` -> `summary.primary_score`.

This project is optimized for PR AUC on imbalanced labels.
In this repo, PR AUC is computed with `sklearn.metrics.average_precision_score`.

Primary score:
- `mean_dev_pr_auc`
- This is the mean dev PR AUC across:
  - `Depression_label`
  - `PTSD_label`

Secondary tie-breakers:
- higher `summary.min_dev_pr_auc`
- then simpler and more defensible code
- then higher `summary.mean_dev_roc_auc`

Higher is always better.

## Repo Structure

- `prepare.py`
  - Fixed validation and dataset summary.
  - Read-only.
- `train.py`
  - Main experiment file.
  - This is the only file you may edit.
- `merged_features.csv`
  - Fixed dataset.
  - Read-only.
- `program.md`
  - Human-authored operating instructions.
  - Read-only.
- `pyproject.toml`
  - Fixed dependencies.
  - Read-only.
- `results.tsv`
  - Append-only local experiment log.
  - Do not commit.
- `results.json` and `artifacts/`
  - Generated outputs.
  - Do not commit.

## Hard Rules

- Only edit `train.py`.
- Never edit `prepare.py`, `program.md`, `pyproject.toml`, or the dataset.
- Never use `Participant`, `split`, `Depression_label`, or `PTSD_label` as features.
- Never use the `test` split for model selection, threshold tuning, acceptance decisions, or early stopping.
- Do not install new packages.
- Keep the run deterministic and reasonably fast on local hardware.
- Prefer small, surgical changes over broad rewrites.
- No data leakage, no split leakage, no participant-specific hacks, no hard-coded dev/test row logic.
- Optimize for PR AUC first. Threshold-dependent metrics such as F1 or balanced accuracy are secondary diagnostics only.

## Workflow

1. Read `prepare.py` and `train.py`.
2. Run:
   - `uv run prepare.py`
3. If `results.tsv` does not exist, create it with this header:

```tsv
commit	primary_score	min_dev_ap	depression_dev_ap	ptsd_dev_ap	status	description
```

4. Run the baseline:
   - `uv run train.py > run.log 2>&1`
5. Read `results.json`.
6. Record the baseline in `results.tsv`.
7. Then iterate forever until the human stops you.

## Acceptance Rule

Default experiment command:
- `uv run train.py > run.log 2>&1`

Focused debugging commands are allowed:
- `uv run train.py --target Depression_label > run.log 2>&1`
- `uv run train.py --target PTSD_label > run.log 2>&1`

But:
- you may use single-target runs only for debugging
- you may accept or reject a code change only after a full `uv run train.py`

Accept a change if:
- `summary.primary_score` improved by at least `0.001`

If the primary score is within `0.001` of the previous best:
- keep it only if code is clearly simpler or safer
- or if `summary.min_dev_pr_auc` improved by at least `0.002`

Otherwise:
- revert the change completely

## Crash Handling

If a run crashes:
- inspect `run.log`
- make one or two targeted fixes
- rerun once
- if it still fails, revert the change and log the failed attempt in `results.tsv`

Do not leave the repo in a broken state.

## Test Split Policy

During research:
- do not run `--include-test`
- do not inspect test metrics

Only when the human explicitly asks for final evaluation:
- run `uv run train.py --include-test > run.log 2>&1`
- then report the final held-out test numbers

## What To Optimize First

Prioritize these ideas before more exotic changes:

- stronger tree ensembles
- better regularization for linear models
- simple target-specific model choices
- robust feature filtering
- calibration when it materially helps ranking quality
- lightweight blending of strong models
- simple feature engineering from existing columns
- better threshold tuning for classification metrics

Important:
- threshold tuning must never be treated as a way to improve PR AUC
- accept or reject changes using dev PR AUC, not F1, not balanced accuracy, not ROC AUC

Prefer:
- interpretable gains
- short diffs
- stable logic

Avoid:
- giant search spaces
- slow nested tuning
- fragile heuristics
- changes that make the script hard to read

## Fast Result Reading

After each run, inspect `results.json`.

The key fields are:
- `summary.primary_score`
- `summary.min_dev_pr_auc`
- `targets.Depression_label.selected_model`
- `targets.Depression_label.dev.pr_auc`
- `targets.PTSD_label.selected_model`
- `targets.PTSD_label.dev.pr_auc`

If needed, extract them quickly with:

```bash
python - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("results.json").read_text())
print("primary_score:", payload["summary"]["primary_score"])
print("min_dev_pr_auc:", payload["summary"]["min_dev_pr_auc"])
for target, info in payload["targets"].items():
    print(target, info["selected_model"], info["dev"]["pr_auc"])
PY
```

## Logging

Every experiment must append one line to `results.tsv`.

Use:
- `status=keep` for accepted runs
- `status=revert` for rejected runs
- `status=crash` for failed runs

`description` should be a short phrase describing the hypothesis, for example:
- `more trees + smaller leaves`
- `drop low-variance features`
- `blend rf and extra trees`

## Completion Condition

Do not stop after one improvement.

Continue iterating until:
- the human interrupts you
- or you have clearly plateaued after several meaningful attempts

When you stop, leave behind:
- the best `train.py`
- a clean working tree
- a filled `results.tsv`
- the latest `results.json`
