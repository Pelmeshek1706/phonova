# Paper GitHub Release Checklist

This checklist tracks what still needs to be added before the Phonova paper branch is ready as a GitHub release or reviewer supplement.

## Current status

- [x] Branch created from `main`: `paper_illia_18_04`.
- [x] Paper feature contract is present in the current schema: `82/82` paper features.
- [x] Current Phonova output schema produces `112` summary-level columns, `39` turn-level columns, and `17` word-level columns.
- [x] Missing paper feature families were implemented in the pipeline:
  - phrase-count summaries: `num_phrases`, `num_one_word_phrases`, `mean_phrases_per_turn`
  - VADER distribution features: `vader_hist_bin_01` through `vader_hist_bin_23`, entropy, tail masses, neutral mass, participant utterance count
  - first-person VADER features: `first_person_sentiment_positive_vader`, `first_person_sentiment_negative_vader`, `first_person_sentiment_overall_vader`
- [x] Unit tests added for paper schema coverage, phrase counts, raw Whisper segment extraction, and VADER distribution summaries.
- [x] Local test suite passed with `PYTHONPATH=src pytest -q`.
- [x] Save the full parity report as a tracked artifact: `paper/PARITY_REPORT.md`.

## Package documentation to add

- [ ] Add a root `README.md` for `phonova` because the package currently has no user-facing README in this repo.
- [ ] Add a paper quickstart showing how to call `speech_characteristics(...)` for Whisper JSON with:
  - `language="en"` or `language="uk"`
  - `speaker_label="participant"`
  - `whisper_turn_mode`
  - `option="coherence"` versus `option="simple"`
- [ ] Document the expected Whisper JSON schema, including `segments`, `words`, speaker/role fields, `text`, and `source_text_en`.
- [ ] Document the exact paper feature subset: all `82` paper model features, separated from the full `112` summary outputs.
- [ ] Add a feature dictionary with definitions, units, aggregation level, and missing-value behavior.
- [ ] Document language support and dependencies for English and Ukrainian sentiment resources.
- [x] Remove or explain debug console output from `speech_characteristics` before release.

## Paper pipeline code to add

- [x] Add a single-command feature extraction script under `paper/scripts/`: `paper/scripts/extract_paper_features.py`.
- [x] Add a script that merges extracted summary features with labels and metadata: `paper/scripts/merge_features_with_labels.py`.
- [x] Add a script that selects the exact `82` paper features from the full Phonova output: `paper/scripts/select_paper_features.py`.
- [x] Add model-training and evaluation scripts that regenerate the paper metrics from saved splits: `paper/scripts/train_evaluate_models.py`.
- [x] Add figure/table generation scripts for the paper and supplement: `paper/scripts/make_tables_and_figures.py`.
- [x] Add deterministic configuration: random seeds, model hyperparameters, cross-validation folds, and metric definitions: `paper/paper_reproducibility_config.json`.
- [x] Add a clean environment file or pinned requirements file for paper metric regeneration: `paper/requirements.txt`.
- [x] Add a smoke-test command that runs the full paper pipeline on a tiny example input: `PYTHONPATH=src python paper/scripts/smoke_test_pipeline.py`.
- [x] Add a reusable WhisperX ASR wrapper converted from the core notebook cells: `paper/scripts/asr_whisperx.py`.
- [x] Add Gemma leakage-cleanup and TranslateGemma Ukrainian rendering scripts:
  - `paper/scripts/gemma_role_cleanup.py`
  - `paper/scripts/translate_gemma_uk.py`
  - `paper/scripts/make_ukrainian_compat_timing.py`
- [x] Add Track A notebook-derived validation scripts:
  - `paper/scripts/track_a_validate_syllables.py`
  - `paper/scripts/track_a_validate_pos_tense.py`
  - `paper/scripts/track_a_evaluate_sentiment.py`

## Data and metadata materials

- [x] Add `DATA_AVAILABILITY.md` explaining which data can be shared, which cannot, and how to request access.
- [x] Add sanitized example input transcripts that are safe to publish.
- [x] Add expected output CSVs for the sanitized examples:
  - word-level output
  - turn-level output
  - summary-level output
- [x] Add a subject/sample manifest for any released or requestable dataset.
- [x] Add label definitions and severity/case-control coding.
- [x] Add train/test or cross-validation split files used in the paper.
- [x] Add checksums for any frozen feature tables or model-input tables.
- [x] Confirm that no private paths, PHI, raw confidential transcripts, `.DS_Store`, `__pycache__`, or local notebook outputs are included.

## Supplementary methods to add

- [ ] Transcript preprocessing protocol:
  - [x] ASR source wrapper: `paper/scripts/asr_whisperx.py`
  - [x] speaker/role cleanup: `paper/scripts/gemma_role_cleanup.py`
  - [x] interviewer removal via participant-only role view
  - [x] translation handling for Ukrainian material: `paper/scripts/translate_gemma_uk.py`
  - [x] Ukrainian proxy timing compatibility view: `paper/scripts/make_ukrainian_compat_timing.py`
  - [x] protocol note: `paper/metadata/transcript_preprocessing_protocol.md`
- [x] Track A input protocol for syllable/SPM, POS/tense, and sentiment
  validators: `paper/metadata/track_a_inputs.md`.
- [ ] Turn and phrase construction protocol.
- [ ] Speaker filtering protocol, especially participant-only feature extraction.
- [ ] VADER distribution feature protocol:
  - raw participant segment input
  - 23-bin histogram definition
  - entropy definition
  - negative/positive tail mass thresholds
  - neutral mass definition
- [ ] First-person sentiment protocol.
- [ ] Coherence feature protocol and model dependencies.
- [ ] Missing data and exclusion rules.
- [ ] Exact statistical analysis protocol for reported paper metrics.

## Supplementary tables and figures to add

- [x] Full feature table for all paper features:
  `paper/results/source_data/paper_82_feature_inventory.csv` and
  `paper/results/tables/table15_full_train_dev_feature_consistency_audit.csv`.
- [x] Mapping table from paper feature names to Phonova output columns:
  `paper/results/source_data/paper_82_feature_to_phonova_mapping.csv`.
- [x] Cohort and demographic/label summary table:
  `paper/results/source_data/cohort_label_split_summary.csv`. Participant-level
  demographics remain governed by the E-DAIC access terms.
- [x] Model performance table with confidence intervals:
  `paper/results/tables/table10_reference_en_frozen_ua_performance_ci.csv`.
- [x] Language-specific performance table:
  `paper/results/source_data/language_specific_performance_summary.csv`.
- [x] Ablation table by feature group:
  `paper/results/tables/table14_sentiment_ablation_summary.csv`,
  `paper/results/tables/table18_broader_fixed_family_ablations_depression.csv`,
  and `paper/results/tables/table19_broader_fixed_family_ablations_ptsd.csv`.
- [x] Feature importance or coefficient table:
  `paper/results/figure_source/figure02_and_figure04_feature_importance_all_coefficients.csv`.
- [x] Confirm calibration/reliability figure status: no calibration/reliability
  figure is used in the current manuscript.
- [x] ROC/PR figure source data status: no ROC/PR curve panels are used in the
  current manuscript; PR-AUC/AUROC metric source values are included in the
  model and language-specific performance CSVs.
- [x] Error-analysis or sensitivity-analysis table:
  `paper/results/tables/table07_heuristic_contamination_screen_summary.csv`,
  `paper/results/tables/table08_adjudicated_random_turn_review_summary.csv`,
  `paper/results/tables/table20_frozen_transfer_sensitivity_proxy_structure.csv`,
  and `paper/results/tables/table21_transcript_condition_lexical_probe.csv`.

## Validation and CI

- [x] Add a parity script that compares current paper scripts against frozen paper feature tables: `paper/scripts/verify_paper_parity.py`.
- [x] Add the parity output as a tracked report: `paper/PARITY_REPORT.md`.
- [ ] Run parity on a clean checkout, not only the working directory.
- [ ] Add GitHub Actions or an equivalent CI check for unit tests.
- [ ] Add a CI smoke test for the sanitized example transcript.
- [ ] Verify package installation in a fresh environment.
- [ ] Verify that package data files for Ukrainian/VADER resources are included in built distributions.

## Release and citation hygiene

- [ ] Add or update `CITATION.cff` with the paper citation when available.
- [ ] Add release notes for the paper branch.
- [x] Audit third-party lexicon/resource licenses and attribution:
  `paper/metadata/third_party_citations_and_licenses.md`.
- [x] Confirm paper-folder license compatibility for bundled sentiment
  resources: no third-party sentiment model weights or VADER lexicon files are
  bundled in `paper/`; scripts download or load dependencies at runtime.
- [ ] Add a stable tag name for the paper release.
- [ ] Add paper DOI, preprint link, or manuscript identifier after publication or submission approval.

## Suggested next order

1. Add README and paper quickstart.
2. Add feature dictionary and paper feature subset file.
3. Add sanitized example input and expected outputs.
4. Add paper scripts for feature extraction and model tables.
5. Save the parity report.
6. Add data availability, citation, and release notes.
7. Run clean-environment verification and tag the release.
