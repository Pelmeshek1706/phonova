# README_DEV

Developer-oriented map of the `airest` repository.

This file complements [README.md](./README.md). The main README is paper/results oriented. This document is meant to answer a different question: what is in the repo, where the important code lives, and which notebooks/folders matter for day-to-day development.

## What This Repository Contains

At a high level, this repository combines:

- research notebooks used to validate Ukrainian NLP and speech features
- an embedded [`openwillis`](./openwillis/README.md) codebase used as the main language pipeline
- experiment scripts for transcript cleanup, translation, sentiment, and downstream modeling
- reduced deliverable packages for paper sections and reproducible tabular experiments

From the main [README.md](./README.md), the major research themes are:

- sentiment analysis on Ukrainian and translated text
- POS and tense validation for Ukrainian
- syllable-count and speech-rate validation
- tangentiality, perplexity, and coherence validation
- downstream predictive modeling for clinical labels

## Suggested Reading Order

If you are new to the repo, start here:

1. [README.md](./README.md) for the research goals and reported results.
2. [README_DEV.md](./README_DEV.md) for the repository layout.
3. [`openwillis/`](./openwillis/) for the main language pipeline and core feature-extraction code.
4. Root notebooks for the main experimental workflows.
5. [`notebooks/`](./notebooks/) and task-specific folders for narrower experiments.

## Root Notebooks

These notebooks sit directly in the repository root and act as the highest-level working notebooks.

### [`whisper_pipeline.ipynb`](./whisper_pipeline.ipynb)

Practical transcription pipeline notebook. It starts from Whisper-style data acquisition and archive handling and is best read as an operational notebook for preparing or replaying ASR/transcription work.

### [`leakage_audit.ipynb`](./leakage_audit.ipynb)

Leakage proxy audit notebook for the paper. Use this when checking whether predictors or preprocessing decisions unintentionally leak label information into downstream modeling.

### [`airest_article.ipynb`](./airest_article.ipynb)

Large consolidated research notebook. It reflects much of the material described in the main README: sentiment, syllables, translation checks, validation runs, and downstream analyses. Treat it as a broad paper-construction notebook rather than a clean production pipeline.

## Main Folders

### [`openwillis/`](./openwillis/)

Main language pipeline.

This is the most important code folder in the repository. It vendors the OpenWillis stack and contains the reusable implementation used by the surrounding notebooks and scripts.

Important implementation entry points:

- [`openwillis/openwillis-speech/src/openwillis/speech/speech_attribute.py`](./openwillis/openwillis-speech/src/openwillis/speech/speech_attribute.py) handles transcript format detection and feature extraction flow for Whisper/Amazon/Vosk-style inputs.
- [`openwillis/openwillis-transcribe/src/openwillis/transcribe/speech_transcribe_whisper.py`](./openwillis/openwillis-transcribe/src/openwillis/transcribe/speech_transcribe_whisper.py) is the core WhisperX transcription wrapper.
- [`openwillis/openwillis-speech/src/openwillis/speech/util/speech/ukrainian_vader.py`](./openwillis/openwillis-speech/src/openwillis/speech/util/speech/ukrainian_vader.py) is the custom Ukrainian sentiment analyzer used by local validation scripts.

### [`notebooks/`](./notebooks/)

Supporting experiment notebooks that break the research into smaller, more focused workflows.

Notable notebooks:

- [`english-transcribed-role-alignment-pipeline.ipynb`](./notebooks/english-transcribed-role-alignment-pipeline.ipynb): aligns role-labeled interview data with Whisper-like transcript structure and writes normalized JSON outputs.
- [`consistency_features_gemma.ipynb`](./notebooks/consistency_features_gemma.ipynb): evaluates feature consistency under Gemma-based processing.
- [`ml_oppenwillis.ipynb`](./notebooks/ml_oppenwillis.ipynb): modeling notebook that imports `openwillis.speech` and `openwillis.transcribe` directly.
- [`ml_baseline.ipynb`](./notebooks/ml_baseline.ipynb): baseline tabular classification experiments with classical ML models.
- [`pos_tense.ipynb`](./notebooks/pos_tense.ipynb): POS and tense validation against UD-style annotated data.
- [`airest_article.ipynb`](./notebooks/airest_article.ipynb): duplicate/working copy of the larger article notebook.

### [`paper_sections_3_4_5_signal_retention_clean/`](./paper_sections_3_4_5_signal_retention_clean/)

Reduced paper package for Sections 3 to 5.

This folder is already documented in its own [README](./paper_sections_3_4_5_signal_retention_clean/README.md). It contains:

- compact EN and UA inputs
- one self-contained notebook for the signal-retention study
- a baseline sweep script
- result tables and figures used by that notebook

Use this folder when you want the cleaned deliverable for the EN-first to UA transfer experiments without the rest of the repository noise.

### [`autoresearch/`](./autoresearch/)

Tabular modeling package for the final classification experiments.

This folder contains:

- [`prepare.py`](./autoresearch/prepare.py) and [`prepare_legacy.py`](./autoresearch/prepare_legacy.py) for dataset preparation
- [`train.py`](./autoresearch/train.py) and [`train_legacy.py`](./autoresearch/train_legacy.py) for experiment runs
- [`merged_features.csv`](./autoresearch/merged_features.csv) as the merged tabular feature source used by the legacy workflow
- configuration and lock files for isolated experiment reproduction

See [autoresearch/README.md](./autoresearch/README.md) for the selected model families and reported dev/test metrics.

### [`sentiment_triplet_inference/`](./sentiment_triplet_inference/)

Focused sentiment comparison package for Whisper/OpenWillis JSON.

It compares:

- classic English VADER
- external `vader-ua`
- the repository’s improved Ukrainian analyzer

The main entry point is [`run_triplet_sentiment_inference.py`](./sentiment_triplet_inference/run_triplet_sentiment_inference.py).

### [`scripts/`](./scripts/)

Preprocessing utilities for transcript cleanup and translation.

Current scripts are grouped under [`scripts/preprocessing/`](./scripts/preprocessing/) and come in two parallel variants:

- Gemma-based cleanup/translation
- OpenAI-based cleanup/translation

Main responsibilities:

- role cleanup for noisy Whisper-style interview transcripts
- split turns into `participant`, `interviewer`, `mixed`, and `unknown`
- preserve timings while normalizing structure
- translate cleaned English transcripts into Ukrainian without fabricating word timings

### [`tests/`](./tests/)

Repository-level test scaffold for contract and feature-group checks.

The folder structure suggests two main test themes:

- `unit_contract/` for Whisper parsing, turn aggregation, and invalid-input behavior
- `feature_groups/` for English and Ukrainian feature-group integrity

In the current checkout, this directory mostly contains cached bytecode and fixture data rather than a full clean source test tree, so treat it as partial test scaffolding rather than a polished standalone suite.

### [`output/`](./output/) and [`tmp/`](./tmp/)

Working directories for generated artifacts, temporary exports, and intermediate JSON/CSV outputs. These are not primary source folders.

### [`graphify-out/`](./graphify-out/)

Derived graph/documentation artifacts generated by an auxiliary analysis workflow. Useful as output, not as source of truth.

## Root-Level Scripts and Files

These scripts are useful operational entry points outside the larger packages.

### [`run_specific_features.py`](./run_specific_features.py)

Runs `openwillis.speech.speech_characteristics(...)` over Whisper-like JSON and exports feature CSVs. It is a direct wrapper around the main language pipeline for targeted feature extraction.

### [`rerun_specific_features.py`](./rerun_specific_features.py)

Recomputes only selected features and merges them back into previously generated CSV outputs.

### [`run_ukrainian_vader_inference.py`](./run_ukrainian_vader_inference.py)

Small demo runner for the custom Ukrainian VADER-like analyzer.

### [`convert_table_to_whisper_like.py`](./convert_table_to_whisper_like.py)

Utility that converts tabular turn-level data into canonical Whisper-like JSON with synthesized word timestamps. Useful when downstream code expects OpenAI/Whisper-style transcript structure.

### [`test_ukrainian_vader.py`](./test_ukrainian_vader.py)

Focused unit tests for the custom Ukrainian sentiment analyzer.

### [`test_code.py`](./test_code.py)

Scratchpad-style exploratory test file. This is not a clean entry point; treat it as a development notebook in `.py` form.

### [`e_daic_leakage_and_translation_report.md`](./e_daic_leakage_and_translation_report.md)

Standalone report for the E-DAIC leakage and translation validation work referenced by the broader paper narrative.

### [`SyllableNucleiv3.praat`](./SyllableNucleiv3.praat)

Praat script used in syllable-related validation discussed in the main README.

## How the Pieces Fit Together

The practical flow of the repository is:

1. Transcribe or normalize data into Whisper-like JSON.
2. Clean speaker roles and, when needed, translate English transcripts to Ukrainian.
3. Run `openwillis` feature extraction over the transcript JSON.
4. Validate language-specific features in notebooks.
5. Export tabular features for downstream modeling and paper figures.

In other words:

- `openwillis/` is the reusable pipeline code
- root notebooks and `notebooks/` are the research workbenches
- `scripts/` are preprocessing helpers
- `autoresearch/` and `paper_sections_3_4_5_signal_retention_clean/` are focused downstream experiment packages

## If You Need One Starting Point

For code changes, start in [`openwillis/`](./openwillis/).

For understanding the full research workflow, start with:

- [README.md](./README.md)
- [`airest_article.ipynb`](./airest_article.ipynb)
- [`whisper_pipeline.ipynb`](./whisper_pipeline.ipynb)

For reproducible downstream modeling, start with:

- [`autoresearch/`](./autoresearch/)
- [`paper_sections_3_4_5_signal_retention_clean/`](./paper_sections_3_4_5_signal_retention_clean/)
