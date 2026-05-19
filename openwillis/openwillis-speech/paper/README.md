# Paper Supplementary Code

This directory contains the deterministic paper configuration used by `paper/scripts/*`.

The default feature set is `paper_82`. It is intentionally fixed in `paper_reproducibility_config.json` so model inputs do not change when Phonova gains additional output columns.

Data availability, label definitions, splits, sample manifest, and checksums are
documented in `paper/DATA_AVAILABILITY.md` and `paper/metadata/`.

Sanitized example transcripts and expected outputs are in `paper/examples/`.

Aggregate manuscript result tables and figure-source CSVs are in
`paper/results/`. This includes CSVs for Tables 1-21, the fixed 82-feature
inventory, the paper-feature to Phonova-column mapping, cohort label/split
counts, model performance tables with confidence intervals, ablations,
coefficient source data, and sensitivity/error-analysis tables.

Minimal smoke test:

```bash
PYTHONPATH=src python paper/scripts/smoke_test_pipeline.py
```

Optional ASR transcription from audio uses the reusable WhisperX wrapper
`paper/scripts/asr_whisperx.py`. Install the separate ASR dependencies only in
an environment where GPU/audio transcription is needed:

```bash
pip install -r paper/asr-requirements.txt
```

`ffmpeg` must be available on `PATH`. Diarization uses pyannote and requires a
Hugging Face token via `HF_TOKEN` or `--hf-token`; use `--no-diarization` for
ASR plus word alignment only. The default input layout matches the paper
notebook convention: `<interview_id>_P/<interview_id>_AUDIO.wav`.

```bash
HF_TOKEN=hf_xxx python paper/scripts/asr_whisperx.py \
  --input-root path/to/wav_only \
  --output-dir outputs/transcribed \
  --language en \
  --model-name auto \
  --device auto
```

To validate file discovery without loading WhisperX:

```bash
python paper/scripts/asr_whisperx.py \
  --input-root path/to/wav_only \
  --output-dir outputs/transcribed \
  --dry-run
```

Track A notebook-derived validation scripts are kept separate from the Phonova
feature pipeline. They document the analyses used to compare syllable/SPM,
Ukrainian POS/tense, and sentiment model behavior against reference annotations.
Dataset locations for each script are documented in
`paper/metadata/track_a_syllable_spm.md`,
`paper/metadata/track_a_pos_tense.md`, and
`paper/metadata/track_a_sentiment.md`.
Install the optional Track A dependencies only when running those validators:

```bash
pip install -r paper/track-a-requirements.txt
```

Syllable segmentation and syllables-per-minute validation:

```bash
python paper/scripts/track_a_validate_syllables.py \
  --input path/to/syllable_validation_items.csv \
  --audio-root path/to/audio \
  --praat-script path/to/SyllableNucleiv3.praat \
  --output-dir outputs/track_a_syllables
```

Part-of-speech tagging and verb-tense validation against Ukrainian UD CoNLL-U
files:

```bash
python paper/scripts/track_a_validate_pos_tense.py \
  --dataset train=path/to/train.conllu \
  --dataset dev=path/to/dev.conllu \
  --dataset test=path/to/test.conllu \
  --output-dir outputs/track_a_pos_tense \
  --stanza-download
```

Sentiment validation can run the notebook model paths or score frozen
predictions without loading transformer models:

```bash
python paper/scripts/track_a_evaluate_sentiment.py \
  --backend ukr-roberta-cosmus \
  --output-dir outputs/track_a_sentiment

python paper/scripts/track_a_evaluate_sentiment.py \
  --input path/to/sentiment_predictions.csv \
  --backend precomputed \
  --prediction-column sentiment_pred_1epoch \
  --output-dir outputs/track_a_sentiment_precomputed
```

Leakage-aware transcript cleanup and Ukrainian rendering are represented as
separate steps so that every derived layer has explicit provenance:

```bash
pip install -r paper/gemma-requirements.txt

python paper/scripts/gemma_role_cleanup.py \
  --input outputs/transcribed \
  --output-dir outputs/cleaned \
  --model google/gemma-3-27b-it

python paper/scripts/translate_gemma_uk.py \
  --input outputs/cleaned/role_labeled \
  --output-dir outputs/translated_uk \
  --model google/translategemma-27b-it

python paper/scripts/make_ukrainian_compat_timing.py \
  --input outputs/translated_uk/role_labeled \
  --output-dir outputs/translated_uk_compat
```

The translated JSON intentionally has empty `word_segments`; the compatibility
view is only for the five proxy timing variables documented in
`paper/metadata/transcript_preprocessing_protocol.md`.

Parity check against frozen paper artifacts. This does not run transformer
sentiment inference; it verifies feature selection, model metrics, thresholds,
and test predictions from the saved paper feature tables:

```bash
PYTHONPATH=src python paper/scripts/verify_paper_parity.py \
  --paper-input-en path/to/summary_en_3_4_2026_expanded_b0_union_features.csv \
  --paper-input-uk path/to/summary_ukr_3_4_2026_expanded_b0_union_features.csv \
  --reference-selected-models path/to/section3_en_selected_models.csv \
  --reference-test-predictions path/to/section3_en_test_predictions.csv \
  --output-dir outputs/paper_parity
```

Full feature extraction on a directory of Whisper JSON files:

```bash
PYTHONPATH=src python paper/scripts/extract_paper_features.py \
  --input path/to/transcripts \
  --output-dir outputs/features \
  --language en \
  --speaker-label participant \
  --whisper-turn-mode auto \
  --option coherence
```

Model metric regeneration from a merged paper feature table:

```bash
PYTHONPATH=src python paper/scripts/select_paper_features.py \
  --input outputs/features_with_labels.csv \
  --output outputs/model_input_paper_82.csv

PYTHONPATH=src python paper/scripts/train_evaluate_models.py \
  --input outputs/model_input_paper_82.csv \
  --output-dir outputs/model_results

PYTHONPATH=src python paper/scripts/make_tables_and_figures.py \
  --results-dir outputs/model_results \
  --output-dir outputs/tables_figures
```
