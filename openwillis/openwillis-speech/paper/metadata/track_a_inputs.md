# Track A Validation Inputs

Track A covers Ukrainian feature-validity checks that are independent of the E-DAIC downstream model pipeline. The scripts are public, but several inputs are restricted, third-party, or model-dependent and are therefore not bundled in this repository.

## Syllable Segmentation And SPM

Script: `paper/scripts/track_a_validate_syllables.py`

Detailed dataset-location note: `paper/metadata/track_a_syllable_spm.md`.

Expected input is a CSV with at least:

- `path`: audio path, relative to `--audio-root` or absolute
- `transcript`: Ukrainian transcript text
- `audio_dur_sec`: audio duration in seconds

The script computes the notebook counters `syll_spacy`, `syll_pyphen`, `syll_nltk`, `syll_praat_like`, `syll_praat_original`, and `spm_spacy`. Praat v3 comparison requires the `SyllableNucleiv3.praat` script supplied by the user via `--praat-script`.

## Part-Of-Speech And Verb Tense

Script: `paper/scripts/track_a_validate_pos_tense.py`

Detailed dataset-location note: `paper/metadata/track_a_pos_tense.md`.

Expected input is one or more Universal Dependencies CoNLL-U files supplied as repeated `--dataset name=path_or_url` arguments. The script reads gold UPOS and `Tense` features, runs spaCy and Stanza Ukrainian pipelines, aligns predicted tokens back to UD token spans, and writes the aggregate UPOS and tense macro-F1 tables.

Model assets are not bundled. Install them in the local environment, for example:

```bash
python -m spacy download uk_core_news_md
python -c "import stanza; stanza.download('uk')"
```

## Sentiment Analysis

Script: `paper/scripts/track_a_evaluate_sentiment.py`

Detailed dataset-location note: `paper/metadata/track_a_sentiment.md`.

Expected input is either the Hugging Face `YShynkarov/COSMUS` dataset or a CSV/JSONL file with:

- `document_content`: text to score
- `annotator_sentiment`: gold label in `negative`, `neutral`, `positive`, or `mixed`

The notebook-derived backends are:

- `ukr-roberta-cosmus`
- `twitter-xlm-roberta`
- `vader-translated`, which expects a `translated_text` column
- `precomputed`, for archived prediction columns
- `hf-pipeline`, for a user-specified Hugging Face sentiment model

Transformer checkpoints are downloaded only when the relevant backend is requested. They are not part of the default smoke test.
