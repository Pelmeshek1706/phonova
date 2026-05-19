# AIREST Phonova

Standalone speech-analysis package extracted for AIREST usage.

## Install

```bash
python -m pip install -e ./phonova
```

For development:

```bash
python -m pip install -r ./phonova/requirements-dev.txt
```

The package will download required NLTK resources on first use. For English and
Ukrainian POS features, spaCy models are also required and will be downloaded by
the package if they are missing.

Optional environment variables:

- `AIREST_COHERENCE_BACKEND`
- `AIREST_BERT_EN_MODEL_ID`
- `AIREST_BERT_MULTI_MODEL_ID`
- `AIREST_BERT_SENTENCE_EN_MODEL_ID`
- `AIREST_BERT_SENTENCE_MULTI_MODEL_ID`
- `AIREST_TOKEN_CACHE_SIZE`
- `HUGGINGFACEHUB_API_TOKEN`

## Usage

```python
from phonova import SpeechAnalyzer

analyzer_ukr = SpeechAnalyzer(language="ua", coherence_backend="gemma")
words_ukr, turns_ukr, summary_ukr = analyzer_ukr.analyze_transcript(
    json_conf=ukr_data,
    option="coherence",
    speaker_label="participant",
    min_coherence_turn_length=2,
    whisper_turn_mode="speaker",
)
```

## Public API

- `from phonova import SpeechAnalyzer`
- `from phonova import SpeechAnalyzerSettings`
- Internal standalone speech namespace: `airest.speech`
