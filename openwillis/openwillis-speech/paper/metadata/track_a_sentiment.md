# Track A Sentiment Data

Script: `paper/scripts/track_a_evaluate_sentiment.py`

## Where To Get Inputs

The notebook sentiment comparison used COSMUS:

- Dataset: `https://huggingface.co/datasets/YShynkarov/COSMUS`
- Script default: `--dataset-name YShynkarov/COSMUS --dataset-split train`

The current dataset card lists license `mit`. No separate formal paper citation
was visible on the current dataset card, so cite the dataset card itself unless
the dataset author provides a publication citation.

The script can also use a local CSV or JSONL file with:

- `document_content`: text to score
- `annotator_sentiment`: gold label in `negative`, `neutral`, `positive`, or `mixed`

For the VADER-on-translated-text path, the local input must also contain:

- `translated_text`: English text used by the VADER backend

## Model Sources

Model backends are loaded only when requested:

- `ukr-roberta-cosmus`: `https://huggingface.co/YShynkarov/ukr-roberta-cosmus-sentiment`
- `twitter-xlm-roberta`: `https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment`
- `vader-translated`: local VADER from `vaderSentiment`, using the caller-provided `translated_text`
- `precomputed`: no model download, reads a numeric prediction column
- `hf-pipeline`: caller-provided Hugging Face sentiment model

## Commands

Run the COSMUS-backed model path:

```bash
python paper/scripts/track_a_evaluate_sentiment.py \
  --backend ukr-roberta-cosmus \
  --output-dir outputs/track_a_sentiment
```

Score archived predictions without transformer inference:

```bash
python paper/scripts/track_a_evaluate_sentiment.py \
  --input path/to/sentiment_predictions.csv \
  --backend precomputed \
  --prediction-column sentiment_pred_1epoch \
  --output-dir outputs/track_a_sentiment_precomputed
```

Citation and license audit: `paper/metadata/third_party_citations_and_licenses.md`.
