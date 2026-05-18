# Triplet Sentiment Inference

This script compares 3 sentiment analyzers on OpenWillis/Whisper JSON:

- `vader` (classic English VADER)
- `vader-ua` (the standard external `vader-ua`)
- `vader-ua-improved` (the improved class from this project)

## Files

- Script: `/Users/pelmeshek1706/Desktop/projects/final_airest_voice/airest/sentiment_triplet_inference/run_triplet_sentiment_inference.py`
- Default outputs: `/Users/pelmeshek1706/Desktop/projects/final_airest_voice/airest/sentiment_triplet_inference/outputs`

## Run (single file)

```bash
airest/.venv/bin/python /Users/pelmeshek1706/Desktop/projects/final_airest_voice/airest/sentiment_triplet_inference/run_triplet_sentiment_inference.py \
  --inputs /Users/pelmeshek1706/Desktop/projects/airest_notebooks/woz_end_whisper_test_ukr/300.json
```

## Run (entire directory)

```bash
airest/.venv/bin/python /Users/pelmeshek1706/Desktop/projects/final_airest_voice/airest/sentiment_triplet_inference/run_triplet_sentiment_inference.py \
  --inputs /Users/pelmeshek1706/Desktop/projects/airest_notebooks/woz_end_whisper_test_ukr
```

## Output filenames

For each input `<n>.json`, the script creates 3 CSV files:

- `sentiment_<n>_vader.csv`
- `sentiment_<n>_vader-ua.csv`
- `sentiment_<n>_vader-ua-improved.csv`

Each CSV contains:

- rows with `row_type=turn` (per segment)
- one row with `row_type=summary` (for the full text)
