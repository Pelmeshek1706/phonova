import openwillis.speech
print()
print("Stay here....")
print(openwillis.speech.__file__)

import openwillis.speech as ows
# import openwillis.transcribe as owt
print("continue...")

import os
import json
import traceback
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import pandas as pd

from openwillis.speech.util.speech import coherence

coherence.COHERENCE_BACKEND = "gemma"   # or "gemma"
print(f"Using coherence backend: {coherence.COHERENCE_BACKEND}")
import torch; print("torch version - ", torch.__version__)
INPUT_DIR   = Path('/Users/pelmeshek1706/Desktop/projects/airest_notebooks/woz_end_whisper_test')
OUTPUT_SOURCE_DIR = Path('/Users/pelmeshek1706/Desktop/projects/airest_notebooks/result_oppenwillis_eng_norm_gemma_updated_sentiment_FIXED_21-2-26')
OUTPUT_DIR  = Path('/Users/pelmeshek1706/Desktop/projects/airest_notebooks/result_oppenwillis_eng_norm_gemma_updated_sentiment_n_fp_FIXED_22-2-26')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

nums = []
for json_file in tqdm(INPUT_DIR.glob('*.json')):
    name = json_file.stem
    print(f"Processing file: {name}")

    if str(name) in nums:
        print(f'Skipping {name}')
        continue
    try:
        # 4.1 - Read JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            transcript_json = json.load(f)

        words_new, turns_new, summary_new = ows.speech_characteristics(
            json_conf=transcript_json,
            option="simple",
            language="en",
            speaker_label="SPEAKER_A",
            feature_groups=["first_person"],
        )
    except RuntimeError as rexc:
        # skip.append(str(name))
        print(f'❌ Skipping {json_file} due to RuntimeError:')
    except Exception as exc:
        # If an error occurs, print the traceback and skip the file
        print(f'❌ Error while processing {json_file}:')
        traceback.print_exc()
        continue
    try:

        words_path_in = OUTPUT_SOURCE_DIR / f"words_{name}.csv"
        turns_path_in = OUTPUT_SOURCE_DIR / f"turns_{name}.csv"
        summ_path_in = OUTPUT_SOURCE_DIR / f"summary_sc_{name}.csv"

        words_path_out = OUTPUT_DIR / f"words_{name}.csv"
        turns_path_out = OUTPUT_DIR / f"turns_{name}.csv"
        summ_path_out = OUTPUT_DIR / f"summary_sc_{name}.csv"

        if not (words_path_in.exists() and turns_path_in.exists() and summ_path_in.exists()):
            print(f"❌ Missing source CSVs for {name}, skipping.")
            continue

        words_old = pd.read_csv(words_path_in)
        turns_old = pd.read_csv(turns_path_in)
        summ_old = pd.read_csv(summ_path_in)

        def _existing(cols, old_df, new_df):
            return [c for c in cols if c in old_df.columns and c in new_df.columns]

        words_cols = _existing(
            ["first_person"],
            words_old,
            words_new,
        )
        turns_cols = _existing(
            [
                "first_person_percentage",
                "first_person_sentiment_positive",
                "first_person_sentiment_negative",
                "first_person_sentiment_positive_vader",
                "first_person_sentiment_negative_vader",
            ],
            turns_old,
            turns_new,
        )
        summ_cols = _existing(
            [
                "first_person_percentage",
                "first_person_sentiment_positive",
                "first_person_sentiment_negative",
                "first_person_sentiment_overall",
                "first_person_sentiment_positive_vader",
                "first_person_sentiment_negative_vader",
                "first_person_sentiment_overall_vader",
            ],
            summ_old,
            summary_new,
        )

        if words_cols:
            words_old[words_cols] = words_new[words_cols].values
        if turns_cols:
            turns_old[turns_cols] = turns_new[turns_cols].values
        if summ_cols:
            summ_old[summ_cols] = summary_new[summ_cols].values

        words_old.to_csv(words_path_out, index=False)
        turns_old.to_csv(turns_path_out, index=False)
        summ_old.to_csv(summ_path_out, index=False)
    except Exception as exc:
        print(f'❌ Error while saving CSVs for {name}:')
        traceback.print_exc()

print('=== Finished ===')
