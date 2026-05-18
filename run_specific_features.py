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
OUTPUT_SOURCE_DIR = Path('/Users/pelmeshek1706/Desktop/projects/airest_notebooks/result_oppenwillis_eng_norm_gemma_test_sentiment-11-2-26')
OUTPUT_DIR  = Path('/Users/pelmeshek1706/Desktop/projects/airest_notebooks/result_oppenwillis_eng_norm_gemma_test_sentiment-11-2-26') # improved_vader_ua - 11-2-26
# OUTPUT_DIR  = Path('/Users/pelmeshek1706/Desktop/projects/airest_notebooks/result_oppenwillis_ukr_norm_gemma_test_sentiment-11-2-26') # default vader - 11-2-26
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _infer_language(*paths: Path) -> str:
    hint = " ".join(str(p).lower() for p in paths)
    if any(token in hint for token in ("ukr", "ukrain", "_ua", "-ua", "/ua/")):
        return "uk"
    return "en"

language = _infer_language(INPUT_DIR, OUTPUT_SOURCE_DIR, OUTPUT_DIR)
print(f"Detected language for run: {language}")



nums = []
files = list(INPUT_DIR.glob("*.json"))  # materialize to know total

for json_file in tqdm(files, total=len(files), desc="Processing"):
    name = json_file.stem
    print(f"Processing file: {name}")

    # if str(name) in nums:
    #     print(f'Skipping {name}')
    #     continue
    if str(name) != "300":
        print(f'Skipping {name}')
        continue
    try:
        # 4.1 - Read JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            transcript_json = json.load(f)

        words_new, turns_new, summary_new = ows.speech_characteristics(
            json_conf=transcript_json,
            option="simple",
            language=language,
            speaker_label="SPEAKER_A",
            feature_groups=["first_person"],
            min_coherence_turn_length=1
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

        words_new.to_csv(words_path_out, index=False)
        turns_new.to_csv(turns_path_out, index=False)
        summary_new.to_csv(summ_path_out, index=False)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] ✅ Saved: words_{name}.csv, turns_{name}.csv, summary_sc_{name}.csv')

    except Exception as exc:
        print(f'❌ Error while saving CSVs for {name}:')
        traceback.print_exc()

print('=== Finished ===')
