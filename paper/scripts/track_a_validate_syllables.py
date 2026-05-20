#!/usr/bin/env python3
"""Track A syllable segmentation and syllables-per-minute validation.

This script converts the syllable/SPM cells from `notebooks/airest_article.ipynb`
into a reusable command. It accepts a restricted validation CSV with transcript,
audio path, and duration columns, computes the notebook syllable counters, and
writes aggregate comparison tables. The input data are not redistributed.
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MIN_PITCH_HZ = 75.0
MIN_SYLLABLE_SEP_SEC = 0.10
PEAK_PROMINENCE_DB = 2.0
INTENSITY_STEP = 0.01
VOWELS_UK = set("аеєиіїоуюяАЕЄИІЇОУЮЯ")
WORD_RE_UK_APOS = re.compile(r"[А-ЩЬЮЯІЇЄҐа-щьюяіїєґʼ'’-]+", re.U)
WORD_RE_UK = re.compile(
    r"[А-ЩЬЮЯІЇЄҐа-щьюяіїєґ]+(?:'[А-ЩЬЮЯІЇЄҐа-щьюяіїєґ]+)?"
    r"(?:-[А-ЩЬЮЯІЇЄҐа-щьюяіїєґ]+)*"
)
APOS_DASH_MAP = str.maketrans({"’": "'", "ʼ": "'", "–": "-", "—": "-"})
UKR_SONORITY = [
    "аеєиіїоуюя",
    "йв",
    "рл",
    "мн",
    "жзшщсхгф",
    "бпдткґчц",
]


@dataclass
class Paths:
    audio_root: str | None = None
    praat_script: str | None = None


_PYPHEN = None
_NLP_UK = None
_SSP_UK = None


def require_module(module_name: str, install_hint: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"Missing optional dependency '{module_name}'. Install {install_hint}.") from exc


def to_text(value: object) -> str:
    return value if isinstance(value, str) else ""


def resolve_audio_path(path: str, audio_root: str | None) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(audio_root or "", path)


def compute_spm(num_syllables: float, duration_seconds: float) -> float:
    if pd.isna(num_syllables) or pd.isna(duration_seconds) or duration_seconds <= 0:
        return float("nan")
    return float(num_syllables) / (float(duration_seconds) / 60.0)


def pyphen_dic():
    global _PYPHEN
    if _PYPHEN is None:
        pyphen = require_module("pyphen", "paper/track-a-requirements.txt")
        _PYPHEN = pyphen.Pyphen(lang="uk_UA")
    return _PYPHEN


def count_syll_pyphen(text: str) -> int:
    text = to_text(text)
    if not text:
        return 0
    total = 0
    dic = pyphen_dic()
    for word in WORD_RE_UK_APOS.findall(text):
        word = word.replace("’", "'").replace("ʼ", "'")
        for part in word.split("-"):
            inserted = dic.inserted(part)
            total += (inserted.count("-") + 1) if inserted else 1
    return int(total)


def syllables_word_uk(word: str) -> int:
    word = word.replace("’", "'").replace("ʼ", "'")
    if not any(ch in VOWELS_UK for ch in word):
        return 1 if re.search(r"[рРлЛ]", word) else 1
    total = 0
    dic = pyphen_dic()
    for part in word.split("-"):
        inserted = dic.inserted(part)
        count = (inserted.count("-") + 1) if inserted else 1
        if re.search(r"(йо|ЙО|ьо|ЬО)", part):
            count = max(1, count)
        total += count
    return int(total)


def build_spacy_uk(spacy_model: str):
    global _NLP_UK
    if _NLP_UK is not None:
        return _NLP_UK
    spacy = require_module("spacy", "paper/track-a-requirements.txt")
    from spacy.language import Language
    from spacy.tokens import Token

    if not Token.has_extension("num_syllables"):
        Token.set_extension("num_syllables", default=0)

    if not Language.has_factory("uk_syllable_counter"):
        @Language.component("uk_syllable_counter")
        def uk_syllable_counter(doc):
            for token in doc:
                if token.is_alpha or WORD_RE_UK_APOS.fullmatch(token.text):
                    token._.num_syllables = syllables_word_uk(token.text)
                else:
                    token._.num_syllables = 0
            return doc

    try:
        nlp = spacy.load(spacy_model)
    except Exception:
        nlp = spacy.blank("uk")
    if "uk_syllable_counter" not in nlp.pipe_names:
        nlp.add_pipe("uk_syllable_counter", last=True)
    _NLP_UK = nlp
    return _NLP_UK


def count_syll_spacy(text: str, spacy_model: str) -> int:
    text = to_text(text)
    if not text:
        return 0
    nlp = build_spacy_uk(spacy_model)
    return int(sum(token._.num_syllables for token in nlp(text)))


def nltk_tokenizer():
    global _SSP_UK
    if _SSP_UK is None:
        nltk = require_module("nltk", "paper/track-a-requirements.txt")
        _SSP_UK = nltk.tokenize.SyllableTokenizer(lang="uk", sonority_hierarchy=UKR_SONORITY)
    return _SSP_UK


def count_syll_nltk(text: str) -> int:
    text = to_text(text)
    if not text:
        return 0
    text = text.translate(APOS_DASH_MAP)
    tokenizer = nltk_tokenizer()
    total = 0
    for word in WORD_RE_UK.findall(text):
        for part in word.split("-"):
            cleaned = part.replace("'", "")
            if cleaned:
                total += len(tokenizer.tokenize(cleaned))
    return int(total)


def count_syllables_praat_like(
    audio_path: str,
    min_pitch_hz: float = MIN_PITCH_HZ,
    time_step_sec: float = INTENSITY_STEP,
    min_separation_sec: float = MIN_SYLLABLE_SEP_SEC,
    prominence_db: float = PEAK_PROMINENCE_DB,
) -> int:
    parselmouth = require_module("parselmouth", "paper/track-a-requirements.txt")
    sound = parselmouth.Sound(audio_path)
    intensity = sound.to_intensity(minimum_pitch=min_pitch_hz, time_step=time_step_sec)
    pitch = sound.to_pitch(time_step=time_step_sec, pitch_floor=min_pitch_hz)
    times = intensity.xs()
    values = np.asarray(intensity.values).flatten()
    if len(values) < 3:
        return 0
    candidates: list[tuple[float, float]] = []
    window = 3
    for idx in range(window, len(values) - window):
        value = values[idx]
        if values[idx - 1] < value > values[idx + 1]:
            local_min = np.min(values[idx - window: idx + window + 1])
            if (value - local_min) >= prominence_db:
                time = times[idx]
                f0 = pitch.get_value_at_time(time)
                if f0 and not math.isnan(f0):
                    candidates.append((time, value))
    if not candidates:
        return 0
    candidates.sort()
    kept: list[tuple[float, float]] = []
    for time, value in candidates:
        if not kept or (time - kept[-1][0]) >= min_separation_sec:
            kept.append((time, value))
        elif value > kept[-1][1]:
            kept[-1] = (time, value)
    return len(kept)


def count_syllables_praat_original(
    audio_path: str,
    praat_script_path: str,
    *,
    detect_filled_pauses: bool = False,
    language: str = "English",
    silence_db: float = -25.0,
    min_dip_db: float = 2.0,
    min_pause_s: float = 0.4,
) -> int:
    parselmouth = require_module("parselmouth", "paper/track-a-requirements.txt")
    from parselmouth.praat import call

    sound = parselmouth.Sound(audio_path)
    result = parselmouth.praat.run_file(
        sound,
        praat_script_path,
        "",
        "None",
        float(silence_db),
        float(min_dip_db),
        float(min_pause_s),
        bool(detect_filled_pauses),
        str(language),
        1.0,
        "Table",
        "OverWriteData",
        False,
    )
    table = result[-1] if isinstance(result, (list, tuple)) else result
    try:
        tsv = call(table, "List", False)
        frame = pd.read_csv(io.StringIO(tsv), sep="\t")
        frame.columns = frame.columns.str.strip()
        if "nsyll" in frame.columns:
            return int(frame.loc[0, "nsyll"])
    except Exception:
        pass

    ncol = call(table, "Get number of columns")
    target_idx = None
    for idx in range(1, ncol + 1):
        label = call(table, "Get column label", idx)
        if str(label).strip().lower() == "nsyll":
            target_idx = idx
            break
    if target_idx is None:
        for idx in range(1, ncol + 1):
            label = call(table, "Get column label", idx)
            if str(label).strip().lower() == "voicedcount":
                target_idx = idx
                break
    if target_idx is None:
        labels = [call(table, "Get column label", idx) for idx in range(1, ncol + 1)]
        raise KeyError(f"nsyll column not found. Columns: {labels}")
    return int(round(float(call(table, "Get value", 1, target_idx))))


def maybe_progress(values: Iterable, label: str):
    try:
        from tqdm import tqdm
        return tqdm(values, desc=label)
    except ImportError:
        return values


def fill_missing_counts(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out[args.text_column] = out[args.text_column].apply(to_text)
    if "syll_pyphen" not in out.columns or args.force:
        out["syll_pyphen"] = [count_syll_pyphen(text) for text in maybe_progress(out[args.text_column], "syll_pyphen")]
    if "syll_spacy" not in out.columns or args.force:
        out["syll_spacy"] = [
            count_syll_spacy(text, args.spacy_model)
            for text in maybe_progress(out[args.text_column], "syll_spacy")
        ]
    if "syll_nltk" not in out.columns or args.force:
        out["syll_nltk"] = [count_syll_nltk(text) for text in maybe_progress(out[args.text_column], "syll_nltk")]

    if args.audio_path_column in out.columns:
        out["_abs_path"] = out[args.audio_path_column].astype(str).map(lambda path: resolve_audio_path(path, args.audio_root))
    else:
        out["_abs_path"] = ""

    if not args.skip_audio:
        if "syll_praat_like" not in out.columns or args.force:
            out["syll_praat_like"] = [
                float("nan") if not os.path.exists(path) else count_syllables_praat_like(path)
                for path in maybe_progress(out["_abs_path"], "syll_praat_like")
            ]
        if "syll_praat_original" not in out.columns or args.force:
            def safe_praat_original(path: str) -> float:
                try:
                    if not os.path.exists(path) or not args.praat_script:
                        return float("nan")
                    return float(count_syllables_praat_original(path, args.praat_script))
                except Exception:
                    if args.strict_audio:
                        raise
                    return float("nan")

            out["syll_praat_original"] = [
                safe_praat_original(path) for path in maybe_progress(out["_abs_path"], "syll_praat_original")
            ]

    if args.duration_column in out.columns:
        out["spm_spacy"] = out.apply(lambda row: compute_spm(row.get("syll_spacy"), row[args.duration_column]), axis=1)
        if "syll_praat_original" in out.columns:
            out["spm_praat_original"] = out.apply(
                lambda row: compute_spm(row.get("syll_praat_original"), row[args.duration_column]), axis=1
            )
            out["d_spm_spacy_vs_praat_original"] = (out["spm_spacy"] - out["spm_praat_original"]).abs()
    return out


def pair_report(df: pd.DataFrame, a: str, b: str, label: str) -> dict[str, float | int | str]:
    if a not in df.columns or b not in df.columns:
        return {"method_pair": label, "n": 0, "mae": float("nan"), "bias": float("nan")}
    pair = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
    if pair.empty:
        return {"method_pair": label, "n": 0, "mae": float("nan"), "bias": float("nan")}
    diff = pair[a] - pair[b]
    row: dict[str, float | int | str] = {
        "method_pair": label,
        "n": int(len(pair)),
        "mae": float(np.mean(np.abs(diff))),
        "bias": float(np.mean(diff)),
        "ba_mean_diff": float(np.mean(diff)),
        "ba_loa_low": float(np.mean(diff) - 1.96 * np.std(diff, ddof=1)) if len(pair) > 1 else float("nan"),
        "ba_loa_high": float(np.mean(diff) + 1.96 * np.std(diff, ddof=1)) if len(pair) > 1 else float("nan"),
    }
    if len(pair) > 1:
        try:
            from scipy.stats import pearsonr, spearmanr
            row["pearson_r"] = float(pearsonr(pair[a], pair[b])[0])
            row["spearman_rho"] = float(spearmanr(pair[a], pair[b])[0])
        except Exception:
            row["pearson_r"] = float("nan")
            row["spearman_rho"] = float("nan")
    return row


def icc2(df: pd.DataFrame, columns: list[str], target_column: str) -> float:
    present = [column for column in columns if column in df.columns]
    if len(present) < 2:
        return float("nan")
    long = df[[target_column] + present].melt(id_vars=target_column, var_name="rater", value_name="score").dropna()
    if long[target_column].nunique() < 2:
        return float("nan")
    pingouin = require_module("pingouin", "paper/track-a-requirements.txt")
    table = pingouin.intraclass_corr(data=long, targets=target_column, raters="rater", ratings="score")
    return float(table.loc[table["Type"] == "ICC2", "ICC"].iloc[0])


def write_outputs(df: pd.DataFrame, args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    item_path = output_dir / "syllable_item_results.csv"
    df.to_csv(item_path, index=False)

    pairs = [
        ("spaCy-uk vs Pyphen", "syll_spacy", "syll_pyphen"),
        ("spaCy-uk vs NLTK", "syll_spacy", "syll_nltk"),
        ("Praat-like vs Praat v3", "syll_praat_like", "syll_praat_original"),
        ("spaCy-uk vs Praat v3", "syll_spacy", "syll_praat_original"),
        ("Pyphen vs Praat v3", "syll_pyphen", "syll_praat_original"),
        ("NLTK vs Praat v3", "syll_nltk", "syll_praat_original"),
    ]
    pair_rows = [pair_report(df, a, b, label) for label, a, b in pairs]
    pair_df = pd.DataFrame(pair_rows)
    overall_icc = icc2(
        df,
        ["syll_spacy", "syll_pyphen", "syll_nltk", "syll_praat_like", "syll_praat_original"],
        args.id_column,
    )
    pair_df["overall_icc_2_1"] = overall_icc
    pair_df.to_csv(output_dir / "syllable_pair_metrics.csv", index=False)
    pair_df[["method_pair", "mae", "overall_icc_2_1"]].to_csv(
        output_dir / "table02_syllable_counting_mae_summary.csv",
        index=False,
    )

    if args.duration_column in df.columns and "spm_praat_original" in df.columns:
        tail_cols = [
            column for column in [
                args.id_column,
                args.audio_path_column,
                args.text_column,
                args.duration_column,
                "syll_spacy",
                "syll_praat_original",
                "spm_spacy",
                "spm_praat_original",
                "d_spm_spacy_vs_praat_original",
            ] if column in df.columns
        ]
        df.sort_values("d_spm_spacy_vs_praat_original", ascending=False).head(args.tail_n)[tail_cols].to_csv(
            output_dir / "tail_spm_top.csv",
            index=False,
        )
    (output_dir / "syllable_validation_meta.json").write_text(
        json.dumps(
            {
                "input": str(args.input),
                "rows": int(len(df)),
                "spacy_model": args.spacy_model,
                "skip_audio": bool(args.skip_audio),
                "praat_script": str(args.praat_script) if args.praat_script else None,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV with transcript, duration, and audio path columns.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--text-column", default="transcript")
    parser.add_argument("--audio-path-column", default="path")
    parser.add_argument("--duration-column", default="audio_dur_sec")
    parser.add_argument("--id-column", default="path")
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--praat-script", default=None, help="Path to SyllableNucleiv3.praat.")
    parser.add_argument("--spacy-model", default="uk_core_news_sm")
    parser.add_argument("--skip-audio", action="store_true", help="Compute text methods only.")
    parser.add_argument("--strict-audio", action="store_true", help="Raise on Praat failures instead of writing NaN.")
    parser.add_argument("--force", action="store_true", help="Recompute columns even if precomputed values exist.")
    parser.add_argument("--tail-n", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    df = pd.read_csv(args.input)
    required = {args.text_column}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input is missing required columns: {sorted(missing)}")
    if args.id_column not in df.columns:
        df[args.id_column] = np.arange(len(df)).astype(str)
    out = fill_missing_counts(df, args)
    write_outputs(out, args)


if __name__ == "__main__":
    main()
