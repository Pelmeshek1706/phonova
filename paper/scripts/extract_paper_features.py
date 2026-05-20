#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phonova.speech.speech_attribute import speech_characteristics

from _common import dataframe_sha256, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract Phonova speech features from Whisper JSON files for the paper pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Whisper JSON file or directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for word/turn/summary CSV outputs.")
    parser.add_argument("--pattern", default="*.json", help="Glob pattern used when --input is a directory.")
    parser.add_argument("--language", default="en", help="Language hint passed to speech_characteristics.")
    parser.add_argument("--speaker-label", default="participant", help="Speaker label to extract.")
    parser.add_argument("--min-turn-length", type=int, default=1)
    parser.add_argument("--min-coherence-turn-length", type=int, default=5)
    parser.add_argument("--option", choices=["simple", "coherence"], default="coherence")
    parser.add_argument("--whisper-turn-mode", choices=["auto", "speaker", "segment"], default="auto")
    parser.add_argument(
        "--feature-groups",
        default=None,
        help="Comma-separated feature groups. Default computes all groups supported by speech_characteristics.",
    )
    parser.add_argument("--id-field", default=None, help="Optional top-level JSON field to use as Participant.")
    return parser


def _input_files(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob(pattern))
        if files:
            return files
    raise FileNotFoundError(f"No input JSON files found at {path} with pattern {pattern}")


def _participant_id(path: Path, payload: dict, id_field: str | None) -> str:
    if id_field and id_field in payload:
        return str(payload[id_field])
    for key in ("Participant", "participant", "participant_id", "id"):
        if key in payload:
            return str(payload[key])
    return path.stem


def _parse_feature_groups(raw: str | None):
    if raw is None or not raw.strip():
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> int:
    args = build_parser().parse_args()
    files = _input_files(args.input, args.pattern)
    feature_groups = _parse_feature_groups(args.feature_groups)

    words_dir = args.output_dir / "words"
    turns_dir = args.output_dir / "turns"
    summary_dir = args.output_dir / "summary"
    for out_dir in (words_dir, turns_dir, summary_dir):
        out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    manifest_rows = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        participant = _participant_id(path, payload, args.id_field)
        word_df, turn_df, summary_df = speech_characteristics(
            payload,
            language=args.language,
            speaker_label=args.speaker_label,
            min_turn_length=args.min_turn_length,
            min_coherence_turn_length=args.min_coherence_turn_length,
            option=args.option,
            feature_groups=feature_groups,
            whisper_turn_mode=args.whisper_turn_mode,
        )

        for df in (word_df, turn_df, summary_df):
            df.insert(0, "source_file", path.name)
            df.insert(0, "Participant", participant)

        safe_id = participant.replace("/", "_")
        word_path = words_dir / f"words_{safe_id}.csv"
        turn_path = turns_dir / f"turns_{safe_id}.csv"
        summary_path = summary_dir / f"summary_{safe_id}.csv"
        word_df.to_csv(word_path, index=False)
        turn_df.to_csv(turn_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        summary_rows.append(summary_df.iloc[0].to_dict())
        manifest_rows.append(
            {
                "Participant": participant,
                "source_file": str(path),
                "word_csv": str(word_path),
                "turn_csv": str(turn_path),
                "summary_csv": str(summary_path),
                "n_word_rows": int(len(word_df)),
                "n_turn_rows": int(len(turn_df)),
            }
        )

    combined = pd.DataFrame(summary_rows)
    manifest = pd.DataFrame(manifest_rows)
    combined_path = args.output_dir / "summary_features.csv"
    manifest_path = args.output_dir / "manifest.csv"
    combined.to_csv(combined_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    write_json(
        args.output_dir / "feature_extraction_meta.json",
        {
            "input": str(args.input),
            "pattern": args.pattern,
            "language": args.language,
            "speaker_label": args.speaker_label,
            "option": args.option,
            "feature_groups": feature_groups,
            "whisper_turn_mode": args.whisper_turn_mode,
            "n_files": int(len(files)),
            "summary_features_csv": str(combined_path),
            "manifest_csv": str(manifest_path),
            "summary_sha256": dataframe_sha256(combined),
        },
    )
    print(f"Processed {len(files)} transcript(s)")
    print(f"Saved combined summary features to {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
