#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from _gemma_common import (
    ALL_ROLES,
    ensure_segments,
    iter_json_files,
    normalize_text,
    participant_id_from_payload,
    read_json,
    safe_num,
    write_json,
)


PROMPT_VERSION = "paper_uk_proxy_timing_v1"
PROXY_TIMING_VARIABLES = [
    "speech_length_words",
    "words_per_min",
    "syllables_per_min",
    "mean_pre_word_pause",
    "mean_pause_variability",
]

TOKEN_RE = re.compile(r"\S+", flags=re.UNICODE)


@dataclass
class FileReport:
    file: str
    n_segments: int
    proxy_segments: int
    proxy_words: int
    preserved_duration_seconds: float
    roles: str


def tokenize_for_proxy(text: str) -> list[str]:
    return [match.group(0) for match in TOKEN_RE.finditer(normalize_text(text))]


def proxy_words_for_segment(seg: dict[str, Any]) -> list[dict[str, Any]]:
    text = normalize_text(seg.get("text", ""))
    tokens = tokenize_for_proxy(text)
    if not tokens:
        return []
    start = safe_num(seg.get("start"), 0.0)
    end = safe_num(seg.get("end"), start)
    duration = max(0.0, end - start)
    step = duration / len(tokens) if tokens else 0.0
    words: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        word_start = start + step * index
        word_end = start + step * (index + 1)
        words.append(
            {
                "word": token,
                "start": word_start,
                "end": word_end,
                "probability": None,
                "timing_mode": "uniform_proxy_from_preserved_segment_duration",
                "source_segment_id": seg.get("id"),
                "source_word_idx": None,
            }
        )
    return words


def parse_roles(values: list[str] | None) -> tuple[str, ...] | None:
    if not values:
        return None
    out: list[str] = []
    for value in values:
        for piece in value.split(","):
            role = piece.strip().lower()
            if not role:
                continue
            if role not in ALL_ROLES:
                raise SystemExit(f"Invalid role in --roles: {role}")
            if role not in out:
                out.append(role)
    return tuple(out)


def build_compat_payload(
    path: Path,
    roles: tuple[str, ...] | None,
    indent: int,
    output_path: Path,
) -> FileReport:
    payload = read_json(path)
    segments = ensure_segments(payload, path)
    selected_segments = [
        seg for seg in segments if roles is None or str(seg.get("role") or "").lower() in roles
    ]

    out_segments: list[dict[str, Any]] = []
    flat_words: list[dict[str, Any]] = []
    total_duration = 0.0
    for seg in selected_segments:
        out_seg = dict(seg)
        proxy_words = proxy_words_for_segment(seg)
        out_seg["words"] = proxy_words
        out_seg["proxy_timing"] = True
        out_seg["proxy_timing_mode"] = "uniform_proxy_from_preserved_segment_duration"
        out_segments.append(out_seg)
        total_duration += max(0.0, safe_num(seg.get("end"), 0.0) - safe_num(seg.get("start"), 0.0))
        for word in proxy_words:
            item = dict(word)
            item["role"] = out_seg.get("role")
            item["segment_id"] = out_seg.get("id")
            flat_words.append(item)

    out = dict(payload)
    out["segments"] = out_segments
    out["word_segments"] = flat_words
    out["text"] = " ".join(normalize_text(seg.get("text", "")) for seg in out_segments).strip()
    out["compat_timing_meta"] = {
        "method": "ukrainian_uniform_proxy_word_timing",
        "prompt_version": PROMPT_VERSION,
        "participant_id": participant_id_from_payload(path, payload),
        "source_file": path.name,
        "roles": list(roles) if roles is not None else "all_roles",
        "timing_mode": "uniform_proxy_from_preserved_segment_duration",
        "source_timing": "translated segment start/end copied from cleaned English segment layer",
        "word_alignment": "proxy_only_not_native_ukrainian_acoustic_alignment",
        "proxy_timing_variables_only": list(PROXY_TIMING_VARIABLES),
        "n_segments_before": len(segments),
        "n_segments_after": len(out_segments),
        "proxy_words": len(flat_words),
    }
    write_json(output_path, out, indent=indent)
    return FileReport(
        file=path.name,
        n_segments=len(segments),
        proxy_segments=len(out_segments),
        proxy_words=len(flat_words),
        preserved_duration_seconds=total_duration,
        roles=",".join(roles) if roles is not None else "all_roles",
    )


def write_report(rows: list[FileReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the paper Ukrainian proxy-timing compatibility view from translated segment-level JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Translated JSON file or directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument(
        "--roles",
        nargs="*",
        default=None,
        help="Optional role filter. Omit to preserve all translated segments.",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-csv", type=Path, default=None)
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    files = iter_json_files(args.input, args.pattern)
    if args.max_files is not None:
        files = files[: args.max_files]
    roles = parse_roles(args.roles)

    reports: list[FileReport] = []
    for index, path in enumerate(files, start=1):
        target = args.output_dir / path.name
        if target.exists() and not args.overwrite:
            print(f"[{index}/{len(files)}] skip existing {path.name}")
            continue
        print(f"[{index}/{len(files)}] process {path.name}")
        reports.append(build_compat_payload(path=path, roles=roles, indent=args.indent, output_path=target))

    if args.report_csv is None:
        args.report_csv = args.output_dir / "ukrainian_compat_timing_manifest.csv"
    if reports:
        write_report(reports, args.report_csv)
    print(json.dumps({"processed_files": len(reports), "report_csv": str(args.report_csv)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
