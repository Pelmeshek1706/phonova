#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run WhisperX ASR, alignment, and optional diarization for paper transcript generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Audio file or directory of audio/interview folders.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for WhisperX JSON outputs.")
    parser.add_argument(
        "--audio-glob",
        default="*_AUDIO.wav",
        help="Audio filename pattern used inside input directories.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively for audio files. By default, only DAIC-style <id>_P folders and root-level files are scanned.",
    )
    parser.add_argument("--language", default="en", help="Language code passed to WhisperX ASR/alignment.")
    parser.add_argument(
        "--model-name",
        default="auto",
        help="'auto' uses large-v3 on CUDA and medium on CPU; otherwise pass a WhisperX model name.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="'auto' uses float16 on CUDA and int8 on CPU; otherwise pass a faster-whisper compute type.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Defaults to 16 on CUDA and 4 on CPU.")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument(
        "--temperatures",
        default="0.0",
        help="Comma-separated decoding temperatures passed as WhisperX asr_options.",
    )
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for pyannote diarization. Defaults to HF_TOKEN env var.")
    parser.add_argument("--no-diarization", action="store_true", help="Skip pyannote diarization and speaker assignment.")
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=6)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-id", action="append", default=[], help="Participant/interview ID to skip. Can be repeated.")
    parser.add_argument("--skip-ids-file", type=Path, default=None, help="Optional text file with one skip ID per line.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many audio files after skip filtering.")
    parser.add_argument("--manifest", type=Path, default=None, help="CSV manifest path. Defaults to <output-dir>/asr_manifest.csv.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve input audio files and write a manifest without loading WhisperX or pyannote.",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failed audio file.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation. Use 0 for compact JSON.")
    return parser


def _parse_temperatures(raw: str) -> list[float]:
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out or [0.0]


def _load_skip_ids(args: argparse.Namespace) -> set[str]:
    skip_ids = {str(item).strip() for item in args.skip_id if str(item).strip()}
    if args.skip_ids_file:
        for line in args.skip_ids_file.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                skip_ids.add(value)
    return skip_ids


def _interview_id_from_path(path: Path) -> str:
    parent_match = re.match(r"^(.+)_P$", path.parent.name, flags=re.IGNORECASE)
    if parent_match:
        return parent_match.group(1)

    stem = path.stem
    audio_match = re.match(r"^(.+)_AUDIO$", stem, flags=re.IGNORECASE)
    if audio_match:
        return audio_match.group(1)
    return stem


def _audio_files(input_root: Path, audio_glob: str, recursive: bool) -> list[tuple[str, Path]]:
    if input_root.is_file():
        return [(_interview_id_from_path(input_root), input_root)]
    if not input_root.exists():
        raise FileNotFoundError(input_root)

    candidates: list[Path] = []
    participant_dirs = [
        path
        for path in input_root.iterdir()
        if path.is_dir() and re.match(r"^.+_P$", path.name, flags=re.IGNORECASE)
    ]
    for participant_dir in sorted(participant_dirs, key=lambda path: path.name):
        matches = sorted(participant_dir.glob(audio_glob))
        if not matches:
            matches = sorted(participant_dir.rglob(audio_glob))
        candidates.extend(matches[:1])

    candidates.extend(sorted(input_root.glob(audio_glob)))
    if recursive:
        candidates.extend(sorted(input_root.rglob(audio_glob)))

    seen = set()
    out = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append((_interview_id_from_path(path), path))
    return out


def _resolve_device(torch_module, requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def _resolve_model_name(requested: str, device: str) -> str:
    if requested != "auto":
        return requested
    return "large-v3" if device == "cuda" else "medium"


def _resolve_compute_type(requested: str, device: str) -> str:
    if requested != "auto":
        return requested
    return "float16" if device == "cuda" else "int8"


def _cleanup_memory(torch_module, device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch_module.cuda.empty_cache()


def _load_asr_runtime(diarization: bool):
    try:
        import torch
        import whisperx
        if diarization:
            from whisperx.diarize import DiarizationPipeline
        else:
            DiarizationPipeline = None
    except ImportError as exc:
        raise SystemExit(
            "Missing ASR dependency. Install optional ASR dependencies with "
            "`pip install -r paper/asr-requirements.txt` and ensure ffmpeg is available."
        ) from exc
    return torch, whisperx, DiarizationPipeline


def _process_one_audio(
    *,
    wav_path: Path,
    asr_model,
    diarize_model,
    align_cache: dict[str, tuple[object, dict]],
    whisperx,
    torch_module,
    device: str,
    language: str,
    batch_size: int,
    min_speakers: int,
    max_speakers: int,
) -> dict[str, Any]:
    audio = whisperx.load_audio(str(wav_path))
    result = asr_model.transcribe(audio, batch_size=batch_size, language=language)
    result["language"] = language

    if language not in align_cache:
        print(f"    loading align model for language={language}", flush=True)
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
        align_cache[language] = (model_a, metadata)

    model_a, metadata = align_cache[language]
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    result["language"] = language

    if diarize_model is not None:
        diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        del diarize_segments

    del audio
    _cleanup_memory(torch_module, device)
    return result


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "interview_id",
        "audio_path",
        "output_json",
        "status",
        "error",
        "n_segments",
        "language",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = build_parser().parse_args()
    skip_ids = _load_skip_ids(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (args.output_dir / "asr_manifest.csv")
    audio_items = _audio_files(args.input_root, args.audio_glob, args.recursive)
    if skip_ids:
        audio_items = [(item_id, path) for item_id, path in audio_items if item_id not in skip_ids]
    if args.limit is not None:
        audio_items = audio_items[: args.limit]

    print(f"input_root={args.input_root}", flush=True)
    print(f"output_dir={args.output_dir}", flush=True)
    print(f"audio_files={len(audio_items)}", flush=True)
    print(f"language={args.language}", flush=True)

    if not audio_items:
        _write_manifest(manifest_path, [])
        print(f"No audio files found. Wrote empty manifest to {manifest_path}", flush=True)
        return 0

    if args.dry_run:
        rows = []
        for interview_id, wav_path in audio_items:
            out_json = args.output_dir / f"{interview_id}.json"
            status = "skipped_existing" if args.skip_existing and out_json.exists() else "pending"
            rows.append(
                {
                    "interview_id": interview_id,
                    "audio_path": str(wav_path),
                    "output_json": str(out_json),
                    "status": status,
                    "error": "",
                    "n_segments": "",
                    "language": args.language,
                }
            )
        _write_manifest(manifest_path, rows)
        print(f"Dry run complete. Wrote manifest to {manifest_path}", flush=True)
        return 0

    diarization = not args.no_diarization
    hf_token = args.hf_token or os.getenv("HF_TOKEN", "")
    if diarization and not hf_token:
        raise SystemExit("HF_TOKEN is required for diarization. Pass --hf-token, set HF_TOKEN, or use --no-diarization.")

    torch_module, whisperx, DiarizationPipeline = _load_asr_runtime(diarization)
    device = _resolve_device(torch_module, args.device)
    model_name = _resolve_model_name(args.model_name, device)
    compute_type = _resolve_compute_type(args.compute_type, device)
    batch_size = args.batch_size or (16 if device == "cuda" else 4)
    temperatures = _parse_temperatures(args.temperatures)

    print(f"device={device}", flush=True)
    print(f"model_name={model_name}", flush=True)
    print(f"compute_type={compute_type}", flush=True)
    print(f"batch_size={batch_size}", flush=True)
    print(f"diarization={diarization}", flush=True)

    asr_options = {"temperatures": temperatures, "beam_size": int(args.beam_size)}
    print("Loading WhisperX ASR model...", flush=True)
    asr_model = whisperx.load_model(model_name, device, compute_type=compute_type, asr_options=asr_options)

    diarize_model = None
    if diarization:
        print("Loading pyannote diarization pipeline...", flush=True)
        diarize_model = DiarizationPipeline(token=hf_token, device=device)

    rows = []
    align_cache: dict[str, tuple[object, dict]] = {}
    processed = failed = skipped_existing = 0

    for index, (interview_id, wav_path) in enumerate(audio_items, start=1):
        out_json = args.output_dir / f"{interview_id}.json"
        if args.skip_existing and out_json.exists():
            print(f"[{index}/{len(audio_items)}] skip {interview_id}: output exists", flush=True)
            skipped_existing += 1
            rows.append(
                {
                    "interview_id": interview_id,
                    "audio_path": str(wav_path),
                    "output_json": str(out_json),
                    "status": "skipped_existing",
                    "error": "",
                    "n_segments": "",
                    "language": args.language,
                }
            )
            continue

        try:
            print(f"[{index}/{len(audio_items)}] transcribing {wav_path}", flush=True)
            result = _process_one_audio(
                wav_path=wav_path,
                asr_model=asr_model,
                diarize_model=diarize_model,
                align_cache=align_cache,
                whisperx=whisperx,
                torch_module=torch_module,
                device=device,
                language=args.language,
                batch_size=batch_size,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
            )
            result["paper_asr_metadata"] = {
                "script": "paper/scripts/asr_whisperx.py",
                "audio_file": str(wav_path),
                "interview_id": interview_id,
                "model_name": model_name,
                "language": args.language,
                "device": device,
                "compute_type": compute_type,
                "batch_size": batch_size,
                "beam_size": int(args.beam_size),
                "temperatures": temperatures,
                "diarization": diarization,
                "min_speakers": int(args.min_speakers),
                "max_speakers": int(args.max_speakers),
            }

            json_kwargs = {"ensure_ascii": False}
            if args.indent > 0:
                json_kwargs["indent"] = args.indent
            out_json.write_text(json.dumps(result, **json_kwargs), encoding="utf-8")
            n_segments = len(result.get("segments", []))
            processed += 1
            rows.append(
                {
                    "interview_id": interview_id,
                    "audio_path": str(wav_path),
                    "output_json": str(out_json),
                    "status": "processed",
                    "error": "",
                    "n_segments": n_segments,
                    "language": args.language,
                }
            )
            del result
            _cleanup_memory(torch_module, device)
        except Exception as exc:
            failed += 1
            print(f"[{index}/{len(audio_items)}] ERROR {interview_id}: {exc}", file=sys.stderr, flush=True)
            rows.append(
                {
                    "interview_id": interview_id,
                    "audio_path": str(wav_path),
                    "output_json": str(out_json),
                    "status": "failed",
                    "error": str(exc),
                    "n_segments": "",
                    "language": args.language,
                }
            )
            _cleanup_memory(torch_module, device)
            if args.fail_fast:
                break

    _write_manifest(manifest_path, rows)
    print(
        f"Done. processed={processed}, failed={failed}, skipped_existing={skipped_existing}, "
        f"manifest={manifest_path}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
