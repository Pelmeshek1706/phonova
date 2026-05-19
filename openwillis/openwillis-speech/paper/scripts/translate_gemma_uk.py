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
    ROLE_INTERVIEWER,
    ROLE_PARTICIPANT,
    ROLE_UNKNOWN,
    GenerationUsage,
    LocalGemmaJSONGenerator,
    build_batches,
    count_words,
    ensure_segments,
    iter_json_files,
    normalize_text,
    participant_id_from_payload,
    read_json,
    safe_num,
    sum_usage,
    write_json,
)


PROMPT_VERSION = "paper_translategemma_uk_v1"
DEFAULT_MODEL = "google/translategemma-27b-it"

TRANSLATION_SYSTEM_INSTRUCTIONS = (
    "You translate cleaned interview transcript segments from English into Ukrainian. "
    "Translate only source_text_en into Ukrainian. Use context_en only to resolve meaning, references, deixis, and ambiguity. "
    "Do not translate or echo context_en. Do not summarize. Do not add or omit information. "
    "Preserve hesitation, brevity, fragmentation, uncertainty, and conversational tone. "
    "If the source is a fragment, translate it as a fragment. Keep natural conversational Ukrainian, not literary rewriting. "
    "Use consistent formal second-person address: 'ви', 'вам', 'вас', 'можете', and related forms. "
    "Do not switch to informal 'ти' forms. Do not output labels, explanations, markdown, or alternative variants with slashes. "
    "Do not output parenthetical gender variants or placeholder forms like 'хотів/хотіла' or 'обірвав(ла)'. "
    "If participant_gender is male or female, use Ukrainian grammatical forms consistent with that gender only for first-person participant self-reference when required by the source. "
    "Do not use participant_gender to change meaning or infer the gender of other people. "
    "If participant_gender is unknown, choose the most neutral natural Ukrainian phrasing possible and do not guess. "
    "If the transcript is noisy or ambiguous, give the most faithful translation possible and lower confidence. "
    "If the source is clearly garbled, preserve uncertainty rather than inventing clarity."
)

TRANSLATION_USER_TEMPLATE = """
Task:
- Translate each item independently into Ukrainian.
- Translate only source_text_en.
- Use context_en only as context; do not translate it.
- For participant items, context_en is the immediately preceding interviewer turn when available.
- For interviewer or unknown items, context_en should be empty.
- participant_gender may be male, female, or unknown. Use it only for first-person participant grammar when needed.
- Return valid JSON only.
- Do not include free text outside the required fields.

Output schema:
{{
  "batch_id": integer,
  "items": [
    {{
      "segment_idx": integer,
      "text_uk": string,
      "confidence": "low" | "medium" | "high",
      "needs_review": boolean
    }}
  ]
}}

Batch metadata:
- file: {file_name}
- batch_id: {batch_id}

Items JSON:
{items_json}
""".strip()

PROMPT_LEAK_RE = re.compile(
    r"^\s*(?:CONTEXT|TARGET|QUESTION|ANSWER|SOURCE|TEXT|ЦІЛЬ|КОНТЕКСТ|ДЖЕРЕЛО|ТЕКСТ)\s*:\s*",
    flags=re.IGNORECASE,
)
PLACEHOLDER_VARIANT_RE = re.compile(r"\b[^\W\d_]{2,}/[^\W\d_]{2,}\b", flags=re.UNICODE)
PAREN_GENDER_RE = re.compile(r"[^\W\d_]{2,}\([^\W\d_]+\)", flags=re.UNICODE)
INFORMAL_UK_RE = re.compile(r"\b(ти|тобі|тебе|твоя|твоє|твої|можеш|хочеш|будеш)\b", flags=re.IGNORECASE)


@dataclass
class FileReport:
    file: str
    n_segments: int
    translated_segments: int
    participant_segments: int
    interviewer_segments: int
    unknown_segments: int
    review_segments: int
    translated_words: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def normalize_gender(value: Any) -> str:
    raw = normalize_text(value).lower()
    if raw in {"male", "m", "man", "masculine", "чоловік", "чоловіча", "ч"}:
        return "male"
    if raw in {"female", "f", "woman", "feminine", "жінка", "жіноча", "ж"}:
        return "female"
    return "unknown"


def load_gender_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Participant" not in reader.fieldnames or "gender" not in reader.fieldnames:
            raise ValueError(f"{path}: expected Participant and gender columns")
        return {
            normalize_text(row.get("Participant", "")): normalize_gender(row.get("gender", ""))
            for row in reader
            if normalize_text(row.get("Participant", ""))
        }


def previous_interviewer_text(segments: list[dict[str, Any]], index: int) -> str:
    for prev in range(index - 1, -1, -1):
        if segments[prev].get("role") == ROLE_INTERVIEWER:
            return normalize_text(segments[prev].get("source_text_en", segments[prev].get("text", "")))
    return ""


def make_payload_item(segments: list[dict[str, Any]], index: int, participant_gender: str) -> dict[str, Any]:
    seg = segments[index]
    role = str(seg.get("role") or ROLE_UNKNOWN)
    item = {
        "segment_idx": index,
        "role": role,
        "start": safe_num(seg.get("start"), 0.0),
        "end": safe_num(seg.get("end"), 0.0),
        "context_en": previous_interviewer_text(segments, index) if role == ROLE_PARTICIPANT else "",
        "source_text_en": normalize_text(seg.get("source_text_en", seg.get("text", ""))),
    }
    if role == ROLE_PARTICIPANT:
        item["participant_gender"] = participant_gender
    return item


def validate_translation_result(payload: dict[str, Any], expected: list[int], batch_id: int) -> dict[int, dict[str, Any]]:
    if int(payload.get("batch_id", -1)) != int(batch_id):
        raise ValueError(f"batch_id mismatch: got {payload.get('batch_id')}, expected {batch_id}")
    rows = payload.get("items", [])
    if not isinstance(rows, list):
        raise ValueError("items must be a list")
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("segment_idx"), int):
            continue
        if row.get("confidence") not in {"low", "medium", "high"}:
            raise ValueError(f"invalid confidence: {row.get('confidence')}")
        out[int(row["segment_idx"])] = row
    missing = [idx for idx in expected if idx not in out]
    if missing:
        raise ValueError(f"missing segment_idx values: {missing}")
    return out


def clean_model_translation(text: Any) -> str:
    value = normalize_text(text)
    value = PROMPT_LEAK_RE.sub("", value)
    return normalize_text(value)


def translation_qa_flags(source_text_en: str, text_uk: str) -> list[str]:
    flags: list[str] = []
    source = normalize_text(source_text_en)
    target = normalize_text(text_uk)
    if not target:
        return ["empty_translation"]
    if PROMPT_LEAK_RE.search(text_uk):
        flags.append("prompt_leak")
    if PLACEHOLDER_VARIANT_RE.search(target):
        flags.append("placeholder_variant")
    if PAREN_GENDER_RE.search(target):
        flags.append("parenthetical_gender_variant")
    if source and target.lower() == source.lower():
        flags.append("unchanged_from_source")
    if INFORMAL_UK_RE.search(target):
        flags.append("informal_address")
    latin_chars = sum(1 for char in target if "A" <= char <= "Z" or "a" <= char <= "z")
    alpha_chars = sum(1 for char in target if char.isalpha())
    if alpha_chars and latin_chars / alpha_chars > 0.45:
        flags.append("too_much_latin_text")
    if source and len(target) > max(40, len(source) * 4):
        flags.append("length_expansion")
    return flags


def translate_segments(
    generator: LocalGemmaJSONGenerator,
    file_name: str,
    segments: list[dict[str, Any]],
    participant_gender: str,
    translate_roles: tuple[str, ...],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[dict[int, dict[str, Any]], GenerationUsage]:
    target_indices = [
        index
        for index, seg in enumerate(segments)
        if str(seg.get("role") or ROLE_UNKNOWN) in translate_roles
        and normalize_text(seg.get("source_text_en", seg.get("text", "")))
    ]
    translated: dict[int, dict[str, Any]] = {}
    usages: list[GenerationUsage] = []
    for batch_id, batch in enumerate(build_batches(target_indices, batch_size)):
        items = [make_payload_item(segments, index, participant_gender) for index in batch]
        user_text = TRANSLATION_USER_TEMPLATE.format(
            file_name=file_name,
            batch_id=batch_id,
            items_json=json.dumps(items, ensure_ascii=False),
        )
        data, usage = generator.generate_json(
            system_text=TRANSLATION_SYSTEM_INSTRUCTIONS,
            user_text=user_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        usages.append(usage)
        mapped = validate_translation_result(data, batch, batch_id)
        for index in batch:
            row = mapped[index]
            text_uk = clean_model_translation(row.get("text_uk", ""))
            if not text_uk:
                raise ValueError(f"empty translation for {file_name} segment {index}")
            flags = translation_qa_flags(segments[index].get("source_text_en", segments[index].get("text", "")), text_uk)
            translated[index] = {
                "segment_idx": index,
                "text_uk": text_uk,
                "confidence": row["confidence"],
                "needs_review": bool(row.get("needs_review", False) or flags),
                "qa_flags": flags,
            }
    return translated, sum_usage(usages)


def source_copy_segments(
    segments: list[dict[str, Any]],
    translate_roles: tuple[str, ...],
) -> tuple[dict[int, dict[str, Any]], GenerationUsage]:
    translated: dict[int, dict[str, Any]] = {}
    for index, seg in enumerate(segments):
        role = str(seg.get("role") or ROLE_UNKNOWN)
        source = normalize_text(seg.get("source_text_en", seg.get("text", "")))
        if role in translate_roles and source:
            translated[index] = {
                "segment_idx": index,
                "text_uk": source,
                "confidence": "low",
                "needs_review": True,
                "qa_flags": ["source_copy_backend"],
            }
    return translated, GenerationUsage()


def build_translated_segments(
    source_segments: list[dict[str, Any]],
    translated: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, seg in enumerate(source_segments):
        source = normalize_text(seg.get("source_text_en", seg.get("text", "")))
        row = translated.get(index)
        role = str(seg.get("role") or ROLE_UNKNOWN)
        item: dict[str, Any] = {
            "id": len(out),
            "start": safe_num(seg.get("start"), 0.0),
            "end": safe_num(seg.get("end"), 0.0),
            "text": row["text_uk"] if row is not None else source,
            "source_text_en": source,
            "role": role,
            "translated": row is not None,
            "translation_confidence": row["confidence"] if row is not None else None,
            "translation_needs_review": bool(row["needs_review"]) if row is not None else False,
            "translation_qa_flags": list(row["qa_flags"]) if row is not None else [],
            "words": [],
        }
        for key in [
            "source_turn_idx",
            "source_word_start_idx",
            "source_word_end_idx",
            "decision_source",
            "turn_role_decision",
            "needs_review",
        ]:
            if key in seg:
                item[key] = seg[key]
        out.append(item)
    return out


def build_view(
    base_data: dict[str, Any],
    translated_segments: list[dict[str, Any]],
    translation_meta: dict[str, Any],
    role: str | None,
) -> dict[str, Any]:
    selected = translated_segments if role is None else [seg for seg in translated_segments if seg.get("role") == role]
    out = dict(base_data)
    out["segments"] = selected
    out["word_segments"] = []
    out["text"] = " ".join(normalize_text(seg.get("text", "")) for seg in selected).strip()
    out["source_text_en"] = " ".join(normalize_text(seg.get("source_text_en", "")) for seg in selected).strip()
    out["translation_meta"] = dict(translation_meta)
    out["translation_meta"]["view_role"] = role if role is not None else "all_roles"
    if "cleanup_meta" in base_data:
        out["source_cleanup_meta"] = base_data["cleanup_meta"]
    if role is None and "turn_decisions" in base_data:
        out["turn_decisions"] = base_data["turn_decisions"]
    return out


def process_file(
    path: Path,
    output_dir: Path,
    *,
    backend: str,
    generator: LocalGemmaJSONGenerator | None,
    gender_map: dict[str, str],
    participant_meta_csv: Path | None,
    model: str,
    translate_roles: tuple[str, ...],
    batch_size: int,
    temperature: float,
    max_new_tokens: int,
    indent: int,
) -> FileReport:
    base_data = read_json(path)
    segments = ensure_segments(base_data, path)
    participant_id = participant_id_from_payload(path, base_data)
    participant_gender = gender_map.get(participant_id, "unknown")

    if backend == "source-copy":
        translated, usage = source_copy_segments(segments, translate_roles)
    else:
        if generator is None:
            raise RuntimeError("local-gemma backend requires a loaded generator")
        translated, usage = translate_segments(
            generator=generator,
            file_name=path.name,
            segments=segments,
            participant_gender=participant_gender,
            translate_roles=translate_roles,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    translated_segments = build_translated_segments(segments, translated)
    translation_meta = {
        "method": "translategemma_segment_translation",
        "prompt_version": PROMPT_VERSION,
        "backend": backend,
        "model": model,
        "target_lang": "Ukrainian",
        "translate_roles": list(translate_roles),
        "participant_id": participant_id,
        "participant_gender": participant_gender,
        "participant_gender_source": str(participant_meta_csv) if participant_meta_csv is not None else None,
        "n_segments_before": len(segments),
        "n_segments_after": len(translated_segments),
        "translated_segments": len(translated),
        "generation_usage": asdict(usage),
        "word_alignment": "not_available_for_translated_text",
        "timing_policy": "segment start/end preserved; translated word timings intentionally omitted",
    }
    views = {
        "role_labeled": build_view(base_data, translated_segments, translation_meta, None),
        "participant_only": build_view(base_data, translated_segments, translation_meta, ROLE_PARTICIPANT),
        "interviewer_only": build_view(base_data, translated_segments, translation_meta, ROLE_INTERVIEWER),
        "unknown_only": build_view(base_data, translated_segments, translation_meta, ROLE_UNKNOWN),
    }
    for subdir, payload in views.items():
        write_json(output_dir / subdir / path.name, payload, indent=indent)

    by_role = {role: [seg for seg in translated_segments if seg.get("role") == role] for role in ALL_ROLES}
    return FileReport(
        file=path.name,
        n_segments=len(translated_segments),
        translated_segments=len(translated),
        participant_segments=len(by_role[ROLE_PARTICIPANT]),
        interviewer_segments=len(by_role[ROLE_INTERVIEWER]),
        unknown_segments=len(by_role[ROLE_UNKNOWN]),
        review_segments=sum(1 for seg in translated_segments if seg.get("translation_needs_review")),
        translated_words=sum(count_words(seg.get("text", "")) for seg in translated_segments),
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )


def parse_roles(values: list[str]) -> tuple[str, ...]:
    if not values:
        return ALL_ROLES
    out: list[str] = []
    for value in values:
        for piece in value.split(","):
            role = piece.strip().lower()
            if not role:
                continue
            if role not in ALL_ROLES:
                raise SystemExit(f"Invalid role in --translate-roles: {role}")
            if role not in out:
                out.append(role)
    return tuple(out)


def write_report(rows: list[FileReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate cleaned English interview segments to Ukrainian with TranslateGemma-style paper rules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Cleaned role-labeled JSON file or directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--backend", choices=["local-gemma", "source-copy"], default="local-gemma")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--participant-meta-csv", type=Path, default=None, help="Optional CSV with Participant and gender columns.")
    parser.add_argument("--translate-roles", nargs="*", default=list(ALL_ROLES))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=5000)
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
    translate_roles = parse_roles(args.translate_roles)
    gender_map = load_gender_map(args.participant_meta_csv)
    generator = None
    if args.backend == "local-gemma":
        generator = LocalGemmaJSONGenerator(args.model, device_map=args.device_map, torch_dtype=args.torch_dtype)

    reports: list[FileReport] = []
    for index, path in enumerate(files, start=1):
        target = args.output_dir / "role_labeled" / path.name
        if target.exists() and not args.overwrite:
            print(f"[{index}/{len(files)}] skip existing {path.name}")
            continue
        print(f"[{index}/{len(files)}] process {path.name}")
        reports.append(
            process_file(
                path=path,
                output_dir=args.output_dir,
                backend=args.backend,
                generator=generator,
                gender_map=gender_map,
                participant_meta_csv=args.participant_meta_csv,
                model=args.model,
                translate_roles=translate_roles,
                batch_size=args.batch_size,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                indent=args.indent,
            )
        )

    if args.report_csv is None:
        args.report_csv = args.output_dir / "translation_manifest.csv"
    if reports:
        write_report(reports, args.report_csv)
    print(json.dumps({"processed_files": len(reports), "report_csv": str(args.report_csv)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
