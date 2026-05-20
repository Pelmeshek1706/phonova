#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from _gemma_common import (
    ALL_ROLES,
    CONCRETE_ROLES,
    ROLE_INTERVIEWER,
    ROLE_MIXED,
    ROLE_PARTICIPANT,
    ROLE_UNKNOWN,
    GenerationUsage,
    LocalGemmaJSONGenerator,
    build_batches,
    ensure_segments,
    iter_json_files,
    normalize_text,
    participant_id_from_payload,
    read_json,
    role_from_value,
    safe_num,
    sum_usage,
    words_to_text,
    write_json,
)


PROMPT_VERSION = "paper_gemma_role_cleanup_v1"
DEFAULT_MODEL = "google/gemma-3-27b-it"

TURN_SYSTEM_INSTRUCTIONS = (
    "You classify transcript turns for leakage-aware cleanup of a two-party clinical-style interview. "
    "Use local conversational context and timestamps to assign each raw WhisperX turn to exactly one role: "
    "participant, interviewer, mixed, or unknown. "
    "Do not rewrite, summarize, translate, normalize, or correct transcript text. "
    "participant means participant speech only, including quoted interviewer speech inside participant narrative. "
    "interviewer means interview prompts, follow-ups, acknowledgments, backchannels, closings, or agent speech. "
    "mixed means the turn plausibly contains both participant and interviewer speech. "
    "unknown means setup chatter, hardware instructions, operator talk, non-speech artifacts, or unreliable role assignment. "
    "Be conservative about dropping participant content. If uncertain between participant and interviewer, choose unknown or mixed."
)

TURN_USER_TEMPLATE = """
Task:
- Classify each turn independently while using prev_text and next_text only for role disambiguation.
- Return valid JSON only.
- Do not include explanations outside the required fields.

Output schema:
{{
  "batch_id": integer,
  "turns": [
    {{
      "turn_idx": integer,
      "role": "participant" | "interviewer" | "mixed" | "unknown",
      "confidence": "low" | "medium" | "high",
      "needs_word_ranges": boolean,
      "reason": "participant_narrative" | "question_or_prompt" | "setup_or_boilerplate" | "mixed_content" | "ambiguous_short_turn" | "backchannel_or_closing" | "reported_speech" | "insufficient_context"
    }}
  ]
}}

Batch metadata:
- file: {file_name}
- batch_id: {batch_id}

Turns JSON:
{turns_json}
""".strip()

WORD_SYSTEM_INSTRUCTIONS = (
    "You resolve mixed transcript turns into contiguous word-index spans while preserving exact original word order. "
    "Input contains indexed WhisperX words from one raw turn plus neighboring context. "
    "Return spans over the provided word indices only. Do not rewrite words, skip words intentionally, reorder words, or invent text. "
    "Use participant for participant content, interviewer for prompts/backchannels/closings, and unknown for ambiguity or setup chatter."
)

WORD_USER_TEMPLATE = """
Task:
- For each turn, assign ordered non-overlapping contiguous spans over the provided word indices.
- Prefer full coverage of every word; use role=unknown for ambiguous leftovers.
- Use inclusive start_word_idx and end_word_idx.
- Return valid JSON only.
- Do not include explanations outside the required fields.

Output schema:
{{
  "batch_id": integer,
  "turns": [
    {{
      "turn_idx": integer,
      "resolution": "participant" | "interviewer" | "mixed" | "unknown",
      "needs_review": boolean,
      "spans": [
        {{
          "role": "participant" | "interviewer" | "unknown",
          "start_word_idx": integer,
          "end_word_idx": integer
        }}
      ]
    }}
  ]
}}

Batch metadata:
- file: {file_name}
- batch_id: {batch_id}

Turns JSON:
{turns_json}
""".strip()


@dataclass
class FileReport:
    file: str
    n_turns: int
    turn_pass_turns: int
    word_pass_turns: int
    participant_segments: int
    interviewer_segments: int
    unknown_segments: int
    participant_words: int
    interviewer_words: int
    unknown_words: int
    review_turns: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def make_turn_payload(segments: list[dict[str, Any]], turn_idx: int) -> dict[str, Any]:
    seg = segments[turn_idx]
    return {
        "turn_idx": int(turn_idx),
        "start": safe_num(seg.get("start"), 0.0),
        "end": safe_num(seg.get("end"), 0.0),
        "text": normalize_text(seg.get("text", "")),
        "prev_text": normalize_text(segments[turn_idx - 1].get("text", "")) if turn_idx > 0 else "",
        "next_text": normalize_text(segments[turn_idx + 1].get("text", "")) if turn_idx + 1 < len(segments) else "",
    }


def make_word_payload(segments: list[dict[str, Any]], turn_idx: int, decision: dict[str, Any]) -> dict[str, Any]:
    seg = segments[turn_idx]
    words = seg.get("words", []) or []
    return {
        "turn_idx": int(turn_idx),
        "text": normalize_text(seg.get("text", "")),
        "prev_text": normalize_text(segments[turn_idx - 1].get("text", "")) if turn_idx > 0 else "",
        "next_text": normalize_text(segments[turn_idx + 1].get("text", "")) if turn_idx + 1 < len(segments) else "",
        "role_hint": decision.get("role", ROLE_MIXED),
        "reason_hint": decision.get("reason", "mixed_content"),
        "confidence_hint": decision.get("confidence", "low"),
        "words": [{"idx": index, "word": normalize_text(word.get("word", ""))} for index, word in enumerate(words)],
    }


def validate_turn_result(payload: dict[str, Any], expected: list[int], batch_id: int) -> dict[int, dict[str, Any]]:
    if int(payload.get("batch_id", -1)) != int(batch_id):
        raise ValueError(f"batch_id mismatch: got {payload.get('batch_id')}, expected {batch_id}")
    rows = payload.get("turns", [])
    if not isinstance(rows, list):
        raise ValueError("turns must be a list")
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("turn_idx"), int):
            continue
        role = row.get("role")
        if role not in {ROLE_PARTICIPANT, ROLE_INTERVIEWER, ROLE_MIXED, ROLE_UNKNOWN}:
            raise ValueError(f"invalid turn role: {role}")
        out[int(row["turn_idx"])] = row
    missing = [idx for idx in expected if idx not in out]
    if missing:
        raise ValueError(f"missing turn_idx values: {missing}")
    return out


def validate_word_result(payload: dict[str, Any], expected: list[int], batch_id: int) -> dict[int, dict[str, Any]]:
    if int(payload.get("batch_id", -1)) != int(batch_id):
        raise ValueError(f"batch_id mismatch: got {payload.get('batch_id')}, expected {batch_id}")
    rows = payload.get("turns", [])
    if not isinstance(rows, list):
        raise ValueError("turns must be a list")
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("turn_idx"), int):
            continue
        out[int(row["turn_idx"])] = row
    missing = [idx for idx in expected if idx not in out]
    if missing:
        raise ValueError(f"missing turn_idx values: {missing}")
    return out


def resolve_existing_roles(segments: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    resolved: dict[int, dict[str, Any]] = {}
    for idx, seg in enumerate(segments):
        role = role_from_value(seg.get("role", seg.get("speaker", ROLE_UNKNOWN)))
        if role == ROLE_MIXED:
            role = ROLE_UNKNOWN
        resolved[idx] = {
            "turn_idx": idx,
            "role": role,
            "confidence": "high" if role != ROLE_UNKNOWN else "low",
            "needs_word_ranges": False,
            "reason": "participant_narrative" if role == ROLE_PARTICIPANT else "question_or_prompt" if role == ROLE_INTERVIEWER else "insufficient_context",
            "decision_source": "existing_role_field",
            "needs_review": role == ROLE_UNKNOWN,
        }
    return resolved


def resolve_turn_roles(
    generator: LocalGemmaJSONGenerator,
    file_name: str,
    segments: list[dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[dict[int, dict[str, Any]], GenerationUsage]:
    resolved: dict[int, dict[str, Any]] = {}
    usages: list[GenerationUsage] = []
    indices = list(range(len(segments)))
    for batch_id, batch in enumerate(build_batches(indices, batch_size)):
        payload = [make_turn_payload(segments, idx) for idx in batch]
        user_text = TURN_USER_TEMPLATE.format(
            file_name=file_name,
            batch_id=batch_id,
            turns_json=json.dumps(payload, ensure_ascii=False),
        )
        data, usage = generator.generate_json(
            system_text=TURN_SYSTEM_INSTRUCTIONS,
            user_text=user_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        usages.append(usage)
        mapped = validate_turn_result(data, batch, batch_id)
        for idx in batch:
            row = mapped[idx]
            resolved[idx] = {
                "turn_idx": idx,
                "role": row["role"],
                "confidence": row.get("confidence", "low"),
                "needs_word_ranges": bool(row["role"] == ROLE_MIXED or row.get("needs_word_ranges", False)),
                "reason": row.get("reason", "insufficient_context"),
                "decision_source": "gemma_turn_pass",
                "needs_review": bool(row.get("confidence") == "low" or row["role"] == ROLE_UNKNOWN),
            }
    return resolved, sum_usage(usages)


def normalize_spans(spans: list[dict[str, Any]], num_words: int) -> list[dict[str, Any]]:
    if num_words <= 0:
        return []
    cleaned: list[dict[str, Any]] = []
    for row in spans:
        role = row.get("role")
        start = row.get("start_word_idx")
        end = row.get("end_word_idx")
        if role not in CONCRETE_ROLES:
            raise ValueError(f"invalid span role: {role}")
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("span indices must be integers")
        if start < 0 or end < start or end >= num_words:
            raise ValueError(f"span out of bounds: {row}")
        cleaned.append({"role": role, "start_word_idx": start, "end_word_idx": end})
    cleaned.sort(key=lambda item: (item["start_word_idx"], item["end_word_idx"]))

    out: list[dict[str, Any]] = []
    cursor = 0
    for span in cleaned:
        start = span["start_word_idx"]
        end = span["end_word_idx"]
        if start < cursor:
            raise ValueError(f"overlapping span: {span}")
        if start > cursor:
            out.append({"role": ROLE_UNKNOWN, "start_word_idx": cursor, "end_word_idx": start - 1})
        if out and out[-1]["role"] == span["role"] and out[-1]["end_word_idx"] + 1 == start:
            out[-1]["end_word_idx"] = end
        else:
            out.append(dict(span))
        cursor = end + 1
    if cursor < num_words:
        out.append({"role": ROLE_UNKNOWN, "start_word_idx": cursor, "end_word_idx": num_words - 1})
    return out


def summarize_spans(spans: list[dict[str, Any]]) -> str:
    concrete = {span["role"] for span in spans if span["role"] != ROLE_UNKNOWN}
    has_unknown = any(span["role"] == ROLE_UNKNOWN for span in spans)
    if len(concrete) == 1 and not has_unknown:
        return next(iter(concrete))
    if not concrete:
        return ROLE_UNKNOWN
    return ROLE_MIXED


def resolve_word_spans(
    generator: LocalGemmaJSONGenerator,
    file_name: str,
    segments: list[dict[str, Any]],
    turn_decisions: dict[int, dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[dict[int, dict[str, Any]], GenerationUsage]:
    target_indices = [idx for idx, row in sorted(turn_decisions.items()) if row["role"] == ROLE_MIXED]
    resolved: dict[int, dict[str, Any]] = {}
    usages: list[GenerationUsage] = []
    for batch_id, batch in enumerate(build_batches(target_indices, batch_size)):
        payload = [make_word_payload(segments, idx, turn_decisions[idx]) for idx in batch]
        user_text = WORD_USER_TEMPLATE.format(
            file_name=file_name,
            batch_id=batch_id,
            turns_json=json.dumps(payload, ensure_ascii=False),
        )
        data, usage = generator.generate_json(
            system_text=WORD_SYSTEM_INSTRUCTIONS,
            user_text=user_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        usages.append(usage)
        mapped = validate_word_result(data, batch, batch_id)
        for idx in batch:
            words = segments[idx].get("words", []) or []
            spans = normalize_spans(list(mapped[idx].get("spans", [])), len(words))
            resolution = summarize_spans(spans)
            resolved[idx] = {
                "turn_idx": idx,
                "resolution": resolution,
                "needs_review": bool(mapped[idx].get("needs_review", False) or resolution == ROLE_UNKNOWN),
                "spans": spans,
                "decision_source": "gemma_word_pass",
            }
    return resolved, sum_usage(usages)


def build_span_segments(
    segments: list[dict[str, Any]],
    turn_decisions: dict[int, dict[str, Any]],
    word_decisions: dict[int, dict[str, Any]],
    strict_word_timing: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    span_segments: list[dict[str, Any]] = []
    turn_records: list[dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        words = seg.get("words", []) or []
        if strict_word_timing and normalize_text(seg.get("text", "")) and not words:
            raise ValueError(f"turn {idx} has text but no WhisperX words; cannot run timing-preserving cleanup")
        decision = turn_decisions[idx]
        word_decision = word_decisions.get(idx)
        if word_decision is not None:
            spans = word_decision["spans"]
            span_source = word_decision["decision_source"]
            review = bool(decision.get("needs_review", False) or word_decision.get("needs_review", False))
        else:
            role = decision["role"] if decision["role"] in CONCRETE_ROLES else ROLE_UNKNOWN
            spans = [{"role": role, "start_word_idx": 0, "end_word_idx": len(words) - 1}] if words else []
            span_source = decision["decision_source"]
            review = bool(decision.get("needs_review", False))

        resolved_spans = []
        for span in spans:
            selected_words = [dict(word) for word in words[span["start_word_idx"] : span["end_word_idx"] + 1]]
            if not selected_words:
                continue
            start = safe_num(selected_words[0].get("start"), safe_num(seg.get("start"), 0.0))
            end = safe_num(selected_words[-1].get("end"), safe_num(seg.get("end"), start))
            text = words_to_text(selected_words)
            out = {
                "id": len(span_segments),
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
                "role": span["role"],
                "source_turn_idx": int(idx),
                "source_word_start_idx": int(span["start_word_idx"]),
                "source_word_end_idx": int(span["end_word_idx"]),
                "decision_source": span_source,
                "turn_role_decision": decision["role"],
                "needs_review": review,
                "words": selected_words,
            }
            span_segments.append(out)
            resolved_spans.append(
                {
                    "role": span["role"],
                    "start_word_idx": int(span["start_word_idx"]),
                    "end_word_idx": int(span["end_word_idx"]),
                    "text": text,
                }
            )
        record: dict[str, Any] = {
            "turn_idx": int(idx),
            "start": safe_num(seg.get("start"), 0.0),
            "end": safe_num(seg.get("end"), 0.0),
            "raw_text": normalize_text(seg.get("text", "")),
            "role": decision["role"],
            "confidence": decision.get("confidence", "low"),
            "reason": decision.get("reason", "insufficient_context"),
            "needs_word_ranges": bool(decision.get("needs_word_ranges", False)),
            "needs_review": review,
            "decision_source": decision["decision_source"],
            "resolved_spans": resolved_spans,
        }
        if word_decision is not None:
            record["word_resolution"] = {
                "resolution": word_decision["resolution"],
                "needs_review": bool(word_decision["needs_review"]),
                "decision_source": word_decision["decision_source"],
            }
        turn_records.append(record)
    return span_segments, turn_records


def build_word_segments(span_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in span_segments:
        for local_idx, word in enumerate(seg.get("words", []) or []):
            item = dict(word)
            item["role"] = seg["role"]
            item["source_turn_idx"] = seg["source_turn_idx"]
            item["source_word_idx"] = seg["source_word_start_idx"] + local_idx
            out.append(item)
    return out


def build_role_view(
    base_data: dict[str, Any],
    span_segments: list[dict[str, Any]],
    cleanup_meta: dict[str, Any],
    turn_records: list[dict[str, Any]],
    role: str | None,
) -> dict[str, Any]:
    selected = span_segments if role is None else [seg for seg in span_segments if seg["role"] == role]
    out_segments: list[dict[str, Any]] = []
    for seg in selected:
        out_segments.append(
            {
                "id": len(out_segments),
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "role": seg["role"],
                "source_turn_idx": seg["source_turn_idx"],
                "source_word_start_idx": seg["source_word_start_idx"],
                "source_word_end_idx": seg["source_word_end_idx"],
                "decision_source": seg["decision_source"],
                "turn_role_decision": seg["turn_role_decision"],
                "needs_review": seg["needs_review"],
                "words": [dict(word) for word in seg.get("words", []) or []],
            }
        )
    out = dict(base_data)
    out["segments"] = out_segments
    out["word_segments"] = build_word_segments(out_segments)
    out["text"] = " ".join(normalize_text(seg.get("text", "")) for seg in out_segments).strip()
    out["cleanup_meta"] = dict(cleanup_meta)
    out["cleanup_meta"]["view_role"] = role if role is not None else "all_roles"
    if role is None:
        out["turn_decisions"] = turn_records
    return out


def process_file(
    path: Path,
    output_dir: Path,
    *,
    backend: str,
    generator: LocalGemmaJSONGenerator | None,
    word_generator: LocalGemmaJSONGenerator | None,
    model: str,
    word_model: str,
    turn_batch_size: int,
    word_batch_size: int,
    temperature: float,
    turn_max_new_tokens: int,
    word_max_new_tokens: int,
    strict_word_timing: bool,
    indent: int,
) -> FileReport:
    base_data = read_json(path)
    segments = ensure_segments(base_data, path)
    participant_id = participant_id_from_payload(path, base_data)

    if backend == "existing-roles":
        turn_decisions = resolve_existing_roles(segments)
        turn_usage = GenerationUsage()
        word_decisions: dict[int, dict[str, Any]] = {}
        word_usage = GenerationUsage()
    else:
        if generator is None or word_generator is None:
            raise RuntimeError("Gemma backend requires loaded generators")
        turn_decisions, turn_usage = resolve_turn_roles(
            generator=generator,
            file_name=path.name,
            segments=segments,
            batch_size=turn_batch_size,
            max_new_tokens=turn_max_new_tokens,
            temperature=temperature,
        )
        word_decisions, word_usage = resolve_word_spans(
            generator=word_generator,
            file_name=path.name,
            segments=segments,
            turn_decisions=turn_decisions,
            batch_size=word_batch_size,
            max_new_tokens=word_max_new_tokens,
            temperature=temperature,
        )

    span_segments, turn_records = build_span_segments(
        segments=segments,
        turn_decisions=turn_decisions,
        word_decisions=word_decisions,
        strict_word_timing=strict_word_timing,
    )
    usage = sum_usage([turn_usage, word_usage])
    cleanup_meta = {
        "method": "gemma_timing_preserving_role_cleanup",
        "prompt_version": PROMPT_VERSION,
        "backend": backend,
        "model": model,
        "word_model": word_model,
        "participant_id": participant_id,
        "n_turns_before": len(segments),
        "n_span_segments_after": len(span_segments),
        "turn_pass_turns": len(turn_decisions),
        "word_pass_turns": len(word_decisions),
        "generation_usage": asdict(usage),
        "text_policy": "output text reconstructed from original WhisperX words only",
    }
    views = {
        "role_labeled": build_role_view(base_data, span_segments, cleanup_meta, turn_records, None),
        "participant_only": build_role_view(base_data, span_segments, cleanup_meta, turn_records, ROLE_PARTICIPANT),
        "interviewer_only": build_role_view(base_data, span_segments, cleanup_meta, turn_records, ROLE_INTERVIEWER),
        "unknown_only": build_role_view(base_data, span_segments, cleanup_meta, turn_records, ROLE_UNKNOWN),
    }
    for subdir, payload in views.items():
        write_json(output_dir / subdir / path.name, payload, indent=indent)

    by_role = {role: [seg for seg in span_segments if seg["role"] == role] for role in ALL_ROLES}
    return FileReport(
        file=path.name,
        n_turns=len(segments),
        turn_pass_turns=len(turn_decisions),
        word_pass_turns=len(word_decisions),
        participant_segments=len(by_role[ROLE_PARTICIPANT]),
        interviewer_segments=len(by_role[ROLE_INTERVIEWER]),
        unknown_segments=len(by_role[ROLE_UNKNOWN]),
        participant_words=sum(len(seg.get("words", []) or []) for seg in by_role[ROLE_PARTICIPANT]),
        interviewer_words=sum(len(seg.get("words", []) or []) for seg in by_role[ROLE_INTERVIEWER]),
        unknown_words=sum(len(seg.get("words", []) or []) for seg in by_role[ROLE_UNKNOWN]),
        review_turns=sum(1 for record in turn_records if record["needs_review"]),
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
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
        description="Run paper-faithful timing-preserving Gemma role cleanup on WhisperX JSON transcripts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSON file or directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--backend", choices=["local-gemma", "existing-roles"], default="local-gemma")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--word-model", default=None, help="Defaults to --model.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--turn-batch-size", type=int, default=12)
    parser.add_argument("--word-batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--turn-max-new-tokens", type=int, default=4000)
    parser.add_argument("--word-max-new-tokens", type=int, default=5000)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict-word-timing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report-csv", type=Path, default=None)
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    files = iter_json_files(args.input, args.pattern)
    if args.max_files is not None:
        files = files[: args.max_files]
    word_model = args.word_model or args.model

    generator = None
    word_generator = None
    if args.backend == "local-gemma":
        generator = LocalGemmaJSONGenerator(args.model, device_map=args.device_map, torch_dtype=args.torch_dtype)
        word_generator = generator if word_model == args.model else LocalGemmaJSONGenerator(
            word_model,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
        )

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
                word_generator=word_generator,
                model=args.model,
                word_model=word_model,
                turn_batch_size=args.turn_batch_size,
                word_batch_size=args.word_batch_size,
                temperature=args.temperature,
                turn_max_new_tokens=args.turn_max_new_tokens,
                word_max_new_tokens=args.word_max_new_tokens,
                strict_word_timing=args.strict_word_timing,
                indent=args.indent,
            )
        )

    if args.report_csv is None:
        args.report_csv = args.output_dir / "cleanup_manifest.csv"
    if reports:
        write_report(reports, args.report_csv)
    print(json.dumps({"processed_files": len(reports), "report_csv": str(args.report_csv)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
