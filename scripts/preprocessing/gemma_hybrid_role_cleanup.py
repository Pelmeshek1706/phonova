#!/usr/bin/env python3
"""
LLM-assisted hybrid cleanup for Whisper-style interview transcripts with word timings.

Pipeline:
1) Read raw transcript turns from `segments`.
2) Run a turn-level LLM pass to classify turns as:
   participant / interviewer / mixed / unknown.
3) Run a word-index LLM pass only for mixed turns.
4) Rebuild role-labeled outputs while preserving original word timings.

Outputs under --output-dir:
- role_labeled/<file>.json
- participant_only/<file>.json
- interviewer_only/<file>.json
- unknown_only/<file>.json

The LLM never invents text for timing-preserving outputs. It only classifies
whole turns or selects contiguous word-index spans from the existing turn.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import pipeline


PROMPT_VERSION = "hybrid_role_cleanup_v2"

ROLE_PARTICIPANT = "participant"
ROLE_INTERVIEWER = "interviewer"
ROLE_MIXED = "mixed"
ROLE_UNKNOWN = "unknown"

CONCRETE_ROLES = (ROLE_PARTICIPANT, ROLE_INTERVIEWER, ROLE_UNKNOWN)

TURN_SYSTEM_INSTRUCTIONS = (
    "You classify interview transcript turns into roles for timing-preserving cleanup. "
    "Input is raw ASR turns from a two-party interview plus possible room-setup chatter. "
    "Valid roles are participant, interviewer, mixed, unknown. "
    "Use neighboring turns only for role disambiguation. "
    "Do not rewrite, summarize, translate, or normalize the transcript text. "
    "Treat room setup, hardware instructions, post-survey chatter, and operator talk as unknown. "
    "Choose mixed when a turn likely contains both participant and interviewer speech. "
    "Choose unknown instead of forcing a wrong whole-turn role."
)

TURN_USER_TASK_TEMPLATE = """
Task:
- Classify each turn independently, using prev/next turn text only as context.
- Roles:
  - participant: the turn is spoken only by the participant, including quoted or reported interviewer speech inside participant narrative.
  - interviewer: the turn is spoken only by the interview agent/interviewer.
  - mixed: the current raw turn plausibly contains both participant and interviewer speech.
  - unknown: whole-turn role cannot be assigned reliably.
- Set needs_word_ranges=true only for mixed turns that need indexed word splitting.
- Treat room setup, hardware instructions, headset/button/survey chatter, doorbell/post-survey chatter, or operator talk outside the interview as unknown.
- Be conservative about dropping participant content. If uncertain, prefer participant or unknown over interviewer.
- Short answers like "yeah", "no", "fine", "born and raised", "my husband" are usually participant unless local context clearly says otherwise.
- Short prompts like "how are you doing today", "tell me more about that", "goodbye", "thanks for sharing your thoughts with me" are interviewer.
- Setup instructions mentioning virtual human, Kinect, headset, buttons, survey, or doorbell are unknown.

Output rules:
- Return valid JSON only.
- Do not include any free-text explanation beyond the required enum fields.

Batch metadata:
- file: {file_name}
- batch_id: {batch_id}

Turns JSON:
{turns_json}
""".strip()

TURN_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "batch_id": {"type": "integer"},
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "turn_idx": {"type": "integer"},
                    "role": {
                        "type": "string",
                        "enum": [
                            ROLE_PARTICIPANT,
                            ROLE_INTERVIEWER,
                            ROLE_MIXED,
                            ROLE_UNKNOWN,
                        ],
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "needs_word_ranges": {"type": "boolean"},
                    "reason": {
                        "type": "string",
                        "enum": [
                            "participant_narrative",
                            "question_or_prompt",
                            "setup_or_boilerplate",
                            "mixed_content",
                            "ambiguous_short_turn",
                            "backchannel_or_closing",
                            "reported_speech",
                            "insufficient_context",
                        ],
                    },
                },
                "required": ["turn_idx", "role", "confidence", "needs_word_ranges", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["batch_id", "turns"],
    "additionalProperties": False,
}

WORD_SYSTEM_INSTRUCTIONS = (
    "You resolve ambiguous transcript turns into role-labeled word spans while preserving exact word order. "
    "Input is a turn, nearby context, and an indexed list of the turn's original ASR words. "
    "Output contiguous word-index spans with roles participant, interviewer, or unknown. "
    "Do not rewrite words. Do not skip or reorder words intentionally. "
    "Prefer full coverage of the indexed words. Use unknown for residual ambiguity instead of guessing."
)

WORD_USER_TASK_TEMPLATE = """
Task:
- For each turn, assign contiguous word-index spans covering the turn's words.
- Every span must use the provided word indices only.
- Spans must be non-overlapping and ordered by index.
- Prefer covering all words; use role=unknown for ambiguous leftovers.
- Use prev/next turn text only for role disambiguation.
- Treat hardware/setup/survey chatter as unknown, not interviewer.
- interviewer is for interview prompts, follow-ups, backchannels, and closings.
- participant is for participant content, even if it quotes interviewer words inside participant narrative.
- mixed or uncertain turns should still be resolved into spans if possible.

Output rules:
- Return valid JSON only.
- Use inclusive start_word_idx/end_word_idx.
- Use resolution:
  - participant/interviewer/unknown when the entire turn is that role
  - mixed when multiple concrete roles appear
  - unknown when the turn still cannot be resolved cleanly

Batch metadata:
- file: {file_name}
- batch_id: {batch_id}

Turns JSON:
{turns_json}
""".strip()

WORD_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "batch_id": {"type": "integer"},
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "turn_idx": {"type": "integer"},
                    "resolution": {
                        "type": "string",
                        "enum": [
                            ROLE_PARTICIPANT,
                            ROLE_INTERVIEWER,
                            ROLE_MIXED,
                            ROLE_UNKNOWN,
                        ],
                    },
                    "needs_review": {"type": "boolean"},
                    "spans": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "enum": [
                                        ROLE_PARTICIPANT,
                                        ROLE_INTERVIEWER,
                                        ROLE_UNKNOWN,
                                    ],
                                },
                                "start_word_idx": {"type": "integer"},
                                "end_word_idx": {"type": "integer"},
                            },
                            "required": ["role", "start_word_idx", "end_word_idx"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["turn_idx", "resolution", "needs_review", "spans"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["batch_id", "turns"],
    "additionalProperties": False,
}


@dataclass
class BatchUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class FileReport:
    file: str
    n_turns: int
    model_turn_pass_turns: int
    word_pass_turns: int
    participant_segments: int
    interviewer_segments: int
    unknown_segments: int
    participant_words: int
    interviewer_words: int
    unknown_words: int
    review_turns: int
    model_input_tokens: int
    model_output_tokens: int
    model_total_tokens: int


def normalize_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def safe_num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def iter_json_files(input_dir: Path) -> Iterable[Path]:
    return sorted(p for p in input_dir.glob("*.json") if p.is_file())


def choose_language_hint(requested: str, data_lang: str) -> str:
    if requested in {"en", "uk"}:
        return requested
    dl = str(data_lang or "").lower()
    if dl.startswith("uk") or dl.startswith("ua"):
        return "uk"
    return "en"


def canonical_model_name(name: str) -> str:
    n = (name or "").strip().lower()
    aliases = {
        "gemma-3-27b-it": "google/gemma-3-27b-it",
        "gemma3-27b-it": "google/gemma-3-27b-it",
        "google/gemma-3-27b-it": "google/gemma-3-27b-it",
    }
    return aliases.get(n, name or "google/gemma-3-27b-it")


def resolve_torch_dtype(name: str) -> Any:
    value = (name or "auto").strip().lower()
    mapping = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if value not in mapping:
        raise SystemExit(f"Unsupported --torch-dtype: {name}")
    return mapping[value]


def flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        return normalize_text(str(content.get("text", "")))
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return normalize_text("\n".join(parts))
    return ""


def extract_generated_text(output: Any) -> str:
    payload = output[0] if isinstance(output, list) and output else output
    if isinstance(payload, str) and payload.strip():
        return payload.strip()
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected local generation payload")

    generated = payload.get("generated_text")
    if isinstance(generated, str) and generated.strip():
        return generated.strip()
    if isinstance(generated, dict):
        content = flatten_message_content(generated.get("content", ""))
        if content:
            return content
    if isinstance(generated, list):
        for item in reversed(generated):
            if not isinstance(item, dict):
                continue
            if item.get("role") not in {None, "assistant"}:
                continue
            content = flatten_message_content(item.get("content", ""))
            if content:
                return content

    content = flatten_message_content(payload.get("content", ""))
    if content:
        return content
    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    raise RuntimeError("No assistant text found in local generation payload")


def parse_json_response(text: str) -> Dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start = t.find("{")
        end = t.rfind("}")
        if start >= 0 and end > start:
            return json.loads(t[start : end + 1])
        raise


class LocalTransformersClient:
    def __init__(
        self,
        device_map: str,
        torch_dtype_name: str,
        attn_implementation: Optional[str],
    ) -> None:
        self.device_map = device_map
        self.torch_dtype_name = torch_dtype_name
        self.torch_dtype = resolve_torch_dtype(torch_dtype_name)
        self.attn_implementation = attn_implementation
        self._pipelines: Dict[str, Any] = {}

    def _get_pipeline(self, model_name: str) -> Any:
        if model_name in self._pipelines:
            return self._pipelines[model_name]

        model_kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation

        pipe_kwargs: Dict[str, Any] = {
            "task": "image-text-to-text",
            "model": model_name,
            "device_map": self.device_map,
            "model_kwargs": model_kwargs,
        }
        if self.torch_dtype != "auto":
            pipe_kwargs["torch_dtype"] = self.torch_dtype

        pipe = pipeline(**pipe_kwargs)
        tokenizer = self._get_tokenizer(pipe)
        if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if eos_token_id is not None:
                tokenizer.pad_token_id = eos_token_id
        self._pipelines[model_name] = pipe
        return pipe

    @staticmethod
    def _get_tokenizer(pipe: Any) -> Any:
        tokenizer = getattr(pipe, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        processor = getattr(pipe, "processor", None)
        if processor is None:
            return None
        return getattr(processor, "tokenizer", None)

    def _estimate_usage(self, pipe: Any, messages: List[Dict[str, Any]], generated_text: str) -> BatchUsage:
        tokenizer = self._get_tokenizer(pipe)
        if tokenizer is None:
            return BatchUsage()

        input_tokens = 0
        try:
            templated = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            input_tokens = len(templated)
        except Exception:
            prompt_text = "\n\n".join(flatten_message_content(msg.get("content", "")) for msg in messages)
            input_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))

        output_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))
        return BatchUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

    def generate_json(
        self,
        model_name: str,
        system_text: str,
        user_text: str,
        schema_name: str,
        schema: Dict[str, Any],
        temperature: float,
        max_output_tokens: int,
    ) -> Tuple[Dict[str, Any], BatchUsage]:
        del schema_name
        schema_text = json.dumps(schema, ensure_ascii=False, sort_keys=True)
        schema_instructions = (
            "Return one JSON object only. It must match this JSON schema exactly. "
            "Do not wrap the answer in markdown.\n"
            f"{schema_text}"
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": f"{system_text}\n\n{schema_instructions}"}]},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]
        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_output_tokens,
            "return_full_text": False,
        }
        if temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
        else:
            generate_kwargs["do_sample"] = False

        pipe = self._get_pipeline(model_name)
        output = pipe(text=messages, **generate_kwargs)
        text = extract_generated_text(output)
        return parse_json_response(text), self._estimate_usage(pipe, messages, text)


def build_batches(items: List[Any], batch_size: int) -> List[List[Any]]:
    if not items:
        return []
    if batch_size <= 0:
        return [items]
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def make_turn_payload(segments: List[Dict[str, Any]], turn_idx: int) -> Dict[str, Any]:
    prev_text = normalize_text(segments[turn_idx - 1].get("text", "")) if turn_idx > 0 else ""
    next_text = normalize_text(segments[turn_idx + 1].get("text", "")) if turn_idx + 1 < len(segments) else ""
    seg = segments[turn_idx]
    return {
        "turn_idx": int(turn_idx),
        "start": safe_num(seg.get("start"), 0.0),
        "end": safe_num(seg.get("end"), 0.0),
        "text": normalize_text(seg.get("text", "")),
        "prev_text": prev_text,
        "next_text": next_text,
    }


def make_word_payload(
    segments: List[Dict[str, Any]],
    turn_idx: int,
    role_hint: str,
    reason_hint: str,
    confidence_hint: str,
) -> Dict[str, Any]:
    seg = segments[turn_idx]
    prev_text = normalize_text(segments[turn_idx - 1].get("text", "")) if turn_idx > 0 else ""
    next_text = normalize_text(segments[turn_idx + 1].get("text", "")) if turn_idx + 1 < len(segments) else ""
    words = seg.get("words", []) or []
    return {
        "turn_idx": int(turn_idx),
        "text": normalize_text(seg.get("text", "")),
        "prev_text": prev_text,
        "next_text": next_text,
        "role_hint": role_hint,
        "reason_hint": reason_hint,
        "confidence_hint": confidence_hint,
        "words": [{"idx": int(i), "word": normalize_text(w.get("word", ""))} for i, w in enumerate(words)],
    }


def validate_turn_batch_result(payload: Dict[str, Any], expected_turn_indices: List[int], batch_id: int) -> Dict[int, Dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Turn batch payload is not an object")
    if int(payload.get("batch_id", -1)) != int(batch_id):
        raise ValueError(f"batch_id mismatch: got {payload.get('batch_id')}, expected {batch_id}")
    rows = payload.get("turns")
    if not isinstance(rows, list):
        raise ValueError("turns must be a list")

    got: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = row.get("turn_idx")
        if not isinstance(idx, int):
            continue
        got[idx] = row

    missing = [i for i in expected_turn_indices if i not in got]
    if missing:
        raise ValueError(f"Missing turn_idx in model output: {missing}")
    return got


def validate_word_batch_result(payload: Dict[str, Any], expected_turn_indices: List[int], batch_id: int) -> Dict[int, Dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Word batch payload is not an object")
    if int(payload.get("batch_id", -1)) != int(batch_id):
        raise ValueError(f"batch_id mismatch: got {payload.get('batch_id')}, expected {batch_id}")
    rows = payload.get("turns")
    if not isinstance(rows, list):
        raise ValueError("turns must be a list")

    got: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = row.get("turn_idx")
        if not isinstance(idx, int):
            continue
        got[idx] = row

    missing = [i for i in expected_turn_indices if i not in got]
    if missing:
        raise ValueError(f"Missing turn_idx in model output: {missing}")
    return got


def call_responses_json_schema(
    client: Any,
    model: str,
    system_text: str,
    user_text: str,
    schema_name: str,
    schema: Dict[str, Any],
    temperature: float,
    max_output_tokens: int,
) -> Tuple[Dict[str, Any], BatchUsage]:
    return client.generate_json(
        model_name=model,
        system_text=system_text,
        user_text=user_text,
        schema_name=schema_name,
        schema=schema,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def resolve_turn_roles(
    client: Any,
    file_name: str,
    segments: List[Dict[str, Any]],
    initial_decisions: Dict[int, Dict[str, Any]],
    model: str,
    batch_size: int,
    temperature: float,
    max_output_tokens: int,
    sleep_seconds: float,
    max_retries: int,
) -> Tuple[Dict[int, Dict[str, Any]], BatchUsage]:
    resolved: Dict[int, Dict[str, Any]] = dict(initial_decisions)
    usage_total = BatchUsage()

    unresolved_idxs = [idx for idx in range(len(segments)) if idx not in resolved]
    batches = build_batches(unresolved_idxs, batch_size)

    for batch_id, batch_idxs in enumerate(batches):
        payload_turns = [make_turn_payload(segments, idx) for idx in batch_idxs]
        user_task = TURN_USER_TASK_TEMPLATE.format(
            file_name=file_name,
            batch_id=batch_id,
            turns_json=json.dumps(payload_turns, ensure_ascii=False),
        )

        mapped: Optional[Dict[int, Dict[str, Any]]] = None
        usage = BatchUsage()
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                data, usage = call_responses_json_schema(
                    client=client,
                    model=model,
                    system_text=TURN_SYSTEM_INSTRUCTIONS,
                    user_text=user_task,
                    schema_name="turn_role_batch",
                    schema=TURN_OUTPUT_SCHEMA,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                mapped = validate_turn_batch_result(data, batch_idxs, batch_id)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < max_retries:
                    time.sleep(min(2.0 * attempt, 8.0))
        usage_total.input_tokens += usage.input_tokens
        usage_total.output_tokens += usage.output_tokens
        usage_total.total_tokens += usage.total_tokens

        if mapped is None:
            print(f"[WARN] turn-pass failure in {file_name} batch={batch_id}: {last_err}")
            for idx in batch_idxs:
                resolved[idx] = {
                    "turn_idx": idx,
                    "role": ROLE_UNKNOWN,
                    "confidence": "low",
                    "needs_word_ranges": True,
                    "reason": "insufficient_context",
                    "decision_source": "turn_pass_fallback",
                    "needs_review": True,
                }
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            continue

        for idx in batch_idxs:
            row = mapped[idx]
            resolved[idx] = {
                "turn_idx": idx,
                "role": row["role"],
                "confidence": row["confidence"],
                "needs_word_ranges": bool(row["role"] == ROLE_MIXED),
                "reason": row["reason"],
                "decision_source": "model_turn_pass",
                "needs_review": bool(row["confidence"] == "low" or row["role"] == ROLE_UNKNOWN),
            }
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return resolved, usage_total


def normalize_and_fill_spans(spans: List[Dict[str, Any]], num_words: int) -> List[Dict[str, Any]]:
    if num_words <= 0:
        return []
    cleaned: List[Dict[str, Any]] = []
    for row in spans:
        role = row.get("role")
        s = row.get("start_word_idx")
        e = row.get("end_word_idx")
        if role not in CONCRETE_ROLES:
            raise ValueError(f"Invalid span role: {role}")
        if not isinstance(s, int) or not isinstance(e, int):
            raise ValueError("Span indices must be integers")
        if s < 0 or e < 0 or s >= num_words or e >= num_words or s > e:
            raise ValueError(f"Span out of bounds: {row}")
        cleaned.append({"role": role, "start_word_idx": s, "end_word_idx": e})

    cleaned.sort(key=lambda x: (x["start_word_idx"], x["end_word_idx"]))
    normalized: List[Dict[str, Any]] = []
    cursor = 0
    for span in cleaned:
        s = span["start_word_idx"]
        e = span["end_word_idx"]
        if s < cursor:
            raise ValueError(f"Overlapping spans detected: {span}")
        if s > cursor:
            normalized.append({"role": ROLE_UNKNOWN, "start_word_idx": cursor, "end_word_idx": s - 1})
        if normalized and normalized[-1]["role"] == span["role"] and normalized[-1]["end_word_idx"] + 1 == s:
            normalized[-1]["end_word_idx"] = e
        else:
            normalized.append(dict(span))
        cursor = e + 1
    if cursor < num_words:
        normalized.append({"role": ROLE_UNKNOWN, "start_word_idx": cursor, "end_word_idx": num_words - 1})
    return normalized


def summarize_span_roles(spans: List[Dict[str, Any]]) -> str:
    concrete = {span["role"] for span in spans if span["role"] != ROLE_UNKNOWN}
    has_unknown = any(span["role"] == ROLE_UNKNOWN for span in spans)
    if len(concrete) == 1 and not has_unknown:
        return next(iter(concrete))
    if len(concrete) == 0:
        return ROLE_UNKNOWN
    if len(concrete) == 1 and has_unknown:
        return ROLE_UNKNOWN
    return ROLE_MIXED


def concrete_fallback_role(role: str) -> str:
    if role in {ROLE_PARTICIPANT, ROLE_INTERVIEWER, ROLE_UNKNOWN}:
        return role
    return ROLE_UNKNOWN


def resolve_word_spans(
    client: Any,
    file_name: str,
    segments: List[Dict[str, Any]],
    turn_decisions: Dict[int, Dict[str, Any]],
    model: str,
    batch_size: int,
    temperature: float,
    max_output_tokens: int,
    sleep_seconds: float,
    max_retries: int,
) -> Tuple[Dict[int, Dict[str, Any]], BatchUsage]:
    target_idxs = [
        idx
        for idx, row in sorted(turn_decisions.items())
        if row["role"] == ROLE_MIXED
    ]
    usage_total = BatchUsage()
    resolved: Dict[int, Dict[str, Any]] = {}
    batches = build_batches(target_idxs, batch_size)

    for batch_id, batch_idxs in enumerate(batches):
        payload_turns = [
            make_word_payload(
                segments=segments,
                turn_idx=idx,
                role_hint=turn_decisions[idx]["role"],
                reason_hint=turn_decisions[idx]["reason"],
                confidence_hint=turn_decisions[idx]["confidence"],
            )
            for idx in batch_idxs
        ]
        user_task = WORD_USER_TASK_TEMPLATE.format(
            file_name=file_name,
            batch_id=batch_id,
            turns_json=json.dumps(payload_turns, ensure_ascii=False),
        )

        mapped: Optional[Dict[int, Dict[str, Any]]] = None
        usage = BatchUsage()
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                data, usage = call_responses_json_schema(
                    client=client,
                    model=model,
                    system_text=WORD_SYSTEM_INSTRUCTIONS,
                    user_text=user_task,
                    schema_name="turn_word_span_batch",
                    schema=WORD_OUTPUT_SCHEMA,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                mapped = validate_word_batch_result(data, batch_idxs, batch_id)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < max_retries:
                    time.sleep(min(2.0 * attempt, 8.0))
        usage_total.input_tokens += usage.input_tokens
        usage_total.output_tokens += usage.output_tokens
        usage_total.total_tokens += usage.total_tokens

        if mapped is None:
            print(f"[WARN] word-pass failure in {file_name} batch={batch_id}: {last_err}")
            for idx in batch_idxs:
                words = segments[idx].get("words", []) or []
                fallback_role = concrete_fallback_role(turn_decisions[idx]["role"])
                spans = (
                    [{"role": fallback_role, "start_word_idx": 0, "end_word_idx": len(words) - 1}]
                    if words
                    else []
                )
                resolved[idx] = {
                    "turn_idx": idx,
                    "resolution": fallback_role,
                    "needs_review": True,
                    "spans": spans,
                    "decision_source": "word_pass_fallback",
                }
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            continue

        for idx in batch_idxs:
            row = mapped[idx]
            words = segments[idx].get("words", []) or []
            try:
                spans = normalize_and_fill_spans(list(row.get("spans", [])), len(words))
                resolution = summarize_span_roles(spans)
                resolved[idx] = {
                    "turn_idx": idx,
                    "resolution": resolution if resolution != ROLE_UNKNOWN else row.get("resolution", ROLE_UNKNOWN),
                    "needs_review": bool(row.get("needs_review", False) or resolution == ROLE_UNKNOWN),
                    "spans": spans,
                    "decision_source": "model_word_pass",
                }
            except Exception as e:  # noqa: BLE001
                fallback_role = concrete_fallback_role(turn_decisions[idx]["role"])
                spans = (
                    [{"role": fallback_role, "start_word_idx": 0, "end_word_idx": len(words) - 1}]
                    if words
                    else []
                )
                print(f"[WARN] invalid word spans in {file_name} turn={idx}: {e}")
                resolved[idx] = {
                    "turn_idx": idx,
                    "resolution": fallback_role,
                    "needs_review": True,
                    "spans": spans,
                    "decision_source": "word_pass_invalid_fallback",
                }
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return resolved, usage_total


def words_to_text(words: List[Dict[str, Any]]) -> str:
    return normalize_text(" ".join(str(w.get("word", "")) for w in words))


def build_turn_spans(
    segments: List[Dict[str, Any]],
    turn_decisions: Dict[int, Dict[str, Any]],
    word_decisions: Dict[int, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    span_segments: List[Dict[str, Any]] = []
    turn_records: List[Dict[str, Any]] = []

    for idx, seg in enumerate(segments):
        decision = dict(turn_decisions[idx])
        word_decision = word_decisions.get(idx)
        words = seg.get("words", []) or []

        if word_decision is not None:
            spans = word_decision["spans"]
            span_source = word_decision["decision_source"]
            review_flag = bool(decision.get("needs_review", False) or word_decision.get("needs_review", False))
        else:
            role = decision["role"]
            if role not in {ROLE_PARTICIPANT, ROLE_INTERVIEWER, ROLE_UNKNOWN}:
                role = ROLE_UNKNOWN
                review_flag = True
                span_source = "turn_pass_unknown_fallback"
            else:
                review_flag = bool(decision.get("needs_review", False))
                span_source = decision["decision_source"]
            spans = (
                [{"role": role, "start_word_idx": 0, "end_word_idx": len(words) - 1}]
                if words
                else []
            )

        span_records: List[Dict[str, Any]] = []
        for span in spans:
            s = span["start_word_idx"]
            e = span["end_word_idx"]
            selected_words = [dict(w) for w in words[s : e + 1]]
            if not selected_words:
                continue
            text = words_to_text(selected_words)
            start = safe_num(selected_words[0].get("start"), safe_num(seg.get("start"), 0.0))
            end = safe_num(selected_words[-1].get("end"), safe_num(seg.get("end"), start))
            role = span["role"]
            span_record = {
                "id": len(span_segments),
                "start": round(float(start), 3),
                "end": round(float(end), 3),
                "text": text,
                "role": role,
                "source_turn_idx": int(idx),
                "source_word_start_idx": int(s),
                "source_word_end_idx": int(e),
                "decision_source": span_source,
                "turn_role_decision": decision["role"],
                "needs_review": review_flag,
                "words": selected_words,
            }
            span_segments.append(span_record)
            span_records.append(
                {
                    "role": role,
                    "start_word_idx": int(s),
                    "end_word_idx": int(e),
                    "text": text,
                }
            )

        turn_record = {
            "turn_idx": int(idx),
            "start": safe_num(seg.get("start"), 0.0),
            "end": safe_num(seg.get("end"), 0.0),
            "raw_text": normalize_text(seg.get("text", "")),
            "role": decision["role"],
            "confidence": decision["confidence"],
            "reason": decision["reason"],
            "needs_word_ranges": bool(decision["needs_word_ranges"]),
            "needs_review": review_flag,
            "decision_source": decision["decision_source"],
            "resolved_spans": span_records,
        }
        if word_decision is not None:
            turn_record["word_resolution"] = {
                "resolution": word_decision["resolution"],
                "needs_review": bool(word_decision["needs_review"]),
                "decision_source": word_decision["decision_source"],
            }
        turn_records.append(turn_record)

    return span_segments, turn_records


def build_word_segments(span_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    word_segments: List[Dict[str, Any]] = []
    for seg in span_segments:
        for local_idx, word in enumerate(seg.get("words", []) or []):
            item = dict(word)
            item["role"] = seg["role"]
            item["source_turn_idx"] = seg["source_turn_idx"]
            item["source_word_idx"] = seg["source_word_start_idx"] + local_idx
            word_segments.append(item)
    return word_segments


def build_role_view(
    base_data: Dict[str, Any],
    span_segments: List[Dict[str, Any]],
    role: Optional[str],
    cleanup_meta: Dict[str, Any],
    turn_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if role is None:
        selected = span_segments
    else:
        selected = [seg for seg in span_segments if seg["role"] == role]

    out_segments: List[Dict[str, Any]] = []
    for seg in selected:
        out_seg = {
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
            "words": [dict(w) for w in seg.get("words", []) or []],
        }
        out_segments.append(out_seg)

    out = dict(base_data)
    out["segments"] = out_segments
    out["word_segments"] = build_word_segments(out_segments)
    out["text"] = " ".join(normalize_text(seg["text"]) for seg in out_segments).strip()
    out["cleanup_meta"] = dict(cleanup_meta)
    out["cleanup_meta"]["view_role"] = role if role is not None else "all_roles"
    if role is None:
        out["turn_decisions"] = turn_records
    return out


def process_file(
    client: Any,
    path: Path,
    output_dir: Path,
    model: str,
    word_model: str,
    language_mode: str,
    turn_batch_size: int,
    word_batch_size: int,
    temperature: float,
    turn_max_output_tokens: int,
    word_max_output_tokens: int,
    sleep_seconds: float,
    max_retries: int,
) -> FileReport:
    base_data = json.loads(path.read_text(encoding="utf-8"))
    segments = base_data.get("segments", [])
    lang_hint = choose_language_hint(language_mode, str(base_data.get("language", "en")))

    turn_decisions: Dict[int, Dict[str, Any]] = {}

    turn_decisions, turn_usage = resolve_turn_roles(
        client=client,
        file_name=path.name,
        segments=segments,
        initial_decisions=turn_decisions,
        model=model,
        batch_size=turn_batch_size,
        temperature=temperature,
        max_output_tokens=turn_max_output_tokens,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
    )

    word_decisions, word_usage = resolve_word_spans(
        client=client,
        file_name=path.name,
        segments=segments,
        turn_decisions=turn_decisions,
        model=word_model,
        batch_size=word_batch_size,
        temperature=temperature,
        max_output_tokens=word_max_output_tokens,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
    )

    span_segments, turn_records = build_turn_spans(segments, turn_decisions, word_decisions)
    cleanup_meta = {
        "method": "gemma_hybrid_role_cleanup",
        "runtime": "local_transformers",
        "prompt_version": PROMPT_VERSION,
        "turn_model": model,
        "word_model": word_model,
        "language_hint": lang_hint,
        "n_turns_before": len(segments),
        "n_role_spans_after": len(span_segments),
        "model_turn_pass_turns": len([idx for idx, row in turn_decisions.items() if row["decision_source"] == "model_turn_pass"]),
        "word_pass_turns": len(word_decisions),
        "model_usage": asdict(
            BatchUsage(
                input_tokens=turn_usage.input_tokens + word_usage.input_tokens,
                output_tokens=turn_usage.output_tokens + word_usage.output_tokens,
                total_tokens=turn_usage.total_tokens + word_usage.total_tokens,
            )
        ),
    }

    role_labeled = build_role_view(base_data, span_segments, None, cleanup_meta, turn_records)
    participant_only = build_role_view(base_data, span_segments, ROLE_PARTICIPANT, cleanup_meta, turn_records)
    interviewer_only = build_role_view(base_data, span_segments, ROLE_INTERVIEWER, cleanup_meta, turn_records)
    unknown_only = build_role_view(base_data, span_segments, ROLE_UNKNOWN, cleanup_meta, turn_records)

    targets = {
        "role_labeled": role_labeled,
        "participant_only": participant_only,
        "interviewer_only": interviewer_only,
        "unknown_only": unknown_only,
    }
    for subdir, payload in targets.items():
        out_path = output_dir / subdir / path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    role_counts = {
        ROLE_PARTICIPANT: [seg for seg in span_segments if seg["role"] == ROLE_PARTICIPANT],
        ROLE_INTERVIEWER: [seg for seg in span_segments if seg["role"] == ROLE_INTERVIEWER],
        ROLE_UNKNOWN: [seg for seg in span_segments if seg["role"] == ROLE_UNKNOWN],
    }

    return FileReport(
        file=path.name,
        n_turns=len(segments),
        model_turn_pass_turns=len([idx for idx, row in turn_decisions.items() if row["decision_source"] == "model_turn_pass"]),
        word_pass_turns=len(word_decisions),
        participant_segments=len(role_counts[ROLE_PARTICIPANT]),
        interviewer_segments=len(role_counts[ROLE_INTERVIEWER]),
        unknown_segments=len(role_counts[ROLE_UNKNOWN]),
        participant_words=sum(len(seg.get("words", []) or []) for seg in role_counts[ROLE_PARTICIPANT]),
        interviewer_words=sum(len(seg.get("words", []) or []) for seg in role_counts[ROLE_INTERVIEWER]),
        unknown_words=sum(len(seg.get("words", []) or []) for seg in role_counts[ROLE_UNKNOWN]),
        review_turns=sum(1 for row in turn_records if row["needs_review"]),
        model_input_tokens=turn_usage.input_tokens + word_usage.input_tokens,
        model_output_tokens=turn_usage.output_tokens + word_usage.output_tokens,
        model_total_tokens=turn_usage.total_tokens + word_usage.total_tokens,
    )


def write_report(rows: List[FileReport], report_csv: Path) -> None:
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "file",
                "n_turns",
                "model_turn_pass_turns",
                "word_pass_turns",
                "participant_segments",
                "interviewer_segments",
                "unknown_segments",
                "participant_words",
                "interviewer_words",
                "unknown_words",
                "review_turns",
                "model_input_tokens",
                "model_output_tokens",
                "model_total_tokens",
            ]
        )
        for row in rows:
            w.writerow(
                [
                    row.file,
                    row.n_turns,
                    row.model_turn_pass_turns,
                    row.word_pass_turns,
                    row.participant_segments,
                    row.interviewer_segments,
                    row.unknown_segments,
                    row.participant_words,
                    row.interviewer_words,
                    row.unknown_words,
                    row.review_turns,
                    row.model_input_tokens,
                    row.model_output_tokens,
                    row.model_total_tokens,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-csv", type=Path, default=None)
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--word-model", type=str, default=None)
    parser.add_argument("--language", choices=["auto", "en", "uk"], default="auto")
    parser.add_argument("--turn-batch-size", type=int, default=12)
    parser.add_argument("--word-batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--turn-max-output-tokens", type=int, default=4000)
    parser.add_argument("--word-max-output-tokens", type=int, default=5000)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default=None,
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.model = canonical_model_name(args.model)
    args.word_model = canonical_model_name(args.word_model or args.model)

    files = list(iter_json_files(args.input_dir))
    if not files:
        raise SystemExit(f"No JSON files found in {args.input_dir}")
    if args.max_files is not None:
        files = files[: args.max_files]

    client = LocalTransformersClient(
        device_map=args.device_map,
        torch_dtype_name=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )

    reports: List[FileReport] = []
    for i, fp in enumerate(files, start=1):
        target = args.output_dir / "role_labeled" / fp.name
        if target.exists() and not args.overwrite:
            print(f"[{i}/{len(files)}] skip existing {fp.name}")
            continue
        print(f"[{i}/{len(files)}] process {fp.name}")
        rep = process_file(
            client=client,
            path=fp,
            output_dir=args.output_dir,
            model=args.model,
            word_model=args.word_model,
            language_mode=args.language,
            turn_batch_size=args.turn_batch_size,
            word_batch_size=args.word_batch_size,
            temperature=args.temperature,
            turn_max_output_tokens=args.turn_max_output_tokens,
            word_max_output_tokens=args.word_max_output_tokens,
            sleep_seconds=args.sleep_seconds,
            max_retries=args.max_retries,
        )
        reports.append(rep)

    if args.report_csv is not None and reports:
        write_report(reports, args.report_csv)

    total = FileReport(
        file="__TOTAL__",
        n_turns=sum(r.n_turns for r in reports),
        model_turn_pass_turns=sum(r.model_turn_pass_turns for r in reports),
        word_pass_turns=sum(r.word_pass_turns for r in reports),
        participant_segments=sum(r.participant_segments for r in reports),
        interviewer_segments=sum(r.interviewer_segments for r in reports),
        unknown_segments=sum(r.unknown_segments for r in reports),
        participant_words=sum(r.participant_words for r in reports),
        interviewer_words=sum(r.interviewer_words for r in reports),
        unknown_words=sum(r.unknown_words for r in reports),
        review_turns=sum(r.review_turns for r in reports),
        model_input_tokens=sum(r.model_input_tokens for r in reports),
        model_output_tokens=sum(r.model_output_tokens for r in reports),
        model_total_tokens=sum(r.model_total_tokens for r in reports),
    )
    print(json.dumps(asdict(total), ensure_ascii=False))


if __name__ == "__main__":
    main()
