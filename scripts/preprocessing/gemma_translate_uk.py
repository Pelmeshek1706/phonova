#!/usr/bin/env python3
"""
LLM-assisted translation of cleaned interview transcript segments into Ukrainian.

Input:
- role-labeled English transcript JSONs, typically from the role-cleanup pipeline

Translation policy:
- participant turns: use the previous interviewer turn as context when available
- interviewer turns: translate standalone by default
- unknown turns: translate standalone as best effort

Outputs under --output-dir:
- role_labeled/<file>.json
- participant_only/<file>.json
- interviewer_only/<file>.json
- unknown_only/<file>.json

Important:
- Segment start/end times are preserved from the English source
- No translated word timings are fabricated
- `word_segments` is intentionally empty in translated outputs
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


PROMPT_VERSION = "translate_uk_v3"


def resolve_default_participant_meta_csv() -> Path:
    rel_path = Path("datasets") / "old" / "dcwoz_eng_new_gemma_merged_full_updated_4-2-26.csv"
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / rel_path
        if candidate.exists():
            return candidate
    return here.parents[2] / rel_path


DEFAULT_PARTICIPANT_META_CSV = resolve_default_participant_meta_csv()

ROLE_PARTICIPANT = "participant"
ROLE_INTERVIEWER = "interviewer"
ROLE_UNKNOWN = "unknown"
ROLE_ORDER = (ROLE_PARTICIPANT, ROLE_INTERVIEWER, ROLE_UNKNOWN)

TRANSLATION_SYSTEM_INSTRUCTIONS = (
    "You translate interview transcript segments from English into Ukrainian. "
    "Each item contains a target segment and optional context. "
    "Translate only source_text_en into Ukrainian. "
    "Use context_en only to resolve meaning, references, deixis, and ambiguity. "
    "Do not translate or echo context_en. "
    "Do not summarize. "
    "Do not add or omit information. "
    "Preserve hesitation, brevity, fragmentation, uncertainty, and conversational tone. "
    "If the source is a fragment, translate it as a fragment. "
    "Keep natural conversational Ukrainian, not literary rewriting. "
    "Use consistent formal second-person address: 'ви', 'вам', 'вас', 'можете', and related forms. "
    "Do not switch to informal 'ти' forms. "
    "Do not output labels, explanations, markdown, or alternative variants with slashes. "
    "Do not output parenthetical gender variants or placeholder forms like 'хотів/хотіла' or 'обірвав(ла)'. "
    "If participant_gender is male or female, use Ukrainian grammatical forms consistent with that gender only for first-person participant self-reference when the source requires a gendered form. "
    "Do not use participant_gender to change meaning or to resolve the gender of other people. "
    "If participant_gender is unknown, choose the most neutral natural Ukrainian phrasing possible and do not guess. "
    "If a literal translation would sound unnatural in Ukrainian, use the closest natural conversational equivalent while preserving meaning. "
    "Translate setup or device-related language naturally, not word-for-word. "
    "For setup, interface, and device actions such as showing, pressing, ringing, or being recorded, prefer the natural Ukrainian phrasing for the situation, not a literal object-by-object translation. "
    "If the transcript is noisy or ambiguous, give the most faithful translation possible and lower confidence. "
    "If the source is clearly garbled, preserve the uncertainty rather than inventing clarity."
)

TRANSLATION_USER_TASK_TEMPLATE = """
Task:
- Translate each item independently into {target_lang}.
- Translate only source_text_en.
- Use context_en only as context; do not translate it.
- For participant items, context_en is usually the previous interviewer turn.
- For interviewer or unknown items, context_en may be empty.
- For participant items, participant_gender may be male, female, or unknown. Use it only for first-person participant grammar when needed.
- Use formal address consistently. Do not switch to informal 'ти' forms.
- If the source is uncertain because of transcript noise, incompleteness, or garbling, still return the best faithful translation but set confidence=low and needs_review=true.
- Return valid JSON only.
- Do not include any free text outside the required schema.

Batch metadata:
- file: {file_name}
- batch_id: {batch_id}

Items JSON:
{items_json}
""".strip()

TRANSLATION_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "batch_id": {"type": "integer"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_idx": {"type": "integer"},
                    "text_uk": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    "needs_review": {"type": "boolean"},
                },
                "required": ["segment_idx", "text_uk", "confidence", "needs_review"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["batch_id", "items"],
    "additionalProperties": False,
}

PROMPT_LEAK_RE = re.compile(
    r"^\s*(?:CONTEXT|TARGET|QUESTION|ANSWER|SOURCE|TEXT|ЦЕЛЬ|ЦІЛЬ|КОНТЕКСТ|ТАРГЕТ)\s*:\s*",
    flags=re.IGNORECASE,
)
PLACEHOLDER_VARIANT_RE = re.compile(
    r"\b[^\W\d_]{2,}/[^\W\d_]{2,}\b",
    flags=re.UNICODE,
)
PAREN_GENDER_RE = re.compile(r"[^\W\d_]{2,}\([^\W\d_]+\)", flags=re.UNICODE)
INFORMAL_UK_RE = re.compile(r"\b(ти|тобі|тебе|тво[яєї]|можеш|хочеш|будеш|переїхав|переїхала)\b", flags=re.IGNORECASE)


@dataclass
class BatchUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


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


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def iter_json_files(input_dir: Path) -> Iterable[Path]:
    return sorted(p for p in input_dir.glob("*.json") if p.is_file())


def normalize_gender(value: Any) -> str:
    raw = normalize_text(str(value or "")).lower()
    if raw in {"male", "m", "man", "masculine", "чоловік", "чоловіча", "ч"}:
        return "male"
    if raw in {"female", "f", "woman", "feminine", "жінка", "жіноча", "ж"}:
        return "female"
    return "unknown"


def load_participant_gender_map(csv_path: Optional[Path]) -> Dict[str, str]:
    if csv_path is None or not csv_path.exists():
        return {}

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path}: missing header row")
        if "Participant" not in reader.fieldnames or "gender" not in reader.fieldnames:
            raise ValueError(f"{csv_path}: expected 'Participant' and 'gender' columns")

        out: Dict[str, str] = {}
        for row in reader:
            participant_id = normalize_text(row.get("Participant", ""))
            if not participant_id:
                continue
            out[participant_id] = normalize_gender(row.get("gender"))
        return out


def canonical_model_name(name: str) -> str:
    n = (name or "").strip().lower()
    aliases = {
        "translategemma-27b-it": "google/translategemma-27b-it",
        "google/translategemma-27b-it": "google/translategemma-27b-it",
        "gemma-3-27b-it": "google/gemma-3-27b-it",
        "google/gemma-3-27b-it": "google/gemma-3-27b-it",
    }
    return aliases.get(n, name or "google/translategemma-27b-it")


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
    payload = text.strip()
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*", "", payload)
        payload = re.sub(r"\s*```$", "", payload)
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start >= 0 and end > start:
            return json.loads(payload[start : end + 1])
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


def count_words(text: str) -> int:
    return len([tok for tok in normalize_text(text).split(" ") if tok])


def previous_interviewer_text(segments: List[Dict[str, Any]], idx: int) -> str:
    for j in range(idx - 1, -1, -1):
        seg = segments[j]
        if seg.get("role") == ROLE_INTERVIEWER:
            return normalize_text(seg.get("text", ""))
    return ""


def make_translation_payload(
    segments: List[Dict[str, Any]],
    segment_idx: int,
    participant_gender: str,
) -> Dict[str, Any]:
    seg = segments[segment_idx]
    role = str(seg.get("role") or ROLE_UNKNOWN)
    context_en = previous_interviewer_text(segments, segment_idx) if role == ROLE_PARTICIPANT else ""
    payload = {
        "segment_idx": int(segment_idx),
        "role": role,
        "start": safe_num(seg.get("start"), 0.0),
        "end": safe_num(seg.get("end"), 0.0),
        "context_en": context_en,
        "source_text_en": normalize_text(seg.get("text", "")),
    }
    if role == ROLE_PARTICIPANT:
        payload["participant_gender"] = participant_gender
    return payload


def validate_translation_batch_result(payload: Dict[str, Any], expected_indices: List[int], batch_id: int) -> Dict[int, Dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Translation batch payload is not an object")
    if int(payload.get("batch_id", -1)) != int(batch_id):
        raise ValueError(f"batch_id mismatch: got {payload.get('batch_id')}, expected {batch_id}")
    rows = payload.get("items")
    if not isinstance(rows, list):
        raise ValueError("items must be a list")

    got: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = row.get("segment_idx")
        if not isinstance(idx, int):
            continue
        got[idx] = row

    missing = [i for i in expected_indices if i not in got]
    if missing:
        raise ValueError(f"Missing segment_idx in model output: {missing}")
    return got


def clean_model_translation(text: str) -> str:
    value = normalize_text(text)
    value = PROMPT_LEAK_RE.sub("", value)
    return normalize_text(value)


def translation_qa_flags(source_text_en: str, text_uk: str) -> List[str]:
    flags: List[str] = []
    src = normalize_text(source_text_en)
    tgt = normalize_text(text_uk)

    if not tgt:
        flags.append("empty_translation")
        return flags
    if PROMPT_LEAK_RE.search(text_uk):
        flags.append("prompt_leak")
    if PLACEHOLDER_VARIANT_RE.search(tgt):
        flags.append("placeholder_variant")
    if PAREN_GENDER_RE.search(tgt):
        flags.append("parenthetical_gender_variant")
    if src and tgt.lower() == src.lower():
        flags.append("unchanged_from_source")
    if INFORMAL_UK_RE.search(tgt):
        flags.append("informal_address")

    latin_chars = sum(1 for ch in tgt if "A" <= ch <= "Z" or "a" <= ch <= "z")
    alpha_chars = sum(1 for ch in tgt if ch.isalpha())
    if alpha_chars and latin_chars / alpha_chars > 0.45:
        flags.append("too_much_latin_text")

    if src and len(tgt) > max(40, len(src) * 4):
        flags.append("length_expansion")
    if "doorbell" in src.lower() and ("дверний дзвінок" in tgt.lower() or "подзвонити в цей" in tgt.lower()):
        flags.append("literal_doorbell_translation")
    return flags


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


def translate_segments(
    client: Any,
    file_name: str,
    segments: List[Dict[str, Any]],
    participant_gender: str,
    translate_roles: Tuple[str, ...],
    model: str,
    target_lang: str,
    batch_size: int,
    temperature: float,
    max_output_tokens: int,
    sleep_seconds: float,
    max_retries: int,
) -> Tuple[Dict[int, Dict[str, Any]], BatchUsage]:
    target_indices = []
    for idx, seg in enumerate(segments):
        role = str(seg.get("role") or ROLE_UNKNOWN)
        text = normalize_text(seg.get("text", ""))
        if role in translate_roles and text:
            target_indices.append(idx)

    translated: Dict[int, Dict[str, Any]] = {}
    usage_total = BatchUsage()
    batches = build_batches(target_indices, batch_size)

    for batch_id, batch_indices in enumerate(batches):
        payload_items = [make_translation_payload(segments, idx, participant_gender) for idx in batch_indices]
        user_task = TRANSLATION_USER_TASK_TEMPLATE.format(
            target_lang=target_lang,
            file_name=file_name,
            batch_id=batch_id,
            items_json=json.dumps(payload_items, ensure_ascii=False),
        )

        mapped: Optional[Dict[int, Dict[str, Any]]] = None
        usage = BatchUsage()
        last_err: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                data, usage = call_responses_json_schema(
                    client=client,
                    model=model,
                    system_text=TRANSLATION_SYSTEM_INSTRUCTIONS,
                    user_text=user_task,
                    schema_name="translation_batch",
                    schema=TRANSLATION_OUTPUT_SCHEMA,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                mapped = validate_translation_batch_result(data, batch_indices, batch_id)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                if "insufficient_quota" in str(e):
                    raise RuntimeError(f"Insufficient quota while translating {file_name} batch={batch_id}: {e}") from e
                if attempt < max_retries:
                    time.sleep(min(2.0 * attempt, 8.0))

        usage_total.input_tokens += usage.input_tokens
        usage_total.output_tokens += usage.output_tokens
        usage_total.total_tokens += usage.total_tokens

        if mapped is None:
            raise RuntimeError(f"Translation failed for {file_name} batch={batch_id}: {last_err}")

        for idx in batch_indices:
            row = mapped[idx]
            clean_text = clean_model_translation(row.get("text_uk", ""))
            if not clean_text:
                raise ValueError(f"Empty translated text for {file_name} segment {idx}")
            qa_flags = translation_qa_flags(segments[idx].get("text", ""), clean_text)
            translated[idx] = {
                "segment_idx": idx,
                "text_uk": clean_text,
                "confidence": row["confidence"],
                "needs_review": bool(row["needs_review"] or bool(qa_flags)),
                "qa_flags": qa_flags,
            }

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return translated, usage_total


def build_translated_segments(
    source_segments: List[Dict[str, Any]],
    translated: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, seg in enumerate(source_segments):
        source_text_en = normalize_text(seg.get("text", ""))
        trans = translated.get(idx)
        text_uk = trans["text_uk"] if trans is not None else source_text_en
        row: Dict[str, Any] = {
            "id": safe_int(seg.get("id", idx), idx),
            "start": safe_num(seg.get("start"), 0.0),
            "end": safe_num(seg.get("end"), 0.0),
            "text": text_uk,
            "source_text_en": source_text_en,
            "role": str(seg.get("role") or ROLE_UNKNOWN),
            "translated": trans is not None,
            "translation_confidence": trans["confidence"] if trans is not None else None,
            "translation_needs_review": bool(trans["needs_review"]) if trans is not None else False,
            "translation_qa_flags": list(trans["qa_flags"]) if trans is not None else [],
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
                row[key] = seg[key]
        out.append(row)
    return out


def build_view(
    base_data: Dict[str, Any],
    translated_segments: List[Dict[str, Any]],
    translation_meta: Dict[str, Any],
    role: Optional[str],
) -> Dict[str, Any]:
    if role is None:
        segments = translated_segments
    else:
        segments = [seg for seg in translated_segments if seg.get("role") == role]

    out: Dict[str, Any] = {
        "segments": segments,
        "word_segments": [],
        "text": " ".join(normalize_text(seg.get("text", "")) for seg in segments).strip(),
        "source_text_en": " ".join(normalize_text(seg.get("source_text_en", "")) for seg in segments).strip(),
        "translation_meta": dict(translation_meta),
    }
    out["translation_meta"]["view_role"] = role if role is not None else "all_roles"

    if "cleanup_meta" in base_data:
        out["source_cleanup_meta"] = base_data["cleanup_meta"]
    if role is None and "turn_decisions" in base_data:
        out["turn_decisions"] = base_data["turn_decisions"]
    return out


def process_file(
    client: Any,
    path: Path,
    output_dir: Path,
    participant_gender_map: Dict[str, str],
    participant_meta_csv: Optional[Path],
    model: str,
    target_lang: str,
    translate_roles: Tuple[str, ...],
    batch_size: int,
    temperature: float,
    max_output_tokens: int,
    sleep_seconds: float,
    max_retries: int,
) -> FileReport:
    base_data = json.loads(path.read_text(encoding="utf-8"))
    segments = base_data.get("segments", []) or []
    if not isinstance(segments, list):
        raise ValueError(f"{path.name}: segments must be a list")
    participant_id = path.stem
    participant_gender = participant_gender_map.get(participant_id, "unknown")

    translated, usage = translate_segments(
        client=client,
        file_name=path.name,
        segments=segments,
        participant_gender=participant_gender,
        translate_roles=translate_roles,
        model=model,
        target_lang=target_lang,
        batch_size=batch_size,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
    )

    translated_segments = build_translated_segments(segments, translated)
    translation_meta = {
        "method": "gemma_translate_uk",
        "runtime": "local_transformers",
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "target_lang": target_lang,
        "translate_roles": list(translate_roles),
        "participant_id": participant_id,
        "participant_gender": participant_gender,
        "participant_gender_source": str(participant_meta_csv) if participant_meta_csv is not None and participant_meta_csv.exists() else None,
        "n_segments_before": len(segments),
        "n_segments_after": len(translated_segments),
        "translated_segments": len(translated),
        "model_usage": asdict(usage),
        "word_alignment": "not_available_for_translated_text",
    }

    role_labeled = build_view(base_data, translated_segments, translation_meta, None)
    participant_only = build_view(base_data, translated_segments, translation_meta, ROLE_PARTICIPANT)
    interviewer_only = build_view(base_data, translated_segments, translation_meta, ROLE_INTERVIEWER)
    unknown_only = build_view(base_data, translated_segments, translation_meta, ROLE_UNKNOWN)

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

    participant_segments = participant_only["segments"]
    interviewer_segments = interviewer_only["segments"]
    unknown_segments = unknown_only["segments"]
    review_segments = sum(1 for seg in translated_segments if seg["translation_needs_review"])

    return FileReport(
        file=path.name,
        n_segments=len(translated_segments),
        translated_segments=len(translated),
        participant_segments=len(participant_segments),
        interviewer_segments=len(interviewer_segments),
        unknown_segments=len(unknown_segments),
        review_segments=review_segments,
        translated_words=sum(count_words(seg.get("text", "")) for seg in translated_segments),
        model_input_tokens=usage.input_tokens,
        model_output_tokens=usage.output_tokens,
        model_total_tokens=usage.total_tokens,
    )


def write_report(rows: List[FileReport], report_csv: Path) -> None:
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file",
                "n_segments",
                "translated_segments",
                "participant_segments",
                "interviewer_segments",
                "unknown_segments",
                "review_segments",
                "translated_words",
                "model_input_tokens",
                "model_output_tokens",
                "model_total_tokens",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.file,
                    row.n_segments,
                    row.translated_segments,
                    row.participant_segments,
                    row.interviewer_segments,
                    row.unknown_segments,
                    row.review_segments,
                    row.translated_words,
                    row.model_input_tokens,
                    row.model_output_tokens,
                    row.model_total_tokens,
                ]
            )


def parse_translate_roles(values: List[str]) -> Tuple[str, ...]:
    if not values:
        return ROLE_ORDER
    parsed: List[str] = []
    for value in values:
        for piece in value.split(","):
            role = piece.strip().lower()
            if not role:
                continue
            if role not in ROLE_ORDER:
                raise SystemExit(f"Invalid role in --translate-roles: {role}")
            if role not in parsed:
                parsed.append(role)
    return tuple(parsed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-csv", type=Path, default=None)
    parser.add_argument(
        "--participant-meta-csv",
        type=Path,
        default=DEFAULT_PARTICIPANT_META_CSV,
        help="CSV with Participant and gender columns. Default: project patient metadata CSV if present.",
    )
    parser.add_argument("--model", type=str, default="google/translategemma-27b-it")
    parser.add_argument("--target-lang", type=str, default="Ukrainian")
    parser.add_argument(
        "--translate-roles",
        nargs="*",
        default=list(ROLE_ORDER),
        help="Roles to translate. Default: participant interviewer unknown",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=5000)
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
    translate_roles = parse_translate_roles(args.translate_roles)
    participant_gender_map = load_participant_gender_map(args.participant_meta_csv)

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
            participant_gender_map=participant_gender_map,
            participant_meta_csv=args.participant_meta_csv,
            model=args.model,
            target_lang=args.target_lang,
            translate_roles=translate_roles,
            batch_size=args.batch_size,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            sleep_seconds=args.sleep_seconds,
            max_retries=args.max_retries,
        )
        reports.append(rep)

    if args.report_csv is not None and reports:
        write_report(reports, args.report_csv)

    total = FileReport(
        file="__TOTAL__",
        n_segments=sum(r.n_segments for r in reports),
        translated_segments=sum(r.translated_segments for r in reports),
        participant_segments=sum(r.participant_segments for r in reports),
        interviewer_segments=sum(r.interviewer_segments for r in reports),
        unknown_segments=sum(r.unknown_segments for r in reports),
        review_segments=sum(r.review_segments for r in reports),
        translated_words=sum(r.translated_words for r in reports),
        model_input_tokens=sum(r.model_input_tokens for r in reports),
        model_output_tokens=sum(r.model_output_tokens for r in reports),
        model_total_tokens=sum(r.model_total_tokens for r in reports),
    )
    print(json.dumps(asdict(total), ensure_ascii=False))


if __name__ == "__main__":
    main()
