from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROLE_PARTICIPANT = "participant"
ROLE_INTERVIEWER = "interviewer"
ROLE_MIXED = "mixed"
ROLE_UNKNOWN = "unknown"

CONCRETE_ROLES = (ROLE_PARTICIPANT, ROLE_INTERVIEWER, ROLE_UNKNOWN)
ALL_ROLES = (ROLE_PARTICIPANT, ROLE_INTERVIEWER, ROLE_UNKNOWN)


@dataclass
class GenerationUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def normalize_text(text: Any) -> str:
    value = str(text or "").replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", value).strip()


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


def iter_json_files(input_path: Path, pattern: str = "*.json") -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(path for path in input_path.glob(pattern) if path.is_file())
        if files:
            return files
    raise FileNotFoundError(f"No JSON files found at {input_path} with pattern {pattern}")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected top-level JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any], indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {"ensure_ascii": False}
    if indent > 0:
        kwargs["indent"] = indent
    path.write_text(json.dumps(payload, **kwargs), encoding="utf-8")


def build_batches(items: list[Any], batch_size: int) -> list[list[Any]]:
    if not items:
        return []
    if batch_size <= 0:
        return [items]
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def parse_json_response(text: str) -> dict[str, Any]:
    payload = normalize_text(text)
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*", "", payload)
        payload = re.sub(r"\s*```$", "", payload)
    if not payload.startswith("{"):
        start = payload.find("{")
        end = payload.rfind("}")
        if start >= 0 and end > start:
            payload = payload[start : end + 1]
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON must be an object")
    return parsed


def count_words(text: Any) -> int:
    return len([tok for tok in normalize_text(text).split(" ") if tok])


def role_from_value(value: Any) -> str:
    role = normalize_text(value).lower()
    if role in {ROLE_PARTICIPANT, "patient", "subject", "client", "candidate"}:
        return ROLE_PARTICIPANT
    if role in {ROLE_INTERVIEWER, "interview", "agent", "ellie", "clinician", "therapist"}:
        return ROLE_INTERVIEWER
    if role == ROLE_MIXED:
        return ROLE_MIXED
    return ROLE_UNKNOWN


def words_to_text(words: list[dict[str, Any]]) -> str:
    return normalize_text(" ".join(str(word.get("word", "")) for word in words))


def ensure_segments(payload: dict[str, Any], path: Path | None = None) -> list[dict[str, Any]]:
    segments = payload.get("segments", [])
    if not isinstance(segments, list):
        label = f"{path}: " if path else ""
        raise ValueError(f"{label}segments must be a list")
    return [seg for seg in segments if isinstance(seg, dict)]


def participant_id_from_payload(path: Path, payload: dict[str, Any]) -> str:
    for key in ("Participant", "participant", "participant_id", "id"):
        if key in payload and normalize_text(payload[key]):
            return normalize_text(payload[key])
    return path.stem


class LocalGemmaJSONGenerator:
    def __init__(
        self,
        model_name: str,
        *,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "Missing Gemma dependency. Install optional dependencies with "
                "`pip install -r paper/gemma-requirements.txt`."
            ) from exc

        dtype: Any = torch_dtype
        if torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float32":
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        self.model_name = model_name

    def _prompt(self, system_text: str, user_text: str):
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    tokenize=True,
                )
            except Exception:
                pass
        text = f"System:\n{system_text}\n\nUser:\n{user_text}\n\nAssistant:\n"
        return self.tokenizer(text, return_tensors="pt").input_ids

    def generate_json(
        self,
        *,
        system_text: str,
        user_text: str,
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> tuple[dict[str, Any], GenerationUsage]:
        import torch

        input_ids = self._prompt(system_text, user_text)
        input_ids = input_ids.to(self.model.device)
        attention_mask = torch.ones_like(input_ids)
        do_sample = bool(temperature and temperature > 0)
        with torch.no_grad():
            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if do_sample:
                generate_kwargs["temperature"] = float(temperature)
            output = self.model.generate(**generate_kwargs)
        new_tokens = output[0, input_ids.shape[-1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        usage = GenerationUsage(
            prompt_tokens=int(input_ids.shape[-1]),
            completion_tokens=int(new_tokens.shape[-1]),
            total_tokens=int(output.shape[-1]),
        )
        return parse_json_response(text), usage


def sum_usage(items: Iterable[GenerationUsage]) -> GenerationUsage:
    total = GenerationUsage()
    for item in items:
        total.prompt_tokens += item.prompt_tokens
        total.completion_tokens += item.completion_tokens
        total.total_tokens += item.total_tokens
    return total
