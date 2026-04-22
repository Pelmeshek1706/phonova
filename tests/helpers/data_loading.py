from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_ROOT = REPO_ROOT / "tests" / "data"
RAW_ROOT = TEST_DATA_ROOT / "raw"
MOCK_ROOT = TEST_DATA_ROOT / "mock"

LANGUAGE_ALIASES = {
    "en": "en",
    "eng": "en",
    "ua": "ua",
    "uk": "ua",
    "ukr": "ua",
}

MOCK_KINDS = ("summary_sc", "turns", "words")


@dataclass(frozen=True)
class MockTriplet:
    summary: pd.DataFrame
    turns: pd.DataFrame
    words: pd.DataFrame


def normalize_language(language: str) -> str:
    key = str(language).strip().lower()
    if key not in LANGUAGE_ALIASES:
        raise ValueError(f"Unsupported language value: {language!r}")
    return LANGUAGE_ALIASES[key]


def _language_root(base: Path, language: str) -> Path:
    canonical = normalize_language(language)
    root = base / canonical
    if not root.exists():
        raise FileNotFoundError(f"Missing language directory: {root}")
    return root


def discover_case_ids(language: str) -> list[int]:
    root = _language_root(RAW_ROOT, language)
    case_ids = sorted(int(path.stem) for path in root.glob("*.json"))
    if not case_ids:
        raise FileNotFoundError(f"No raw JSON fixtures found in {root}")
    return case_ids


def discover_mock_case_ids(language: str) -> list[int]:
    root = _language_root(MOCK_ROOT, language)
    case_ids = sorted(
        int(path.stem.split("_")[-1])
        for path in root.glob("summary_sc_*.csv")
    )
    if not case_ids:
        raise FileNotFoundError(f"No mock CSV fixtures found in {root}")
    return case_ids


def discover_paired_case_ids(language: str) -> list[int]:
    raw_ids = set(discover_case_ids(language))
    mock_ids = set(discover_mock_case_ids(language))
    paired = sorted(raw_ids & mock_ids)
    if not paired:
        raise FileNotFoundError(f"No paired case IDs found for {language}")
    return paired


def discover_cross_language_case_ids() -> list[int]:
    en_ids = set(discover_paired_case_ids("en"))
    ua_ids = set(discover_paired_case_ids("ua"))
    paired = sorted(en_ids & ua_ids)
    if not paired:
        raise FileNotFoundError("No paired EN/UA case IDs found")
    return paired


def load_raw_json(language: str, case_id: int) -> dict:
    root = _language_root(RAW_ROOT, language)
    path = root / f"{int(case_id)}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw JSON fixture: {path}")
    return json.loads(path.read_text())


def load_mock_csv_triplet(language: str, case_id: int) -> MockTriplet:
    root = _language_root(MOCK_ROOT, language)
    frames = {}
    for kind in MOCK_KINDS:
        path = root / f"{kind}_{int(case_id)}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing mock CSV fixture: {path}")
        frames[kind] = pd.read_csv(path)
    return MockTriplet(summary=frames["summary_sc"], turns=frames["turns"], words=frames["words"])


def iter_case_ids(language: str) -> Iterable[int]:
    return iter(discover_paired_case_ids(language))
