from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ThresholdProfile:
    name: str
    neg: float
    pos: float
    description: str


# Standard 3-class sentiment labeling profiles for compound scores.
THRESHOLD_PROFILES: Dict[str, ThresholdProfile] = {
    "paper_strict": ThresholdProfile(
        name="paper_strict",
        neg=-0.5,
        pos=0.5,
        description="Strict paper-style cutoffs (neutral band [-0.5, 0.5]).",
    ),
    "vader_default": ThresholdProfile(
        name="vader_default",
        neg=-0.05,
        pos=0.05,
        description="Classic VADER cutoffs.",
    ),
    "moderate": ThresholdProfile(
        name="moderate",
        neg=-0.1,
        pos=0.1,
        description="Moderate cutoffs between paper_strict and vader_default.",
    ),
}

DEFAULT_THRESHOLD_PROFILE = "paper_strict"


def available_profile_names() -> List[str]:
    """Return the names of the registered sentiment threshold profiles."""
    return sorted(THRESHOLD_PROFILES.keys())


def get_threshold_profile(name: str) -> ThresholdProfile:
    """Look up a threshold profile by name."""
    try:
        return THRESHOLD_PROFILES[name]
    except KeyError as exc:  # pragma: no cover
        known = ", ".join(available_profile_names())
        raise KeyError(f"Unknown threshold profile '{name}'. Known profiles: {known}") from exc


def classify_compound(compound: float, profile: ThresholdProfile) -> str:
    """Classify a compound sentiment score with the supplied thresholds."""
    if compound >= profile.pos:
        return "positive"
    if compound <= profile.neg:
        return "negative"
    return "neutral"
