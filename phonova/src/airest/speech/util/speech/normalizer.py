from __future__ import annotations

import re
from typing import List
import simplemma 


class UkrainianMorphNormalizer:
    """
    UA token normalization for lexicon lookup.

    Strategy:
    - Surface lowercase form.
    - Lemma via simplemma (if available).
    - Heuristic stem/lemma candidates as a   fallback.
    """

    _SUFFIXES = (
        "юватимуться",
        "уватимуться",
        "уватиметься",
        "юватиметься",
        "итимуться",
        "атимуться",
        "ятимуться",
        "еться",
        "ються",
        "ого",
        "ому",
        "ими",
        "ій",
        "им",
        "их",
        "ою",
        "ею",
        "ами",
        "ями",
        "ові",
        "еві",
        "єві",
        "ах",
        "ях",
        "ів",
        "їв",
        "ев",
        "ям",
        "єм",
        "ом",
        "ем",
        "ий",
        "ій",
    )
    _UA_WORD_RE = re.compile(r"^[а-щьюяґєії'-]+$", flags=re.IGNORECASE)

    def __init__(self) -> None:
        """Record whether optional simplemma lemmatization is available."""
        self.has_simplemma = simplemma is not None

    def lemmatize(self, token: str) -> str:
        """Return a lowercase lemma for a Ukrainian token when possible."""
        tok = token.lower()
        if not tok:
            return tok
        if not self.has_simplemma:
            return tok
        lemma = simplemma.lemmatize(tok, lang="uk")
        if not lemma:
            return tok
        return str(lemma).lower()

    def _heuristic_forms(self, token: str) -> List[str]:
        """Generate conservative stem-like variants for fallback lexicon matching."""
        tok = token.lower()
        out: List[str] = []
        if not self._UA_WORD_RE.match(tok):
            return out

        for suffix in self._SUFFIXES:
            if len(tok) - len(suffix) < 3:
                continue
            if tok.endswith(suffix):
                stem = tok[: -len(suffix)]
                if len(stem) < 3:
                    continue
                out.append(stem)
                # Conservative suffix restoration to reduce false positives from short/function words.
                out.append(stem + "а")
                out.append(stem + "я")
                out.append(stem + "ий")
                out.append(stem + "ій")
                break

        return out

    @staticmethod
    def _unique(values: List[str]) -> List[str]:
        """Preserve order while removing empty and duplicate values."""
        seen = set()
        unique_values: List[str] = []
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            unique_values.append(value)
        return unique_values

    def word_candidates(self, token: str) -> List[str]:
        """Return normalized lookup candidates for a single token."""
        tok = token.lower()
        lemma = self.lemmatize(tok)
        candidates = [tok, lemma]
        candidates.extend(self._heuristic_forms(tok))
        return self._unique(candidates)

    def phrase_candidates(self, tokens: List[str]) -> List[str]:
        """Return surface, lemma, and stem-like variants for a token phrase."""
        if not tokens:
            return []
        surface = " ".join(tokens)
        lemmas = [self.lemmatize(token) for token in tokens]
        lemma_phrase = " ".join(lemmas)
        stems: List[str] = []
        for token in tokens:
            tok = token.lower()
            heur = self._heuristic_forms(tok)
            stems.append(heur[0] if heur else tok)
        stem_phrase = " ".join(stems)
        return self._unique([surface, lemma_phrase, stem_phrase])
