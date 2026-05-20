from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from vaderSentiment.vaderSentiment import (
    C_INCR,
    N_SCALAR,
    SentiText,
    SPECIAL_CASES as EN_SPECIAL_CASES,
    SentimentIntensityAnalyzer as EnglishSentimentIntensityAnalyzer,
    normalize,
)
from .lexicon import load_tonsum_lexicon, summarize_lexicon
from .normalizer import UkrainianMorphNormalizer


B_INCR = 0.293
B_DECR = -0.293
ALPHA = 15.0


class UkrainianSentimentIntensityAnalyzerImproved:
    """
    VADER-like sentiment analyzer for Ukrainian.
    """

    BOOSTER_DICT = {
        "дуже": B_INCR,
        "вельми": B_INCR,
        "вкрай": B_INCR,
        "надзвичайно": B_INCR,
        "неймовірно": B_INCR,
        "максимально": B_INCR,
        "сильно": B_INCR,
        "абсолютно": B_INCR,
        "цілком": B_INCR,
        "майже": B_DECR,
        "ледве": B_DECR,
        "трохи": B_DECR,
        "дещо": B_DECR,
        "частково": B_DECR,
        "злегка": B_DECR,
    }

    NEGATIONS = {
        "не",
        "ані",
        "ні",
        "жоден",
        "жодна",
        "жодне",
        "жодного",
        "жодної",
        "ніколи",
        "ніде",
        "нікуди",
        "нітрохи",
        "немає",
        "нема",
        "без",
    }

    CONTRASTIVE_CONJ = {"але", "проте", "однак", "зате"}
    KIND_OF_BIGRAMS = {("ніби", "то"), ("типу", "того")}
    NEVER_TERMS = {"ніколи"}
    SO_THIS_TERMS = {"так", "це"}
    WITHOUT_TERMS = {"без"}
    DOUBT_TERMS = {"сумніву", "сумнів"}
    LEAST_TERMS = {"найменш", "щонайменш"}
    AT_VERY_TERMS = {"дуже", "аж"}

    def __init__(
        self,
        lexicon_path: str | Path | None = None,
        lexicon_scale_to_vader: bool = False,
        alpha: float = ALPHA,
    ) -> None:
        """Load the Ukrainian sentiment lexicon and initialize rule tables."""
        if lexicon_path is None:
            lexicon_path = (
                Path(__file__).resolve().parent.parent / "data" / "vader_en_openai_uk_only_unweighted.tsv"
            )
        self.lexicon_path = Path(lexicon_path)
        raw_lexicon = load_tonsum_lexicon(self.lexicon_path)
        if lexicon_scale_to_vader:
            raw_lexicon = {token: score * 2.0 for token, score in raw_lexicon.items()}

        self.alpha = float(alpha)
        self.normalizer = UkrainianMorphNormalizer()
        self.lexicon_stats = summarize_lexicon(raw_lexicon)
        self.lexicon: Dict[str, float] = {}
        self.emojis = EnglishSentimentIntensityAnalyzer().emojis
        self.special_cases = dict(EN_SPECIAL_CASES)
        for token, score in raw_lexicon.items():
            normalized = token.strip().lower()
            if not normalized:
                continue
            if " " in normalized:
                self.special_cases[normalized] = score
            else:
                self.lexicon[normalized] = score

    @staticmethod
    def _dedupe(values: Sequence[str]) -> List[str]:
        """Preserve order while removing empty and duplicate string variants."""
        seen = set()
        out: List[str] = []
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    def _token_forms(self, token: str) -> List[str]:
        """Return normalized token variants for lexicon and rule lookup."""
        lower = token.lower()
        candidates = [lower]
        candidates.extend(self.normalizer.word_candidates(lower))
        return self._dedupe(candidates)

    def _lookup_lexicon_valence(self, token: str) -> float | None:
        """Look up a token's sentiment valence while skipping control terms."""
        token_lower = token.lower()
        if (
            token_lower in self.BOOSTER_DICT
            or token_lower in self.NEGATIONS
            or token_lower in self.CONTRASTIVE_CONJ
            or token_lower in self.LEAST_TERMS
            or token_lower in self.AT_VERY_TERMS
        ):
            return None

        for form in self._token_forms(token):
            # Do not treat control words (negations/boosters/conjunctions) as sentiment-bearing lexicon items.
            if (
                form in self.BOOSTER_DICT
                or form in self.NEGATIONS
                or form in self.CONTRASTIVE_CONJ
                or form in self.LEAST_TERMS
                or form in self.AT_VERY_TERMS
            ):
                continue
            if form in self.lexicon:
                return self.lexicon[form]
        return None

    def _is_booster(self, token: str) -> bool:
        """Check whether a token acts as an intensity booster or dampener."""
        for form in self._token_forms(token):
            if form in self.BOOSTER_DICT:
                return True
        return False

    def _punctuation_emphasis(self, text: str) -> float:
        """Estimate additional sentiment emphasis contributed by punctuation."""
        ep_count = min(text.count("!"), 4)
        ep_amplifier = ep_count * 0.292

        qm_count = text.count("?")
        if qm_count > 1:
            if qm_count <= 3:
                qm_amplifier = qm_count * 0.18
            else:
                qm_amplifier = 0.96
        else:
            qm_amplifier = 0.0
        return ep_amplifier + qm_amplifier

    def _scalar_inc_dec(self, word: str, valence: float, is_cap_diff: bool) -> float:
        """Compute the booster adjustment contributed by a nearby modifier."""
        scalar = 0.0
        for form in self._token_forms(word):
            if form in self.BOOSTER_DICT:
                scalar = self.BOOSTER_DICT[form]
                break
        if scalar == 0.0:
            return scalar

        if valence < 0:
            scalar *= -1

        if word.isupper() and is_cap_diff:
            if valence > 0:
                scalar += C_INCR
            else:
                scalar -= C_INCR
        return scalar

    def _negated(self, input_words: Sequence[str]) -> bool:
        """Return True when any token variant is recognized as a negation."""
        for word in input_words:
            for form in self._token_forms(str(word)):
                if form in self.NEGATIONS:
                    return True
        return False

    def _negation_check(self, valence: float, words_norm: Sequence[str], start_i: int, i: int) -> float:
        """Adjust valence according to nearby negation patterns."""
        if start_i == 0:
            if self._negated([words_norm[i - 1]]):
                valence *= N_SCALAR
        elif start_i == 1:
            if words_norm[i - 2] in self.NEVER_TERMS and words_norm[i - 1] in self.SO_THIS_TERMS:
                valence *= 1.25
            elif words_norm[i - 2] in self.WITHOUT_TERMS and words_norm[i - 1] in self.DOUBT_TERMS:
                valence = valence
            elif self._negated([words_norm[i - 2]]):
                valence *= N_SCALAR
        elif start_i == 2:
            # Keep upstream VADER operator precedence: (A and B) or C.
            if (words_norm[i - 3] in self.NEVER_TERMS and words_norm[i - 2] in self.SO_THIS_TERMS) or (
                words_norm[i - 1] in self.SO_THIS_TERMS
            ):
                valence *= 1.25
            elif words_norm[i - 3] in self.WITHOUT_TERMS and (
                words_norm[i - 2] in self.DOUBT_TERMS or words_norm[i - 1] in self.DOUBT_TERMS
            ):
                valence = valence
            elif self._negated([words_norm[i - 3]]):
                valence *= N_SCALAR
        return valence

    def _special_idioms_check(self, valence: float, words_norm: Sequence[str], i: int) -> float:
        """Apply idiom-specific overrides and booster phrases around a token."""
        onezero = f"{words_norm[i - 1]} {words_norm[i]}"
        twoonezero = f"{words_norm[i - 2]} {words_norm[i - 1]} {words_norm[i]}"
        twoone = f"{words_norm[i - 2]} {words_norm[i - 1]}"
        threetwoone = f"{words_norm[i - 3]} {words_norm[i - 2]} {words_norm[i - 1]}"
        threetwo = f"{words_norm[i - 3]} {words_norm[i - 2]}"

        sequences = [onezero, twoonezero, twoone, threetwoone, threetwo]
        for seq in sequences:
            if seq in self.special_cases:
                valence = self.special_cases[seq]
                break

        if len(words_norm) - 1 > i:
            zeroone = f"{words_norm[i]} {words_norm[i + 1]}"
            if zeroone in self.special_cases:
                valence = self.special_cases[zeroone]
        if len(words_norm) - 1 > i + 1:
            zeroonetwo = f"{words_norm[i]} {words_norm[i + 1]} {words_norm[i + 2]}"
            if zeroonetwo in self.special_cases:
                valence = self.special_cases[zeroonetwo]

        for ngram in (threetwoone, threetwo, twoone):
            if ngram in self.BOOSTER_DICT:
                valence += self.BOOSTER_DICT[ngram]
        return valence

    def _least_check(self, valence: float, words_norm: Sequence[str], i: int) -> float:
        """Handle VADER's least-based negation rule for the current token."""
        if i > 1 and words_norm[i - 1] in self.LEAST_TERMS and words_norm[i - 1] not in self.lexicon:
            if words_norm[i - 2] not in self.AT_VERY_TERMS:
                valence *= N_SCALAR
        elif i > 0 and words_norm[i - 1] in self.LEAST_TERMS and words_norm[i - 1] not in self.lexicon:
            valence *= N_SCALAR
        return valence

    def _sentiment_valence(
        self,
        valence: float,
        sentitext: SentiText,
        item: str,
        i: int,
        sentiments: List[float],
        words_norm: Sequence[str],
    ) -> List[float]:
        """Update the running sentiment list for a single token."""
        is_cap_diff = sentitext.is_cap_diff
        words_and_emoticons = sentitext.words_and_emoticons
        item_lower = item.lower()
        lex_valence = self._lookup_lexicon_valence(item_lower)

        if lex_valence is not None:
            valence = lex_valence

            if item_lower in {"не", "ні"} and i != len(words_and_emoticons) - 1:
                if self._lookup_lexicon_valence(words_and_emoticons[i + 1].lower()) is not None:
                    valence = 0.0

            if item.isupper() and is_cap_diff:
                if valence > 0:
                    valence += C_INCR
                else:
                    valence -= C_INCR

            for start_i in range(0, 3):
                if i > start_i and self._lookup_lexicon_valence(words_and_emoticons[i - (start_i + 1)].lower()) is None:
                    s = self._scalar_inc_dec(words_and_emoticons[i - (start_i + 1)], valence, is_cap_diff)
                    if start_i == 1 and s != 0:
                        s *= 0.95
                    if start_i == 2 and s != 0:
                        s *= 0.9
                    valence += s
                    valence = self._negation_check(valence, words_norm, start_i, i)
                    if start_i == 2:
                        valence = self._special_idioms_check(valence, words_norm, i)

            valence = self._least_check(valence, words_norm, i)

        sentiments.append(valence)
        return sentiments

    def _but_check(self, words_norm: Sequence[str], sentiments: List[float]) -> List[float]:
        """Reweight sentiments around contrastive conjunctions like 'але'."""
        bi = None
        for idx, word in enumerate(words_norm):
            if word in self.CONTRASTIVE_CONJ:
                bi = idx
                break
        if bi is None:
            return sentiments

        # Keep this behavior close to reference VADER implementation.
        for sentiment in sentiments:
            si = sentiments.index(sentiment)
            if si < bi:
                sentiments.pop(si)
                sentiments.insert(si, sentiment * 0.5)
            elif si > bi:
                sentiments.pop(si)
                sentiments.insert(si, sentiment * 1.5)
        return sentiments

    @staticmethod
    def _sift_sentiment_scores(sentiments: List[float]) -> Tuple[float, float, float]:
        """Split raw sentiment contributions into positive, negative, and neutral totals."""
        pos_sum = 0.0
        neg_sum = 0.0
        neu_count = 0.0
        for score in sentiments:
            if score > 0:
                pos_sum += score + 1.0
            if score < 0:
                neg_sum += score - 1.0
            if score == 0:
                neu_count += 1.0
        return pos_sum, neg_sum, neu_count

    def _score_valence(self, sentiments: List[float], text: str) -> Dict[str, float]:
        """Convert token-level sentiment contributions into final VADER scores."""
        if sentiments:
            sum_s = float(sum(sentiments))
            punct_emph = self._punctuation_emphasis(text)
            if sum_s > 0:
                sum_s += punct_emph
            elif sum_s < 0:
                sum_s -= punct_emph

            compound = normalize(sum_s, self.alpha)
        else:
            compound = 0.0

        if not sentiments:
            return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": round(compound, 4)}

        pos_sum, neg_sum, neu_count = self._sift_sentiment_scores(sentiments)
        if pos_sum > abs(neg_sum):
            pos_sum += punct_emph
        elif pos_sum < abs(neg_sum):
            neg_sum -= punct_emph

        total = pos_sum + abs(neg_sum) + neu_count
        if total == 0:
            return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": round(compound, 4)}

        pos = abs(pos_sum / total)
        neg = abs(neg_sum / total)
        neu = abs(neu_count / total)

        return {
            "neg": round(neg, 3),
            "neu": round(neu, 3),
            "pos": round(pos, 3),
            "compound": round(compound, 4),
        }

    def polarity_scores(self, text: str) -> Dict[str, float]:
        """Score Ukrainian text with VADER-style lexical and rule-based heuristics."""
        if text is None:
            text = ""
        text = str(text)

        # Keep emoji handling aligned with the EN VADER reference implementation.
        text_no_emoji = ""
        prev_space = True
        for chr_ in text:
            if chr_ in self.emojis:
                description = self.emojis[chr_]
                if not prev_space:
                    text_no_emoji += " "
                text_no_emoji += description
                prev_space = False
            else:
                text_no_emoji += chr_
                prev_space = chr_ == " "
        text = text_no_emoji.strip()

        sentitext = SentiText(text)
        words_and_emoticons = sentitext.words_and_emoticons
        # Keep rule logic close to VADER by using lowercased surface tokens (not lemmas) for context rules.
        words_norm = [str(w).lower() for w in words_and_emoticons]

        sentiments: List[float] = []
        for i, item in enumerate(words_and_emoticons):
            valence = 0.0
            if self._is_booster(item):
                sentiments.append(valence)
                continue
            if i < len(words_and_emoticons) - 1:
                if (words_norm[i], words_norm[i + 1]) in self.KIND_OF_BIGRAMS:
                    sentiments.append(valence)
                    continue

            sentiments = self._sentiment_valence(
                valence=valence,
                sentitext=sentitext,
                item=item,
                i=i,
                sentiments=sentiments,
                words_norm=words_norm,
            )

        sentiments = self._but_check(words_norm, sentiments)
        return self._score_valence(sentiments, text)


# Backward-compatible aliases used in the project.
UkrainianSentimentIntensityAnalyzerImproved = UkrainianSentimentIntensityAnalyzerImproved
SentimentIntensityAnalyzerUK = UkrainianSentimentIntensityAnalyzerImproved
