# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
import pandas as pd
import numpy as np
import string
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from lexicalrichness import LexicalRichness
import spacy
import torch
from transformers import AutoTokenizer, XLMRobertaTokenizer, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# NLTK Tag list
TAG_DICT = {"PRP": "Pronoun", "PRP$": "Pronoun", "VB": "Verb", "VBD": "Verb", "VBG": "Verb", "VBN": "Verb", "VBP": "Verb", 
            "VBZ": "Verb", "JJ": "Adjective", "JJR": "Adjective", "JJS": "Adjective", "NN": "Noun", "NNP": "Noun", "NNS": "Noun",
            "RB": "Adverb", "RBR": "Adverb", "RBS": "Adverb", "DT": "Determiner"}

TAG_DICT_T = {
    "ua" : {
    "CCONJ": "Conjunction",
    "PRON": "Pronoun",
    "NOUN": "Noun",
    "ADJ": "Adjective",
    "PUNCT": "Punctuation",
    "VERB": "Verb",
    "AUX": "Auxiliary",
    "ADV": "Adverb",
    "ADP": "Adposition",
    "SCONJ": "Subordinating Conjunction",
    "NUM": "Numeral",
    "PROPN": "Proper Noun"
},
    "uk" : {
    "CCONJ": "Conjunction",
    "PRON": "Pronoun",
    "NOUN": "Noun",
    "ADJ": "Adjective",
    "PUNCT": "Punctuation",
    "VERB": "Verb",
    "AUX": "Auxiliary",
    "ADV": "Adverb",
    "ADP": "Adposition",
    "SCONJ": "Subordinating Conjunction",
    "NUM": "Numeral",
    "PROPN": "Proper Noun"
},
    'en' : {
    "PRP": "Pronoun",
    "PRP$": "Pronoun",
    "VB": "Verb",
    "VBD": "Verb",
    "VBG": "Verb",
    "VBN": "Verb",
    "VBP": "Verb",
    "VBZ": "Verb",
    "JJ": "Adjective",
    "JJR": "Adjective",
    "JJS": "Adjective",
    "NN": "Noun",
    "NNP": "Noun",
    "NNS": "Noun",
    "RB": "Adverb",
    "RBR": "Adverb",
    "RBS": "Adverb",
    "DT": "Determiner"}
}

FIRST_PERSON_PRONOUNS = ["I", "me", "my", "mine", "myself"]
FIRST_PERSON_PRONOUNS_T = {'en' : {"I", "me", "my", "mine", "myself"},
                           'ua' : {"я", "мене", "мені", "мною", "мій", "моя", "мої", "моє"},
                           'uk' : {"я", "мене", "мені", "мною", "мій", "моя", "мої", "моє"},}
FIRST_PERSON_PRONOUNS_T_LOWER = {
    lang: {p.lower() for p in pronouns}
    for lang, pronouns in FIRST_PERSON_PRONOUNS_T.items()
}
PRESENT = ["VBP", "VBZ"]
PAST = ["VBD", "VBN"]

VADER_SENTIMENT_COLS = {
    "neg": "sentiment_vader_neg",
    "neu": "sentiment_vader_neu",
    "pos": "sentiment_vader_pos",
    "compound": "sentiment_vader_overall",
}

FIRST_PERSON_VADER_COLS = {
    "positive": "first_person_sentiment_positive_vader",
    "negative": "first_person_sentiment_negative_vader",
    "overall": "first_person_sentiment_overall_vader",
}

DEFAULT_SUMMARY_SENTIMENT_ALPHA = 0.0
DEFAULT_SUMMARY_SENTIMENT_EPS = 1e-8
_TURN_SENTIMENT_TOKEN_COUNT_COL = "__turn_sentiment_token_count"
_SPACY_BATCH_SIZE_TURNS = 64
_SPACY_BATCH_SIZE_WORDS = 256
MATTR_WINDOWS = (5, 10, 25, 50, 100)
FUNCTION_WORD_POS = {"PRON", "DET", "ADP", "CCONJ", "SCONJ"}


def _normalize_lang(lang: Optional[str]) -> str:
    """Normalize language hints to the lexical module's supported codes."""
    normalized = (lang or "en").lower().strip()
    if normalized in {"ua", "uk"}:
        return normalized
    return "en"


def _spacy_model_name(lang: str) -> str:
    """Return the spaCy model name for a normalized language code."""
    return "uk_core_news_sm" if lang in {"ua", "uk"} else "en_core_web_sm"


@lru_cache(maxsize=2)
def _get_spacy_nlp_cached(model_name: str):
    """Load and cache a lightweight spaCy pipeline for lexical features."""
    # Keep tokenization + POS + morphology while dropping heavy, unused components.
    return spacy.load(model_name, disable=["parser", "ner", "textcat"])


def get_spacy_nlp(lang: str = "en"):
    """Return the cached spaCy pipeline for the requested language."""
    normalized_lang = _normalize_lang(lang)
    return _get_spacy_nlp_cached(_spacy_model_name(normalized_lang))


class MultilingualSentiment:
    """
    Multilingual sentiment analyzer with sliding-window support for long texts.
    polarity_scores(text) -> {"neg": float, "neu": float, "pos": float, "compound": float}
    """

    def __init__(
        self,
        model_id: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        *,
        window_tokens: Optional[int] = None,
        overlap_tokens: int = 64,
        batch_size: int = 16,
        device: Optional[str] = None,
    ):
        """Initialize the transformer sentiment pipeline and runtime device."""
        tokenizer = model_id
        pipeline_kwargs = {
            "task": "sentiment-analysis",
            "model": model_id,
            "tokenizer": tokenizer,
            "top_k": None,
            "framework": "pt",
        }
        if "xlm-roberta" in str(model_id).lower():
            # Prefer the fast tokenizer in environments where it is available.
            # Fall back to the slow tokenizer for runtimes that need sentencepiece.
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            except Exception:
                try:
                    tokenizer = XLMRobertaTokenizer.from_pretrained(model_id)
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to initialize the XLM-R tokenizer. Install sentencepiece "
                        "(or transformers[sentencepiece]) in the runtime environment."
                    ) from exc
            pipeline_kwargs["tokenizer"] = tokenizer

        self._pipe = pipeline(**pipeline_kwargs)
        self.tokenizer = self._pipe.tokenizer
        self.model = self._pipe.model

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.window_tokens = window_tokens
        self.overlap_tokens = overlap_tokens
        self.batch_size = batch_size

    @staticmethod
    def _label_index_map(model) -> Dict[str, int]:
        """Map model label ids to the canonical neg/neu/pos positions."""
        id2label = getattr(model.config, "id2label", None) or {}
        mapping: Dict[str, int] = {}

        for idx, label in id2label.items():
            label_name = str(label).lower()
            if "neg" in label_name:
                mapping["neg"] = int(idx)
            elif "neu" in label_name:
                mapping["neu"] = int(idx)
            elif "pos" in label_name:
                mapping["pos"] = int(idx)

        if set(mapping.keys()) != {"neg", "neu", "pos"}:
            mapping = {"neg": 0, "neu": 1, "pos": 2}

        return mapping

    @staticmethod
    def _vader_normalize(score: float, alpha: float = 15.0) -> float:
        """Project a raw score into the VADER-style compound range."""
        normalized = score / ((score * score + alpha) ** 0.5)
        if normalized < -1.0:
            return -1.0
        if normalized > 1.0:
            return 1.0
        return normalized

    @torch.inference_mode()
    def polarity_scores(self, text: str) -> dict:
        """Score text sentiment, chunking long inputs into overlapping windows."""
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)

        if not text.strip():
            return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}

        window_tokens = self.window_tokens
        if window_tokens is None:
            window_tokens = int(getattr(self.tokenizer, "model_max_length", 512))
            if window_tokens > 2048:
                window_tokens = 512

        if self.overlap_tokens >= window_tokens:
            raise ValueError("overlap_tokens must be < window_tokens")

        enc = self.tokenizer(
            text,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=window_tokens,
            stride=self.overlap_tokens,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        idx_map = self._label_index_map(self.model)

        probs_list: List[torch.Tensor] = []
        weights_list: List[torch.Tensor] = []
        n = input_ids.size(0)

        for i in range(0, n, self.batch_size):
            ids_batch = input_ids[i : i + self.batch_size]
            mask_batch = attention_mask[i : i + self.batch_size]

            logits = self.model(input_ids=ids_batch, attention_mask=mask_batch).logits
            probs = torch.softmax(logits, dim=-1)

            probs_list.append(probs)
            weights_list.append(mask_batch.sum(dim=1).float().clamp_min(1.0))

        probs_all = torch.cat(probs_list, dim=0)
        weights = torch.cat(weights_list, dim=0)
        avg = (probs_all * weights.unsqueeze(1)).sum(dim=0) / weights.sum()

        neg = float(avg[idx_map["neg"]].item())
        neu = float(avg[idx_map["neu"]].item())
        pos = float(avg[idx_map["pos"]].item())
        compound = self._vader_normalize(pos - neg, alpha=15.0)

        return {
            "neg": neg,
            "neu": neu,
            "pos": pos,
            "compound": compound,
        }


@lru_cache(maxsize=1)
def get_multilingual_sentiment_analyzer() -> MultilingualSentiment:
    """Instantiate the multilingual transformer sentiment analyzer."""
    return MultilingualSentiment()


@lru_cache(maxsize=4)
def get_vader_sentiment_analyzer(lang: str = "en") -> Any:
    """Return the language-appropriate VADER sentiment analyzer."""
    normalized_lang = (lang or "en").lower()
    if normalized_lang in {"uk", "ua"}:
        try:
            from .ukrainian_vader import SentimentIntensityAnalyzerUK

            return SentimentIntensityAnalyzerUK()
        except Exception as exc:
            logger.warning(
                "Failed to initialize Ukrainian VADER analyzer for lang=%s (%s). "
                "Falling back to default VADER.",
                normalized_lang,
                exc,
            )
    return SentimentIntensityAnalyzer()


def _sentiment_values(scores: dict) -> List[float]:
    """Extract sentiment components in neg, neu, pos, compound order."""
    return [
        scores.get("neg", np.nan),
        scores.get("neu", np.nan),
        scores.get("pos", np.nan),
        scores.get("compound", np.nan),
    ]

def _count_turn_tokens(text: str, tokenizer) -> float:
    """Count tokenizer tokens for a turn without adding special tokens."""
    if text is None:
        return 0.0
    if not isinstance(text, str):
        text = str(text)
    if not text.strip():
        return 0.0

    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )
        token_ids = encoded.get("input_ids", [])
        return float(len(token_ids))
    except Exception:
        return float(len(text.split()))

def _aggregate_turn_sentiment_scores(
    turn_df: pd.DataFrame,
    *,
    length_col: str,
    neg_col: str,
    neu_col: str,
    pos_col: str,
    alpha: float = DEFAULT_SUMMARY_SENTIMENT_ALPHA,
    eps: float = DEFAULT_SUMMARY_SENTIMENT_EPS,
) -> Optional[Dict[str, float]]:
    """Aggregate per-turn sentiment columns into a length-weighted summary."""
    required_cols = [length_col, neg_col, neu_col, pos_col]
    if any(c not in turn_df.columns for c in required_cols):
        return None

    clean = turn_df[required_cols].apply(pd.to_numeric, errors="coerce")
    lengths = clean[length_col].to_numpy(dtype=float)
    probs = clean[[neg_col, neu_col, pos_col]].to_numpy(dtype=float)

    mask = np.isfinite(lengths) & np.all(np.isfinite(probs), axis=1)
    if not np.any(mask):
        return None

    lengths = np.maximum(lengths[mask], 0.0)
    probs = probs[mask]

    weights_raw = np.power(lengths + float(eps), float(alpha))
    if not np.all(np.isfinite(weights_raw)) or float(np.sum(weights_raw)) <= 0.0:
        weights_raw = np.ones_like(lengths, dtype=float)

    weights = weights_raw / np.sum(weights_raw)
    avg_scores = np.sum(probs * weights[:, None], axis=0)

    neg, neu, pos = avg_scores.tolist()
    return {
        "neg": float(neg),
        "neu": float(neu),
        "pos": float(pos),
        "compound": float(pos - neg),
    }


def _extract_mattr_lemmas(text, lemmatizer) -> List[str]:
    """Extract normalized lemma tokens for MATTR calculations."""
    text = text if isinstance(text, str) else ("" if text is None else str(text))
    if not text.strip():
        return []

    doc = lemmatizer(text)
    lemmas: List[str] = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        lemma = token.lemma_.strip().lower()
        if not lemma:
            continue
        if lemma == "-pron-":
            lemma = token.text.strip().lower()
        if lemma:
            lemmas.append(lemma)
    return lemmas


def get_mattrs(text, lemmatizer, window_sizes=MATTR_WINDOWS):
    """Compute MATTR for multiple windows using a single tokenize/lemmatize pass."""
    if not window_sizes:
        return []

    lemmas = _extract_mattr_lemmas(text, lemmatizer)
    if not lemmas:
        return [np.nan] * len(window_sizes)

    lex_richness = LexicalRichness(" ".join(lemmas))
    if lex_richness.words <= 0:
        return [np.nan] * len(window_sizes)

    mattrs: List[float] = []
    for ws in window_sizes:
        try:
            ws_int = max(1, int(ws))
        except Exception:
            mattrs.append(np.nan)
            continue
        mattrs.append(lex_richness.mattr(window_size=min(ws_int, lex_richness.words)))

    return mattrs


def get_mattr(text, lemmatizer, window_size=50):
    """
    ------------------------------------------------------------------------------------------------------
    This function calculates the Moving Average Type-Token Ratio (MATTR)
     of the input text using the
     LexicalRichness library.

    Parameters:
    ...........
    text : str
        The input text to be analyzed.
    lemmatizer : spacy lemmatizer
        The lemmatizer to be used in the calculation.
    window_size : int
        The size of the window to be used in the calculation.

    Returns:
    ...........
    mattr : float
        The calculated MATTR value.

    ------------------------------------------------------------------------------------------------------
    """

    return get_mattrs(text, lemmatizer, window_sizes=(window_size,))[0]

def get_tag(word_df, word_list, measures, lang='en', nlp=None):
    """
    ------------------------------------------------------------------------------------------------------

    This function performs part-of-speech
     tagging on the input text using NLTK, and returns
     word-level part-of-speech tags.

    Parameters:
    ...........
    word_df: pandas dataframe
        A dataframe containing word summary information.
    word_list: list
        List of transcribed text at the word level.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    word_df: pandas dataframe
        The updated word_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    # tag_list = nltk.pos_tag(word_list)
    tag_list = get_tag_l(word_list, lang=lang, nlp=nlp, batch_size=_SPACY_BATCH_SIZE_WORDS)
    
    # tag_list_pos = [TAG_DICT_T[lang][tag[1]] if tag[1] in TAG_DICT_T[lang].keys() else "Other" for tag in tag_list] # change TAG_LIST
    tag_list_pos = [tag[1] for tag in tag_list]
    word_df[measures["part_of_speech"]] = tag_list_pos
    # words['first_person']
    word_df[measures["first_person"]] = [
        True if str(word).lower() in FIRST_PERSON_PRONOUNS_T_LOWER[lang] else np.nan
        for word, pos, _ in tag_list
    ]  # [word in FIRST_PERSON_PRONOUNS_T[lang] for word in word_list]
    # make non pronouns NaN
    allowed_tags = ["Pronoun", "DET"]
    word_df[measures["first_person"]] = word_df[measures["first_person"]].where(word_df[measures["part_of_speech"]].isin(allowed_tags), np.nan)
    # word_df[measures["first_person"]] = word_df[measures["first_person"]].where(word_df[measures["part_of_speech"]] == "Pronoun", np.nan)

    tag_list_verb = [verb_tense if pos == "Verb" else np.nan for _, pos, verb_tense in tag_list] # ["Present" if tag[1] in PRESENT else "Past" if tag[1] in PAST else "Other" for tag in tag_list]
    word_df[measures["verb_tense"]] = tag_list_verb
    # make non verbs NaN
    word_df[measures["verb_tense"]] = word_df[measures["verb_tense"]].where(word_df[measures["part_of_speech"]] == "Verb", np.nan)

    return word_df

def calculate_first_person_sentiment(
    df,
    measures,
    *,
    first_person_col: Optional[str] = None,
    pos_col: Optional[str] = None,
    neg_col: Optional[str] = None,
):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates a measure of the influence of sentiment on the use of first person pronouns.

    Parameters:
    ...........
    df: pandas dataframe
        A dataframe containing summary information.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    res1: list
        A list containing the calculated measure of the influence of positive sentiment on the use of first person pronouns.
    res2: list
        A list containing the calculated measure of the influence of negative sentiment on the use of first person pronouns.

    ------------------------------------------------------------------------------------------------------
    """
    
    first_person_col = first_person_col or measures["first_person_percentage"]
    pos_col = pos_col or measures["pos"]
    neg_col = neg_col or measures["neg"]

    res1: List[float] = []
    res2: List[float] = []
    for _, row in df.iterrows():
        perc = row.get(first_person_col, np.nan)
        pos = row.get(pos_col, np.nan)
        neg = row.get(neg_col, np.nan)

        if pd.isna(perc) or pd.isna(pos) or pd.isna(neg):
            res1.append(np.nan)
            res2.append(np.nan)
            continue

        res1.append((100 - perc) * pos)
        res2.append(perc * neg)

    return res1, res2

# def calculate_first_person_percentage(text, lang = 'en'):
#     """
#     ------------------------------------------------------------------------------------------------------

#     This function calculates the percentage of first person pronouns in the input text.

#     Parameters:
#     ...........
#     text: str
#         The input text to be analyzed.

#     Returns:
#     ...........
#     float
#         The calculated percentage of first person pronouns in the input text.

#     ------------------------------------------------------------------------------------------------------
#     """
#     lang = 'ua'
#     words = nltk.word_tokenize(text)
#     tags = nltk.pos_tag(words)
#     # filter out non pronouns
#     pronouns = [tag[0] for tag in tags if tag[1] == "PRP" or tag[1] == "PRP$"]
#     if len(pronouns) == 0:
#         return np.nan

#     first_person_pronouns = len([word for word in pronouns if word in FIRST_PERSON_PRONOUNS_T[lang]])
#     return (first_person_pronouns / len(pronouns)) * 100

def calculate_first_person_percentage(text, lang='en', nlp=None):
    """
    Calculates the percentage of first person pronouns in the input text.
    
    Parameters:
        text (str): The input text to be analyzed.
        lang (str): Language code ('en' for English, 'ua' or 'uk' for Ukrainian).
    
    Returns:
        float: The percentage of first person pronouns in the text, or np.nan if no tokens.
    """
    normalized_lang = _normalize_lang(lang)
    nlp = nlp or get_spacy_nlp(normalized_lang)
    text = text if isinstance(text, str) else ("" if text is None else str(text))
    doc = nlp(text)
    total_tokens = len(doc)
    if total_tokens == 0:
        return np.nan

    first_person_pronouns = FIRST_PERSON_PRONOUNS_T_LOWER.get(normalized_lang, FIRST_PERSON_PRONOUNS_T_LOWER["en"])
    first_person_count = sum(1 for token in doc if token.text.lower() in first_person_pronouns)
    
    return (first_person_count / total_tokens) * 100

def calculate_first_person_percentage_batch(texts, lang='en', nlp=None, batch_size=_SPACY_BATCH_SIZE_TURNS):
    """Compute first-person pronoun ratios for a batch of texts."""
    normalized_lang = _normalize_lang(lang)
    nlp = nlp or get_spacy_nlp(normalized_lang)
    first_person_pronouns = FIRST_PERSON_PRONOUNS_T_LOWER.get(normalized_lang, FIRST_PERSON_PRONOUNS_T_LOWER["en"])

    if texts is None:
        return []

    normalized_texts = [text if isinstance(text, str) else ("" if text is None else str(text)) for text in texts]
    output: List[float] = []
    for doc in nlp.pipe(normalized_texts, batch_size=max(1, int(batch_size))):
        total_tokens = len(doc)
        if total_tokens == 0:
            output.append(np.nan)
            continue
        first_person_count = sum(1 for token in doc if token.text.lower() in first_person_pronouns)
        output.append((first_person_count / total_tokens) * 100)
    return output


def get_first_person_turn(turn_df, turn_list, measures, lang='en', nlp=None):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates measures related to the first person pronouns in each turn.
     Specifically, it calculates the percentage of first person pronouns in each turn,
     and the influence of sentiment on the use of first person pronouns.

    Parameters:
    ...........
    turn_df: pandas dataframe
        A dataframe containing turn summary information.
    turn_list: list
        List of transcribed text at the turn level.

    Returns:
    ...........
    turn_df: pandas dataframe
        The updated turn_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    first_person_percentages = calculate_first_person_percentage_batch(
        turn_list,
        lang=lang,
        nlp=nlp,
        batch_size=_SPACY_BATCH_SIZE_TURNS,
    )

    turn_df[measures["first_person_percentage"]] = first_person_percentages

    first_pos, first_neg = calculate_first_person_sentiment(turn_df, measures)

    turn_df[measures["first_person_sentiment_positive"]] = first_pos
    turn_df[measures["first_person_sentiment_negative"]] = first_neg

    if VADER_SENTIMENT_COLS["pos"] in turn_df.columns and VADER_SENTIMENT_COLS["neg"] in turn_df.columns:
        vader_pos, vader_neg = calculate_first_person_sentiment(
            turn_df,
            measures,
            pos_col=VADER_SENTIMENT_COLS["pos"],
            neg_col=VADER_SENTIMENT_COLS["neg"],
        )
        turn_df[FIRST_PERSON_VADER_COLS["positive"]] = vader_pos
        turn_df[FIRST_PERSON_VADER_COLS["negative"]] = vader_neg

    return turn_df

def get_first_person_summ(summ_df, turn_df, full_text, measures, lang='en', nlp=None):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates measures related to the first person pronouns in the transcript.

    Parameters:
    ...........
    summ_df: pandas dataframe
        A dataframe containing summary information.
    turn_df: pandas dataframe
        A dataframe containing turn summary information.
    full_text: str
        The full transcribed text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    summ_df: pandas dataframe
        The updated summ_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """

    def _compute_first_person_summary(
        pos_col: str,
        neg_col: str,
        out_pos_col: str,
        out_neg_col: str,
        out_overall_col: str,
    ) -> None:
        """Populate summary-level first-person sentiment columns from turn scores."""
        if len(turn_df) > 0:
            if out_pos_col not in turn_df.columns or out_neg_col not in turn_df.columns:
                fp_pos, fp_neg = calculate_first_person_sentiment(
                    turn_df,
                    measures,
                    pos_col=pos_col,
                    neg_col=neg_col,
                )
                turn_df[out_pos_col] = fp_pos
                turn_df[out_neg_col] = fp_neg

            summ_df[out_pos_col] = turn_df[out_pos_col].mean(skipna=True)
            summ_df[out_neg_col] = turn_df[out_neg_col].mean(skipna=True)

            first_person_sentiment = []
            for _, row in turn_df.iterrows():
                pos_val = row.get(pos_col, np.nan)
                neg_val = row.get(neg_col, np.nan)
                if pd.isna(pos_val) or pd.isna(neg_val):
                    first_person_sentiment.append(np.nan)
                elif pos_val > neg_val:
                    first_person_sentiment.append(row.get(out_pos_col, np.nan))
                else:
                    first_person_sentiment.append(row.get(out_neg_col, np.nan))

            non_na_sentiment = [value for value in first_person_sentiment if not pd.isna(value)]
            if non_na_sentiment:
                summ_df[out_overall_col] = float(np.mean(non_na_sentiment))
            else:
                summ_df[out_overall_col] = np.nan
        else:
            fp_pos, fp_neg = calculate_first_person_sentiment(
                summ_df,
                measures,
                pos_col=pos_col,
                neg_col=neg_col,
            )
            summ_df[out_pos_col] = fp_pos
            summ_df[out_neg_col] = fp_neg

            pos_val = summ_df[pos_col].iloc[0] if pos_col in summ_df.columns else np.nan
            neg_val = summ_df[neg_col].iloc[0] if neg_col in summ_df.columns else np.nan
            if pd.isna(pos_val) or pd.isna(neg_val):
                summ_df[out_overall_col] = np.nan
            elif pos_val > neg_val:
                summ_df[out_overall_col] = summ_df[out_pos_col].iloc[0]
            else:
                summ_df[out_overall_col] = summ_df[out_neg_col].iloc[0]

    original_summ_df = summ_df.copy(deep=True)
    summ_df[measures["first_person_percentage"]] = calculate_first_person_percentage(full_text, lang=lang, nlp=nlp)
    try:
        _compute_first_person_summary(
            measures["pos"],
            measures["neg"],
            measures["first_person_sentiment_positive"],
            measures["first_person_sentiment_negative"],
            measures["first_person_sentiment_overall"],
        )

        has_vader_scores = (
            (VADER_SENTIMENT_COLS["pos"] in turn_df.columns and VADER_SENTIMENT_COLS["neg"] in turn_df.columns)
            or (VADER_SENTIMENT_COLS["pos"] in summ_df.columns and VADER_SENTIMENT_COLS["neg"] in summ_df.columns)
        )
        if has_vader_scores:
            _compute_first_person_summary(
                VADER_SENTIMENT_COLS["pos"],
                VADER_SENTIMENT_COLS["neg"],
                FIRST_PERSON_VADER_COLS["positive"],
                FIRST_PERSON_VADER_COLS["negative"],
                FIRST_PERSON_VADER_COLS["overall"],
            )

        return summ_df
    except Exception:
        logger.exception("Error while computing first-person summary features")
        return original_summ_df


# def get_tag_l(full_text, lang='en'):
#     nlp = spacy.load("uk_core_news_sm") if (lang in ['uk', 'ua']) else spacy.load("en_core_web_sm")
#     if type(full_text) == list:
#         full_text = " ".join(full_text)
#     doc = nlp(full_text)
    
#     # Get the original tags and map them using our dictionary if available
#     pos_tags = [(token.text, TAG_DICT_T[lang].get(token.tag_, token.tag_)) for token in doc]
#     return pos_tags

def count_space_tokens(text, lang='en', nlp=None):
    """Count tokens that spaCy tags as whitespace."""
    normalized_lang = _normalize_lang(lang)
    nlp = nlp or get_spacy_nlp(normalized_lang)
    text = text if isinstance(text, str) else ("" if text is None else str(text))
    doc = nlp(text)
    return sum(1 for token in doc if token.pos_ == "SPACE")


def _extract_pos_and_verb_tense(token, lang: str):
    """Map a token to normalized POS and verb-tense labels."""
    if lang in {'ua', 'uk'}:
        pos = TAG_DICT_T[lang].get(token.pos_, token.pos_)
        if token.pos_ in {"VERB", "AUX"}:
            tense_vals = token.morph.get("Tense")
            if tense_vals:
                if "Past" in tense_vals:
                    verb_tense = "Past"
                elif "Pres" in tense_vals:
                    verb_tense = "Present"
                else:
                    verb_tense = "Other"
            else:
                verb_tense = "Other"
        else:
            verb_tense = None
    else:
        pos = TAG_DICT_T[lang].get(token.tag_, token.tag_)
        if token.tag_ in PRESENT:
            verb_tense = "Present"
        elif token.tag_ in PAST:
            verb_tense = "Past"
        else:
            verb_tense = "Other"
    return pos, verb_tense


def calculate_prop_verb_past(text, lang='en', nlp=None):
    """
    Compute proportion of past-tense verbs among all verbs and auxiliaries.
    Returns NaN when no verbs are present.
    """
    normalized_lang = _normalize_lang(lang)
    nlp = nlp or get_spacy_nlp(normalized_lang)
    text = text if isinstance(text, str) else ("" if text is None else str(text))
    if not text.strip():
        return np.nan

    try:
        # Lemmatization is not required here and can fail on malformed/noisy tokens.
        if "lemmatizer" in nlp.pipe_names:
            with nlp.select_pipes(disable=["lemmatizer"]):
                doc = nlp(text)
        else:
            doc = nlp(text)
    except Exception as exc:
        logger.warning(f"Failed to parse text for prop_verb_past: {exc}")
        return np.nan
    total_verbs = 0
    past_verbs = 0

    for token in doc:
        if token.is_space or token.is_punct:
            continue
        if token.pos_ not in {"VERB", "AUX"}:
            continue
        total_verbs += 1
        tense_vals = token.morph.get("Tense")
        verb_tense = None
        if tense_vals:
            if "Past" in tense_vals:
                verb_tense = "Past"
            elif "Pres" in tense_vals:
                verb_tense = "Present"
            elif "Fut" in tense_vals:
                verb_tense = "Future"
            else:
                verb_tense = "Other"
        else:
            _, verb_tense = _extract_pos_and_verb_tense(token, normalized_lang)
        if verb_tense == "Past":
            past_verbs += 1

    if total_verbs <= 0:
        return np.nan

    return float(past_verbs / total_verbs)


def calculate_prop_function_words(text, lang='en', nlp=None):
    """
    Compute proportion of function words among all non-space, non-punctuation tokens.
    """
    normalized_lang = _normalize_lang(lang)
    nlp = nlp or get_spacy_nlp(normalized_lang)
    text = text if isinstance(text, str) else ("" if text is None else str(text))
    if not text.strip():
        return np.nan

    try:
        # Lemmatization is not required here and can fail on malformed/noisy tokens.
        if "lemmatizer" in nlp.pipe_names:
            with nlp.select_pipes(disable=["lemmatizer"]):
                doc = nlp(text)
        else:
            doc = nlp(text)
    except Exception as exc:
        logger.warning(f"Failed to parse text for prop_function_words: {exc}")
        return np.nan

    pos_counts = doc.count_by(spacy.attrs.POS)
    space_id = doc.vocab.strings["SPACE"]
    punct_id = doc.vocab.strings["PUNCT"]

    total_tokens = sum(
        count
        for pos_id, count in pos_counts.items()
        if pos_id not in {space_id, punct_id}
    )

    function_tokens = 0
    for pos in FUNCTION_WORD_POS:
        pos_id = doc.vocab.strings[pos]
        function_tokens += pos_counts.get(pos_id, 0)

    if total_tokens <= 0:
        return np.nan

    return float(function_tokens / total_tokens)


def get_tag_l(full_text, lang='en', nlp=None, batch_size=_SPACY_BATCH_SIZE_WORDS):
    """
    Returns a list of tuples (text, pos, verb_tense).
    If given a list, produce exactly one tag per input element to keep
    alignment with word_df rows.
    """
    normalized_lang = _normalize_lang(lang)
    nlp = nlp or get_spacy_nlp(normalized_lang)

    tags = []

    # Maintain 1:1 alignment when a list of words is provided
    if isinstance(full_text, list):
        normalized_words = [w if isinstance(w, str) else ("" if w is None else str(w)) for w in full_text]
        for original_word, doc in zip(
            full_text,
            nlp.pipe(normalized_words, batch_size=max(1, int(batch_size))),
        ):
            # Pick first non-space token if available, else fall back to empty
            token = next((t for t in doc if t.pos_ != "SPACE"), None)
            if token is None:
                # No token produced (e.g., empty/space-only). Mark as Other/None keeping alignment
                pos = "Other"
                verb_tense = None
                text = original_word
            else:
                text = token.text
                pos, verb_tense = _extract_pos_and_verb_tense(token, normalized_lang)
            tags.append((text, pos, verb_tense))
        return tags

    # If given a single string, process as a whole and filter out SPACE tokens
    full_text = full_text if isinstance(full_text, str) else ("" if full_text is None else str(full_text))
    doc = nlp(full_text)
    for token in doc:
        if token.pos_ == "SPACE":
            continue
        pos, verb_tense = _extract_pos_and_verb_tense(token, normalized_lang)
        tags.append((token.text, pos, verb_tense))

    return tags

def get_pos_tag(df_list, text_list, measures, lang="en"):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the part-of-speech measures
        and adds them to the output dataframes

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    text_list: list
        List of transcribed text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_list: list
        List of updated pandas dataframes.

    ------------------------------------------------------------------------------------------------------
    """
    try:
        word_df, turn_df, summ_df = df_list
        word_list, turn_list, full_text = text_list
        normalized_lang = _normalize_lang(lang)
        nlp = get_spacy_nlp(normalized_lang)

        word_df = get_tag(word_df, word_list, measures, lang=normalized_lang, nlp=nlp)
        summ_df[measures["prop_verb_past"]] = calculate_prop_verb_past(full_text, lang=normalized_lang, nlp=nlp)
        summ_df[measures["prop_function_words"]] = calculate_prop_function_words(full_text, lang=normalized_lang, nlp=nlp)

        if len(turn_list) > 0:
            turn_df = get_first_person_turn(turn_df, turn_list, measures, lang=normalized_lang, nlp=nlp)

        next_summ_df = get_first_person_summ(summ_df, turn_df, full_text, measures, normalized_lang, nlp=nlp)
        if isinstance(next_summ_df, pd.DataFrame):
            summ_df = next_summ_df
        else:
            logger.error("get_first_person_summ returned non-DataFrame; keeping previous summary dataframe")

        df_list = [word_df, turn_df, summ_df]
    except Exception as e:
        logger.info(f"Error in pos tag feature calculation: {e}")
    finally:
        return df_list

def get_sentiment(
    df_list,
    text_list,
    measures,
    lang='en',
    summary_sentiment_alpha: float = DEFAULT_SUMMARY_SENTIMENT_ALPHA,
    summary_sentiment_eps: float = DEFAULT_SUMMARY_SENTIMENT_EPS,
):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates sentiment scores of the input text using
     multilingual XLM-R (main output columns) and VADER (extra *_vader columns).
     Summary sentiment is aggregated from turn-level sentiment using
     weighted averaging over turns:
     w_i = ((l_i + eps)^alpha) / sum_j((l_j + eps)^alpha),
     p_c = sum_i w_i * p_{i,c},
     where l_i is the token count of turn i.

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    text_list: list
        List of transcribed text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.
    summary_sentiment_alpha: float
        Length-strength weight parameter (alpha). Default is 0.0 (uniform weights).
    summary_sentiment_eps: float
        Small constant added to turn lengths before exponentiation.

    Returns:
    ...........
    df_list: list
        List of updated pandas dataframes.

    ------------------------------------------------------------------------------------------------------
    """
    try:
        word_df, turn_df, summ_df = df_list
        _, turn_list, full_text = text_list
        normalized_lang = _normalize_lang(lang)
        lemmatizer = get_spacy_nlp(normalized_lang)
        
        vader_sentiment = get_vader_sentiment_analyzer(lang=normalized_lang)
        sentiment_cols = [measures["neg"], measures["neu"], measures["pos"], measures["compound"]]
        mattr_cols = [measures["speech_mattr_5"], measures["speech_mattr_10"], measures["speech_mattr_25"], measures["speech_mattr_50"], measures["speech_mattr_100"]]
        vader_cols = [
            VADER_SENTIMENT_COLS["neg"],
            VADER_SENTIMENT_COLS["neu"],
            VADER_SENTIMENT_COLS["pos"],
            VADER_SENTIMENT_COLS["compound"],
        ]
        turn_token_counts: Dict[int, float] = {}
        sentiment = None

        try:
            sentiment = get_multilingual_sentiment_analyzer()
        except Exception as e:
            logger.info(f"Transformer sentiment analyzer unavailable: {e}")

        for idx, u in enumerate(turn_list):
            try:
                vader_dict = vader_sentiment.polarity_scores(u)
                mattrs = get_mattrs(u, lemmatizer, window_sizes=MATTR_WINDOWS)
                turn_df.loc[idx, mattr_cols] = mattrs
                turn_df.loc[idx, vader_cols] = _sentiment_values(vader_dict)
                if sentiment is not None:
                    turn_token_counts[idx] = _count_turn_tokens(u, sentiment.tokenizer)
                    sentiment_dict = sentiment.polarity_scores(u)
                    turn_df.loc[idx, sentiment_cols] = _sentiment_values(sentiment_dict)

            except Exception as e:
                logger.info(f"Error in sentiment analysis: {e}")
                continue

        if sentiment is not None:
            turn_for_summary = turn_df.copy()
            turn_for_summary[_TURN_SENTIMENT_TOKEN_COUNT_COL] = pd.Series(turn_token_counts, dtype=float)

            sentiment_dict = _aggregate_turn_sentiment_scores(
                turn_for_summary,
                length_col=_TURN_SENTIMENT_TOKEN_COUNT_COL,
                neg_col=measures["neg"],
                neu_col=measures["neu"],
                pos_col=measures["pos"],
                alpha=summary_sentiment_alpha,
                eps=summary_sentiment_eps,
            )
            if sentiment_dict is None:
                sentiment_dict = sentiment.polarity_scores(full_text)
            summ_df.loc[0, sentiment_cols] = _sentiment_values(sentiment_dict)

        # vader_dict = _aggregate_turn_sentiment_scores(
        #     turn_for_summary,
        #     length_col=_TURN_SENTIMENT_TOKEN_COUNT_COL,
        #     neg_col=VADER_SENTIMENT_COLS["neg"],
        #     neu_col=VADER_SENTIMENT_COLS["neu"],
        #     pos_col=VADER_SENTIMENT_COLS["pos"],
        #     alpha=summary_sentiment_alpha,
        #     eps=summary_sentiment_eps,
        # )
        # if vader_dict is None:
        vader_dict = vader_sentiment.polarity_scores(full_text)

        mattrs = get_mattrs(full_text, lemmatizer, window_sizes=MATTR_WINDOWS)

        summ_df.loc[0, mattr_cols] = mattrs
        summ_df.loc[0, vader_cols] = _sentiment_values(vader_dict)
        df_list = [word_df, turn_df, summ_df]
    except Exception as e:
        logger.info(f"Error in sentiment feature calculation: {e}")
    finally:
        return df_list

def calculate_repetitions(words_texts, phrases_texts):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the percentage of repeated words and phrases in the input lists.

    Parameters:
    ...........
    words_texts: list
        List of transcribed text at the word level.
    phrases_texts: list
        List of transcribed text at the phrase level.

    Returns:
    ...........
    word_reps_perc: float
        The percentage of repeated words in the input lists.
    phrase_reps_perc: float
        The percentage of repeated phrases in the input lists.

    ------------------------------------------------------------------------------------------------------
    """
    def calculate_percentage_repetitions(text_list, window_size):
        """Helper function to calculate the percentage of repetitions in a sliding window."""
        if len(text_list) <= window_size:
            reps = len(text_list) - len(set(text_list))
            return 100 * reps / len(text_list) if len(text_list) > 0 else 0
        else:
            reps_list = [
                100 * (len(window) - len(set(window))) / len(window)
                for i in range(len(text_list) - window_size + 1)
                for window in [text_list[i:i + window_size]]
            ]
            return np.mean(reps_list)

    # Clean words and phrases: remove punctuation, convert to lowercase, and filter out empty strings
    words_texts = [word.translate(str.maketrans('', '', string.punctuation)).lower() for word in words_texts if word.strip()]
    phrases_texts = [phrase.translate(str.maketrans('', '', string.punctuation)).lower() for phrase in phrases_texts if phrase.strip()]

    # Calculate repetition percentages for words (sliding window of 10 words) and phrases (sliding window of 3 phrases)
    word_reps_perc = calculate_percentage_repetitions(words_texts, window_size=10)
    phrase_reps_perc = calculate_percentage_repetitions(phrases_texts, window_size=3) if phrases_texts else np.nan

    return word_reps_perc, phrase_reps_perc

def get_repetitions(df_list, utterances_speaker, utterances_speaker_filtered, measures):
    """
    This function calculates the percentage of repeated words and phrases in the input text
    and adds them to the output dataframes.

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    utterances_speaker: pandas dataframe
        A dataframe containing the turns extracted from the JSON object for the specified speaker.
    utterances_speaker_filtered: pandas dataframe
        A dataframe containing the turns extracted from the JSON object for the specified speaker
        after filtering out turns with less than min_turn_length words.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_list: list
        List of updated pandas dataframes.
    """
    
    try:
        word_df, turn_df, summ_df = df_list

        # turn-level
        if not turn_df.empty:
            for i in range(len(utterances_speaker_filtered)):
                row = utterances_speaker_filtered.iloc[i]
                words_texts = row[measures['words_texts']]
                phrases_texts = row[measures['phrases_texts']]

                word_reps_perc, phrase_reps_perc = calculate_repetitions(words_texts, phrases_texts)

                turn_df.loc[i, measures['word_repeat_percentage']] = word_reps_perc
                turn_df.loc[i, measures['phrase_repeat_percentage']] = phrase_reps_perc

            # Calculate summary-level statistics
            summ_df[measures['word_repeat_percentage']] = turn_df[measures['word_repeat_percentage']].mean(skipna=True)
            summ_df[measures['phrase_repeat_percentage']] = turn_df[measures['phrase_repeat_percentage']].mean(skipna=True)
        else:
            words_texts = [word for words in utterances_speaker[measures['words_texts']] for word in words]
            phrases_texts = [phrase for phrases in utterances_speaker[measures['phrases_texts']] for phrase in phrases]

            word_reps_perc, phrase_reps_perc = calculate_repetitions(words_texts, phrases_texts)

            summ_df[measures['word_repeat_percentage']] = word_reps_perc
            summ_df[measures['phrase_repeat_percentage']] = phrase_reps_perc

        df_list = [word_df, turn_df, summ_df]
    except Exception as e:
        logger.info(f"Error in calculating repetitions: {e}")
    finally:
        return df_list
