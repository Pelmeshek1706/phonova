# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
import gc
import logging
import math
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BertTokenizer, BertModel, BertForMaskedLM
from huggingface_hub import login
import os

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if token:
    try:
        login(token)
    except Exception:
        pass



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
# Suppress warnings from transformers
logging.getLogger("transformers").setLevel(logging.ERROR)


# Select coherence backend: "gemma" (default) or "bert".
# Override via env OPENWILLIS_COHERENCE_BACKEND or by setting COHERENCE_BACKEND at runtime.
COHERENCE_BACKEND = os.getenv("OPENWILLIS_COHERENCE_BACKEND", "gemma").strip().lower()

DEFAULT_EMBEDDING_MODEL_ID = "google/embeddinggemma-300m"
DEFAULT_PPL_MODEL_ID = "google/gemma-3-270m"
BERT_EN_MODEL_ID = os.getenv("OPENWILLIS_BERT_EN_MODEL_ID", "bert-base-cased")
BERT_MULTI_MODEL_ID = os.getenv("OPENWILLIS_BERT_MULTI_MODEL_ID", "bert-base-multilingual-uncased")
BERT_SENTENCE_EN_MODEL_ID = os.getenv(
    "OPENWILLIS_BERT_SENTENCE_EN_MODEL_ID",
    "sentence-transformers/all-MiniLM-L6-v2",
)
BERT_SENTENCE_MULTI_MODEL_ID = os.getenv(
    "OPENWILLIS_BERT_SENTENCE_MULTI_MODEL_ID",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
PPL_MAX_TOKENS = 2048
EMBEDDING_BATCH_SIZE = 256
WINDOW_BATCH_SIZE = 512
MIN_EMBEDDING_BATCH_SIZE = 8
WORD_STREAM_CHUNK_SIZE = 32
TOKEN_CACHE_SIZE = int(os.getenv("OPENWILLIS_TOKEN_CACHE_SIZE", "512"))
PREVIOUS_SPEAKER_SIMILARITY_MIN_TURN_LENGTH = 1
_PPL_TOKEN_CACHE: "OrderedDict[str, torch.Tensor]" = OrderedDict()
_BERT_TOKEN_CACHE: "OrderedDict[str, torch.Tensor]" = OrderedDict()


def _select_torch_device(explicit: Optional[str] = None) -> torch.device:
    """Return an available torch device, preferring CUDA, then MPS, otherwise CPU."""
    if explicit:
        return torch.device(explicit)

    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _maybe_enable_tf32(device: torch.device) -> None:
    """Enable TF32 where it is supported to speed up matmul-heavy workloads."""
    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


_MODEL_CACHE: Dict[str, Dict[str, object]] = {}
_BERT_CACHE: Dict[str, Dict[str, object]] = {}


def _cache_get(cache: "OrderedDict[str, torch.Tensor]", key: str) -> Optional[torch.Tensor]:
    """Fetch a cached tensor and refresh its LRU position."""
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_put(cache: "OrderedDict[str, torch.Tensor]", key: str, value: torch.Tensor) -> None:
    """Insert a tensor into the LRU cache and evict the oldest entries."""
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > TOKEN_CACHE_SIZE:
        cache.popitem(last=False)


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embedding rows, returning an empty array for invalid input."""
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def _cosine_for_offset(normalized_embeddings: np.ndarray, k: int) -> np.ndarray:
    """Compute cosine similarity between embeddings separated by k positions."""
    if k <= 0:
        raise ValueError("k must be > 0")
    n = normalized_embeddings.shape[0]
    if n <= k:
        return np.zeros((0,), dtype=np.float32)
    return np.einsum("ij,ij->i", normalized_embeddings[:-k], normalized_embeddings[k:]).astype(np.float32)


def _adjacent_turn_similarity_from_texts(
    utterance_texts: List[str],
    sentence_encoder: Optional[SentenceTransformer],
) -> Optional[np.ndarray]:
    """Encode turns and compute cosine similarity with the immediately previous turn."""
    if sentence_encoder is None or len(utterance_texts) == 0:
        return None

    utterances_embeddings = sentence_encoder.encode(
        utterance_texts,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    normalized_turns = _normalize_embeddings(utterances_embeddings)
    adjacent_turn_similarity = np.full((len(utterance_texts),), np.nan, dtype=np.float32)
    if normalized_turns.shape[0] > 1:
        adjacent_turn_similarity[1:] = _cosine_for_offset(normalized_turns, 1)
    return adjacent_turn_similarity


def _previous_speaker_turn_similarity(
    dialogue_utterances_filtered,
    speaker_label: Optional[str],
    previous_speaker_min_turn_length: int,
    sentence_encoder: Optional[SentenceTransformer],
    measures: dict,
) -> List[float]:
    """
    Compute similarity between each target-speaker turn and the immediately
    previous turn from a different speaker in the full dialogue sequence.
    """
    if dialogue_utterances_filtered is None or len(dialogue_utterances_filtered) == 0:
        return []

    if speaker_label is None:
        target_rows = dialogue_utterances_filtered
    else:
        target_rows = dialogue_utterances_filtered[
            dialogue_utterances_filtered[measures['speaker_label']] == speaker_label
        ]

    if len(target_rows) == 0:
        return []

    adjacent_turn_similarity = _adjacent_turn_similarity_from_texts(
        dialogue_utterances_filtered[measures['utterance_text']].values.tolist(),
        sentence_encoder,
    )
    if adjacent_turn_similarity is None:
        return [np.nan] * len(target_rows)

    previous_speaker_similarity = []
    for i, row in dialogue_utterances_filtered.iterrows():
        current_speaker = row[measures['speaker_label']]
        if speaker_label is not None and current_speaker != speaker_label:
            continue

        if len(row[measures['words_texts']]) < previous_speaker_min_turn_length:
            previous_speaker_similarity.append(np.nan)
            continue

        if i == 0:
            previous_speaker_similarity.append(np.nan)
            continue

        previous_row = dialogue_utterances_filtered.iloc[i - 1]
        previous_speaker = previous_row[measures['speaker_label']]
        if (
            previous_speaker == current_speaker
            or len(previous_row[measures['words_texts']]) < previous_speaker_min_turn_length
        ):
            previous_speaker_similarity.append(np.nan)
            continue

        previous_speaker_similarity.append(float(adjacent_turn_similarity[i]))

    return previous_speaker_similarity


def _word_coherence_from_embeddings(word_embeddings: np.ndarray) -> Tuple[List[float], List[float], List[float], Dict[int, List[float]]]:
    """Derive per-word coherence windows and variability metrics from embeddings."""
    normalized = _normalize_embeddings(word_embeddings)
    n = int(normalized.shape[0])
    if n == 0:
        empty_var = {k: [] for k in range(2, 11)}
        return [], [], [], empty_var

    offsets = {k: _cosine_for_offset(normalized, k) for k in range(1, 11)}

    word_coherence: List[float] = [np.nan] * n
    if n > 1:
        word_coherence[1:] = offsets[1].tolist()

    word_coherence_5: List[float] = [np.nan] * n
    if n > 5:
        for j in range(2, n - 2):
            total = 1.0
            total += float(offsets[1][j - 1]) + float(offsets[1][j])
            total += float(offsets[2][j - 2]) + float(offsets[2][j])
            word_coherence_5[j] = total / 5.0

    word_coherence_10: List[float] = [np.nan] * n
    if n > 10:
        for j in range(5, n - 5):
            total = 1.0
            for d in range(1, 6):
                total += float(offsets[d][j - d]) + float(offsets[d][j])
            word_coherence_10[j] = total / 11.0

    variability: Dict[int, List[float]] = {}
    for k in range(2, 11):
        if n > k:
            variability[k] = offsets[k].tolist() + [np.nan] * k
        else:
            variability[k] = [np.nan] * n

    return word_coherence, word_coherence_5, word_coherence_10, variability


def _phrase_tangeniality_from_embeddings(phrase_embeddings: np.ndarray) -> Tuple[float, float]:
    """Compute first- and second-order phrase tangentiality scores."""
    normalized = _normalize_embeddings(phrase_embeddings)
    n = int(normalized.shape[0])
    if n == 0:
        return np.nan, np.nan

    sentence_tangeniality1 = np.nan
    sentence_tangeniality2 = np.nan

    if n > 1:
        first_order = _cosine_for_offset(normalized, 1)
        if first_order.size > 0:
            sentence_tangeniality1 = float(np.mean(first_order))
    if n > 2:
        second_order = _cosine_for_offset(normalized, 2)
        if second_order.size > 0:
            sentence_tangeniality2 = float(np.mean(second_order))

    return sentence_tangeniality1, sentence_tangeniality2


def _get_backend() -> str:
    """Resolve the configured coherence backend, defaulting invalid values to gemma."""
    backend = (COHERENCE_BACKEND or "gemma").strip().lower()
    if backend not in {"gemma", "bert"}:
        logger.warning("Unknown COHERENCE_BACKEND=%s; defaulting to gemma.", backend)
        return "gemma"
    return backend


def _load_sentence_encoder(device: torch.device) -> SentenceTransformer:
    """Instantiate the sentence embedding model on the requested device."""
    target = device.type
    # SentenceTransformer expects a string device identifier
    try:
        return SentenceTransformer(DEFAULT_EMBEDDING_MODEL_ID, device=target)
    except Exception:
        if target != "cpu":
            # Fallback to CPU if the requested accelerator is unsupported
            return SentenceTransformer(DEFAULT_EMBEDDING_MODEL_ID, device="cpu")
        raise


def _resolve_dtype(device: torch.device) -> torch.dtype:
    """Choose a safe inference dtype for the selected torch device."""
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        # Gemma kernels on MPS are unstable in float16/bfloat16; force float32
        return torch.float32
    return torch.float32


def _load_lm_bundle(device: torch.device) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Instantiate the Gemma language model and tokenizer on the requested device."""
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_PPL_MODEL_ID)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = _resolve_dtype(device)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_PPL_MODEL_ID,
            torch_dtype=dtype,
        )
    except (TypeError, ValueError):
        # Fallback for older transformer versions or unsupported dtypes
        model = AutoModelForCausalLM.from_pretrained(DEFAULT_PPL_MODEL_ID)

    if device.type == "mps":
        model = model.to(device, dtype=torch.float32)
    else:
        model = model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    return tokenizer, model


def get_model_bundle(language: str, device_hint: Optional[str] = None) -> Dict[str, object]:
    """Lazy-load and cache the Gemma models required for coherence metrics."""
    cache_key = (language or "").lower() or "default"
    bundle = _MODEL_CACHE.get(cache_key)
    if bundle:
        return bundle

    device = _select_torch_device(device_hint)
    _maybe_enable_tf32(device)

    sentence_encoder = None
    try:
        sentence_encoder = _load_sentence_encoder(device)
    except Exception as exc:
        logger.warning("Failed to load sentence encoder on %s: %s", device, exc)

    try:
        tokenizer, lm_model = _load_lm_bundle(device)
    except Exception as exc:
        logger.warning("Failed to load Gemma LM on %s: %s", device, exc)
        tokenizer, lm_model = None, None

    bundle = {
        "device": device,
        "sentence_encoder": sentence_encoder,
        "tokenizer": tokenizer,
        "lm_model": lm_model,
        "model_max_length": int(getattr(lm_model.config, "max_position_embeddings", 32768)) if lm_model else None,
    }

    _MODEL_CACHE[cache_key] = bundle
    return bundle


def _resolve_bert_model_id(language: str, measures: Dict[str, object]) -> Optional[str]:
    """Resolve the token-level BERT model id for the requested language."""
    if language in measures.get("english_langs", []):
        return BERT_EN_MODEL_ID
    if language in measures.get("supported_langs_bert", []):
        return BERT_MULTI_MODEL_ID
    return None


def _resolve_sentence_encoder_id(language: str, measures: Dict[str, object]) -> Optional[str]:
    """Resolve the sentence-embedding model id for the requested language."""
    if language in measures.get("english_langs", []):
        return BERT_SENTENCE_EN_MODEL_ID
    if language in measures.get("supported_langs_sentence_embeddings", []):
        return BERT_SENTENCE_MULTI_MODEL_ID
    return None


def get_bert_bundle(language: str, measures: Dict[str, object], device_hint: Optional[str] = None) -> Dict[str, object]:
    """Lazy-load and cache the BERT models required for coherence metrics."""
    cache_key = (language or "").lower() or "default"
    bundle = _BERT_CACHE.get(cache_key)
    if bundle:
        return bundle

    device = _select_torch_device(device_hint)
    _maybe_enable_tf32(device)

    sentence_encoder = None
    sentence_model_id = _resolve_sentence_encoder_id(language, measures)
    if sentence_model_id:
        try:
            sentence_encoder = SentenceTransformer(sentence_model_id, device=device.type)
        except Exception:
            if device.type != "cpu":
                sentence_encoder = SentenceTransformer(sentence_model_id, device="cpu")

    tokenizer = word_model = mlm_model = None
    model_id = _resolve_bert_model_id(language, measures)
    if model_id:
        tokenizer = BertTokenizer.from_pretrained(model_id)
        word_model = BertModel.from_pretrained(model_id)
        mlm_model = BertForMaskedLM.from_pretrained(model_id)
        word_model.eval()
        mlm_model.eval()
        word_model.to(device)
        mlm_model.to(device)

    bundle = {
        "device": device,
        "sentence_encoder": sentence_encoder,
        "tokenizer": tokenizer,
        "word_model": word_model,
        "mlm_model": mlm_model,
    }

    _BERT_CACHE[cache_key] = bundle
    return bundle

def get_word_embeddings(word_list, sentence_encoder):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the word embeddings for the input text using the Gemma sentence encoder.

    Parameters:
    ...........
    word_list: list
        List of transcribed text at the word level.
    sentence_encoder: SentenceTransformer
        Sentence embedding model used for encoding.

    Returns:
    ...........
    word_embeddings: numpy array
        The calculated word embeddings.

    ------------------------------------------------------------------------------------------------------
    """
    if sentence_encoder is None or len(word_list) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    def _safe_encode(bs: int) -> np.ndarray:
        """Encode the current word batch with a lower-bounded batch size."""
        return _encode_in_chunks(sentence_encoder, word_list, max(bs, 1))

    try:
        return _safe_encode(EMBEDDING_BATCH_SIZE)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise RuntimeError(f"Error in embedding calculation: {exc}") from exc

        logger.warning(
            "Sentence encoder OOM on device %s; retrying on CPU with smaller batches.",
            getattr(sentence_encoder, "device", "unknown"),
        )

        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()  # type: ignore[attr-defined]
        except Exception:
            pass

        try:
            sentence_encoder.to("cpu")  # type: ignore[attr-defined]
        except Exception:
            pass

        reduced_bs = max(MIN_EMBEDDING_BATCH_SIZE, EMBEDDING_BATCH_SIZE // 4)
        return _safe_encode(reduced_bs)
    except Exception as exc:
        raise RuntimeError(f"Error in embedding calculation: {exc}") from exc

def get_word_coherence_utterance(row, sentence_encoder, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates word coherence measures for a single utterance.

    Parameters:
    ...........
    row: pandas dataframe
        A dataframe containing the turns extracted from the JSON object for the specified speaker.
    sentence_encoder: SentenceTransformer
        A sentence encoder model (Gemma embeddings).
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    word_coherence: list
        A list containing the calculated semantic similarity of each word to the immediately preceding word.
    word_coherence_5: list
        A list containing the calculated semantic similarity of each word in 5-words window.
    word_coherence_10: list
        A list containing the calculated semantic similarity of each word in 10-words window.
    word_word_variability: dict
        A dictionary containing the calculated word-to-word variability at k inter-word distances.

    ------------------------------------------------------------------------------------------------------
    """
    words_texts = row[measures['words_texts']]
    if len(words_texts) == 0:
        # return empty lists if no words in the utterance
        return [np.nan]*len(words_texts), [np.nan]*len(words_texts), [np.nan]*len(words_texts), {k: [np.nan]*len(words_texts) for k in range(2, 11)}

    word_embeddings = get_word_embeddings(words_texts, sentence_encoder)
    return _word_coherence_from_embeddings(word_embeddings)


def get_word_embeddings_bert(word_list, tokenizer: BertTokenizer, model: BertModel) -> np.ndarray:
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the word embeddings for the input text using BERT.

    Parameters:
    ...........
    word_list: list
        List of transcribed text at the word level.
    tokenizer: BertTokenizer
        A tokenizer object for BERT.
    model: BertModel
        A BERT model object.

    Returns:
    ...........
    word_embeddings: numpy array
        The calculated word embeddings.

    ------------------------------------------------------------------------------------------------------
    """
    if len(word_list) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    device = next(model.parameters()).device if model is not None else torch.device("cpu")

    def _embed_chunk(chunk: List[str]) -> np.ndarray:
        """Embed a chunk of tokens with BERT and mean-pool the hidden states."""
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(1).detach().cpu().numpy()

    if len(word_list) >= 512:
        word_embeddings = []
        for i in range(0, len(word_list), 512):
            chunk = word_list[i:i + 512]
            word_embeddings.append(_embed_chunk(chunk))
        return np.concatenate(word_embeddings, axis=0)

    return _embed_chunk(word_list)


def get_word_coherence_utterance_bert(row, tokenizer, model, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates word coherence measures for a single utterance using BERT embeddings.

    Parameters:
    ...........
    row: pandas dataframe
        A dataframe containing the turns extracted from the JSON object for the specified speaker.
    tokenizer: BertTokenizer
        A tokenizer object for BERT.
    model: BertModel
        A BERT model object.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    word_coherence: list
        A list containing the calculated semantic similarity of each word to the immediately preceding word.
    word_coherence_5: list
        A list containing the calculated semantic similarity of each word in 5-words window.
    word_coherence_10: list
        A list containing the calculated semantic similarity of each word in 10-words window.
    word_word_variability: dict
        A dictionary containing the calculated word-to-word variability at k inter-word distances.

    ------------------------------------------------------------------------------------------------------
    """
    words_texts = row[measures['words_texts']]
    if len(words_texts) == 0:
        return [np.nan] * len(words_texts), [np.nan] * len(words_texts), [np.nan] * len(words_texts), {k: [np.nan] * len(words_texts) for k in range(2, 11)}

    word_embeddings = get_word_embeddings_bert(words_texts, tokenizer, model)
    return _word_coherence_from_embeddings(word_embeddings)

def get_word_coherence_summary(word_df, summ_df, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function summarizes the word coherence measures at the summary level.

    Parameters:
    ...........
    word_df: pandas dataframe
        A dataframe containing word summary information.
    summ_df: pandas dataframe
        A dataframe containing summary information.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    summ_df: pandas dataframe
        The updated summ_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """

    summ_df[measures['word_coherence_mean']] = word_df[measures['word_coherence']].mean(skipna=True)
    summ_df[measures['word_coherence_var']] = word_df[measures['word_coherence']].var(skipna=True)
    summ_df[measures['word_coherence_5_mean']] = word_df[measures['word_coherence_5']].mean(skipna=True)
    summ_df[measures['word_coherence_5_var']] = word_df[measures['word_coherence_5']].var(skipna=True)
    summ_df[measures['word_coherence_10_mean']] = word_df[measures['word_coherence_10']].mean(skipna=True)
    summ_df[measures['word_coherence_10_var']] = word_df[measures['word_coherence_10']].var(skipna=True)
    for k in range(2, 11):
        summ_df[measures[f'word_coherence_variability_{k}_mean']] = word_df[measures[f'word_coherence_variability_{k}']].mean(skipna=True)
        summ_df[measures[f'word_coherence_variability_{k}_var']] = word_df[measures[f'word_coherence_variability_{k}']].var(skipna=True)

    return summ_df

def append_nan_values(coherence_lists, row_len):
    """
    ------------------------------------------------------------------------------------------------------

    Helper function for appending NaN values to the coherence lists.

    Parameters:
    ...........
    coherence_lists: dict
        A dictionary containing the coherence lists.
    row_len: int
        The length of the row.

    Returns:
    ...........
    coherence_lists: dict
        The updated coherence lists.

    ------------------------------------------------------------------------------------------------------
    """
    coherence_lists['word_coherence'] += [np.nan] * row_len
    coherence_lists['word_coherence_5'] += [np.nan] * row_len
    coherence_lists['word_coherence_10'] += [np.nan] * row_len
    for k in range(2, 11):
        coherence_lists['variability'][k] += [np.nan] * row_len

    return coherence_lists

def get_word_coherence(df_list, utterances_speaker, min_coherence_turn_length, language, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates word coherence measures

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    utterances_speaker: pandas dataframe
        A dataframe containing the turns extracted from the JSON object for the specified speaker.
    min_coherence_turn_length: int
        Minimum number of words in a turn for word coherence analysis.
    language: str
        Language of the transcribed text.
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
        backend = _get_backend()
        coherence_fn = None
        sentence_encoder: Optional[SentenceTransformer] = None
        tokenizer: Optional[BertTokenizer] = None
        word_model: Optional[BertModel] = None

        if backend == "bert":
            bundle = get_bert_bundle(language, measures)
            tokenizer = bundle.get("tokenizer")
            word_model = bundle.get("word_model")
            if tokenizer is None or word_model is None:
                logger.info(f"BERT models not available for language {language}; skipping word coherence analysis.")
                return df_list
            coherence_fn = lambda row: get_word_coherence_utterance_bert(row, tokenizer, word_model, measures)
        else:
            bundle = get_model_bundle(language)
            sentence_encoder = bundle.get("sentence_encoder")
            if sentence_encoder is None:
                logger.info(f"Sentence encoder not available for language {language}; skipping word coherence analysis.")
                return df_list
            coherence_fn = lambda row: get_word_coherence_utterance(row, sentence_encoder, measures)

        overall_lists = _new_coherence_lists()

        chunk_rows = []

        def _process_chunk(rows_chunk):
            """Compute coherence metrics for a buffered chunk of utterance rows."""
            if not rows_chunk:
                return

            chunk_lists = _new_coherence_lists()
            eligible_words: List[List[str]] = []
            for row in rows_chunk:
                words = row[measures['words_texts']]
                if len(words) >= min_coherence_turn_length:
                    eligible_words.append(words)

            row_embeddings: List[Optional[np.ndarray]] = [None] * len(rows_chunk)
            if eligible_words:
                flat_words = [w for words in eligible_words for w in words]
                try:
                    if backend == "bert" and tokenizer is not None and word_model is not None:
                        flat_embeddings = get_word_embeddings_bert(flat_words, tokenizer, word_model)
                    elif backend != "bert" and sentence_encoder is not None:
                        flat_embeddings = get_word_embeddings(flat_words, sentence_encoder)
                    else:
                        flat_embeddings = None

                    if flat_embeddings is not None:
                        offset = 0
                        for row_idx, row in enumerate(rows_chunk):
                            words = row[measures['words_texts']]
                            if len(words) < min_coherence_turn_length:
                                continue
                            count = len(words)
                            row_embeddings[row_idx] = flat_embeddings[offset:offset + count]
                            offset += count
                except Exception as exc:
                    logger.info(f"Error in batch word embedding analysis: {exc}")
                    row_embeddings = [None] * len(rows_chunk)

            for row_idx, row in enumerate(rows_chunk):
                words = row[measures['words_texts']]
                try:
                    if len(words) < min_coherence_turn_length:
                        append_nan_values(chunk_lists, len(words))
                        continue

                    emb = row_embeddings[row_idx]
                    if emb is not None:
                        coherence, coherence_5, coherence_10, variability = _word_coherence_from_embeddings(emb)
                    else:
                        coherence, coherence_5, coherence_10, variability = coherence_fn(row)

                    chunk_lists['word_coherence'] += coherence
                    chunk_lists['word_coherence_5'] += coherence_5
                    chunk_lists['word_coherence_10'] += coherence_10
                    for k in range(2, 11):
                        chunk_lists['variability'][k] += variability[k]

                except Exception as exc:
                    logger.info(f"Error in word coherence analysis for row: {exc}")
                    append_nan_values(chunk_lists, len(words))

            _extend_coherence_lists(overall_lists, chunk_lists)
            _release_accelerator_cache()

        for _, row in utterances_speaker.iterrows():
            chunk_rows.append(row)
            if len(chunk_rows) >= WORD_STREAM_CHUNK_SIZE:
                _process_chunk(chunk_rows)
                chunk_rows = []

        _process_chunk(chunk_rows)

        # Update word_df with calculated coherence values
        word_df[measures['word_coherence']] = overall_lists['word_coherence']
        word_df[measures['word_coherence_5']] = overall_lists['word_coherence_5']
        word_df[measures['word_coherence_10']] = overall_lists['word_coherence_10']
        for k in range(2, 11):
            word_df[measures[f'word_coherence_variability_{k}']] = overall_lists['variability'][k]

        # Update the summary-level dataframe
        summ_df = get_word_coherence_summary(word_df, summ_df, measures)

        df_list = [word_df, turn_df, summ_df]
    except Exception as e:
        logger.info(f"Error in word coherence analysis: {e}")
    finally:
        return df_list

def calculate_perplexity(
    text: str,
    model: Optional[AutoModelForCausalLM],
    tokenizer: Optional[AutoTokenizer],
    model_max_length: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the perplexity of the input text using the Gemma causal language model.

    Parameters:
    ...........
    text: str
        The input text to be analyzed.
    model: AutoModelForCausalLM
        A Gemma causal language model.
    tokenizer: AutoTokenizer
        The tokenizer paired with the causal language model.

    Returns:
    ...........
    Tuple of floats
        The calculated perplexity of the input text (global teacher-forced perplexity).
        The calculated windowed perplexity with window size 2.
        The calculated windowed perplexity with window size 5.
        The calculated windowed perplexity with window size 7.

    ------------------------------------------------------------------------------------------------------
    """
    if model is None or tokenizer is None:
        return (np.nan, np.nan, np.nan, np.nan)

    if not isinstance(text, str):
        return (np.nan, np.nan, np.nan, np.nan)

    clean_text = re.sub(r"\s+", " ", text.strip())
    if len(clean_text) == 0:
        return (np.nan, np.nan, np.nan, np.nan)

    first_param = next(model.parameters(), None)
    device = first_param.device if first_param is not None else torch.device("cpu")
    cached_ids = _cache_get(_PPL_TOKEN_CACHE, clean_text)
    if cached_ids is None:
        tokens = tokenizer(
            clean_text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=PPL_MAX_TOKENS,
        )
        cached_ids = tokens["input_ids"].detach().cpu()
        _cache_put(_PPL_TOKEN_CACHE, clean_text, cached_ids)

    input_ids = cached_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    if input_ids.size(1) < 2:
        return (np.nan, np.nan, np.nan, np.nan)

    max_len_model = model_max_length or int(getattr(model.config, "max_position_embeddings", input_ids.size(1)))
    effective_len = min(max_len_model, PPL_MAX_TOKENS)
    if input_ids.size(1) > effective_len:
        input_ids = input_ids[:, :effective_len]
        attention_mask = attention_mask[:, :effective_len]

    def _teacher_forced_perplexity(ids: torch.Tensor, mask: torch.Tensor) -> float:
        """Compute full-sequence teacher-forced perplexity for token ids."""
        with torch.inference_mode():
            outputs = model(input_ids=ids, attention_mask=mask)
            logits = outputs.logits[:, :-1, :].float()
        labels = ids[:, 1:]
        attn = mask[:, 1:].to(logits.dtype)
        attn_sum = attn.sum()
        if attn_sum <= 0:
            return float("nan")
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        nll = (nll * attn).sum() / attn_sum
        return float(torch.exp(nll).item())

    def _windowed_perplexity(ids: torch.Tensor, k: int) -> float:
        """Estimate next-token perplexity from limited left-context windows."""
        ids = ids[:, :min(ids.size(1), PPL_MAX_TOKENS)]
        seq_len = ids.size(1)
        if seq_len < 2:
            return float("nan")

        k_eff = min(k, max_len_model - 1)
        if k_eff <= 0:
            return float("nan")

        pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_token is None:
            fallback_token = tokenizer.eos_token or tokenizer.pad_token or tokenizer.bos_token
            if fallback_token is not None:
                pad_token = tokenizer.convert_tokens_to_ids(fallback_token)
            else:
                pad_token = 0

        log_prob_chunks = []
        for start in range(1, seq_len, WINDOW_BATCH_SIZE):
            positions = list(range(start, min(seq_len, start + WINDOW_BATCH_SIZE)))
            if not positions:
                continue

            lens, next_tokens, sequences = [], [], []
            for pos in positions:
                window = min(k_eff, pos)
                lens.append(window)
                next_tokens.append(int(ids[0, pos].item()))
                sequences.append(ids[0, pos - window:pos])

            max_window = max(lens) if lens else 0
            if max_window == 0:
                continue

            batch = torch.full((len(sequences), max_window), pad_token, dtype=torch.long, device=device)
            for row_idx, seq in enumerate(sequences):
                if lens[row_idx] == 0:
                    continue
                batch[row_idx, -lens[row_idx]:] = seq

            attn_mask = (batch != pad_token).long()
            next_tensor = torch.tensor(next_tokens, device=device, dtype=torch.long)

            with torch.inference_mode():
                outputs = model(input_ids=batch, attention_mask=attn_mask)
                # Context tokens are right-aligned, so the last real token sits at max_window - 1
                # for every row regardless of its individual context length.
                idx = torch.full((batch.size(0),), max_window - 1, device=device, dtype=torch.long)
                logits = outputs.logits[torch.arange(batch.size(0), device=device), idx].float()
                log_probs = torch.log_softmax(logits, dim=-1)
                log_prob_chunks.append(log_probs[torch.arange(batch.size(0), device=device), next_tensor])

        if not log_prob_chunks:
            return float("nan")

        stacked = torch.cat(log_prob_chunks).float()
        finite_mask = torch.isfinite(stacked)
        if not torch.any(finite_mask):
            return float("nan")

        stacked = stacked[finite_mask]
        return float(torch.exp(-stacked.mean()).item())

    global_ppl = _teacher_forced_perplexity(input_ids, attention_mask)
    ppl_2 = _windowed_perplexity(input_ids, 2)
    ppl_5 = _windowed_perplexity(input_ids, 5)
    ppl_7 = _windowed_perplexity(input_ids, 7)

    return global_ppl, ppl_2, ppl_5, ppl_7


def calculate_perplexity_bert(
    text: str,
    model: Optional[BertForMaskedLM],
    tokenizer: Optional[BertTokenizer],
) -> Tuple[float, float, float, float]:
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the pseudo-perplexity of the input text using BERT (masked LM).

    Parameters:
    ...........
    text: str
        The input text to be analyzed.
    model: BertForMaskedLM
        A BERT masked language model.
    tokenizer: BertTokenizer
        A BERT tokenizer.

    Returns:
    ...........
    Tuple of floats
        The calculated pseudo-perplexity of the input text.
        The calculated pseudo-perplexity of the input text using 2 words before and after the masked token.
        The calculated pseudo-perplexity of the input text using 5 words before and after the masked token.
        The calculated pseudo-perplexity of the input text using 7 words before and after the masked token.

    ------------------------------------------------------------------------------------------------------
    """
    if model is None or tokenizer is None:
        return np.nan, np.nan, np.nan, np.nan

    if not isinstance(text, str):
        return np.nan, np.nan, np.nan, np.nan

    clean_text = re.sub(r"\s+", " ", text.strip())
    if len(clean_text) == 0 or len(clean_text.split()) < 2:
        return np.nan, np.nan, np.nan, np.nan

    max_len_model = int(getattr(model.config, "max_position_embeddings", 512))
    cache_key = f"{max_len_model}:{clean_text}"
    cached_ids = _cache_get(_BERT_TOKEN_CACHE, cache_key)
    if cached_ids is None:
        tokens = tokenizer(
            clean_text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=max_len_model,
        )
        cached_ids = tokens.input_ids.detach().cpu()
        _cache_put(_BERT_TOKEN_CACHE, cache_key, cached_ids)
    input_ids = cached_ids

    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")
    input_ids = input_ids.to(device)

    seq_len = int(input_ids.size(1))
    if seq_len < 2:
        return np.nan, np.nan, np.nan, np.nan

    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        return np.nan, np.nan, np.nan, np.nan

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def _adjust_bounds(i, window_size, seq_len, max_len):
        """Clamp a masked-LM context window to sequence and model limits."""
        start = max(0, i - window_size)
        end = min(seq_len - 1, i + window_size)
        L = end - start + 1
        if L > max_len:
            left_budget = min(i - start, (max_len - 1) // 2)
            right_budget = max_len - 1 - left_budget
            start = i - left_budget
            end = min(i + right_budget, seq_len - 1)
            L = end - start + 1
            if L > max_len:
                start = end - (max_len - 1)
        return int(start), int(end)

    def _batched_log_probs_for_window(window_size, batch_size=64):
        """Collect masked-token log probabilities for a fixed context window."""
        all_log_probs = []
        indices = list(range(seq_len))
        with torch.no_grad():
            for b_start in range(0, seq_len, batch_size):
                b_end = min(seq_len, b_start + batch_size)
                batch_idx = indices[b_start:b_end]

                segs = []
                adj_positions = []
                true_ids = []
                maxL = 0
                for i in batch_idx:
                    start, end = _adjust_bounds(i, window_size, seq_len, max_len_model)
                    seg = input_ids[0, start:end + 1].clone()
                    idx_adj = i - start
                    seg[idx_adj] = mask_id
                    segs.append(seg)
                    adj_positions.append(idx_adj)
                    true_ids.append(int(input_ids[0, i].item()))
                    if seg.numel() > maxL:
                        maxL = int(seg.numel())

                if not segs:
                    continue

                B = len(segs)
                batch_inputs = torch.full((B, maxL), pad_id, dtype=input_ids.dtype, device=device)
                attn_mask = torch.zeros((B, maxL), dtype=torch.long, device=device)
                for r, seg in enumerate(segs):
                    L = int(seg.numel())
                    batch_inputs[r, :L] = seg
                    attn_mask[r, :L] = 1

                outputs = model(input_ids=batch_inputs, attention_mask=attn_mask)
                logits = outputs.logits  # [B, maxL, V]
                rows = torch.arange(len(adj_positions), device=device)
                pos_logits = logits[rows, torch.tensor(adj_positions, device=device), :]
                pos_log_probs = torch.log_softmax(pos_logits, dim=-1)
                gathered = pos_log_probs[rows, torch.tensor(true_ids, device=device)]
                all_log_probs.extend(gathered.detach().cpu().tolist())

        return all_log_probs

    log_probs_256 = _batched_log_probs_for_window(min(256, max_len_model - 1))
    log_probs_2 = _batched_log_probs_for_window(2)
    log_probs_5 = _batched_log_probs_for_window(5)
    log_probs_7 = _batched_log_probs_for_window(7)

    perplexity = float(np.exp(-np.mean(log_probs_256))) if len(log_probs_256) else np.nan
    perplexity_5 = float(np.exp(-np.mean(log_probs_2))) if len(log_probs_2) else np.nan
    perplexity_11 = float(np.exp(-np.mean(log_probs_5))) if len(log_probs_5) else np.nan
    perplexity_15 = float(np.exp(-np.mean(log_probs_7))) if len(log_probs_7) else np.nan

    return perplexity, perplexity_5, perplexity_11, perplexity_15

def calculate_phrase_tangeniality(
    phrases_texts,
    utterance_text,
    sentence_encoder,
    lm_model,
    tokenizer,
    phrase_embeddings: Optional[np.ndarray] = None,
):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the semantic similarity of each phrase to the immediately preceding phrase,
    the semantic similarity of each phrase to the phrase 2 turns before, and the model-based perplexity of the turn.

    Parameters:
    ...........
    phrases_texts: list
        List of transcribed text at the phrase level.
    utterance_text: str
        The full transcribed text.
    sentence_encoder: SentenceTransformer
        A SentenceTransformer model.
    lm_model: AutoModelForCausalLM | BertForMaskedLM
        A language model for perplexity (Gemma or BERT depending on backend).
    tokenizer: AutoTokenizer | BertTokenizer
        A tokenizer paired with the language model.

    Returns:
    ...........
    sentence_tangeniality1: float
        The semantic similarity of each phrase to the immediately preceding phrase.
    sentence_tangeniality2: float
        The semantic similarity of each phrase to the phrase 2 turns before.
    perplexity: float
        The perplexity of the turn (global teacher-forced for Gemma, pseudo-perplexity for BERT).
    perplexity_5: float
        The windowed perplexity of the turn with window size 2.
    perplexity_11: float
        The windowed perplexity of the turn with window size 5.
    perplexity_15: float
        The windowed perplexity of the turn with window size 7.

    ------------------------------------------------------------------------------------------------------
    """
    sentence_tangeniality1 = np.nan
    sentence_tangeniality2 = np.nan
    if sentence_encoder is not None and len(phrases_texts) > 0:
        embeddings = phrase_embeddings
        if embeddings is None:
            embeddings = sentence_encoder.encode(
                phrases_texts,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        sentence_tangeniality1, sentence_tangeniality2 = _phrase_tangeniality_from_embeddings(embeddings)

    perplexity, perplexity_5, perplexity_11, perplexity_15 = np.nan, np.nan, np.nan, np.nan
    if tokenizer is not None and lm_model is not None:
        if _get_backend() == "bert":
            perplexity, perplexity_5, perplexity_11, perplexity_15 = calculate_perplexity_bert(
                utterance_text,
                lm_model,
                tokenizer,
            )
        else:
            max_len_model = int(getattr(lm_model.config, "max_position_embeddings", 32768))
            perplexity, perplexity_5, perplexity_11, perplexity_15 = calculate_perplexity(
                utterance_text,
                lm_model,
                tokenizer,
                model_max_length=max_len_model,
            )

    return sentence_tangeniality1, sentence_tangeniality2, perplexity, perplexity_5, perplexity_11, perplexity_15

def calculate_slope(y):
    """
    ------------------------------------------------------------------------------------------------------
    This function calculates the slope
     of the input list using linear regression

    Parameters:
    ...........
    y: list
        A list of values.

    Returns:
    ...........
    float
        The calculated slope of the input list.

    ------------------------------------------------------------------------------------------------------
    """
    # remove NaNs
    y = [val for val in y if not np.isnan(val)]

    x = range(len(y))
    try:
        slope, _ = np.polyfit(x, y, 1)
    except:
        slope = np.nan

    return slope

def init_model(language, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function initializes the appropriate models and tokenizers based on language.

    Parameters:
    ...........
    language: str
        Language of the transcribed text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    sentence_encoder: SentenceTransformer
        A SentenceTransformer model.
    tokenizer: AutoTokenizer | BertTokenizer
        A tokenizer compatible with the selected backend.
    lm_model: AutoModelForCausalLM | BertForMaskedLM
        A language model for perplexity (Gemma or BERT).

    ------------------------------------------------------------------------------------------------------
    """
    sentence_encoder, tokenizer, lm_model = None, None, None

    backend = _get_backend()
    if backend == "bert":
        bundle = get_bert_bundle(language, measures)
        sentence_encoder = bundle.get("sentence_encoder")
        tokenizer = bundle.get("tokenizer")
        lm_model = bundle.get("mlm_model")

        if sentence_encoder is None:
            if language in measures.get("supported_langs_sentence_embeddings", set()):
                logger.info(f"Sentence encoder is unavailable for language {language}.")
            else:
                logger.info(f"Language {language} not supported for phrase coherence analysis")

        if tokenizer is None or lm_model is None:
            if language in measures.get("supported_langs_bert", set()) or language in measures.get("english_langs", set()):
                logger.info(f"BERT language model unavailable for language {language}.")
            else:
                logger.info(f"Language {language} not supported for perplexity analysis")

        return sentence_encoder, tokenizer, lm_model

    bundle = get_model_bundle(language)
    sentence_encoder = bundle.get("sentence_encoder")
    tokenizer = bundle.get("tokenizer")
    lm_model = bundle.get("lm_model")

    if sentence_encoder is None and language not in measures.get("english_langs", set()):
        if language in measures.get("supported_langs_sentence_embeddings", set()):
            logger.info(f"Sentence encoder is unavailable for language {language}.")
        else:
            logger.info(f"Language {language} not supported for phrase coherence analysis")

    if (tokenizer is None or lm_model is None) and language not in measures.get("english_langs", set()):
        if language in measures.get("supported_langs_bert", set()):
            logger.info(f"Gemma language model unavailable for language {language}.")
        else:
            logger.info(f"Language {language} not supported for perplexity analysis")

    return sentence_encoder, tokenizer, lm_model

def calculate_turn_coherence(
    utterances_filtered,
    turn_df,
    min_coherence_turn_length,
    speaker_label,
    sentence_encoder,
    lm_model,
    tokenizer,
    measures,
    dialogue_utterances_filtered=None,
):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates turn coherence measures for the specified speaker.

    Parameters:
    ...........
    utterances_filtered: pandas dataframe
        A dataframe containing the turns extracted from the JSON object
        after filtering out turns with less than min_turn_length words of the specified speaker.
    turn_df: pandas dataframe
        A dataframe containing turn summary information.
    min_coherence_turn_length: int
        Minimum number of words in a turn for word coherence analysis.
    speaker_label: str
        Speaker label
    sentence_encoder: SentenceTransformer
        A SentenceTransformer model.
    lm_model: AutoModelForCausalLM
        A Gemma causal language model.
    tokenizer: AutoTokenizer
        The tokenizer paired with the causal language model.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    turn_df: pandas dataframe
        The updated turn_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    # speaker_label = ""
    # turn-to-turn tangentiality only needs adjacent cosine similarities
    utterances_texts = utterances_filtered[measures['utterance_text']].values.tolist()
    adjacent_turn_similarity = _adjacent_turn_similarity_from_texts(
        utterances_texts,
        sentence_encoder,
    )
    phrase_embeddings_by_row: Dict[object, np.ndarray] = {}
    previous_speaker_turn_similarity_list = _previous_speaker_turn_similarity(
        dialogue_utterances_filtered,
        speaker_label,
        PREVIOUS_SPEAKER_SIMILARITY_MIN_TURN_LENGTH,
        sentence_encoder,
        measures,
    )
    if sentence_encoder:
        flat_phrases: List[str] = []
        phrase_row_indices: List[object] = []
        phrase_row_counts: List[int] = []
        for row_idx, row in utterances_filtered.iterrows():
            if len(row[measures['words_texts']]) < min_coherence_turn_length:
                continue
            phrases = row[measures['phrases_texts']]
            if len(phrases) == 0:
                continue
            phrase_row_indices.append(row_idx)
            phrase_row_counts.append(len(phrases))
            flat_phrases.extend(phrases)

        if flat_phrases:
            all_phrase_embeddings = _encode_in_chunks(
                sentence_encoder,
                flat_phrases,
                EMBEDDING_BATCH_SIZE,
            )
            offset = 0
            for row_idx, count in zip(phrase_row_indices, phrase_row_counts):
                phrase_embeddings_by_row[row_idx] = all_phrase_embeddings[offset:offset + count]
                offset += count

    # Initialize coherence lists
    sentence_tangeniality1_list, sentence_tangeniality2_list = [], []
    perplexity_list, perplexity_5_list, perplexity_11_list, perplexity_15_list = [], [], [], []
    turn_to_turn_tangeniality_list = []
    if len(previous_speaker_turn_similarity_list) != len(utterances_filtered):
        previous_speaker_turn_similarity_list = [np.nan] * len(utterances_filtered)
    for i, row in utterances_filtered.iterrows():
        current_speaker = row[measures['speaker_label']]
        if speaker_label is not None and current_speaker != speaker_label:
            continue
        if len(row[measures['words_texts']]) < min_coherence_turn_length:
            sentence_tangeniality1_list.append(np.nan)
            sentence_tangeniality2_list.append(np.nan)
            perplexity_list.append(np.nan)
            perplexity_5_list.append(np.nan)
            perplexity_11_list.append(np.nan)
            perplexity_15_list.append(np.nan)
            turn_to_turn_tangeniality_list.append(np.nan)
            continue

        phrases_texts = row[measures['phrases_texts']]
        utterance_text = row[measures['utterance_text']]

        sentence_tangeniality1, sentence_tangeniality2, perplexity, perplexity_5, perplexity_11, perplexity_15 = calculate_phrase_tangeniality(
            phrases_texts,
            utterance_text,
            sentence_encoder,
            lm_model,
            tokenizer,
            phrase_embeddings=phrase_embeddings_by_row.get(i),
        )

        sentence_tangeniality1_list.append(sentence_tangeniality1)
        sentence_tangeniality2_list.append(sentence_tangeniality2)
        perplexity_list.append(perplexity)
        perplexity_5_list.append(perplexity_5)
        perplexity_11_list.append(perplexity_11)
        perplexity_15_list.append(perplexity_15)

        if i == 0 or len(utterances_filtered.iloc[i - 1][measures['words_texts']]) < min_coherence_turn_length or adjacent_turn_similarity is None:
            turn_to_turn_tangeniality_list.append(np.nan)
        else:
            turn_to_turn_tangeniality_list.append(float(adjacent_turn_similarity[i]))

    turn_df[measures['sentence_tangeniality1']] = sentence_tangeniality1_list
    turn_df[measures['sentence_tangeniality2']] = sentence_tangeniality2_list
    turn_df[measures['perplexity']] = perplexity_list
    turn_df[measures['perplexity_5']] = perplexity_5_list
    turn_df[measures['perplexity_11']] = perplexity_11_list
    turn_df[measures['perplexity_15']] = perplexity_15_list
    turn_df[measures['turn_to_turn_tangeniality']] = turn_to_turn_tangeniality_list
    turn_df[measures['turn_to_previous_speaker_turn_similarity']] = previous_speaker_turn_similarity_list

    return turn_df

def get_phrase_coherence(
    df_list,
    utterances_filtered,
    min_coherence_turn_length,
    speaker_label,
    language,
    measures,
    dialogue_utterances_filtered=None,
):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates turn coherence measures

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    utterances_filtered: pandas dataframe
        A dataframe containing the turns extracted from the JSON object
        after filtering out turns with less than min_turn_length words of the specified speaker.
    min_coherence_turn_length: int
        Minimum number of words in a turn for word coherence analysis.
    speaker_label: str
        Speaker label
    language: str
        Language of the transcribed text.
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

        sentence_encoder, tokenizer, lm_model = init_model(language, measures)
        if not sentence_encoder and not tokenizer and not lm_model:
            logger.info(f"Language {language} not supported for phrase coherence nor perplexity analysis")
            return df_list

        # turn-level
        if len(turn_df) > 0:
            turn_df = calculate_turn_coherence(
                utterances_filtered,
                turn_df,
                min_coherence_turn_length,
                speaker_label,
                sentence_encoder,
                lm_model,
                tokenizer,
                measures,
                dialogue_utterances_filtered=dialogue_utterances_filtered,
            )

            for measure in [
                'sentence_tangeniality1',
                'sentence_tangeniality2',
                'perplexity',
                'perplexity_5',
                'perplexity_11',
                'perplexity_15',
                'turn_to_turn_tangeniality',
                'turn_to_previous_speaker_turn_similarity',
            ]:
                if turn_df[measures[measure]].isnull().all():
                    continue
                summ_df[measures[measure + '_mean']] = turn_df[measures[measure]].mean(skipna=True)
                summ_df[measures[measure + '_var']] = turn_df[measures[measure]].var(skipna=True)

            if not turn_df[measures['turn_to_turn_tangeniality']].isnull().all():
                summ_df[measures['turn_to_turn_tangeniality_slope']] = calculate_slope(turn_df[measures['turn_to_turn_tangeniality']])
            if not turn_df[measures['turn_to_previous_speaker_turn_similarity']].isnull().all():
                summ_df[measures['turn_to_previous_speaker_turn_similarity_slope']] = calculate_slope(
                    turn_df[measures['turn_to_previous_speaker_turn_similarity']]
                )

        df_list = [word_df, turn_df, summ_df]
    except Exception as e:
        logger.info(f"Error in phrase coherence analysis: {e}")
    finally:
        return df_list
def _encode_in_chunks(
    encoder: SentenceTransformer,
    texts: List[str],
    batch_size: int,
) -> np.ndarray:
    """Encode texts in smaller batches to control peak memory usage."""
    if len(texts) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    outputs: List[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        if len(chunk) == 0:
            continue
        # oom here
        chunk_emb = encoder.encode(
            chunk,
            batch_size=min(batch_size, len(chunk)),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        outputs.append(np.asarray(chunk_emb, dtype=np.float32))

    if not outputs:
        return np.zeros((0, encoder.get_sentence_embedding_dimension()), dtype=np.float32)

    return np.vstack(outputs)
def _release_accelerator_cache() -> None:
    """Clear CUDA or MPS caches after heavy coherence computations."""
    has_accelerator = False
    if torch.cuda.is_available():
        has_accelerator = True
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        has_accelerator = True
        try:
            torch.mps.empty_cache()  # type: ignore[attr-defined]
        except Exception:
            pass
    if has_accelerator:
        gc.collect()


def _new_coherence_lists() -> Dict[str, object]:
    """Create empty accumulators for word-level coherence outputs."""
    return {
        "word_coherence": [],
        "word_coherence_5": [],
        "word_coherence_10": [],
        "variability": {k: [] for k in range(2, 11)},
    }


def _extend_coherence_lists(target: Dict[str, object], source: Dict[str, object]) -> None:
    """Append one coherence accumulator into another in place."""
    target["word_coherence"].extend(source["word_coherence"])
    target["word_coherence_5"].extend(source["word_coherence_5"])
    target["word_coherence_10"].extend(source["word_coherence_10"])
    target_var = target["variability"]
    source_var = source["variability"]
    for k in range(2, 11):
        target_var[k].extend(source_var[k])
