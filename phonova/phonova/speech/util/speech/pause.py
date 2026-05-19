# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
import pandas as pd
import numpy as np
import logging
from functools import lru_cache

import nltk
import string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

### Basic staff for ukr lang
ukrainian_hierarchy = [ # add 'ь'
    "аеєиіїоуюя",   # голосні української мови
    "лмнрй",       # носові, рідкі та напівголосні (наприклад, й)
    "зсжшщ",       # фрикативні
    "бвгґдкптфхцч"  # змичні (решта приголосних)
]

_PUNCT_TRANSLATOR = str.maketrans("", "", string.punctuation)


def _normalize_syllable_lang(lang):
    """Map incoming language aliases to the supported syllable tokenizers."""
    normalized = (lang or "en").lower()
    return "uk" if normalized in {"ua", "uk"} else "en"


@lru_cache(maxsize=2)
def _get_syllable_tokenizer(lang):
    """Create the syllable tokenizer for the requested language."""
    if lang == "uk":
        return nltk.tokenize.SyllableTokenizer(lang="uk", sonority_hierarchy=ukrainian_hierarchy)
    return nltk.tokenize.SyllableTokenizer()


def _syllable_tokens(text):
    """Normalize text and split it into tokens before syllable counting."""
    clean_text = text.translate(_PUNCT_TRANSLATOR).lower()
    if not clean_text.strip():
        return []
    stripped = clean_text.strip()
    # Fast path for the hot word-level loop where inputs are single tokens.
    if stripped == clean_text and " " not in stripped and "\t" not in stripped and "\n" not in stripped:
        return [stripped]
    return [tok for tok in nltk.word_tokenize(clean_text) if tok]


def _count_syllables_from_tokens(tokens, syllable_tokenizer):
    """Count syllables across a token sequence using the provided tokenizer."""
    return sum(len(syllable_tokenizer.tokenize(token)) for token in tokens)


def _pause_series_non_negative(values):
    """Convert pause values to numeric seconds and clamp negatives to zero."""
    return pd.to_numeric(values, errors="coerce").clip(lower=0.0)


def _bounded_speech_percentage(pause_sum_seconds, duration_minutes):
    """
    Convert silence duration and turn/file duration to speech percentage,
    clamped to [0, 100].
    """
    if duration_minutes is None:
        return np.nan
    try:
        duration_minutes = float(duration_minutes)
    except Exception:
        return np.nan
    if not np.isfinite(duration_minutes) or duration_minutes <= 0:
        return np.nan

    duration_seconds = duration_minutes * 60.0
    speech_pct = 100.0 * (1.0 - (float(pause_sum_seconds) / duration_seconds))
    if not np.isfinite(speech_pct):
        return np.nan
    return float(np.clip(speech_pct, 0.0, 100.0))


def get_num_of_syllables(text, lang = 'en'):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the number of syllables in the input text.

    Parameters:
    ...........
    text: str
        The input text.

    Returns:
    ...........
    syllable_count: int
        The number of syllables in the input text.

    ---------------------------------------------------------------------------------------
    """


    normalized_lang = _normalize_syllable_lang(lang)
    syllable_tokenizer = _get_syllable_tokenizer(normalized_lang)
    text = text if isinstance(text, str) else ("" if text is None else str(text))
    tokens = _syllable_tokens(text)

    return _count_syllables_from_tokens(tokens, syllable_tokenizer)


def get_num_of_syllables_batch(text_list, lang='en'):
    """Count syllables for each text item in a batch."""
    normalized_lang = _normalize_syllable_lang(lang)
    syllable_tokenizer = _get_syllable_tokenizer(normalized_lang)
    output = []
    for text in text_list:
        norm_text = text if isinstance(text, str) else ("" if text is None else str(text))
        tokens = _syllable_tokens(norm_text)
        output.append(_count_syllables_from_tokens(tokens, syllable_tokenizer))
    return output

def calculate_pause_features_for_word(word_df, df_diff, word_list, turn_index, measures, lang):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates various pause-related speech characteristic
        features at the word level and adds them to the output dataframe word_df.

    Parameters:
    ...........
    word_df: pandas dataframe
        A dataframe containing word summary information
    df_diff: pandas dataframe
        A dataframe containing the word-level information
            from the JSON response.
    word_list: list
        List of transcribed text at the word level.
    turn_index: list
        A list containing the indices of the first and last word
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    word_df: pandas dataframe
        The updated word_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    turn_starts = [pindex[0] for pindex in turn_index]
    word_pauses = _pause_series_non_negative(df_diff[measures["pause"]])
    word_df[measures["word_pause"]] = word_pauses.where(~df_diff[measures["old_index"]].isin(turn_starts), np.nan)
    
    word_df[measures["num_syllables"]] = pd.Series(get_num_of_syllables_batch(word_list, lang=lang))
    return word_df

def calculate_pause_features_for_turn(df_diff, df, text_level, index_list, time_index, measures, language):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates various pause-related speech
     characteristic features at the turn
     level and adds them to the output dataframe df.

    Parameters:
    ...........
    df_diff: pandas dataframe
        A dataframe containing the word-level information from the JSON response.
    df: pandas dataframe
        A dataframe containing turn summary information
    text_level: list
        List of transcribed text at the turn level.
    index_list: list
        A list containing the indices of the first and last word in each turn.
    time_index: list
        A list containing the names of the columns in json that contain
         the start and end times of each word.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df: pandas dataframe
        The updated df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    turn_syllable_counts = get_num_of_syllables_batch(text_level, lang=language)
    for j, index in enumerate(index_list):
        try:
            rng = range(index[0], index[1] + 1)
            turn_data = df_diff[df_diff[measures["old_index"]].isin(rng)]

            raw_pauses = pd.to_numeric(turn_data[measures["pause"]], errors="coerce").to_numpy(dtype=float)
            pauses = np.clip(raw_pauses[1:], a_min=0.0, a_max=None) if len(raw_pauses) > 1 else np.array([], dtype=float)
            finite_pauses = pauses[np.isfinite(pauses)]
            turn_duration = (float(turn_data.iloc[-1][time_index[1]]) - float(turn_data.iloc[0][time_index[0]])) / 60

            df.loc[j, measures[f"turn_minutes"]] = turn_duration
            df.loc[j, measures[f"turn_words"]] = len(turn_data)

            if len(finite_pauses) > 0:
                df.loc[j, measures["pause_var"]] = np.var(finite_pauses) if len(finite_pauses) > 1 else 0
                df.loc[j, measures["pause_meandur"]] = np.mean(finite_pauses)

            if turn_duration > 0:
                speech_percentage = _bounded_speech_percentage(np.sum(finite_pauses), turn_duration)
                if np.isfinite(speech_percentage):
                    df.loc[j, measures["speech_percentage"]] = speech_percentage

                if language in measures["english_langs"] or language in ['uk', 'ua']:
                    syllable_rate = (turn_syllable_counts[j] / turn_duration)
                    df.loc[j, measures["syllable_rate"]] = syllable_rate

                df.loc[j, measures["word_rate"]] = len(turn_data) / turn_duration
        except Exception as e:
            logger.info(f"Error in pause feature calculation for turn {j}: {e}")
            continue

    return df

def get_pause_feature_turn(turn_df, df_diff, turn_list, turn_index, time_index, measures, language):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates various pause-related speech characteristic
        features at the turn level and adds them to the output dataframe turn_df.

    Parameters:
    ...........
    turn_df: pandas dataframe
        A dataframe containing turn summary information
    df_diff: pandas dataframe
        A dataframe containing the word-level information
            from the JSON response.
    turn_list: list
        List of transcribed text at the turn level.
    turn_index: list
        A list containing the indices of the first and last word
            in each turn.
    time_index: list
        A list containing the names of the columns in json that contain
            the start and end times of each word.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    turn_df: pandas dataframe
        The updated turn_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    turn_starts = [uindex[0] for uindex in turn_index]
    df_diff_turn = df_diff[df_diff[measures["old_index"]].isin(turn_starts)].reset_index(drop=True)
    raw_turn_pause = pd.to_numeric(df_diff_turn[measures["pause"]], errors="coerce")

    turn_df = turn_df.reset_index(drop=True)
    turn_df[measures["turn_pause"]] = raw_turn_pause.clip(lower=0.0)
    # Keep overlap signal from raw pauses; zero-gap boundaries are not interruptions.
    turn_df[measures["interrupt_flag"]] = (raw_turn_pause < 0)

    turn_df = calculate_pause_features_for_turn(df_diff, turn_df, turn_list, turn_index, time_index, measures, language)
    return turn_df

def update_summ_df(df_diff, summ_df, full_text, time_index, word_df, turn_df, measures, language):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates various pause-related speech characteristic
     features at the file level and adds them to the output dataframe summ_df.

    Parameters:
    ...........
    df_diff: pandas dataframe
        A dataframe containing the word-level information
         from the JSON response.
    summ_df: pandas dataframe
        A dataframe containing the speech characteristics of the input text.
    time_index: list
        A list containing the names of the columns in json
         that contain the start and end times of each word.
    word_df: pandas dataframe
        A dataframe containing word summary information
    turn_df: pandas dataframe
        A dataframe containing turn summary information
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    summ_df: pandas dataframe
        The updated summ_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    if len(turn_df) > 0:
        speech_minutes = turn_df[measures["turn_minutes"]].sum()
        speech_words = turn_df[measures["turn_words"]].sum()
    else:
        speech_minutes = (float(df_diff.iloc[-1][time_index[1]]) - float(df_diff.iloc[0][time_index[0]])) / 60
        speech_words = len(df_diff)

    summ_df[measures["speech_minutes"]] = [speech_minutes]    
    summ_df[measures["speech_words"]] = [speech_words]

    if speech_minutes > 0:
        summ_df[measures["word_rate"]] = speech_words / speech_minutes
        summ_df[measures["syllable_rate"]] = get_num_of_syllables(full_text, lang=language) / speech_minutes
        file_length_values = pd.to_numeric(summ_df[measures["file_length"]], errors="coerce")
        if len(file_length_values) > 0 and np.isfinite(file_length_values.iloc[0]) and file_length_values.iloc[0] > 0:
            speech_pct = 100.0 * (speech_minutes / float(file_length_values.iloc[0]))
            summ_df[measures["speech_percentage"]] = float(np.clip(speech_pct, 0.0, 100.0))
        else:
            summ_df[measures["speech_percentage"]] = np.nan

    if len(word_df[measures["word_pause"]]) > 1:
        summ_df[measures["word_pause_mean"]] = word_df[measures["word_pause"]].mean(skipna=True)
        summ_df[measures["word_pause_var"]] = word_df[measures["word_pause"]].var(skipna=True)

    if len(turn_df) > 0:
        summ_df[measures["num_turns"]] = len(turn_df)
        summ_df[measures["turn_minutes_mean"]] = turn_df[measures["turn_minutes"]].mean(skipna=True)

        summ_df[measures["turn_words_mean"]] = turn_df[measures["turn_words"]].mean(skipna=True)
        summ_df[measures["turn_pause_mean"]] = turn_df[measures["turn_pause"]].mean(skipna=True)

        summ_df[measures["num_one_word_turns"]] = len(turn_df[turn_df[measures["turn_words"]] == 1])
        summ_df[measures["num_interrupts"]] = len(turn_df[turn_df[measures["interrupt_flag"]]==True])

    return summ_df

def get_pause_feature(json_conf, df_list, text_list, turn_index, measures, time_index, language):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates various pause-related
     speech characteristic features

    Parameters:
    ...........
    json_conf: list
        JSON response object.
    df_list: list
        List of pandas dataframes: word_df, turn_df, summ_df
    text_list: list
        List of transcribed text: split into words, turns, and full text.
    turn_index: list
        List of indices for text_list.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.
    time_index: list
        timepoint index (start/end)
    language: str
        Language of the transcribed text.

    Returns:
    ...........
    df_feature: list
        List of updated pandas dataframes (word_df, turn_df and summ_df)

    ------------------------------------------------------------------------------------------------------
    """
    try:
        if len(json_conf) <= 0:
            return df_list

        word_df, turn_df, summ_df = df_list
        word_list, turn_list, full_text = text_list
        df_diff = pd.DataFrame(json_conf)

        # Calculate the pause time between; each word and add the results to pause_list
        if measures["pause"] not in df_diff.columns:
            df_diff[measures["pause"]] = df_diff[time_index[0]].astype(float) - df_diff[time_index[1]].astype(float).shift(1)
        else:
            df_diff[measures["pause"]] = pd.to_numeric(df_diff[measures["pause"]], errors="coerce")

        # word-level analysis

        word_df = calculate_pause_features_for_word(word_df, df_diff, word_list, turn_index, measures, lang=language)

        # turn-level analysis
        if len(turn_index) > 0:
            turn_df = get_pause_feature_turn(turn_df, df_diff, turn_list, turn_index, time_index, measures, language)

        # file-level analysis
        summ_df = update_summ_df(df_diff, summ_df, full_text, time_index, word_df, turn_df, measures, language)
        df_feature = [word_df, turn_df, summ_df]
        return df_feature
    except Exception as e:
        logger.info(f"Error in pause feature calculation: {e}")
        return df_list
