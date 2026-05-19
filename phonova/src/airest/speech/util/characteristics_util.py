# author:    Vijay Yadav, Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
import logging
import itertools
import sys

import pandas as pd
import numpy as np
import nltk
import spacy
import subprocess
from .speech.pause import get_pause_feature
from .speech.lexical import get_repetitions, get_sentiment, get_pos_tag
from .speech.coherence import get_word_coherence, get_phrase_coherence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
# Suppress warnings from transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

def create_empty_dataframes(measures):
    """
    ------------------------------------------------------------------------------------------------------

    Creating empty measures dataframes

    Parameters:
    ...........
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    tuple: pandas dataframe
        An empty dataframe for word, turn and summary measures

    ------------------------------------------------------------------------------------------------------
    """

    word_df = pd.DataFrame(columns=[measures["word_pause"], measures["num_syllables"], measures["part_of_speech"],
                                    measures["first_person"], measures["verb_tense"],
                                    measures["word_coherence"], measures["word_coherence_5"], measures["word_coherence_10"],
                                    measures["word_coherence_variability_2"], measures["word_coherence_variability_3"],
                                    measures["word_coherence_variability_4"], measures["word_coherence_variability_5"],
                                    measures["word_coherence_variability_6"], measures["word_coherence_variability_7"],
                                    measures["word_coherence_variability_8"], measures["word_coherence_variability_9"],
                                    measures["word_coherence_variability_10"]
                                    ])
    turn_df = pd.DataFrame(columns=[measures["turn_pause"], measures["turn_minutes"], measures["turn_words"], 
                                    measures["word_rate"], measures["syllable_rate"], measures["speech_percentage"], 
                                    measures["pause_meandur"], measures["pause_var"], measures["pos"], measures["neg"], 
                                    measures["neu"], measures["compound"], measures["pos_vader"], measures["neg_vader"],
                                    measures["neu_vader"], measures["compound_vader"], measures["speech_mattr_5"],
                                    measures["speech_mattr_10"], measures["speech_mattr_25"], measures["speech_mattr_50"], measures["speech_mattr_100"],
                                    measures["first_person_percentage"], measures["first_person_sentiment_positive"], measures["first_person_sentiment_negative"],
                                    measures["first_person_sentiment_positive_vader"], measures["first_person_sentiment_negative_vader"],
                                    measures["first_person_sentiment_overall_vader"], measures["phrase_count"],
                                    measures["word_repeat_percentage"], measures["phrase_repeat_percentage"],
                                    measures["sentence_tangeniality1"], measures["sentence_tangeniality2"],
                                    measures["turn_to_turn_tangeniality"], measures["turn_to_previous_speaker_turn_similarity"], measures["perplexity"],
                                    measures["perplexity_5"], measures["perplexity_11"], measures["perplexity_15"],
                                    measures["interrupt_flag"]])

    summ_df = pd.DataFrame(
        columns=[measures["file_length"], measures["speech_minutes"], measures["speech_words"], measures["word_rate"],
                measures["syllable_rate"], measures["word_pause_mean"], measures["word_pause_var"], 
                measures["speech_percentage"], measures["pos"], measures["neg"], measures["neu"], measures["compound"],
                measures["pos_vader"], measures["neg_vader"], measures["neu_vader"], measures["compound_vader"],
                *[measures[f"vader_hist_bin_{idx:02d}"] for idx in range(1, 24)],
                measures["vader_hist_entropy"], measures["vader_neg_tail_mass"], measures["vader_pos_tail_mass"],
                measures["vader_neutral_mass"], measures["participant_utterance_count"],
                measures["speech_mattr_5"], measures["speech_mattr_10"], measures["speech_mattr_25"], measures["speech_mattr_50"], measures["speech_mattr_100"],
                measures["first_person_percentage"], measures["prop_verb_past"], measures["prop_function_words"], measures["first_person_sentiment_positive"],
                measures["first_person_sentiment_negative"], measures["first_person_sentiment_overall"],
                measures["first_person_sentiment_positive_vader"], measures["first_person_sentiment_negative_vader"],
                measures["first_person_sentiment_overall_vader"],
                measures["word_repeat_percentage"], measures["phrase_repeat_percentage"],
                measures["word_coherence_mean"], measures["word_coherence_var"],
                measures["word_coherence_5_mean"], measures["word_coherence_5_var"],
                measures["word_coherence_10_mean"], measures["word_coherence_10_var"],
                measures["word_coherence_variability_2_mean"], measures["word_coherence_variability_2_var"],
                measures["word_coherence_variability_3_mean"], measures["word_coherence_variability_3_var"],
                measures["word_coherence_variability_4_mean"], measures["word_coherence_variability_4_var"],
                measures["word_coherence_variability_5_mean"], measures["word_coherence_variability_5_var"],
                measures["word_coherence_variability_6_mean"], measures["word_coherence_variability_6_var"],
                measures["word_coherence_variability_7_mean"], measures["word_coherence_variability_7_var"],
                measures["word_coherence_variability_8_mean"], measures["word_coherence_variability_8_var"],
                measures["word_coherence_variability_9_mean"], measures["word_coherence_variability_9_var"],
                measures["word_coherence_variability_10_mean"], measures["word_coherence_variability_10_var"],
                measures["num_turns"], measures["num_one_word_turns"], measures["num_phrases"],
                measures["num_one_word_phrases"], measures["turn_minutes_mean"],
                measures["turn_words_mean"], measures["phrase_count_mean"], measures["turn_pause_mean"], measures["speaker_percentage"],
                measures["sentence_tangeniality1_mean"], measures["sentence_tangeniality1_var"],
                measures["sentence_tangeniality2_mean"], measures["sentence_tangeniality2_var"],
                measures["turn_to_turn_tangeniality_mean"], measures["turn_to_turn_tangeniality_var"],
                measures["turn_to_turn_tangeniality_slope"], measures["turn_to_previous_speaker_turn_similarity_mean"],
                measures["turn_to_previous_speaker_turn_similarity_var"], measures["turn_to_previous_speaker_turn_similarity_slope"],
                measures["perplexity_mean"], measures["perplexity_var"],
                measures["perplexity_5_mean"], measures["perplexity_5_var"], measures["perplexity_11_mean"],
                measures["perplexity_11_var"], measures["perplexity_15_mean"], measures["perplexity_15_var"],
                measures["num_interrupts"]])

    return word_df, turn_df, summ_df

def create_index_column(item_data, measures):
    """
    This function creates an index column in the JSON response object.

    Parameters:
    item_data: dict
        JSON response object.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    item_data: dict
        The updated JSON response object.
    """
    all_words = list(itertools.chain.from_iterable([item.get("words", []) for item in item_data]))

    for index, word in enumerate(all_words):
        word[measures["old_index"]] = index

    return item_data

def download_nltk_resources():
    """
    ------------------------------------------------------------------------------------------------------

    This function downloads the
     required NLTK resources for processing text data.

    ------------------------------------------------------------------------------------------------------
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger")

def download_spacy_models():
    """Ensure the English and Ukrainian spaCy models are available locally."""
    models = ["uk_core_news_sm", "en_core_web_sm"]
    for model in models:
        try:
            spacy.load(model)
        except OSError:
            # Install into the same interpreter/env that runs this process.
            subprocess.run([sys.executable, "-m", "spacy", "download", model], check=True)

def download_nltk_resources():
    """Ensure the NLTK punkt and POS-tagger resources are installed."""
    resources = ["punkt", "averaged_perceptron_tagger"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"taggers/{resource}")
        except LookupError:
            nltk.download(resource)

def download_ua_resources():
    """Download the NLP resources required by the Ukrainian text pipeline."""
    # try:
    #     nltk.data.find("tokenizers/punkt")
    #     nltk.download('punkt_tab')
    #     nltk.download('averaged_perceptron_tagger')
    #     # spacy.load("uk_core_news_sm")
    #     # spacy.load("en_core_web_sm")
    # except LookupError:
    #     nltk.download("punkt")
    #     nltk.download('punkt_tab')

    # try:
    #     nltk.data.find("taggers/averaged_perceptron_tagger")
    # except LookupError:
    #     nltk.download("averaged_perceptron_tagger")
    try: 
        download_nltk_resources()
        download_spacy_models()
    except Exception as ex:
        print("EXCEPTION while downloading recources: ", ex)
        
def process_utterance(utterances, current_utterance, utterance_texts, current_words, words_texts, current_speaker, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function processes the utterance and splits it into phrases.

    Parameters:
    ...........
    utterances: list
        A list of utterances.
    current_utterance: list
        A list of current utterance indices.
    utterance_texts: list
        A list of utterance texts.
    current_words: list
        A list of current word indices.
    words_texts: list
        A list of word texts.
    current_speaker: str
        Speaker label
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    utterances: list
        A list of utterances.

    ------------------------------------------------------------------------------------------------------
    """
    # split utterance into phrases
    phrases = nltk.tokenize.sent_tokenize(' '.join(utterance_texts))
    word_counts = np.array([len(phrase.split()) for phrase in phrases])
    # Compute the start indices using cumulative sum and add the starting index of the first phrase
    start_indices = np.cumsum(np.concatenate(([0], word_counts[:-1]))) + current_utterance[0]
    # Compute the end indices by adding word counts to start indices and subtracting 1
    end_indices = start_indices + word_counts - 1
    # Zip start and end indices to create the phrase index tuples
    phrases_idxs = np.column_stack((start_indices, end_indices))

    utterances.append({
        measures['utterance_ids']: (current_utterance[0], current_utterance[-1]),
        measures['utterance_text']: ' '.join(utterance_texts),
        measures['phrases_ids']: phrases_idxs,
        measures['phrases_texts']: phrases.copy(),
        measures['words_ids']: current_words.copy(),
        measures['words_texts']: words_texts.copy(),
        measures['speaker_label']: current_speaker,
    })

    return utterances

def create_turns_aws(item_data, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function creates a dataframe of turns from the JSON response object for AWS.

    Parameters:
    ...........
    item_data: dict
        JSON response object.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    utterances: pandas dataframe
        A dataframe containing the turns extracted from the JSON object,
            along with word, phrase and utterance indices and texts.

    ------------------------------------------------------------------------------------------------------
    """

    utterances, current_utterance, utterance_texts = [], [], []
    current_words, words_texts = [], []
    current_speaker = None
    utterance_id = 0

    for item in item_data:
        # Check if the item is a continuation of the current speaker's turn
        if item['speaker_label'] == current_speaker:
            current_utterance.append(utterance_id)
            utterance_texts.append(item['alternatives'][0]['content'])
            if 'start_time' in item and 'end_time' in item:
                current_words.append(utterance_id)
                words_texts.append(item['alternatives'][0]['content'])
        else:
            # If not, save the current utterance (if any) and start a new one
            if current_utterance:
                utterances = process_utterance(utterances, current_utterance, utterance_texts, current_words, words_texts, current_speaker, measures)
                current_utterance.clear()
                utterance_texts.clear()
                current_words.clear()
                words_texts.clear()
            
            current_speaker = item['speaker_label']
            current_utterance.append(utterance_id)
            utterance_texts.append(item['alternatives'][0]['content'])
            if 'start_time' in item and 'end_time' in item:
                current_words.append(utterance_id)
                words_texts.append(item['alternatives'][0]['content'])
        
        utterance_id += 1

    # Don't forget to add the last utterance if the loop ends
    if current_utterance:
        utterances = process_utterance(utterances, current_utterance, utterance_texts, current_words, words_texts, current_speaker, measures)

    return pd.DataFrame(utterances)

def filter_json_transcribe_aws(item_data, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function filters the JSON response object to only include items with start_time and end_time.

    Parameters:
    ...........
    item_data: dict
        JSON response object.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    filter_json: list
        The updated JSON response object.

    ------------------------------------------------------------------------------------------------------
    """
    filter_json = [item for item in item_data if "start_time" in item and "end_time" in item]
    filter_json = pause_calculation(filter_json, measures, ['start_time', 'end_time'])

    return filter_json

def _default_phrase_payload(idxs, text, words_texts):
    """Build a fallback phrase span and text when Whisper phrases are missing."""
    if not idxs:
        return [], []

    fallback_text = (text or "").strip()
    if not fallback_text:
        fallback_text = " ".join(words_texts).strip()
    return [(idxs[0], idxs[-1])], [fallback_text]

def _extract_phrase_payload(item, idxs, words_texts, text):
    """Extract Whisper phrase spans/text or fall back to a full-segment payload."""
    fallback_ids, fallback_texts = _default_phrase_payload(idxs, text, words_texts)
    raw_phrases = item.get("phrases")
    if not isinstance(raw_phrases, list) or len(raw_phrases) == 0:
        return fallback_ids, fallback_texts

    parsed_ranges = []
    parsed_texts = []
    for phrase in raw_phrases:
        if not isinstance(phrase, dict):
            continue

        try:
            word_start = int(phrase.get("word_start"))
            word_end = int(phrase.get("word_end"))
        except (TypeError, ValueError):
            continue

        if word_start < 0 or word_end < word_start or word_end >= len(idxs):
            continue

        phrase_text = (phrase.get("text") or "").strip()
        if not phrase_text:
            phrase_text = " ".join(words_texts[word_start:word_end + 1]).strip()
        parsed_ranges.append((word_start, word_end))
        parsed_texts.append(phrase_text)

    if len(parsed_ranges) == 0:
        return fallback_ids, fallback_texts

    ordering = sorted(range(len(parsed_ranges)), key=lambda pos: (parsed_ranges[pos][0], parsed_ranges[pos][1]))
    parsed_ranges = [parsed_ranges[pos] for pos in ordering]
    parsed_texts = [parsed_texts[pos] for pos in ordering]

    expected_start = 0
    for word_start, word_end in parsed_ranges:
        if word_start != expected_start:
            return fallback_ids, fallback_texts
        expected_start = word_end + 1

    if expected_start != len(idxs):
        return fallback_ids, fallback_texts

    phrase_ids = [(idxs[word_start], idxs[word_end]) for word_start, word_end in parsed_ranges]
    return phrase_ids, parsed_texts

def normalize_whisper_turn_mode(whisper_turn_mode):
    """Normalize and validate the Whisper turn construction mode."""
    if whisper_turn_mode is None:
        return "auto"

    turn_mode = str(whisper_turn_mode).strip().lower()
    if turn_mode not in {"auto", "speaker", "segment"}:
        raise ValueError(
            "Invalid whisper_turn_mode. Please use 'auto', 'speaker', or 'segment'"
        )
    return turn_mode

def _build_whisper_turn_row(item, measures):
    """Convert a single Whisper segment into one turn row."""
    words = [w for w in item.get("words", []) if "start" in w]
    idxs = [w[measures["old_index"]] for w in words]
    if not idxs:
        return None

    text = (item.get("text") or "").strip()
    words_texts = [w.get("word", "") for w in words]
    phrase_ids, phrase_texts = _extract_phrase_payload(item, idxs, words_texts, text)
    return {
        measures["utterance_ids"]: (idxs[0], idxs[-1]),
        measures["utterance_text"]: text,
        measures["phrases_ids"]: phrase_ids,
        measures["phrases_texts"]: phrase_texts,
        measures["words_ids"]: idxs,
        measures["words_texts"]: words_texts,
        measures["speaker_label"]: item.get("speaker"),
    }

def create_turns_whisper(item_data, measures, whisper_turn_mode="auto"):
    """Convert Whisper segments into the turn-level dataframe schema."""
    data = []
    turn_mode = normalize_whisper_turn_mode(whisper_turn_mode)
    has_speaker = any(("speaker" in s and s["speaker"] is not None) for s in item_data)
    use_speaker_turns = has_speaker and turn_mode in {"auto", "speaker"}

    if not use_speaker_turns:
        # Segment-level turns, even when speaker labels are present.
        for item in item_data:
            row = _build_whisper_turn_row(item, measures)
            if row is not None:
                data.append(row)
        return pd.DataFrame(data)

    # Diarized turns: group consecutive segments by speaker.
    current_speaker = None
    aggregated_text = ""
    aggregated_ids = []
    word_ids, word_texts = [], []
    phrase_ids, phrase_texts = [], []

    for item in item_data:
        row = _build_whisper_turn_row(item, measures)
        if row is None:
            continue

        speaker = row[measures["speaker_label"]]
        utterance_text = row[measures["utterance_text"]]
        row_word_ids = row[measures["words_ids"]]
        row_word_texts = row[measures["words_texts"]]
        row_phrase_ids = row[measures["phrases_ids"]]
        row_phrase_texts = row[measures["phrases_texts"]]

        if speaker == current_speaker:
            if utterance_text:
                aggregated_text = f"{aggregated_text} {utterance_text}".strip() if aggregated_text else utterance_text
            aggregated_ids.extend(row_word_ids)
            word_ids.extend(row_word_ids)
            word_texts.extend(row_word_texts)
            phrase_ids.extend(row_phrase_ids)
            phrase_texts.extend(row_phrase_texts)
            continue

        if aggregated_ids:
            data.append({
                measures["utterance_ids"]: (aggregated_ids[0], aggregated_ids[-1]),
                measures["utterance_text"]: aggregated_text.strip(),
                measures["phrases_ids"]: phrase_ids,
                measures["phrases_texts"]: phrase_texts,
                measures["words_ids"]: word_ids,
                measures["words_texts"]: word_texts,
                measures["speaker_label"]: current_speaker,
            })

        current_speaker = speaker
        aggregated_text = utterance_text
        aggregated_ids = row_word_ids.copy()
        word_ids = row_word_ids.copy()
        word_texts = row_word_texts.copy()
        phrase_ids = row_phrase_ids.copy()
        phrase_texts = row_phrase_texts.copy()

    if aggregated_ids:
        data.append({
            measures["utterance_ids"]: (aggregated_ids[0], aggregated_ids[-1]),
            measures["utterance_text"]: aggregated_text.strip(),
            measures["phrases_ids"]: phrase_ids,
            measures["phrases_texts"]: phrase_texts,
            measures["words_ids"]: word_ids,
            measures["words_texts"]: word_texts,
            measures["speaker_label"]: current_speaker,
        })

    return pd.DataFrame(data)

def pause_calculation(filter_json, measures, time_index):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the pause duration between each item.

    Parameters:
    ...........
    filter_json: list
        JSON response object.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    filter_json: list
        The updated JSON response object.

    ------------------------------------------------------------------------------------------------------
    """
    for i, item in enumerate(filter_json):
        if i > 0:
            item[measures["pause"]] = float(item[time_index[0]]) - float(filter_json[i - 1][time_index[1]])
        
        else:
            item[measures["pause"]] = np.nan
    return filter_json

def filter_json_transcribe(item_data, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function filters the JSON response object to only include items with start and end time.

    Parameters:
    ...........
    item_data: dict
        JSON response object.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    filter_json: list
        The updated JSON response object.

    ------------------------------------------------------------------------------------------------------
    """
    item_data2 = []
    for item in item_data:
        try:
            if "speaker" in item:
                speaker = item["speaker"]
            else:
                speaker = ""
            words = item["words"]
            
            for j, w in enumerate(words):# update speaker labels
                words[j]["speaker"] = speaker
            
            item_data2 += words
        except Exception as e:
            logger.info(f"Failed to filter word: {e}")
    
    filter_json = [item for item in item_data2 if "start" in item and "end" in item]
    filter_json = pause_calculation(filter_json, measures, ['start', 'end'])

    return filter_json

def calculate_file_feature(json_data, model, speakers):
    """
    ------------------------------------------------------------------------------------------------------

    Calculate file features based on JSON data.

    Parameters:
    ...........
    json_conf: list
        JSON response object.
    model: str
        model name (vosk/aws/whisper)
    speakers: str
        speakers label

    Returns:
    ...........
    tuple: A tuple containing two values - the total file length and the percentage of time spent speaking.

    ------------------------------------------------------------------------------------------------------
    """
    
    if model == 'aws':
        segments = json_data.get('items', [])
        file_length = max(float(segment.get("end_time", "0")) for segment in segments)
        
        if speakers is None:
            return file_length/60, np.NaN

        speaking_time = sum(float(segment.get("end_time", "0") or "0") - float(segment.get("start_time", "0") or "0")
                           for segment in segments if segment.get("speaker_label", "") in speakers)
    else:
        segments = json_data.get('segments', [])
        file_length = max(segment.get('end', 0) for segment in segments)
        
        if speakers is None:
            return file_length/60, np.NaN
        speaking_time = sum(segment['end'] - segment['start'] for segment in segments if segment.get('speaker', '') in speakers)

    speaking_pct = (speaking_time / file_length) * 100
    return file_length/60, speaking_pct

def create_text_list(utterances_speaker, speaker_label, min_turn_length, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function creates a list of transcribed text at the word, turn, and full text levels,
        and a list of the indices of the first and last word in each turn.

    Parameters:
    ...........
    utterances_speaker: pandas dataframe
        A dataframe containing the turns extracted from the JSON object
         for the specified speaker.
    speaker_label: str
        Speaker label
    min_turn_length: int
        minimum words required in each turn
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    text_list: list
        List of transcribed text at the word, turn, and full text levels.
    turn_indices: list
        A list containing the indices of the first and last word in each turn.

    ------------------------------------------------------------------------------------------------------

    """

    full_text = " ".join(utterances_speaker[measures['utterance_text']].tolist())
    
    word_list = sum(utterances_speaker[measures['words_texts']].tolist(), [])

    valid_turns = utterances_speaker[utterances_speaker[measures['words_texts']].apply(len) >= min_turn_length]
    turn_list = valid_turns[measures['utterance_text']].tolist() if speaker_label is not None else []
    turn_indices = valid_turns[measures['utterance_ids']].tolist() if speaker_label is not None else []

    text_list = [word_list, turn_list, full_text]

    if speaker_label is not None and len(turn_indices) <= 0:
        raise ValueError(f"No utterances found for speaker {speaker_label} with minimum length {min_turn_length}")

    return text_list, turn_indices

def add_phrase_count_features(df_list, utterances_speaker, min_turn_length, measures):
    """
    Populate turn-level and summary-level phrase-count features.
    """
    word_df, turn_df, summ_df = df_list

    valid_turns = utterances_speaker[
        utterances_speaker[measures["words_texts"]].apply(len) >= min_turn_length
    ].reset_index(drop=True)

    phrase_counts = valid_turns[measures["phrases_texts"]].apply(
        lambda phrases: len(phrases) if isinstance(phrases, (list, tuple, np.ndarray)) else 0
    )
    one_word_phrase_counts = valid_turns[measures["phrases_texts"]].apply(
        lambda phrases: sum(
            1
            for phrase in phrases
            if isinstance(phrase, str) and len(phrase.split()) == 1
        )
        if isinstance(phrases, (list, tuple, np.ndarray))
        else 0
    )

    turn_df = turn_df.reset_index(drop=True)
    turn_df[measures["phrase_count"]] = phrase_counts.to_numpy(dtype=float)

    if len(phrase_counts) > 0:
        summ_df.loc[0, measures["num_phrases"]] = int(phrase_counts.sum())
        summ_df.loc[0, measures["num_one_word_phrases"]] = int(one_word_phrase_counts.sum())
        summ_df.loc[0, measures["phrase_count_mean"]] = float(phrase_counts.mean())

    return [word_df, turn_df, summ_df]

def extract_segment_texts_for_speaker(json_conf, speaker_label=None, source="whisper", language=None):
    """
    Extract raw segment texts for utterance-distribution features.
    """
    if not isinstance(json_conf, dict):
        return None

    if source == "whisper":
        segments = json_conf.get("segments")
    elif source == "aws":
        segments = json_conf.get("items")
    else:
        segments = json_conf.get("result") or json_conf.get("segments")

    if not isinstance(segments, list):
        return None

    def segment_role(segment):
        return str(
            segment.get("role")
            or segment.get("speaker")
            or segment.get("speaker_label")
            or segment.get("turn_role_decision")
            or ""
        ).strip()

    def sort_key(segment):
        start = segment.get("start")
        if start is None:
            start = segment.get("start_time")
        try:
            start_value = float(start)
        except (TypeError, ValueError):
            start_value = np.inf
        try:
            id_value = int(segment.get("id", 0))
        except (TypeError, ValueError):
            id_value = 0
        return start_value, id_value

    texts = []
    for segment in sorted(segments, key=sort_key):
        if speaker_label is not None and segment_role(segment).lower() != str(speaker_label).strip().lower():
            continue
        alternatives = segment.get("alternatives")
        alternative_text = ""
        if isinstance(alternatives, list) and alternatives:
            alternative_text = alternatives[0].get("content", "")
        if str(language or "").lower() in {"uk", "ua"}:
            text = segment.get("text") or segment.get("source_text_en") or alternative_text
        else:
            text = segment.get("source_text_en") or segment.get("text") or alternative_text
        text = str(text).strip()
        if text:
            texts.append(text)

    return texts

def filter_speaker(utterances, json_conf, speaker_label, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function filters the utterances based on the speaker label.

    Parameters:
    ...........
    utterances: pandas dataframe
        A dataframe containing the turns extracted from the JSON object.
    json_conf: list
        JSON response object.
    speaker_label: str
        Speaker label
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    utterances_speaker: pandas dataframe
        A dataframe containing the turns extracted from the JSON object for the specified speaker.
    json_conf_speaker: list
        JSON response object for the specified speaker.

    ------------------------------------------------------------------------------------------------------
    """
    utterances_speaker = utterances.copy()
    json_conf_speaker = json_conf.copy()
    if speaker_label is not None:
        utterances_speaker = utterances[utterances[measures['speaker_label']] == speaker_label]
        json_conf_speaker = [item for item in json_conf if item.get("speaker_label", "") == speaker_label or item.get("speaker", "") == speaker_label]

        if len(utterances_speaker) <= 0:
            raise ValueError(f"No utterances found for speaker {speaker_label}")

    return utterances_speaker, json_conf_speaker

def filter_length(utterances, utterances_speaker, speaker_label, min_turn_length, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function filters the utterances based on the minimum turn length.

    Parameters:
    ...........
    utterances: pandas dataframe
        A dataframe containing the turns extracted from the JSON object.
    utterances_speaker: pandas dataframe
        A dataframe containing the turns extracted from the JSON object for the specified speaker.
    speaker_label: str
        Speaker label
    min_turn_length: int
        minimum words required in each turn
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    utterances_filtered: pandas dataframe
        A dataframe containing the turns extracted from the JSON object after filtering based on the minimum turn length.
    utterances_speaker_filtered: pandas dataframe
        A dataframe containing the turns extracted from the JSON object for the specified speaker after filtering based on the minimum turn length.

    ------------------------------------------------------------------------------------------------------
    """
    # filter utterances with minimum length
    utterances_speaker_filtered = utterances_speaker[utterances_speaker[measures['words_texts']].apply(lambda x: len(x) >= min_turn_length)].reset_index(drop=True)
    # filter utterances with minimum length for speaker
    utterances_filtered = utterances.copy()
    if speaker_label is not None:
        utterances_filtered = utterances_filtered.iloc[0:0]
        for i in range(len(utterances)):
            if utterances.iloc[i][measures['speaker_label']] != speaker_label or len(utterances.iloc[i][measures['words_texts']]) >= min_turn_length:
                utterances_filtered = pd.concat([utterances_filtered, utterances.iloc[i:i+1]])

        utterances_filtered = utterances_filtered.reset_index(drop=True)

    return utterances_filtered, utterances_speaker_filtered

def process_language_feature(
    df_list,
    transcribe_info,
    speaker_label,
    min_turn_length,
    min_coherence_turn_length,
    language,
    time_index,
    option,
    measures,
    feature_groups=None,
    speaker_filter_label=None,
    coherence_speaker_label=None,
    raw_json_conf=None,
    source=None,
):
    """
    ------------------------------------------------------------------------------------------------------

    This function processes the language features from json response.

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    transcribe_info: list
        transcribed info
    min_turn_length: int
        minimum words required in each turn
    min_coherence_turn_length: int
        minimum words required in each turn for coherence analysis
    speaker_label: str
        Speaker label
    time_index: list
        timepoint index (start/end)
    language: str
        Language of the transcribed text.
    option: str
        Option for processing language features
         which to be processed
    feature_groups: list[str] | set[str] | None
        Optional feature group selector. When provided, only these groups are computed.
        Supported groups: "structure", "pause", "repetition", "coherence", "sentiment", "first_person".
    speaker_filter_label: str | None
        Speaker scope for strict filtering of utterances and word-level json payloads.
        When None, language features are computed over the whole interview.
    coherence_speaker_label: str | None
        Speaker scope for phrase coherence. When None, phrase coherence is
        computed over the full dialogue turn sequence.
    raw_json_conf: dict | None
        Original transcript payload, used for raw-segment distribution features.
    source: str | None
        Transcript source name.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_list: list
        List of pandas dataframes (word_df, turn_df and summ_df)

    ------------------------------------------------------------------------------------------------------
    """
    json_conf, utterances = transcribe_info

    if feature_groups is None:
        feature_groups = {"structure", "pause", "repetition", "coherence", "sentiment", "first_person"}
    else:
        if isinstance(feature_groups, str):
            feature_groups = {feature_groups}
        feature_groups = {str(f).strip().lower() for f in feature_groups if f}

    want_pause = "pause" in feature_groups
    want_structure = "structure" in feature_groups
    want_repetition = "repetition" in feature_groups
    want_coherence = "coherence" in feature_groups and option == "coherence"
    # first_person relies on sentiment scores (pos/neg)
    want_sentiment = "sentiment" in feature_groups or "first_person" in feature_groups
    want_first_person = "first_person" in feature_groups

    # filter speaker in json_conf and utterances
    utterances_speaker, json_conf_speaker = filter_speaker(
        utterances,
        json_conf,
        speaker_filter_label,
        measures,
    )
    # create text list and turn indices
    text_list, turn_indices = create_text_list(utterances_speaker, speaker_label, min_turn_length, measures)
    # filter utterances with minimum length
    utterances_dialogue_filtered, utterances_speaker_filtered = filter_length(
        utterances,
        utterances_speaker,
        speaker_filter_label,
        min_turn_length,
        measures,
    )

    if want_structure:
        df_list = add_phrase_count_features(df_list, utterances_speaker, min_turn_length, measures)
    if want_pause:
        df_list = get_pause_feature(json_conf_speaker, df_list, text_list, turn_indices, measures, time_index, language)
    if want_repetition:
        df_list = get_repetitions(df_list, utterances_speaker, utterances_speaker_filtered, measures)

    if want_coherence:
        df_list = get_word_coherence(df_list, utterances_speaker, min_coherence_turn_length, language, measures)
        df_list = get_phrase_coherence(
            df_list,
            utterances_speaker_filtered,
            min_coherence_turn_length,
            coherence_speaker_label,
            language,
            measures,
            dialogue_utterances_filtered=utterances_dialogue_filtered,
        )

    if language in measures["english_langs"] or language in ['uk', 'ua']:
        if want_sentiment:
            vader_distribution_texts = extract_segment_texts_for_speaker(
                raw_json_conf,
                speaker_filter_label,
                source,
                language,
            )
            df_list = get_sentiment(
                df_list,
                text_list,
                measures,
                lang=language,
                vader_distribution_texts=vader_distribution_texts,
            )
        if want_first_person:
            df_list = get_pos_tag(df_list, text_list, measures, lang=language)

    return df_list
