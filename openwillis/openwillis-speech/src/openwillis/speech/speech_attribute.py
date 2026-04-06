# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages
import os
import json
import logging

import numpy as np
import pandas as pd
from .util import characteristics_util as cutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def get_config(filepath, json_file):
    """
    ------------------------------------------------------------------------------------------------------

    This function reads the configuration file containing the column names for the output dataframes,
    and returns the contents of the file as a dictionary.

    Parameters:
    ...........
    filepath : str
        The path to the configuration file.
    json_file : str
        The name of the configuration file.

    Returns:
    ...........
    measures: A dictionary containing the names of the columns in the output dataframes.

    ------------------------------------------------------------------------------------------------------
    """
    dir_name = os.path.dirname(filepath)
    measure_path = os.path.abspath(os.path.join(dir_name, f"config/{json_file}"))

    file = open(measure_path)
    measures = json.load(file)
    return measures


def is_amazon_transcribe(json_conf):
    """
    ------------------------------------------------------------------------------------------------------
    This function checks if the json response object is from Amazon Transcribe.
    Parameters:
    ...........
    json_conf: dict
        JSON response object.
    Returns:
    ...........
    bool: True if the json response object
     is from Amazon Transcribe, False otherwise.
    ------------------------------------------------------------------------------------------------------
    """
    return "jobName" in json_conf and "results" in json_conf


def is_whisper_transcribe(json_conf):
    """
    ------------------------------------------------------------------------------------------------------

    This function checks if the json response object is from Whisper Transcribe.

    Parameters:
    ...........
    json_conf: dict
        JSON response object.

    Returns:
    ...........
    bool: True if the json response object
     is from Whisper Transcribe, False otherwise.

    ------------------------------------------------------------------------------------------------------
    """
    if "segments" in json_conf:
        if len(json_conf["segments"])>0:

            if "words" in json_conf["segments"][0]:
                return True
    return False

def filter_transcribe(json_conf, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function extracts the text and filters the JSON data for Amazon Transcribe json response objects.
    Also, it filters the JSON data based on the speaker label if provided.

    Parameters:
    ...........
    json_conf: dict
        aws transcribe json response.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    filter_json: list
        The filtered JSON object containing
        only the relevant data for processing.
    utterances: pd.DataFrame
        The utterances extracted from the JSON object.

    ------------------------------------------------------------------------------------------------------
    """
    item_data = json_conf["results"]["items"]
    
    for i, item in enumerate(item_data): # create_index_column
        item[measures["old_index"]] = i

    utterances = cutil.create_turns_aws(item_data, measures)

    filter_json = cutil.filter_json_transcribe_aws(item_data, measures)

    return filter_json, utterances

def filter_whisper(json_conf, measures, whisper_turn_mode="auto"):
    """
    ------------------------------------------------------------------------------------------------------

    This function extracts the text and filters the JSON data
        for Whisper Transcribe json response objects.
        Also, it filters the JSON data based on the speaker label if provided.

    Parameters:
    ...........
    json_conf: dict
        whisper transcribe json response.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.
    whisper_turn_mode: str
        Whisper turn construction mode:
        "auto" keeps the current behavior, "speaker" groups consecutive segments
        by speaker, and "segment" keeps one segment per turn.

    Returns:
    ...........
    filter_json: list
        The filtered JSON object containing
        only the relevant data for processing.
    utterances: pd.DataFrame
        The utterances extracted from the JSON object.
        
    Raises:
    ...........
    ValueError: If the speaker label is not found in the json response object.

    ------------------------------------------------------------------------------------------------------
    """
    item_data = json_conf["segments"]

    item_data = cutil.create_index_column(item_data, measures)
    utterances = cutil.create_turns_whisper(
        item_data,
        measures,
        whisper_turn_mode=whisper_turn_mode,
    )
    
    filter_json = cutil.filter_json_transcribe(item_data, measures)

    return filter_json, utterances

def filter_vosk(json_conf, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function extracts the text for json_conf objects
     from sources other than Amazon Transcribe.

    Parameters:
    ...........
    json_conf: dict
        The input text in the form of a JSON object.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    json_conf: list
        The filtered JSON object containing
        only the relevant data for processing.
    utterances: pd.DataFrame
        The utterances extracted from the JSON object.

    ------------------------------------------------------------------------------------------------------
    """
    # make a dictionary to map old indices to new indices
    words = []
    words_ids = []
    for i, item in enumerate(json_conf):
        item[measures["old_index"]] = i

        if "word" in item:
            words.append(item["word"])
            words_ids.append(i)

    text = " ".join(words)

    utterances = pd.DataFrame({
        measures["utterance_ids"]: [(0, len(json_conf) - 1)],
        measures["utterance_text"]: [text],
        measures['words_ids']: [words_ids],
        measures['words_texts']: [words],
        measures['phrases_ids']: [[]],
        measures['phrases_texts']: [[]],
        measures['speaker_label']: [""],

    })

    return json_conf, utterances

def common_summary_feature(df_summ, json_data, model, speaker_label):
    """
    ------------------------------------------------------------------------------------------------------

    Calculate file features based on JSON data.

    Parameters:
    ...........
    json_conf: list
        JSON response object.
    summ_df: pandas dataframe
        A dataframe containing summary information on the speech
    model: str
        model name
    speaker_label: str
        Speaker label

    Returns:
    ...........
    summ_df: pandas dataframe
        A dataframe containing summary information on the speech

    ------------------------------------------------------------------------------------------------------
    """
    try:
        if model == 'vosk':
            if len(json_data) > 0 and 'end' in json_data[-1]:

                last_dict = json_data[-1]
                df_summ['file_length'] = [last_dict['end']]

        else:
            if model == 'aws':
                json_data = json_data["results"]
                fl_length, spk_pct = cutil.calculate_file_feature(json_data, model, speaker_label)

            else:
                fl_length, spk_pct = cutil.calculate_file_feature(json_data, model, speaker_label)
            
            df_summ['file_length'] = [fl_length]
            df_summ['speaker_percentage'] = [spk_pct]# if speaker_label is not None else df_summ['speaker_percentage']
            
    except Exception as e:
        logger.info("Error in file length calculation")
    return df_summ

def process_transcript(
    df_list,
    json_conf,
    measures,
    min_turn_length,
    min_coherence_turn_length,
    speaker_label,
    source,
    language,
    option,
    feature_groups=None,
    whisper_turn_mode="auto",
):
    """
    ------------------------------------------------------------------------------------------------------
    
    Process transcript
    
    Parameters:
    ...........
    df_list: list, :
        contains pandas dataframe
    json_conf: dict
        Transcribed json file
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.
    min_turn_length: int
        minimum words required in each turn
    min_coherence_turn_length: int
        minimum words required in each turn for coherence calculation
    speaker_label: str
        Speaker label
    source: str
        model name
    language: str
        Language type
    option: str
        option for which measures to calculate
            can be 'simple' or 'coherence'
    feature_groups: list[str] | set[str] | None
        Optional feature group selector. When provided, only these groups are computed.
        Supported groups: "pause", "repetition", "coherence", "sentiment", "first_person".
    whisper_turn_mode: str
        Whisper turn construction mode:
        "auto" keeps the current behavior, "speaker" groups consecutive segments
        by speaker, and "segment" keeps one segment per turn.
    
    Returns:
    ...........
    df_list: list
        contains pandas dataframe
    
    ------------------------------------------------------------------------------------------------------
    """
    common_summary_feature(df_list[2], json_conf, source, speaker_label)

    if source == 'whisper':
        info = filter_whisper(json_conf, measures, whisper_turn_mode=whisper_turn_mode)

    elif source == 'aws':
        info = filter_transcribe(json_conf, measures)

    else:
        info = filter_vosk(json_conf, measures)

    if len(info[0]) > 0 and len(info[1]) > 0:
        df_list = cutil.process_language_feature(
            df_list,
            info,
            speaker_label,
            min_turn_length,
            min_coherence_turn_length,
            language,
            get_time_columns(source),
            option,
            measures,
            feature_groups=feature_groups,
            speaker_filter_label=speaker_label,
            coherence_speaker_label=speaker_label,
        )
    return df_list

def get_time_columns(source):
    """
    ------------------------------------------------------------------------------------------------------
    
    get time columns
    
    Parameters:
    ...........
    source: str
        model name
    
    Returns:
    ...........
    object: list
        time index name
        
    ------------------------------------------------------------------------------------------------------
    """
    if source == 'aws':
        return ["start_time", "end_time"]
    else:
        return ["start", "end"]

def speech_characteristics(
    json_conf,
    language="en",
    speaker_label=None,
    min_turn_length=1,
    min_coherence_turn_length=5,
    option='coherence',
    feature_groups=None,
    whisper_turn_mode="auto",
):
    """
    ------------------------------------------------------------------------------------------------------

    Speech Characteristics

    Parameters:
    ...........
    json_conf: dict
        Transcribed json file
    language: str
        Language type
    speaker_label: str
        Speaker label
    min_turn_length: int
        minimum words required in each turn
    min_coherence_turn_length: int
        minimum words required in each turn for coherence calculation
    option: str
        option for which measures to calculate
         can be 'simple' or 'coherence'
    feature_groups: list[str] | set[str] | None
        Optional feature group selector. When provided, only these groups are computed.
        Supported groups: "pause", "repetition", "coherence", "sentiment", "first_person".
    whisper_turn_mode: str
        Whisper turn construction mode.
        Supported values:
        "auto":
            Default Whisper behavior. If diarization labels are present, consecutive
            segments are merged into speaker turns. Downstream language/coherence
            features continue to respect `speaker_label` when it is provided. If
            diarization labels are absent, this falls back to one-segment-per-turn
            behavior.
        "speaker":
            Force diarized speaker turns by merging consecutive Whisper segments with
            the same speaker label. Downstream language/coherence features continue
            to respect `speaker_label`, while `speaker_percentage` still reflects
            the requested `speaker_label`.
        "segment":
            Keep each Whisper segment as its own turn. Downstream language/coherence
            features continue to respect `speaker_label` when it is provided, so
            segment and speaker modes differ only in turn construction.

    Returns:
    ...........
    df_list: list, contains:
        word_df: pandas dataframe
            A dataframe containing word summary information
        turn_df: pandas dataframe
            A dataframe containing turn summary information
        summ_df: pandas dataframe
            A dataframe containing summary information on the speech

    ------------------------------------------------------------------------------------------------------
    """
    try:
        # Load configuration measures
        measures = get_config(os.path.abspath(__file__), "text.json")
        whisper_turn_mode = cutil.normalize_whisper_turn_mode(whisper_turn_mode)
        df_list = cutil.create_empty_dataframes(measures)

        if option  not in ['simple', 'coherence']:
            raise ValueError("Invalid option. Please use 'simple' or 'coherence'")

        if bool(json_conf):
            language = language[:2].lower() if (language and len(language) >= 2) else "na"

            if language in measures["english_langs"]:
                cutil.download_nltk_resources()
            
            if language == "ua" or language == "uk":
                cutil.download_ua_resources()

            if is_whisper_transcribe(json_conf):
                df_list = process_transcript(
                    df_list,
                    json_conf,
                    measures,
                    min_turn_length,
                    min_coherence_turn_length,
                    speaker_label,
                    'whisper',
                    language,
                    option,
                    feature_groups=feature_groups,
                    whisper_turn_mode=whisper_turn_mode,
                )

            elif is_amazon_transcribe(json_conf):
                df_list = process_transcript(
                    df_list,
                    json_conf,
                    measures,
                    min_turn_length,
                    min_coherence_turn_length,
                    speaker_label,
                    'aws',
                    language,
                    option,
                    feature_groups=feature_groups,
                    whisper_turn_mode=whisper_turn_mode,
                )

            else:
                df_list = process_transcript(
                    df_list,
                    json_conf,
                    measures,
                    min_turn_length,
                    min_coherence_turn_length,
                    speaker_label,
                    'vosk',
                    language,
                    option,
                    feature_groups=feature_groups,
                    whisper_turn_mode=whisper_turn_mode,
                )

    except Exception as e:
        logger.info(f"Error in Speech Characteristics {e}")

    finally:
        for df in df_list:
            df.loc[0] = np.nan if df.empty else df.loc[0]

    return df_list
