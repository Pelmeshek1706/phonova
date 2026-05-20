# Transcript Preprocessing Protocol

This protocol records the paper-facing Track B transcript derivation path. The scripts are reusable, but the E-DAIC audio, raw transcripts, cleaned transcripts, translated transcripts, and derived participant-level feature tables are restricted and are not redistributed.

## 1. ASR and Diarization

Audio is re-transcribed with WhisperX using `paper/scripts/asr_whisperx.py`.

- Force English transcription with `--language en`.
- Preserve WhisperX segment start/end times and word-level alignment.
- Assign speaker labels with pyannote diarization when `HF_TOKEN` is provided.
- Write one WhisperX JSON file per participant/interview plus a manifest.

## 2. Gemma Leakage Cleanup

Role cleanup is performed with `paper/scripts/gemma_role_cleanup.py` using the local instruction-tuned checkpoint `google/gemma-3-27b-it`.

The cleanup is timing-preserving:

- The model is not allowed to rewrite, translate, normalize, summarize, or correct transcript text.
- Pass 1 labels each raw WhisperX turn as `participant`, `interviewer`, `mixed`, or `unknown`.
- Pass 2 reprocesses only `mixed` turns as indexed original WhisperX words.
- Pass 2 returns contiguous spans with roles `participant`, `interviewer`, or `unknown`.
- Participant-only, interviewer-only, unknown-only, and full role-labeled views are rebuilt from the original WhisperX words.
- Each rebuilt segment retains source turn and source word-span provenance.

Interviewer-like material includes prompts, follow-ups, acknowledgments, backchannels, closings, setup chatter, and instruction-like turns. Ambiguous material is retained as `unknown` rather than forced into participant speech.

## 3. TranslateGemma Ukrainian Rendering

Ukrainian rendering is performed with `paper/scripts/translate_gemma_uk.py` using `google/translategemma-27b-it`.

- Translation operates at cleaned segment level.
- Each segment is translated independently.
- Participant segments may receive only the immediately preceding interviewer turn as optional context.
- The translated JSON preserves segment start/end times, role labels, source English text, and cleanup provenance.
- Translated `word_segments` are intentionally empty because English-to-Ukrainian word timing alignment would be artificial.

## 4. Ukrainian Compatibility Timing

For downstream feature extraction only, `paper/scripts/make_ukrainian_compat_timing.py` builds a separate compatibility view.

- Preserved segment duration is uniformly partitioned across Ukrainian whitespace tokens.
- The generated words are marked `uniform_proxy_from_preserved_segment_duration`.
- These timings are proxy-only and are not native Ukrainian acoustic word alignments.
- Only five Ukrainian timing variables depend on this compatibility layer:
  - `speech_length_words`
  - `words_per_min`
  - `syllables_per_min`
  - `mean_pre_word_pause`
  - `mean_pause_variability`

All other Ukrainian text features should be interpreted as translated-text features using preserved segment structure, not as newly aligned Ukrainian audio features.
