# E-DAIC Leakage, Re-Transcription, Cleanup, and Translation Report

## Scope

This document summarizes why transcript leakage is a critical problem for the E-DAIC style interview setup, how we audited that leakage in multiple transcript versions, how we rebuilt the English transcript pipeline with WhisperX, how we reduced leakage with a local Gemma 3 27B cleanup stage, and how we translated the cleaned transcripts into Ukrainian with TranslateGemma.

The leakage percentages reported here are **proxy estimates**, not exact ground-truth measurements. They come from the paper-facing audit notebook `leakage_audit.ipynb`, which uses:

- a **strict interviewer-like heuristic** as the primary estimate
- a **broad interviewer-like heuristic** as an upper-bound sensitivity check
- a **temporal overlap sanity check** against interviewer spans in the cleaned diarization output

All corpus-level numbers below are **corpus-weighted** over the 136 interviews shared by the audited transcript versions.

## Why leakage is a crucial problem in this dataset

Leakage means that text produced by the interviewer, by setup chatter, or by mixed turns is retained inside the transcript that is later treated as the participant's speech. This is a serious problem for E-DAIC style modeling for several reasons:

1. It contaminates the participant transcript with language that does not reflect the participant's mental state.
2. It injects systematic prompt-like wording into the participant side, for example questions, second-person prompts, closings, setup instructions, or operator chatter.
3. It can distort lexical, semantic, discourse, coherence, and style features by making the participant transcript look more interviewer-like than it really is.
4. It can leak interview protocol structure into downstream models, which is especially problematic when a dataset is used for depression prediction or related clinical modeling.
5. It undermines interpretability: if a model responds to question prompts, closings, or procedural speech, it is no longer clear whether the learned signal comes from participant behavior or from annotation/transcription artifacts.

In short, leakage is not a cosmetic transcript issue. It can change both the statistical properties of the corpus and the meaning of downstream results.

## Why the original participant transcript is especially hard to fix

For the "original" condition in this audit, we used the participant-only transcript version stored in `datasets/old/woz_end_whisper_test`. In that format, the JSON contains a flat list of segments and a top-level transcript text, but it does **not** preserve a parallel interviewer stream or a per-segment role label.

That creates a structural problem:

- the transcript is already reduced to the participant view
- leaked interviewer text can still appear inside that participant view
- but there is no aligned interviewer transcript in the same file to compare against
- therefore leakage cannot be corrected exactly at the released transcript level

This means the original participant-only transcript is not just noisy. It is **un-auditable at span level** once the interviewer stream has been discarded. If a segment says something interviewer-like, we cannot determine from that file alone whether it is:

- truly participant speech
- quoted interviewer speech inside participant narrative
- interviewer speech that leaked into the participant transcript
- room/setup chatter
- an ASR or segmentation artifact

That limitation is one of the main reasons we rebuilt the transcription pipeline from audio.

## Leakage audit method

We audited three English transcript versions:

1. `original_edaic_proxy`
   The participant-only transcript version in `datasets/old/woz_end_whisper_test`.
2. `raw_whisper_participant_speaker`
   Our WhisperX re-transcription, reduced to the diarized speaker inferred to be the participant.
3. `fixed_diarization_participant_only`
   The output after the Gemma cleanup stage, keeping only segments classified as participant.

### Primary metric: strict heuristic leakage

The main paper metric is the percentage of words flagged by a conservative interviewer-like heuristic. This heuristic looks for:

- direct question marks
- question prefixes such as `how`, `what`, `why`, `can you`, `tell me`
- short second-person prompts
- imperative/setup-style prompts
- stock interviewer phrases such as `I'd love to hear all about it`, `repeat that`, `goodbye`, `thank you for sharing your thoughts with me`

### Secondary metric: broad heuristic leakage

The broad metric is a more permissive version of the same idea. It is useful as an upper bound, but it produces more false positives. In particular, it can flag participant speech that starts with words like `when`, `what`, or `where`, and it can also overreact to discourse markers such as `you know`.

### Timing sanity check: overlap with interviewer spans

For transcript versions with timestamps, we also measure how much their active duration overlaps interviewer spans in the cleaned role-labeled output. This is not an independent source of truth, but it is a useful sanity check.

Important caveat:

- the overlap metric uses the cleaned role-labeled diarization as the reference
- therefore the overlap value for the final `participant_only` cleaned transcript should go to approximately zero by construction

For that reason, the **strict heuristic word percentage** is the main number to cite.

## Findings in the original participant-only transcript

The original participant-only transcript version already contains substantial interviewer-like language.

Examples of flagged original segments include:

- `I'd love to hear all about it`
- `repeat that`
- `spell Klaus`
- `how do your best friend describe`
- `thank you`

These examples matter because they show that the problem is not limited to small punctuation artifacts. The released participant transcript can contain:

- direct interviewer prompts
- boilerplate interview phrases
- imperative requests
- short setup-like instructions

Since the original format does not preserve a role stream, these leaked fragments cannot be corrected reliably from the released participant transcript alone.

## Why we re-transcribed from audio

Because the released participant transcript is structurally hard to audit and correct, we built a fresh transcription pipeline directly from the interview audio.

The transcription notebook is `whisper_pipeline.ipynb`. The pipeline does the following:

1. Loads the interview audio from folders of the form `<id>_P/<id>_AUDIO.wav`.
2. Uses WhisperX for ASR.
3. Forces English transcription.
4. Uses WhisperX alignment to obtain word-level timing.
5. Runs pyannote diarization through `DiarizationPipeline`.
6. Assigns a speaker label to each aligned word with `whisperx.assign_word_speakers`.
7. Saves one JSON file per interview with both `segments` and `word_segments`.

The key configuration choices in the notebook are:

- model: `large-v3` on GPU, `medium` on CPU fallback
- compute type: `float16` on GPU, `int8` on CPU
- batch size: `16` on GPU, `4` on CPU
- language forced to English
- diarization speaker bounds: `min_speakers=1`, `max_speakers=6`

This step gives us something the original participant-only transcript does not have:

- timestamped segments
- word-level timings
- speaker labels
- a recoverable full conversation stream before participant-only reduction

That is what makes a real cleanup pass possible.

## Leakage after WhisperX re-transcription

Even after retranscribing from audio, leakage is still present if we simply take the diarized speaker that appears to be the participant.

In the audit notebook, the participant speaker for raw WhisperX is inferred by:

1. taking the cleaned role-labeled output
2. mapping each cleaned segment back to its raw Whisper turn via `source_turn_idx`
3. counting which diarized raw speaker contributed the most participant words
4. treating that raw speaker as the participant speaker for that interview

This gives a fair participant-speaker baseline, but it is still imperfect. Raw diarization alone does not solve:

- mixed turns
- diarization errors
- adjacent interviewer and participant speech merged into one ASR turn
- short backchannels and closings
- setup chatter

Examples of flagged raw Whisper participant-speaker segments include direct interviewer-style questions:

- `Why did you move to LA?`
- `Are you still doing that?`
- `What's your dream job?`
- `Do you travel?`
- `Can you be a little bit more specific?`

So retranscription helps recover structure, but by itself it does not remove interviewer leakage.

## Gemma 3 27B cleanup pipeline

To correct the raw WhisperX transcript while preserving timings, we wrote a local cleanup pipeline using `google/gemma-3-27b-it` in `scripts/preprocessing/gemma_hybrid_role_cleanup.py`.

### Design goal

The cleanup stage is constrained to be **timing-preserving**. It is not allowed to invent text or rewrite the transcript. It can only:

- classify turns
- split ambiguous turns into contiguous word spans
- assign those spans to `participant`, `interviewer`, or `unknown`

This is a critical design choice. It keeps the cleaned transcript aligned with the original Whisper word timings.

### Two-pass cleanup procedure

The cleanup pipeline has two LLM passes.

#### Pass 1: turn-level role classification

For each raw Whisper turn, Gemma sees:

- the current turn text
- the previous turn
- the next turn
- timestamps

It must classify the turn as one of:

- `participant`
- `interviewer`
- `mixed`
- `unknown`

The prompt explicitly tells the model to:

- keep participant narrative even if it quotes interviewer speech
- mark room setup, hardware instructions, survey chatter, and operator talk as `unknown`
- prefer `participant` or `unknown` over an incorrect `interviewer` label when uncertain

#### Pass 2: word-span resolution for mixed turns

If a turn is classified as `mixed`, Gemma receives:

- the raw turn text
- local context
- an indexed list of the exact original words in the turn

It then returns contiguous word-index spans labeled as:

- `participant`
- `interviewer`
- `unknown`

Because the model outputs only index spans, the pipeline can rebuild the cleaned transcript from the original Whisper words without changing their timing or order.

### Output structure

The cleanup pipeline writes four JSON views:

- `role_labeled/<file>.json`
- `participant_only/<file>.json`
- `interviewer_only/<file>.json`
- `unknown_only/<file>.json`

Each cleaned segment preserves:

- text reconstructed from the original words
- segment start and end times
- the original words with timings
- `source_turn_idx`
- `source_word_start_idx`
- `source_word_end_idx`
- decision metadata such as `decision_source`, `turn_role_decision`, and `needs_review`

This makes the cleanup output auditable and suitable for downstream timing-aware analysis.

## Leakage after Gemma cleanup

After the Gemma cleanup stage, leakage drops sharply.

### Corpus-weighted leakage results

| Version | Strict leakage (words) | Broad leakage (words) | Strict leakage (duration) | Broad leakage (duration) | Overlap sanity check |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original participant-only transcript proxy | 9.743% | 15.586% | 5.011% | 9.242% | 9.222% |
| Raw WhisperX participant speaker | 10.737% | 12.107% | 9.587% | 10.857% | 9.032% |
| Gemma-cleaned participant only | 2.099% | 3.500% | 2.058% | 3.433% | 0.000% |

### Interpretation

The important pattern is stable:

- the original participant-only transcript already contains substantial interviewer-like content
- raw WhisperX plus participant-speaker selection still leaves substantial leakage
- the Gemma cleanup stage reduces leakage to a low residual level

For the paper, the best headline number is:

- **strict leakage by words = 2.099%** for the final cleaned participant-only transcript

The overlap metric is zero after cleanup because the final view explicitly removes interviewer spans. The stricter and more informative residual estimate is therefore the remaining heuristic text signal.

## Why the remaining cleaned leakage is likely mostly false positive

The residual cleaned percentage is small, and most of the flagged cleaned segments do not look like true interviewer carryover.

Among the 638 broad-flagged cleaned segments:

- 460 were triggered by question-like prefixes or question-mark patterns
- 112 were triggered by short second-person patterns
- only 69 were triggered by hard interviewer phrases or imperative prompt cues

So approximately 89.7% of the remaining flags are question-like or short second-person heuristic triggers, which are exactly the categories most likely to overfire on participant narrative.

### Representative likely false positives

These are examples of cleaned participant segments that the heuristic flags even though they plausibly remain genuine participant speech:

1. `when i was younger my dream job was to be an ambassador ...`
   Flagged because it starts with `when`, not because it is clearly interviewer speech.

2. `what i'm most proud of ... one of the things i'm most proud of has been a dad ...`
   Flagged because it starts with `what`, but it is plainly participant self-report.

3. `When somebody talks to you ...`
   Flagged by the second-person heuristic, but the segment is part of participant narrative rather than a prompt to the interlocutor.

4. `... sometimes I wake up and I have to ask my mom, like, did we go here?`
   Flagged because it contains a question mark, yet the question is quoted inside participant narrative.

5. `You know ... is that a good getaway spot still right now?`
   Flagged as question-like, but it is better interpreted as a self-directed rhetorical continuation than as interviewer text.

### Borderline residual cases

A smaller subset does look like real leftover contamination or mixed-turn concatenation, for example segments containing:

- `tell me more about that`
- `thank you goodbye have a good day`
- `virtual human`

So the correct interpretation is not that cleanup is perfect. It is that:

- residual leakage is **low**
- the majority of the remaining heuristic flags are likely false positives
- the remaining true errors are relatively sparse and often tied to mixed-turn or noisy ASR conditions

## Ukrainian translation pipeline with TranslateGemma

After English cleanup, we translate the cleaned transcript into Ukrainian with `google/translategemma-27b-it` using the local `transformers` pipeline in `scripts/preprocessing/gemma_translate_uk.py`.

### Input and output design

The translation pipeline expects cleaned English JSON input, typically the role-labeled output of the cleanup stage.

It writes four translated views:

- `role_labeled/<file>.json`
- `participant_only/<file>.json`
- `interviewer_only/<file>.json`
- `unknown_only/<file>.json`

Each translated segment preserves:

- the original segment start time
- the original segment end time
- the original role
- the original English text in `source_text_en`

### How translation is performed

The translation prompt is designed to translate the segment faithfully while using limited context only when necessary.

For each segment:

- `participant` segments receive the **previous interviewer turn** as `context_en` when available
- `interviewer` and `unknown` segments are translated without that participant-specific context by default
- `participant_gender` is optionally supplied from metadata and is used only when first-person Ukrainian grammar requires a gendered form

The prompt explicitly instructs the model to:

- translate only `source_text_en`
- use `context_en` only for disambiguation
- preserve conversational tone, uncertainty, and fragmentation
- use formal Ukrainian address consistently
- avoid informal `ти` forms
- avoid placeholder slash variants and parenthetical gender variants
- avoid hallucinating clarity when the English source is noisy

The script also runs lightweight post-generation QA checks for issues such as:

- prompt leakage
- unchanged English text
- informal address
- excessive Latin script
- suspicious length expansion

Segments that trigger those QA checks are marked with `translation_needs_review`.

## Limitations of the translation stage

The translation stage is useful, but it has important structural limitations.

### 1. Word timings cannot be preserved

The translated JSON intentionally writes:

- translated segment text
- preserved segment-level `start` and `end`
- **empty `word_segments`**

This is the correct choice. Once the English transcript is translated into Ukrainian, the original English word-level alignment no longer maps one-to-one onto the translated words.

Therefore we do **not** fabricate translated word timings.

### 2. Whisper-style downstream paths may not work on translated JSON

Any downstream component that expects Whisper-like `segments[].words` with aligned word timings cannot directly treat the translated JSON as a native Whisper output.

The translated files preserve:

- segment timing
- role labels
- source text traceability

but they do **not** preserve:

- translated word timings
- translated diarized word speaker assignments

### 3. Translation can smooth or reshape cues

Even with a faithful prompt, translation can alter:

- sentence length
- punctuation
- hesitation markers
- idiomatic phrasing
- local lexical cues

That means the translated corpus should be treated as a derived analysis layer, not as a timing-faithful replacement for the English ASR.

### 4. Some ambiguity is irreducible

If the English source segment is noisy, incomplete, or already mixed, the translation can only preserve that ambiguity, not eliminate it. The translation layer cannot repair upstream ASR or diarization errors that survive cleanup.

## Practical conclusion

The main conclusions supported by this workflow are:

1. The participant-only E-DAIC transcript condition already contains meaningful interviewer leakage.
2. Because that original format discards the aligned interviewer stream, leakage cannot be precisely audited or corrected there.
3. Re-transcribing from audio with WhisperX is necessary because it restores timestamps, words, and speaker structure.
4. Raw diarization alone is not enough; interviewer-like leakage remains substantial after simple participant-speaker selection.
5. The Gemma 3 27B cleanup stage substantially reduces leakage while preserving original word timings and transcript traceability.
6. The residual leakage in the cleaned participant-only view is low, and most remaining flags are more consistent with heuristic false positives than with clear interviewer carryover.
7. TranslateGemma provides a practical Ukrainian translation layer over the cleaned English transcript, but translated word-level timing cannot be preserved and should not be fabricated.

## Artifacts used for this report

- `leakage_audit.ipynb`
- `whisper_pipeline.ipynb`
- `scripts/preprocessing/gemma_hybrid_role_cleanup.py`
- `scripts/preprocessing/gemma_translate_uk.py`
- `tmp/leakage_audit_paper_2026-03-26/paper_table.csv`
- `tmp/leakage_audit_paper_2026-03-26/fixed_examples.csv`

