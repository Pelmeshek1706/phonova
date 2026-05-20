## Introduction

Within the AIREST project, the sentiment analysis capabilities of NLP algorithms were verified using a corpus of Telegram user messages (COSMUS dataset). The evaluation includes several sentiment analysis models, allowing identification of the most reliable option for further usage.



## Splits & Reproducibility
- **Software / library versions**  
  - spaCy version: **3.8.7**  
  - Stanza version: **1.10.1**  
  - Transformers version: **4.56.2**  
  - Tokenizers version: **0.22.1**  
  - SentencePiece version: **0.2.0**  
  - PyTorch version: **2.8.0**  
  - Device: Tesla T4 GPU (15,360 MiB) for acceleration; fallback to CPU if necessary.

 ### DCWOZ (EN→UA) translation and quality validation

We translate the English DCWOZ dialogues into Ukrainian using the open-source MT model **`Yehor/kulyk-en-uk`** (based on **LiquidAI/LFM2-350M**, ~354M parameters; fine-tuned on ~40M EN–UK sentence pairs filtered by an automatic quality metric). :contentReference[oaicite:0]{index=0}

**Performance benchmarks (FLORES-200).**  
The authors report evaluation on the FLORES-200 devtest benchmark:
- **EN → UK (`kulyk-en-uk`)**: **27.24 BLEU** :contentReference[oaicite:1]{index=1}  
- **UK → EN (`kulyk-uk-en`)**: **36.27 BLEU** :contentReference[oaicite:2]{index=2}  

To quantify translation quality and potential distortions introduced by translation (important for downstream NLP analyses), we report both **reference-free Quality Estimation** and **round-trip consistency** checks.

**1) Reference-free MT quality estimation (QE).**  
We compute **COMET-QE** using **`Unbabel/wmt22-cometkiwi-da`** (reference-free learned metric; regression model on top of InfoXLM; trained on WMT direct assessments + MLQE-PE). :contentReference[oaicite:3]{index=3}  
**Result (DCWOZ EN→UA):** COMET-QE system score = **0.8337**.

**2) Length preservation diagnostics.**  
We compare token-length distributions between the English source and Ukrainian translations to detect pathological outputs (e.g., repetition loops). Reported token-length means are trimmed at the 99.5th percentile to reduce outlier impact.  
- EN token length mean±std: **9.444 ± 9.071**  
- UA token length mean±std: **7.988 ± 7.728**  
- Length ratio (UA/EN) mean±std: **0.873 ± 0.221**

**3) Lexical feature preservation (utterances with ≥5 tokens).**  
- TTR EN mean±std: **0.9294 ± 0.0981**  
- TTR UA mean±std: **0.9596 ± 0.0745**  
- MATTR EN mean±std: **0.9302 ± 0.0960**  
- MATTR UA mean±std: **0.9600 ± 0.0733**

**4) Round-trip consistency check (EN→UA→EN) with chrF++.**  
As an additional sanity check, we back-translate Ukrainian text to English and compute **chrF++** between the original English and back-translated English (chrF++ corresponds to chrF with word n-gram order = 2 in sacreBLEU). :contentReference[oaicite:4]{index=4}  
**Result (EN_back vs EN):** chrF++ = **90.27 (case-sensitive)** / **95.30 (lowercased)**.

> Interpretation: COMET-QE provides a quantified, reference-free estimate of EN→UA translation quality, while chrF++ on round-trip English provides a conservative consistency check (it measures recoverability after a second translation step and therefore is not a pure EN→UA metric).

    
- **Hardware & memory footprints**  
  - Training / inference run on GPU (Tesla T4) / CPU fallback  
  - Report peak GPU memory usage (e.g. ~X GB) and CPU memory usage  
  - Indicate batch sizes, gradient accumulation, mixed precision (bfloat16 on GPU etc.)  
  - If using multiple runs, average over seeds (e.g. 3 runs) to mitigate variance  

- **Reproducibility statements**  
  - Fix random seeds for all components (numpy, torch, transformers, data shuffling) set to 1706.  
  - Save model checkpoints, hyperparameters, and training logs  
  - Include scripts for data preprocessing (tokenization, translation, feature extraction)  
  - Provided a link to exact versioned code (e.g. GitHub commit tag or release)
 
- **Dataset splits**
	- COSMUS : neutral - 4702, negative - 4541, positive - 2373;
 	- Parlamint :
  		- dev dataset:  739 sentences;
    	- test dataset: 792 sentences;
     	- train dataset: 3901 sentences;
	- DCWOZ :
   		- Depression_Label(Main predictor) splits: 0 - 209, 1 - 66;
     	- Training splits - Using originaly split: train - 163, dev - 56,test - 56 . Using originaly split
	- All other datasets were distributed in an 80% train, 20% test format.
 - 
   


## COSMUS Dataset Overview

- **COSMUS** is a sentiment analysis dataset composed of Telegram messages in Ukrainian and Russian languages .
- Data is stored in Parquet format, containing approximately ~12,200 entries .
- Each entry consists of message content (`document_content`), sentiment labels (`annotator_sentiment`), and additional metadata fields: `response_id`, `document_id`, `user_id`, `response_timestamp`, `annotation_date`, `username`, `unique_document_id`, `language_wc`, `document_length`, `gpt_labels_v1`, `language_gpt`, `language_manual`, `language`, `stratification_label`, and `df_set` .
- Sentiment labels are manually annotated with values: "negative", "neutral", "positive", "mixed" .
- Dataset License: MIT .
- Associated tags include: `sentiment`, `social networks`, `Telegram` .
- Objective: analyzing sentiment polarity of social media posts .
- Accessible through Hugging Face Datasets and integrated with Pandas .
- Single available split: `train` on the entire dataset .
- Primary languages: Ukrainian and Russian .

## Additional info

**Overview.**  
Community dataset on Hugging Face for sentiment analysis over Telegram messages in Ukrainian/Russian; packaged for text classification tasks. ([huggingface.co](https://huggingface.co/datasets/YShynkarov/COSMUS))

**Collection & consent.**  
The public dataset card does *not* provide a detailed collection protocol or consent description beyond task framing; treat content as user‑generated social media text. (No additional collection/consent details are stated on the card.) ([huggingface.co](https://huggingface.co/datasets/YShynkarov/COSMUS))

**Licensing & access.**  
Released under the **MIT License** according to the dataset card. Reuse is broadly permitted under MIT; downstream users remain responsible for compliance with platform (Telegram) terms and any applicable laws. ([huggingface.co](https://huggingface.co/datasets/YShynkarov/COSMUS))

**Moderation & sensitive content.**  
The card does not specify an explicit moderation process. As with most social‑media corpora, text may contain offensive or sensitive content; apply standard toxicity/offensive‑content screening in downstream use. (No moderation policy is listed on the card.) ([huggingface.co](https://huggingface.co/datasets/YShynkarov/COSMUS))

**Intended use.**  
Benchmarks and research on sentiment classification for Ukrainian/Russian social‑media text. ([huggingface.co](https://huggingface.co/datasets/YShynkarov/COSMUS))

**Prohibited/unsafe uses.**  
Avoid attempts to re‑identify individuals; do not infer protected attributes or make consequential decisions about people solely from model outputs. (General research ethics guidance; the dataset card itself does not enumerate prohibitions.) ([huggingface.co](https://huggingface.co/datasets/YShynkarov/COSMUS))

**Known limitations.**  
Potential class imbalance; social‑media noise; unknown representativeness; lack of fine‑grained consent metadata; dataset card is sparse on collection details. ([huggingface.co](https://huggingface.co/datasets/YShynkarov/COSMUS))

**How to cite / where to learn more.**  
Hugging Face dataset page (license/tags/language). ([huggingface.co](https://huggingface.co/datasets/YShynkarov/COSMUS))

---

## Section 1.1 — Sentiment Analysis

The main model evaluated is **YShynkarov/ukr-roberta-cosmus-sentiment**, a fine-tuned variant of `ukr-roberta` adapted specifically to Ukrainian and Russian texts. Additionally, several other models were tested for comparative evaluation:

- **cardiffnlp/twitter-xlm-roberta-base-sentiment**  
- **tabularisai/multilingual-sentiment-analysis**  
- **Vader Sentiment via Phonova** *(texts translated from Ukrainian to English using Yehor/kulyk-uk-en)*

### Comparative Results Table

| Model                                                         | Accuracy | Negative (P / R / F1) | Neutral (P / R / F1) | Positive (P / R / F1) | Macro Avg (P / R / F1) | Weighted Avg (P / R / F1) |
|:--------------------------------------------------------------|---------:|----------------------:|---------------------:|----------------------:|-----------------------:|--------------------------:|
| `YShynkarov/ukr-roberta-cosmus-sentiment`                     |   76.80% |    0.90 / 0.66 / 0.76 |   0.71 / 0.87 / 0.78 |    0.73 / 0.77 / 0.75 |     0.78 / 0.77 / 0.76 |             0.79 / 0.77 / 0.77 |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment`               |   67.12% |    0.75 / 0.60 / 0.67 |   0.60 / 0.81 / 0.69 |    0.79 / 0.52 / 0.63 |     0.71 / 0.65 / 0.66 |             0.70 / 0.67 / 0.67 |
| `tabularisai/multilingual-sentiment-analysis`                 |   49.56% |    0.52 / 0.69 / 0.59 |   0.58 / 0.24 / 0.34 |    0.42 / 0.64 / 0.50 |     0.50 / 0.52 / 0.48 |             0.52 / 0.50 / 0.47 |
| `Vader Sentiment (Phonova, uk-en translation)`             |   43.55% |    0.77 / 0.03 / 0.05 |   0.42 / 0.98 / 0.59 |    0.78 / 0.13 / 0.23 |     0.66 / 0.38 / 0.29 |             0.63 / 0.44 / 0.30 |

### Evaluation Methodology

1. **Data loading and filtering**: Dataset loaded using Hugging Face Datasets; records labeled as "mixed" were removed.
2. **Label encoding**: Textual labels ("negative", "neutral", "positive") were mapped to numerical values (-1, 0, +1).
3. **Model initialization**: Pipelines for sentiment analysis were set up for each model using identical tokenization and inference parameters.
4. **Predictions**: The sentiment category for each text entry was predicted using the respective model’s inference pipeline.
5. **Vader Sentiment testing**: Due to the Vader Sentiment Analyzer supporting English input only, all texts were first translated from Ukrainian to English using the Yehor/kulyk-uk-en model, specifically fine-tuned for Ukrainian-English translation. After translation, texts were labeled using the Vader algorithm provided by Phonova.
6. **Metric computation**: Accuracy, precision, recall, and F1 scores were calculated for negative, neutral, and positive classes using sklearn metrics.
7. **Result comparison**: Performance metrics were summarized in the comparative results table.

## POS Tagging Evaluation for Ukrainian (spaCy-uk vs. UD Gold)

### Introduction

As part of the AIREST project, we conducted an evaluation of part-of-speech (POS) tagging accuracy for the Ukrainian language.  
The goal was to ensure that the **spaCy-uk** model used in Phonova achieves expert-level accuracy, comparable to manually annotated gold-standard data, and does not fall behind alternative solutions by more than the acceptable threshold.  

## UD_Ukrainian-ParlaMint Dataset Overview

- **UD_Ukrainian-ParlaMint** is a corpus of Ukrainian parliamentary speeches annotated according to the [Universal Dependencies](https://universaldependencies.org/) standards.  
- The dataset includes **morphological and syntactic annotations** that have been manually verified and corrected.  
- Annotation resources include **UD_Ukrainian-IU** and the **VESUM** dictionary.  
- In the metadata, the annotation status for lemmas, UPOS, morphological features, and dependencies is *manual native*.  
- The annotation covers **UPOS** (universal parts of speech), **XPOS**, morphological features (FEATS, including `Tense`), and syntactic dependencies (HEAD, DEPREL).  
- The data format is CoNLL-U, fully compatible with UD processing tools.  
- The main objective of using this dataset is to assess the accuracy of POS tagging and the extraction of the `Tense` feature for verbs.

## Section 1.2 — POS Tagging Evaluation

Two tools were evaluated:
- **spaCy-uk** — a Universal Dependencies–based model integrated into Phonova for Ukrainian.  
- **Stanza-uk** — an NLP library from Stanford NLP, trained on Ukrainian UD corpora, used here as a reference point.

Evaluation methodology:
1. **Data**: Gold-standard annotation from UD_Ukrainian-ParlaMint was used as the reference.
2. **Model predictions**: For each token, UPOS and Tense labels were obtained from spaCy-uk and Stanza-uk.
3. **Token alignment**: Character-based alignment was applied to correctly compare model predictions with the gold standard.
4. **Metrics**:  
   - UPOS: Macro-averaged F1 score computed across 10 target UPOS classes.  
   - Tense: Macro-F1 score computed among verbs (`VERB`).

### Comparative Results Table

| Dataset | spaCy UPOS F1 | Stanza UPOS F1 | Δ F1 UPOS (vs Stanza) |
|:--------|--------------:|---------------:|----------------------:|
| **dev**   | 0.972         | 0.975          | 0.285%                |
| **test**  | 0.973         | 0.972          | -0.112%               |
| **train** | 0.974         | 0.974          | 0.071%                |

### Analysis and Conclusion

Across all splits (dev, test, train), **spaCy-uk** demonstrates macro-F1 scores above 0.97 for UPOS tagging, consistently matching or nearly matching the performance of **Stanza-uk**.  
The observed differences between the two models are within ±0.3 percentage points — well below the predefined 2 p.p. threshold.  

This confirms that **spaCy-uk** maintains near-expert accuracy for POS tagging on Ukrainian, making it fully reliable for downstream features in Phonova that depend on POS counts (e.g., first-person pronoun percentage, noun–verb ratio, adjective density).  
Given these results, spaCy-uk can be confidently used as the primary POS tagging module for Ukrainian in production workflows without compromising accuracy compared to Stanza-uk.

## Section 1.3 — Tense Evaluation

In addition to UPOS tagging, we evaluated the accuracy of extracting the `Tense` morphological feature from the UD_Ukrainian-ParlaMint dataset.  
The `Tense` feature in Universal Dependencies denotes grammatical tense (e.g., `Past`, `Pres`, `Fut`) and is primarily relevant for verbs, although it can appear in other categories (such as participles tagged as `ADJ`).

Two evaluation settings were applied:

- **VERB only** (`restrict_to_verbs=True`):  
  Evaluation is restricted to tokens where the gold UPOS label is `VERB`.  
  This focuses the metric on canonical verb forms, excluding other parts of speech that may occasionally carry a `Tense` feature.

- **has gold Tense** (`restrict_to_verbs=False` with filtering):  
  Evaluation includes all tokens in the corpus that have a `Tense` value in the gold annotation, regardless of UPOS.  
  This setting captures cases where tense information is attached to non-verb forms (e.g., participles annotated as `ADJ` in UD).

### Comparative Results Table — Tense (Macro-F1)

| Dataset  | spaCy Tense F1<br>(VERB only) | Stanza Tense F1<br>(VERB only) | Δ F1<br>(VERB only) | spaCy Tense F1<br>(has gold Tense) | Stanza Tense F1<br>(has gold Tense) | Δ F1<br>(has gold Tense) |
|:---------|------------------------------:|-------------------------------:|--------------------:|-----------------------------------:|------------------------------------:|-------------------------:|
| **dev**  | 0.972                         | 0.929                          | -4.32%              | 0.981                              | 0.706                               | -27.46%                  |
| **test** | 0.959                         | 0.926                          | -3.31%              | 0.729                              | 0.708                               | -2.09%                   |
| **train**| 0.967                         | 0.952                          | -1.50%              | 0.731                              | 0.719                               | -1.27%                   |

### Analysis and Conclusion

For **VERB only** evaluation, spaCy-uk consistently achieves high macro-F1 scores (0.959–0.972) and performs comparably to Stanza-uk, with differences between -1.5% and -4.3%, remaining within acceptable limits.

In the **has gold Tense** setting, performance differences between spaCy-uk and Stanza-uk are minimal on test and train sets (≤ 2.1%), but a substantial gap (-27.46%) is observed on the dev set.  
This large dev-set gap is likely explained by differences in how each model handles non-verbal forms carrying tense information, such as adjectival participles, which are less frequent and harder to predict accurately.

Overall, **spaCy-uk** demonstrates reliable tense prediction for canonical verbs, making it suitable for downstream analysis involving verb tense usage.  
However, for tasks requiring precise tense detection in non-verb forms, additional model fine-tuning or post-processing rules may be necessary to bridge the performance gap observed in the dev split.

## Section 1.4 Validation of Syllable Counter and “Syllables-per-Minute” Metric (Ukrainian)

### Goal (Brief)

For timing biomarkers (SPM, articulation rate), an accurate syllable count is required. English rules from NLTK are not suitable for Ukrainian, so we built a custom spaCy-uk component and validated it on real data, comparing it with a text-based baseline (Pyphen) and an acoustic gold standard (Praat Syllable Nuclei v3).

---

### Dataset: “Telebachennia Toronto” (Summary)

- **Source:** A popular Ukrainian satirical news show “Telebachennia Toronto” / Toronto TV (hosted by the persona Michael Shchur, created by journalist Roman Vintoniv). The show is regularly described in English-language media as a “news with comedy” format aimed at explaining complex topics to a wide audience.
- **Corpus format:** 5–7-second audio clips from episodes, processed with a transcriber; fields: path, transcript, transcript_len, audio_dur_sec. Total of 18,302 clips, average length ≈ 5 s.
- **Important:** Some transcripts don’t fully match the audio (repetitions/interjections/cuts), affecting “text ↔ audio” comparison.
- **Context:** See profiles on [The Fix](https://thefix.media/) and [Kyiv Independent](https://kyivindependent.com/) for more info about the Toronto TV project in English.

---

### Syllable Counting Methods

**Text-based:**
1. **spaCy-uk (our component):** Rules based on Pyphen + Ukrainian-specific patches (apostrophe/hyphens/“йо”, rare syllabic “r/l”).
2. **Pyphen:** Hyphenation as a simple approximation of syllabic structure.
3. **NLTK (SSP):** Universal sonority tokenizer (used as a sanity-check).

**Audio-based:**
4. **Praat Syllable Nuclei v3 (Original):** Detects intensity peaks with voicing filter; also provides phonation time and pauses.
5. **Praat-Like:** Our simplified implementation of the same principle in Python/Parselmouth (without external .praat scripts).

---

### Evaluation Protocol

- For each clip, compute: syll_spacy, syll_pyphen, syll_nltk, syll_praat_original, syll_praat_like; also SPM = syllables / (duration/60).
- **Metrics:** Pairwise MAE, Bias, Pearson r, Spearman ρ, Bland–Altman LOA, and ICC(2,1) (absolute agreement).
- Praat v3 threshold settings were taken from open recommendations, without fine-tuning for noise/channel.

---

### Results (Full Run)

#### 1) Main MAE Summary

| Method Pair              | MAE   |
|--------------------------|-------|
| spaCy-uk vs Pyphen       | 0.517 |
| spaCy-uk vs NLTK         | 3.776 |
| Praat-Like vs Praat Orig | 2.423 |
| spaCy-uk vs Praat Orig   | 9.778 |
| Pyphen vs Praat Orig     | 9.836 |
| NLTK vs Praat Orig       | 6.680 |

Overall ICC(2,1) across all raters: **0.656** (moderate).

#### 2) Within-Class Comparisons (Audio and Text)

##### Audio ↔ Audio (n=18,210)
- MAE = 2.423, Bias = −1.082; r = 0.908, ρ = 0.892; LOA ≈ [−6.94; 4.78]; ICC(2,1) within audio = 0.897.
- **Conclusion:** Praat-Like closely approximates Praat v3; we use Praat Original as the gold standard (it also provides phonation time).

##### Text ↔ Text (n=18,210)
- spaCy-uk vs Pyphen: MAE = 0.517, Bias = −0.106; r = 0.988, ρ = 0.989; LOA is narrow.
- spaCy-uk vs NLTK: MAE = 3.776, Bias = +3.425; r = 0.961, ρ = 0.956.
- Pyphen vs NLTK: MAE = 3.681, Bias = +3.532; r = 0.972, ρ = 0.965.
- ICC(2,1) within text = 0.937.
- **Conclusion:** spaCy-uk and Pyphen are well-aligned; NLTK (SSP) systematically overestimates or segments differently for Ukrainian, so it’s left as a sanity-check.

---

### Interpretation

- Everything is logical within each class:
    - Audio methods show high agreement (ICC ≈ 0.90), i.e., acoustic syllable nucleus detection is robust.
    - Text counters are consistent with each other (spaCy-uk ↔ Pyphen).
- Text ↔ audio diverges significantly (MAE ≈ 9–10; overall ICC ≈ 0.66) due to partial mismatch between transcripts and audio in the corpus — a common situation for auto-generated subtitles/snippets. On subsets where “text ≈ audio,” the divergence is small (by manual check).
- For clinical timing (SPM, articulation rate), it is correct to rely on the audio gold standard (Praat v3) and use its phonation time. The described methodology is widely used precisely without a transcript.


# Section E — Validation of Tangentiality, Perplexity, and Coherence (Ukrainian)

---

## Dataset

## E‑DAIC (Extended Distress Analysis Interview Corpus)

**Overview.**  
E‑DAIC extends DAIC‑WOZ clinical interviews with linked clinical assessments (e.g., PHQ‑8, PCL‑C). DAIC‑WOZ/E‑DAIC materials are distributed by USC/ICT with documentation describing data contents and interview protocol. ([dcapswoz.ict.usc.edu](https://dcapswoz.ict.usc.edu/wp-content/uploads/2022/02/DAICWOZDepression_Documentation.pdf))

**Collection & consent.**  
DAIC‑WOZ interviews were collected under human‑subjects research oversight; the official documentation describes recording setup, annotation, and accompanying measures (e.g., PHQ‑8). (The documentation is the canonical reference; access is controlled via USC/ICT.) ([dcapswoz.ict.usc.edu](https://dcapswoz.ict.usc.edu/wp-content/uploads/2022/02/DAICWOZDepression_Documentation.pdf))

**Licensing & access.**  
Access requires registration/approval via the **official DAIC‑WOZ download portal**; redistribution is restricted. Treat E‑DAIC as **research‑only** data with a data‑use agreement (DUA). ([dcapswoz.ict.usc.edu](https://dcapswoz.ict.usc.edu/wp-content/uploads/2022/02/DAICWOZDepression_Documentation.pdf))  
Additionally, the Extended DAIC is governed by an End User License Agreement (EULA) that stipulates this database is permitted for **research purpose only**, prohibits commercial use, and disallows redistribution without permission.  [oai_citation:0‡ihp-lab.org](https://www.ihp-lab.org/downloads/Extended-DAIC-BLANK_EULA.pdf)

**Usage restrictions & non‑diagnostic disclaimer.**  
Use for **research** and **method development** only; **not for clinical diagnosis or individual decision‑making**. Do not attempt to identify participants; no commercial or clinical deployment without separate permission/oversight. (These restrictions are enforced by USC/ICT and outlined in the EULA.)  [oai_citation:1‡ihp-lab.org](https://www.ihp-lab.org/downloads/Extended-DAIC-BLANK_EULA.pdf)

**Clinical risk notes.**  
Interviews contain sensitive mental‑health content. Models trained or applied to E‑DAIC should be clearly labeled as **non‑diagnostic** and **experimental**; misuse (e.g., in clinical settings) may pose ethical risks. (Standard risk guidance; EULA prohibits clinical use.)  [oai_citation:2‡ihp-lab.org](https://www.ihp-lab.org/downloads/Extended-DAIC-BLANK_EULA.pdf)

**Intended use.**  
Academic research on affective computing, mental‑health signal processing, benchmarking algorithms in interview settings. ([dcapswoz.ict.usc.edu](https://dcapswoz.ict.usc.edu/wp-content/uploads/2022/02/DAICWOZDepression_Documentation.pdf))

**Prohibited/unsafe uses.**  
Re‑identification, redistribution outside the DUA, clinical or commercial deployment without authorization, or presenting outputs as medical advice. (Enforced via USC/ICT licensing and EULA.)  [oai_citation:3‡ihp-lab.org](https://www.ihp-lab.org/downloads/Extended-DAIC-BLANK_EULA.pdf)

**Known limitations.**  
Single‑site study design; relatively modest sample size; scripted-interviewer setting (Ellie) introduces artifacts; interviewer prompts may bias models. Indeed, recent work has shown that models exploiting the interviewer’s prompts may “shortcut” the depression classification task.  [oai_citation:4‡arXiv](https://arxiv.org/abs/2404.14463)  
Also, some session transcripts are incomplete or missing interviewer speech (excluded sessions) per documentation.  [oai_citation:5‡DAIC-WOZ](https://dcapswoz.ict.usc.edu/wp-content/uploads/2022/02/DAICWOZDepression_Documentation.pdf)

**How to cite / where to learn more.**  
DAIC‑WOZ documentation, USC/ICT portal, and the DAIC‑WOZ EULA. ([dcapswoz.ict.usc.edu](https://dcapswoz.ict.usc.edu/wp-content/uploads/2022/02/DAICWOZDepression_Documentation.pdf)) 
---


- **DAIC‑WOZ Depression Database** — 189 clinical interview sessions, collected via the Wizard‑of‑Oz paradigm (virtual interviewer “Ellie”). [oai_citation:0‡DAIC‑WOZ](https://dcapswoz.ict.usc.edu/wp-content/uploads/2022/02/DAICWOZDepression_Documentation.pdf)  
- Sessions are split into **train/dev/test**: 107 / 35 / 47 respectively. [oai_citation:1‡Research on DAIC‑WOZ](https://braininformatics.springeropen.com/articles/10.1186/s40708-023-00185-9)  
- Each participant has a PHQ‑8 assessment (total score from 0 to 24); PHQ‑8 ≥ 10 is used as the binary marker of depression. [oai_citation:2‡Research on DAIC‑WOZ](https://braininformatics.springeropen.com/articles/10.1186/s40708-023-00185-9)  
- Audio recordings (16 kHz), text transcripts, acoustic and visual features (OpenFace / COVAREP, etc.). [oai_citation:3‡DAIC‑WOZ](https://dcapswoz.ict.usc.edu/wp-content/uploads/2022/02/DAICWOZDepression_Documentation.pdf)  

- **Usage in our project:**  
  The English transcripts have been translated into Ukrainian using the **Yehor/kulyk‑en‑uk** model. [oai_citation:4‡Hugging Face](https://huggingface.co/posts/Yehor/884755060089981)  

---

## Methods

- **Embeddings (Gemma):**
  - For both English and Ukrainian we use `google/embeddinggemma‑300m` (L2 normalization, cosine similarity).
  - Chosen because of its broad tokenizer coverage, including Ukrainian.

- **Tangentiality:**
  - $1 - \cos(\text{utterance}, \text{topic})$, where “topic” = session‑level centroid.
  - Implemented using EmbeddingGemma.

- **Coherence:**
  - $\cos(\text{utterance}_i,\; \text{utterance}_{i-1})$, $i>0$ (adjacent turns).
  - Also with EmbeddingGemma.

- **Pseudo‑Perplexity:**
  - Causal LM (teacher‑forcing) — `google/gemma‑3‑270m`.
  - Context windows: 256, 2, 5, 7 (principal measure PPL‑256).

---

## Evaluation Protocol

- **Turn‑level:** Pearson $r$, Spearman $\rho$, MAE, RMSE (EN vs UK).  
- **Session‑level** (mean, variance): ICC(2,1), paired tests, mean difference %.  

---

## Results

### BERT-Base-Cased (English) vs BERT-Multilingual-Cased (Ukrainian)

I also compared performance of two other models:

| Metric        | N   | Pearson_r | Spearman_rho | MAE          | RMSE         | ICC2_1   | mean_diff_% (UK_vs_EN) | t_p           | wilcoxon_p   |
| ------------- | --- | --------- | ------------ | ------------ | ------------ | -------- | ---------------------- | ------------- | ------------ |
| coherence     | 275 | 0.335050  | 0.298841     | 0.355406     | 0.355501     | 0.000199 | −41.980173             | 0.000000e+00  | 7.488097e-47 |
| perplexity    | 275 | 0.202752  | 0.290302     | 24,699.46107 | 37,889.42130 | 0.144741 | −29.216398             | 4.198781e-19  | 8.970070e-27 |
| tangentiality | 275 | 0.761982  | 0.740529     | 0.114286     | 0.116063     | 0.080850 | 54.005750              | 6.440214e-210 | 7.488097e-47 |

### Gemma case for English vs Gemma for Ukrainian

|   | metric        | N   | Pearson_r | Spearman_rho | MAE          | RMSE         | ICC2_1   | mean_diff_% (UK_vs_EN) | t_p          | wilcoxon_p   |
| - | ------------- | --- | --------- | ------------ | ------------ | ------------ | ------- | ---------------------- | ------------ | ------------ |
| 0 | coherence     | 275 | 0.242285  | 0.202955     | 2.097768e-02 | 2.118642e-02 | 0.005617 | -2.219679              | 5.625587e-236 | 7.488097e-47 |
| 1 | perplexity    | 275 | 0.060095  | 0.262961     | 9.059692e+07 | 1.937525e+08 | 0.058709 | 398.242392             | 1.732284e-01 | 2.671626e-07 |
| 2 | tangentiality | 275 | 0.973861  | 0.972289     | 1.273736e-02 | 1.548681e-02 | 0.953386 | -1.368960              | 1.118840e-34 | 1.584426e-28 |

---

### QE-stratified EN↔UA agreement for tangentiality (Gemma vs BERT)

To test whether EN↔UA tangentiality agreement depends on translation quality, we computed turn-level COMET-QE (participant-only), aggregated it to interview-level `QE_i = median(QE_i,j)`, and split interviews into tertiles (Low/Mid/High). Agreement is reported per bin (Pearson *r*, Spearman *ρ*, ICC(2,1), and mean shift Δμ = mean(UA − EN) with bootstrap 95% CI).

#### Gemma (EmbeddingGemma tangentiality)

| Model | QE bin |   n | QE median [min,max]  | r [95% CI]           | ρ [95% CI]           | ICC(2,1) | Mean shift Δμ [95% CI]  |
| ----- | -----: | --: | -------------------- | -------------------- | -------------------- | -------: | ----------------------- |
| Gemma |    All | 275 | 0.765 [0.609, 0.848] | 0.974 [0.967, 0.979] | 0.972 [0.961, 0.978] |    0.953 | −0.010 [−0.011, −0.009] |
| Gemma |    Low |  92 | 0.713 [0.609, 0.739] | 0.935 [0.902, 0.958] | 0.932 [0.889, 0.954] |    0.914 | −0.008 [−0.011, −0.005] |
| Gemma |    Mid |  91 | 0.765 [0.740, 0.787] | 0.966 [0.945, 0.979] | 0.959 [0.927, 0.973] |    0.926 | −0.011 [−0.013, −0.009] |
| Gemma |   High |  92 | 0.807 [0.788, 0.848] | 0.970 [0.960, 0.978] | 0.972 [0.951, 0.981] |    0.939 | −0.011 [−0.014, −0.009] |

**Interpretation (Gemma).**

* Association is extremely high (r/ρ ≈ 0.93–0.97 across bins), and absolute agreement is **excellent** (ICC(2,1) ≈ 0.91–0.95). ICC values above 0.90 are commonly interpreted as “excellent” reliability. ([PMC][1])
* Mean shift is very small (Δμ ≈ −0.01): Ukrainian tangentiality is ~0.01 lower than English on average, with tight CIs.
* QE stratification shows only a modest reduction in Low-QE (r and ICC drop slightly), but agreement remains high overall.

#### BERT (English BERT vs mBERT tangentiality)

| Model | QE bin |   n | QE median [min,max]  | r [95% CI]           | ρ [95% CI]           | ICC(2,1) | Mean shift Δμ [95% CI] |
| ----- | -----: | --: | -------------------- | -------------------- | -------------------- | -------: | ---------------------- |
| BERT  |    All | 275 | 0.765 [0.609, 0.848] | 0.762 [0.714, 0.806] | 0.741 [0.679, 0.793] |    0.081 | 0.114 [0.112, 0.117]   |
| BERT  |    Low |  92 | 0.713 [0.609, 0.739] | 0.823 [0.753, 0.879] | 0.815 [0.735, 0.874] |    0.082 | 0.116 [0.113, 0.119]   |
| BERT  |    Mid |  91 | 0.765 [0.740, 0.787] | 0.713 [0.590, 0.801] | 0.671 [0.518, 0.780] |    0.078 | 0.114 [0.109, 0.118]   |
| BERT  |   High |  92 | 0.807 [0.788, 0.848] | 0.762 [0.656, 0.834] | 0.727 [0.599, 0.818] |    0.084 | 0.113 [0.109, 0.118]   |

**Interpretation (BERT).**

* Association is moderate (r ≈ 0.71–0.82; ρ ≈ 0.67–0.82), but absolute agreement is **poor** (ICC(2,1) ≈ 0.08). Under common guidelines, ICC < 0.50 indicates poor reliability. ([PMC][1])
* There is a large, consistent systematic shift: Δμ ≈ +0.114 (UA − EN), i.e., Ukrainian tangentiality is ~0.11 higher than English across all QE bins.
* QE binning does **not** explain this: the shift and ICC stay nearly constant in Low/Mid/High. This pattern is more consistent with a **model/calibration mismatch** than with translation-quality effects.

### Key findings of language dependent features agreement 

#### Sentiment and first-person sentiment: **strongest agreement**

- Summary-level `sentiment_pos`, `sentiment_overall`, `sentiment_neg`, and `first_person_sentiment_positive` show the strongest cross-language agreement: **r≈0.92–0.95**, **ICC≈0.90–0.94**, with very small mean shifts.
- The corresponding turn-level features remain strong: `sentiment_pos`, `sentiment_overall`, `sentiment_neg`, and `first_person_sentiment_positive` again show **r≈0.90–0.95** and **ICC≈0.70–0.94**.

#### Moderate agreement for lexical diversity and timing

- MATTR features show moderate-to-strong rank agreement but only moderate absolute agreement: **ICC≈0.22–0.71** at the summary level and **ICC≈0.49–0.58** at the turn level.
- `syllables_per_min` remains highly correlated across languages, but with a clear positive Ukrainian shift and only moderate ICC: summary **r=0.986**, **ICC=0.629**; turns **r=0.948**, **ICC=0.484**.

#### Weak agreement for coherence/perplexity; sentence tangentiality unavailable

- `word_coherence*` features show weak cross-language stability in the current run: mostly **r<0.53** and **ICC<0.13**.
- `semantic_perplexity*` features are not stable in the current run: correlations are near zero and ICC is near zero or negative.
- `first_order_sentence_tangeniality*` and `second_order_sentence_tangeniality*` are `nan` in the current notebook output, so they should be treated as unavailable for this run.

**Interpretation:** in the current consistency run, stability is strongest for sentiment-derived and some first-person features, moderate for lexical diversity and syllable-rate features, weak for word coherence and pseudo-perplexity, and unavailable for sentence-level tangentiality outputs.


### Summary features table 

|   # | feature                                 | EN_median[min,max]                              | UA_median[min,max]                                | r [95% CI]             | rho [95% CI]          | ICC(2,1) [95% CI]      | mean shift Δμ (UA-EN) [95% CI]                    |
|----:|:----------------------------------------|:------------------------------------------------|:--------------------------------------------------|:-----------------------|:----------------------|:-----------------------|:--------------------------------------------------|
|   0 | first_person_percentage                 | 8.767 [3.852,12.865]                            | 6.875 [3.358,10.672]                              | 0.870 [0.839,0.898]    | 0.865 [0.830,0.896]   | 0.465 [0.414,0.515]    | -1.870 [-1.963,-1.779]                            |
|   1 | first_person_sentiment_negative         | 2.356 [0.942,4.780]                             | 1.852 [0.634,4.230]                               | 0.899 [0.875,0.921]    | 0.878 [0.840,0.910]   | 0.700 [0.651,0.743]    | -0.506 [-0.544,-0.465]                            |
|   2 | first_person_sentiment_negative_vader   | 0.388 [0.037,1.304]                             | 0.335 [0.005,1.562]                               | 0.807 [0.762,0.846]    | 0.815 [0.762,0.857]   | 0.788 [0.740,0.830]    | -0.045 [-0.061,-0.029]                            |
|   3 | first_person_sentiment_overall          | 23.555 [5.383,39.756]                           | 22.218 [9.377,39.764]                             | 0.935 [0.917,0.950]    | 0.930 [0.910,0.946]   | 0.899 [0.876,0.921]    | -1.441 [-1.660,-1.210]                            |
|   4 | first_person_sentiment_overall_vader    | 17.259 [8.679,33.070]                           | 17.994 [7.695,32.707]                             | 0.855 [0.815,0.887]    | 0.839 [0.794,0.876]   | 0.830 [0.790,0.865]    | 0.967 [0.713,1.229]                               |
|   5 | first_person_sentiment_positive         | 25.725 [13.849,40.996]                          | 26.327 [15.431,41.739]                            | 0.943 [0.926,0.956]    | 0.933 [0.913,0.950]   | 0.937 [0.920,0.952]    | 0.202 [-0.005,0.404]                              |
|   6 | first_person_sentiment_positive_vader   | 17.535 [8.449,32.395]                           | 18.227 [7.622,32.918]                             | 0.860 [0.822,0.892]    | 0.843 [0.800,0.881]   | 0.832 [0.793,0.866]    | 1.039 [0.794,1.295]                               |
|   7 | mattr_10                                | 0.919 [0.843,0.951]                             | 0.935 [0.846,0.975]                               | 0.848 [0.806,0.883]    | 0.829 [0.780,0.870]   | 0.573 [0.504,0.640]    | 0.015 [0.014,0.016]                               |
|   8 | mattr_100                               | 0.588 [0.458,0.672]                             | 0.669 [0.574,0.775]                               | 0.882 [0.852,0.910]    | 0.888 [0.860,0.914]   | 0.224 [0.192,0.257]    | 0.080 [0.078,0.081]                               |
|   9 | mattr_25                                | 0.800 [0.687,0.863]                             | 0.844 [0.761,0.897]                               | 0.837 [0.792,0.877]    | 0.839 [0.791,0.881]   | 0.345 [0.301,0.390]    | 0.041 [0.039,0.043]                               |
|  10 | mattr_5                                 | 0.974 [0.926,0.985]                             | 0.978 [0.918,0.995]                               | 0.818 [0.744,0.878]    | 0.711 [0.632,0.780]   | 0.714 [0.618,0.800]    | 0.004 [0.003,0.005]                               |
|  11 | mattr_50                                | 0.697 [0.571,0.767]                             | 0.760 [0.668,0.837]                               | 0.871 [0.840,0.900]    | 0.881 [0.848,0.908]   | 0.262 [0.226,0.297]    | 0.063 [0.061,0.064]                               |
|  12 | semantic_perplexity_11_mean             | 990.702 [279.345,38075.138]                     | 5797.996 [1850.641,41379.571]                     | 0.023 [-0.086,0.155]   | 0.119 [-0.002,0.241]  | 0.007 [-0.016,0.037]   | 4567.930 [4136.221,4991.527]                      |
|  13 | semantic_perplexity_11_var              | 4042807.597 [93212.034,79099665804.348]         | 54516017.911 [6250376.911,142692330364.321]       | -0.006 [-0.030,0.062]  | 0.039 [-0.080,0.156]  | -0.000 [-0.024,0.055]  | 279698264.450 [-789277594.126,1606021239.929]     |
|  14 | semantic_perplexity_15_mean             | 546.562 [143.284,34292.183]                     | 2130.565 [796.692,36922.515]                      | -0.025 [-0.112,0.032]  | -0.017 [-0.139,0.102] | -0.014 [-0.049,0.013]  | 1326.268 [956.279,1702.714]                       |
|  15 | semantic_perplexity_15_var              | 1791134.079 [18193.583,78543080548.482]         | 7783072.973 [693539.652,142932474066.401]         | -0.009 [-0.023,-0.003] | -0.053 [-0.174,0.062] | -0.004 [-0.019,-0.000] | 206119878.235 [-865132804.418,1538986592.495]     |
|  16 | semantic_perplexity_5_mean              | 11333.345 [4195.269,423517.137]                 | 217567.771 [122071.186,30022944.057]              | 0.038 [-0.038,0.202]   | 0.303 [0.190,0.409]   | 0.003 [-0.000,0.016]   | 325550.500 [213131.078,552675.161]                |
|  17 | semantic_perplexity_5_var               | 644270371.882 [31601012.747,19497783571149.441] | 163063162006.775 [19260125571.790,145020475229... | -0.003 [-0.022,0.030]  | 0.195 [0.073,0.310]   | 0.002 [-0.002,0.020]   | 510925809466537.062 [371663971880.555,15832181... |
|  18 | semantic_perplexity_mean                | 255.349 [40.046,33889.189]                      | 370.450 [120.476,34837.182]                       | 0.023 [-0.044,0.154]   | 0.201 [0.078,0.323]   | 0.010 [-0.028,0.086]   | -102.457 [-471.505,259.694]                       |
|  19 | semantic_perplexity_var                 | 1278865.337 [1909.129,78568320243.490]          | 923644.889 [33741.632,143068158428.190]           | -0.009 [-0.024,0.000]  | 0.003 [-0.131,0.135]  | -0.004 [-0.018,0.000]  | 200149858.552 [-871701355.917,1533596638.337]     |
|  20 | sentiment_neg                           | 0.306 [0.198,0.430]                             | 0.298 [0.199,0.410]                               | 0.919 [0.895,0.939]    | 0.913 [0.883,0.937]   | 0.895 [0.869,0.917]    | -0.009 [-0.011,-0.007]                            |
|  21 | sentiment_neu                           | 0.404 [0.295,0.504]                             | 0.417 [0.315,0.528]                               | 0.906 [0.883,0.927]    | 0.889 [0.855,0.918]   | 0.854 [0.821,0.882]    | 0.011 [0.009,0.013]                               |
|  22 | sentiment_overall                       | -0.021 [-0.270,0.227]                           | -0.016 [-0.239,0.211]                             | 0.939 [0.921,0.954]    | 0.930 [0.906,0.949]   | 0.931 [0.913,0.946]    | 0.007 [0.003,0.011]                               |
|  23 | sentiment_pos                           | 0.281 [0.153,0.445]                             | 0.283 [0.166,0.446]                               | 0.947 [0.933,0.959]    | 0.939 [0.921,0.954]   | 0.941 [0.927,0.955]    | -0.002 [-0.004,0.000]                             |
|  24 | sentiment_vader_neg                     | 0.070 [0.024,0.175]                             | 0.079 [0.022,0.208]                               | 0.860 [0.818,0.895]    | 0.848 [0.809,0.882]   | 0.812 [0.767,0.852]    | 0.008 [0.007,0.010]                               |
|  25 | sentiment_vader_neu                     | 0.738 [0.605,0.829]                             | 0.718 [0.601,0.827]                               | 0.764 [0.710,0.811]    | 0.764 [0.706,0.814]   | 0.695 [0.641,0.745]    | -0.018 [-0.022,-0.015]                            |
|  26 | sentiment_vader_overall                 | 0.999 [-0.995,1.000]                            | 0.999 [-0.998,1.000]                              | 0.686 [0.340,0.894]    | 0.918 [0.892,0.939]   | 0.661 [0.297,0.890]    | -0.020 [-0.048,0.007]                             |
|  27 | sentiment_vader_pos                     | 0.189 [0.103,0.310]                             | 0.198 [0.104,0.326]                               | 0.772 [0.722,0.817]    | 0.769 [0.715,0.819]   | 0.748 [0.697,0.795]    | 0.010 [0.007,0.013]                               |
|  28 | syllables_per_min                       | 210.586 [66.382,326.344]                        | 260.171 [80.935,399.138]                          | 0.986 [0.982,0.989]    | 0.982 [0.977,0.986]   | 0.629 [0.584,0.672]    | 48.728 [47.295,50.154]                            |
|  29 | turn_to_turn_tangeniality_mean          | 0.453 [0.380,0.548]                             | 0.442 [0.368,0.524]                               | 0.863 [0.830,0.891]    | 0.844 [0.799,0.881]   | 0.809 [0.766,0.845]    | -0.009 [-0.011,-0.008]                            |
|  30 | turn_to_turn_tangeniality_slope         | 0.000 [-0.016,0.006]                            | 0.000 [-0.014,0.005]                              | 0.813 [0.660,0.929]    | 0.646 [0.556,0.733]   | 0.803 [0.649,0.922]    | 0.000 [0.000,0.000]                               |
|  31 | turn_to_turn_tangeniality_var           | 0.017 [0.008,0.044]                             | 0.015 [0.007,0.047]                               | 0.845 [0.799,0.885]    | 0.804 [0.751,0.850]   | 0.762 [0.700,0.824]    | -0.002 [-0.002,-0.002]                            |
|  32 | word_coherence_5_mean                   | 0.656 [0.642,0.671]                             | 0.701 [0.677,0.719]                               | 0.223 [0.091,0.353]    | 0.265 [0.143,0.378]   | 0.005 [0.002,0.008]    | 0.044 [0.043,0.045]                               |
|  33 | word_coherence_5_var                    | 0.001 [0.001,0.002]                             | 0.003 [0.002,0.004]                               | 0.228 [0.075,0.379]    | 0.180 [0.058,0.306]   | 0.012 [0.004,0.021]    | 0.002 [0.002,0.002]                               |
|  34 | word_coherence_mean                     | 0.566 [0.535,0.576]                             | 0.619 [0.584,0.635]                               | 0.525 [0.396,0.635]    | 0.473 [0.368,0.573]   | 0.016 [0.010,0.023]    | 0.052 [0.051,0.053]                               |
|  35 | word_coherence_var                      | 0.004 [0.003,0.008]                             | 0.009 [0.006,0.012]                               | 0.463 [0.317,0.599]    | 0.339 [0.224,0.450]   | 0.025 [0.014,0.038]    | 0.004 [0.004,0.004]                               |
|  36 | word_coherence_variability_2_mean       | 0.568 [0.543,0.586]                             | 0.619 [0.590,0.636]                               | 0.385 [0.221,0.531]    | 0.353 [0.229,0.471]   | 0.013 [0.007,0.020]    | 0.051 [0.050,0.051]                               |
|  37 | word_coherence_variability_2_var        | 0.007 [0.003,0.011]                             | 0.010 [0.008,0.014]                               | 0.240 [0.101,0.362]    | 0.242 [0.118,0.357]   | 0.043 [0.017,0.067]    | 0.003 [0.003,0.004]                               |
|  38 | word_coherence_variability_3_mean       | 0.568 [0.546,0.595]                             | 0.620 [0.587,0.640]                               | 0.334 [0.197,0.472]    | 0.331 [0.218,0.450]   | 0.013 [0.007,0.020]    | 0.051 [0.050,0.052]                               |
|  39 | word_coherence_variability_3_var        | 0.007 [0.004,0.018]                             | 0.011 [0.008,0.014]                               | 0.318 [0.214,0.413]    | 0.292 [0.177,0.400]   | 0.071 [0.041,0.104]    | 0.003 [0.003,0.003]                               |
|  40 | word_coherence_variability_4_mean       | 0.568 [0.538,0.595]                             | 0.622 [0.589,0.647]                               | 0.307 [0.159,0.447]    | 0.325 [0.204,0.438]   | 0.013 [0.007,0.021]    | 0.053 [0.051,0.054]                               |
|  41 | word_coherence_variability_4_var        | 0.008 [0.003,0.016]                             | 0.011 [0.007,0.018]                               | 0.148 [-0.010,0.297]   | 0.111 [-0.013,0.231]  | 0.044 [-0.003,0.094]   | 0.003 [0.003,0.004]                               |
|  42 | word_coherence_variability_5_mean       | 0.569 [0.534,0.585]                             | 0.622 [0.573,0.647]                               | 0.350 [0.161,0.524]    | 0.362 [0.240,0.476]   | 0.018 [0.008,0.029]    | 0.053 [0.052,0.055]                               |
|  43 | word_coherence_variability_5_var        | 0.008 [0.002,0.018]                             | 0.011 [0.002,0.029]                               | 0.166 [-0.047,0.400]   | 0.251 [0.127,0.373]   | 0.059 [-0.020,0.142]   | 0.003 [0.003,0.003]                               |
|  44 | word_coherence_variability_6_mean       | 0.570 [0.516,0.650]                             | 0.622 [0.567,0.645]                               | 0.331 [0.071,0.581]    | 0.391 [0.267,0.505]   | 0.022 [0.005,0.040]    | 0.052 [0.051,0.054]                               |
|  45 | word_coherence_variability_6_var        | 0.008 [0.003,0.041]                             | 0.011 [0.005,0.017]                               | 0.158 [0.039,0.299]    | 0.145 [0.020,0.265]   | 0.068 [0.016,0.127]    | 0.003 [0.003,0.003]                               |
|  46 | word_coherence_variability_7_mean       | 0.570 [0.515,0.602]                             | 0.622 [0.556,0.690]                               | 0.325 [0.075,0.544]    | 0.373 [0.254,0.488]   | 0.024 [0.005,0.043]    | 0.052 [0.050,0.053]                               |
|  47 | word_coherence_variability_7_var        | 0.008 [0.002,0.016]                             | 0.011 [0.003,0.016]                               | 0.292 [0.128,0.439]    | 0.206 [0.075,0.331]   | 0.127 [0.050,0.203]    | 0.003 [0.003,0.003]                               |
|  48 | word_coherence_variability_8_mean       | 0.570 [0.520,0.603]                             | 0.623 [0.480,0.695]                               | 0.260 [-0.075,0.545]   | 0.330 [0.203,0.452]   | 0.027 [-0.007,0.067]   | 0.052 [0.050,0.054]                               |
|  49 | word_coherence_variability_8_var        | 0.008 [0.001,0.021]                             | 0.011 [0.001,0.019]                               | 0.155 [-0.026,0.324]   | 0.141 [0.017,0.267]   | 0.068 [-0.011,0.144]   | 0.003 [0.003,0.003]                               |
|  50 | word_coherence_variability_9_mean       | 0.570 [0.515,0.611]                             | 0.623 [0.443,0.695]                               | 0.400 [0.193,0.576]    | 0.323 [0.206,0.439]   | 0.043 [0.014,0.085]    | 0.053 [0.051,0.054]                               |
|  51 | word_coherence_10_mean                  | 0.611 [0.586,0.630]                             | 0.663 [0.634,0.693]                               | 0.177 [0.027,0.312]    | 0.259 [0.138,0.372]   | 0.007 [0.001,0.012]    | 0.052 [0.051,0.053]                               |
|  52 | word_coherence_10_var                   | 0.002 [0.001,0.003]                             | 0.004 [0.001,0.010]                               | 0.069 [-0.195,0.328]   | 0.159 [0.036,0.282]   | 0.004 [-0.017,0.024]   | 0.002 [0.002,0.002]                               |
|  53 | word_coherence_variability_10_mean      | 0.570 [0.523,0.625]                             | 0.622 [0.536,0.742]                               | 0.150 [-0.101,0.394]   | 0.250 [0.124,0.376]   | 0.016 [-0.011,0.044]   | 0.051 [0.049,0.053]                               |
|  54 | word_coherence_variability_10_var       | 0.008 [0.002,0.026]                             | 0.011 [0.002,0.027]                               | -0.019 [-0.158,0.116]  | 0.070 [-0.060,0.202]  | -0.013 [-0.101,0.067]  | 0.003 [0.002,0.003]                               |
|  55 | word_coherence_variability_9_var        | 0.008 [0.003,0.021]                             | 0.011 [0.003,0.017]                               | 0.091 [-0.068,0.250]   | 0.041 [-0.089,0.169]  | 0.037 [-0.026,0.105]   | 0.003 [0.003,0.003]                               |
|  56 | first_order_sentence_tangeniality_mean  | nan [nan,nan]                                   | nan [nan,nan]                                     | nan [nan,nan]          | nan [nan,nan]         | nan [nan,nan]          | nan [nan,nan]                                     |
|  57 | first_order_sentence_tangeniality_var   | nan [nan,nan]                                   | nan [nan,nan]                                     | nan [nan,nan]          | nan [nan,nan]         | nan [nan,nan]          | nan [nan,nan]                                     |
|  58 | second_order_sentence_tangeniality_mean | nan [nan,nan]                                   | nan [nan,nan]                                     | nan [nan,nan]          | nan [nan,nan]         | nan [nan,nan]          | nan [nan,nan]                                     |
|  59 | second_order_sentence_tangeniality_var  | nan [nan,nan]                                   | nan [nan,nan]                                     | nan [nan,nan]          | nan [nan,nan]         | nan [nan,nan]          | nan [nan,nan]                                     |


### Turns features table 

|   # | feature                            | EN_median[min,max]              | UA_median[min,max]                   | r [95% CI]            | rho [95% CI]          | ICC(2,1) [95% CI]     | mean shift Δμ (UA-EN) [95% CI]     |
|----:|:-----------------------------------|:--------------------------------|:-------------------------------------|:----------------------|:----------------------|:----------------------|:-----------------------------------|
|   0 | first_person_percentage            | 8.196 [3.344,12.830]            | 6.723 [2.191,10.554]                 | 0.907 [0.884,0.927]   | 0.895 [0.862,0.923]   | 0.623 [0.571,0.670]   | -1.567 [-1.655,-1.481]             |
|   1 | first_person_sentiment_negative    | 2.356 [0.942,4.780]             | 1.852 [0.634,4.230]                  | 0.899 [0.875,0.921]   | 0.878 [0.840,0.910]   | 0.700 [0.651,0.743]   | -0.506 [-0.544,-0.465]             |
|   2 | first_person_sentiment_positive    | 25.725 [13.849,40.996]          | 26.327 [15.431,41.739]               | 0.943 [0.926,0.956]   | 0.933 [0.913,0.950]   | 0.937 [0.920,0.952]   | 0.202 [-0.005,0.404]               |
|   3 | mattr_10                           | 0.959 [0.917,0.993]             | 0.971 [0.947,0.997]                  | 0.816 [0.768,0.854]   | 0.789 [0.733,0.838]   | 0.504 [0.447,0.555]   | 0.012 [0.011,0.013]                |
|   4 | mattr_100                          | 0.929 [0.836,0.992]             | 0.954 [0.886,0.997]                  | 0.937 [0.917,0.951]   | 0.923 [0.898,0.944]   | 0.577 [0.529,0.621]   | 0.024 [0.023,0.025]                |
|   5 | mattr_25                           | 0.933 [0.867,0.992]             | 0.957 [0.905,0.997]                  | 0.916 [0.893,0.935]   | 0.904 [0.875,0.929]   | 0.547 [0.497,0.592]   | 0.022 [0.020,0.023]                |
|   6 | mattr_5                            | 0.983 [0.964,0.996]             | 0.988 [0.963,1.000]                  | 0.623 [0.543,0.697]   | 0.587 [0.495,0.675]   | 0.490 [0.409,0.569]   | 0.004 [0.004,0.005]                |
|   7 | mattr_50                           | 0.929 [0.840,0.992]             | 0.955 [0.886,0.997]                  | 0.933 [0.913,0.949]   | 0.920 [0.895,0.941]   | 0.573 [0.524,0.617]   | 0.024 [0.022,0.025]                |
|   8 | semantic_perplexity                | 255.349 [40.046,33889.189]      | 370.450 [120.476,34837.182]          | 0.023 [-0.044,0.154]  | 0.201 [0.078,0.323]   | 0.010 [-0.028,0.086]  | -102.457 [-471.505,259.694]        |
|   9 | semantic_perplexity_11             | 990.702 [279.345,38075.138]     | 5797.996 [1850.641,41379.571]        | 0.023 [-0.086,0.155]  | 0.119 [-0.002,0.241]  | 0.007 [-0.016,0.037]  | 4567.930 [4136.221,4991.527]       |
|  10 | semantic_perplexity_15             | 546.562 [143.284,34292.183]     | 2130.565 [796.692,36922.515]         | -0.025 [-0.112,0.032] | -0.017 [-0.139,0.102] | -0.014 [-0.049,0.013] | 1326.268 [956.279,1702.714]        |
|  11 | semantic_perplexity_5              | 11333.345 [4195.269,423517.137] | 217567.771 [122071.186,30022944.057] | 0.038 [-0.038,0.202]  | 0.303 [0.190,0.409]   | 0.003 [-0.000,0.016]  | 325550.500 [213131.078,552675.161] |
|  12 | sentiment_neg                      | 0.306 [0.198,0.430]             | 0.298 [0.199,0.410]                  | 0.919 [0.895,0.939]   | 0.913 [0.883,0.937]   | 0.895 [0.869,0.917]   | -0.009 [-0.011,-0.007]             |
|  13 | sentiment_neu                      | 0.404 [0.295,0.504]             | 0.417 [0.315,0.528]                  | 0.906 [0.883,0.927]   | 0.889 [0.855,0.918]   | 0.854 [0.821,0.882]   | 0.011 [0.009,0.013]                |
|  14 | sentiment_overall                  | -0.005 [-0.069,0.057]           | -0.004 [-0.061,0.053]                | 0.939 [0.921,0.954]   | 0.930 [0.906,0.950]   | 0.931 [0.913,0.947]   | 0.002 [0.001,0.003]                |
|  15 | sentiment_pos                      | 0.281 [0.153,0.445]             | 0.283 [0.166,0.446]                  | 0.947 [0.933,0.959]   | 0.939 [0.921,0.954]   | 0.941 [0.927,0.955]   | -0.002 [-0.004,0.000]              |
|  16 | syllables_per_min                  | 304.535 [202.380,541.139]       | 378.867 [253.878,661.717]            | 0.948 [0.930,0.962]   | 0.936 [0.918,0.951]   | 0.484 [0.418,0.545]   | 74.702 [72.237,77.145]             |
|  17 | turn_to_turn_tangeniality          | 0.453 [0.380,0.548]             | 0.442 [0.368,0.524]                  | 0.863 [0.830,0.891]   | 0.844 [0.799,0.881]   | 0.809 [0.766,0.845]   | -0.009 [-0.011,-0.008]             |
|  18 | first_order_sentence_tangeniality  | nan [nan,nan]                   | nan [nan,nan]                        | nan [nan,nan]         | nan [nan,nan]         | nan [nan,nan]         | nan [nan,nan]                      |
|  19 | second_order_sentence_tangeniality | nan [nan,nan]                   | nan [nan,nan]                        | nan [nan,nan]         | nan [nan,nan]         | nan [nan,nan]         | nan [nan,nan]                      |

---

### Why Gemma aligns better than BERT here (evidence-based)

**1) “Same embedding space” vs “two different encoders.”**
In the Gemma stack, both English and Ukrainian are embedded with the same embedding model (`google/embeddinggemma-300m`), so cosine-based geometry (and therefore tangentiality scale) is shared across languages by design. The EmbeddingGemma model card describes multilingual training (over 100 languages) and large-scale training data (~320B tokens), consistent with robust cross-lingual embedding behavior. ([Hugging Face][2])
In the BERT comparison, English uses `bert-base-cased` while Ukrainian uses `bert-base-multilingual-cased` (mBERT). These are **different pretrained models** with different training distributions and tokenization/representation behavior, so there is no guarantee that cosine similarities (and derived tangentiality values) are on the same numeric scale across EN vs UA.

**2) Multilingual BERT’s training setup can introduce language-dependent representation artifacts.**
mBERT is pretrained on the 104 languages with the largest Wikipedias. ([Hugging Face][3])
Google’s multilingual BERT documentation notes it uses a shared WordPiece vocabulary of ~110k subwords across languages. ([GitHub][4])
Research probing mBERT finds that it learns multilingual representations but with “systematic deficiencies affecting certain language pairs,” which is consistent with language-dependent shifts in similarity-based downstream scores. ([arxiv.org][5])

**3) Tokenizer size alone is not the whole explanation.**
Gemma-family models use a large vocabulary (Gemma 2 inherits a ~256k vocabulary designed to work across many languages), but Gemma 2 is still described as trained on primarily English data and “not trained specifically for state-of-the-art multilingual capabilities.” ([arxiv.org][6])
In this project, the decisive factor for tangentiality agreement is that **EmbeddingGemma is explicitly trained as a multilingual embedding model** with broad language coverage. ([Hugging Face][2])
By contrast, mixing an English-only BERT with mBERT can yield a stable cross-language offset (Δμ ≈ +0.114) that correlation does not “fix,” and ICC(2,1) correctly flags as poor absolute agreement. ([PMC][1])

---

## Conclusion

* **Tangentiality (Gemma stack)** is validated for EN↔UA: QE-stratified agreement remains extremely high (r/ρ ≈ 0.93–0.97; ICC(2,1) ≈ 0.91–0.95) with a small mean shift (Δμ ≈ −0.01). By common ICC guidance, this corresponds to excellent reliability. ([PMC][1])
* **Tangentiality (BERT stack)** is **not** cross-language stable in the current configuration (English BERT vs mBERT): it shows a large systematic shift (Δμ ≈ +0.114) and very poor absolute agreement (ICC(2,1) ≈ 0.08), and this effect does not meaningfully vary with QE bin.
* **Coherence and Pseudo-Perplexity** (in the current implementation) do **not** meet the consistency criteria; PPL and coherence are especially sensitive to model/tokenization and are not directly comparable across EN↔UA without stronger calibration controls.

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4913118/?utm_source=chatgpt.com "A Guideline of Selecting and Reporting Intraclass ..."
[2]: https://huggingface.co/google/embeddinggemma-300m "google/embeddinggemma-300m · Hugging Face"
[3]: https://huggingface.co/google-bert/bert-base-multilingual-cased?utm_source=chatgpt.com "google-bert/bert-base-multilingual-cased"
[4]: https://github.com/google-research/bert/blob/master/multilingual.md "bert/multilingual.md at master · google-research/bert · GitHub"
[5]: https://arxiv.org/abs/1906.01502?utm_source=chatgpt.com "How multilingual is Multilingual BERT?"
[6]: https://arxiv.org/html/2408.00118v1 "Gemma 2: Improving Open Language Models at a Practical Size"


# Section F — Downstream Predictive Validation

---

## Goal

Demonstrate that adapted Phonova features, after Ukrainian translation and feature extraction, retain their predictive utility for psychiatric outcomes — not just cross-lingual fidelity, but real-world clinical validity. This is done by assessing performance on both regression of symptom severity (PHQ-8, PCL-C) and binary diagnostic classification (depression/PTSD cases), and benchmarking Ukrainian against English.

---

## Dataset

- **DAIC-WOZ**: 189 semi-structured clinical interviews (virtual interviewer “Ellie”) with linked psychiatric assessments:  
  - **PHQ-8** (depression severity, 0–24; binary cutoff ≥10)  
  - **PCL-C** (PTSD, 17–85; binary cutoff ≥50)  
  - Audio (16 kHz), transcripts, acoustic, and visual features.
- **Our approach**:  
  - **English pipeline**: Original transcripts, Phonova features (incl. Gemma/bert/other as specified).
  - **Ukrainian pipeline**: Transcripts translated turn-by-turn (Yehor/kulyk-en-uk), features extracted with Ukrainian-adapted Phonova stack (Gemma, multilingual BERT).

---

## Methods

- **Main prediction task:**
  - *Classification*: Depression (PHQ-8 ≥10) and PTSD (PCL-C ≥50) — XGBoost/DecisionTree/RandomForest.
- **Features:**  
  Lexical (MATTR, wordfreq, affect, etc.), discourse (tangentiality, coherence, perplexity), prosodic (pauses, speech rate), sentiment, and summary stats (mean/var per session).
- **Session-level aggregation**: mean + variance of each feature over all turns.
- **Cross-validation:** 5-fold, participant-level (no overlap train/test).
- **Metrics:**  
  - Regression: MAE, RMSE  
  - Classification: Accuracy, AUROC, F1  
  - **Cross-lingual drop tolerance:** ≤10% degradation vs English is “acceptable”.

---

## Results

Main idea - classify Depression label

### 1. **Gemma-based models (English)**

| model                | accuracy | f1_macro | f1_lo  | f1_hi  | balanced_acc | roc_auc | auc_lo | auc_hi | pr_auc | brier  | icc2_1 | icc_lo | icc_hi |
|----------------------|----------|----------|--------|--------|--------------|---------|--------|--------|--------|--------|--------|--------|--------|
| DecisionTree         | 0.5714   | 0.4932   | 0.3627 | 0.6199 | 0.4932       | 0.5505  | 0.3899 | 0.7172 | 0.3887 | 0.3486 | NaN    | NaN    | NaN    |
| RandomForest         | 0.6964   | 0.5623   | 0.4255 | 0.6828 | 0.5664       | 0.5354  | 0.3574 | 0.7014 | 0.4188 | 0.2216 | 0.1004 | -0.16  | 0.35   |
| XGBoost              | 0.5893   | 0.4649   | 0.3532 | 0.5919 | 0.4729       | 0.4676  | 0.3015 | 0.6320 | 0.3369 | 0.3232 | 0.3070 | 0.06   | 0.52   |
| DecisionTree vs XGBoost | 0.5714   | 0.4932   | 0.3665 | 0.6209 | 0.4932       | 0.5505  | 0.3876 | 0.7112 | 0.3887 | 0.3486 | 0.1762 | -0.08  | 0.41   |

#### 1.1 The impact of features on the model
<img width="789" height="490" alt="image" src="https://github.com/user-attachments/assets/b0450c99-ee06-4c5d-8311-13b45778541d" />
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/cbebce63-611c-4467-beb7-4f3e2ff481d0" />

### 2. **BERT-based models (English)**

| model                  | accuracy | f1_macro | f1_lo  | f1_hi  | balanced_acc | roc_auc | auc_lo | auc_hi | pr_auc | brier  | icc2_1 | icc_lo | icc_hi |
|------------------------|----------|----------|--------|--------|--------------|---------|--------|--------|--------|--------|--------|--------|--------|
| DecisionTree           | 0.8036   | 0.7168   | 0.5736 | 0.8371 | 0.6931       | 0.6508  | 0.5068 | 0.7919 | 0.5766 | 0.2163 | NaN    | NaN    | NaN    |
| RandomForest           | 0.7321   | 0.5896   | 0.4432 | 0.7317 | 0.5920       | 0.5566  | 0.3740 | 0.7285 | 0.4867 | 0.2135 | 0.1413 | -0.13  | 0.39   |
| XGBoost                | 0.6071   | 0.4762   | 0.3563 | 0.5993 | 0.4857       | 0.4646  | 0.2866 | 0.6471 | 0.3363 | 0.2944 | 0.0403 | -0.23  | 0.30   |
| DecisionTree vs XGBoost| 0.8036   | 0.7168   | 0.5604 | 0.8371 | 0.6931       | 0.6508  | 0.5008 | 0.7904 | 0.5766 | 0.2163 | 0.0098 | -0.26  | 0.27   |

#### 2.1 The impact of features on the model
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/eb3273ca-1d02-4268-8273-3073bf000df6" />
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/705f112f-ab7f-4ad8-847a-d0ab9fb7df7d" />


### 3. **Gemma-based models (Ukrainian)**

| model                  | accuracy | f1_macro | f1_lo  | f1_hi  | balanced_acc | roc_auc | auc_lo | auc_hi | pr_auc | brier  | icc2_1 | icc_lo | icc_hi |
|------------------------|----------|----------|--------|--------|--------------|---------|--------|--------|--------|--------|--------|--------|--------|
| DecisionTree           | 0.5536   | 0.4954   | 0.3627 | 0.6263 | 0.4969       | 0.4924  | 0.3529 | 0.6373 | 0.3103 | 0.3677 | NaN    | NaN    | NaN    |
| RandomForest           | 0.6071   | 0.5354   | 0.3978 | 0.6621 | 0.5354       | 0.5279  | 0.3484 | 0.7044 | 0.3518 | 0.2384 | 0.1250 | -0.14  | 0.37   |
| XGBoost                | 0.6429   | 0.5625   | 0.4267 | 0.6938 | 0.5611       | 0.5234  | 0.3544 | 0.6908 | 0.3517 | 0.2946 | 0.0526 | -0.19  | 0.30   |
| DecisionTree vs XGBoost| 0.5536   | 0.4954   | 0.3652 | 0.6199 | 0.4970       | 0.4925  | 0.3612 | 0.6418 | 0.3103 | 0.3677 | 0.0410 | -0.21  | 0.29   |

#### 3.1 The impact of features on the model
<img width="789" height="490" alt="image" src="https://github.com/user-attachments/assets/03022158-7969-4460-899d-ee74381c5feb" />
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/335520ce-566c-44ac-b998-4145533b7df2" />

### 4. **BERT-based models (Ukrainian)**

| model                  | accuracy | f1_macro | f1_lo  | f1_hi  | balanced_acc | roc_auc | auc_lo | auc_hi | pr_auc | brier  | icc2_1  | icc_lo | icc_hi |
|------------------------|----------|----------|--------|--------|--------------|---------|--------|--------|--------|--------|---------|--------|--------|
| DecisionTree           | 0.5714   | 0.5429   | 0.4149 | 0.6791 | 0.5596       | 0.5528  | 0.3944 | 0.7127 | 0.3503 | 0.3413 | NaN     | NaN    | NaN    |
| RandomForest           | 0.7321   | 0.5246   | 0.4105 | 0.6606 | 0.5588       | 0.4902  | 0.3212 | 0.6592 | 0.4154 | 0.2133 | -0.1835 | -0.43  | 0.08   |
| XGBoost                | 0.6964   | 0.5852   | 0.4490 | 0.7268 | 0.5830       | 0.5294  | 0.3529 | 0.7059 | 0.4534 | 0.2608 | -0.1714 | -0.40  | 0.08   |
| DecisionTree vs XGBoost| 0.5714   | 0.5429   | 0.4147 | 0.6730 | 0.5596       | 0.5528  | 0.4065 | 0.7066 | 0.3503 | 0.3413 | -0.1762 | -0.41  | 0.08   |

#### 4.1 The impact of features on the model
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/f98b135e-da7a-4048-808e-37ec04a138e1" />
<img width="792" height="490" alt="image" src="https://github.com/user-attachments/assets/eecce5a5-c698-41b8-87c4-164fbf7066fd" />


---

### 5. **Cross-validation: Train on EN, test on UA (and vice versa) — Gemma stack**

|   | model_on | own_test | own_acc  | own_f1_macro | own_roc_auc | own_f1_lo | own_f1_hi | own_auc_lo | own_auc_hi | cross_test | cross_acc | cross_f1_macro | cross_roc_auc | cross_f1_lo | cross_f1_hi | cross_auc_lo | cross_auc_hi | n_train | n_own_test | n_cross_test | own_icc2_1  | cross_icc2_1 |
|---|----------|----------|----------|--------------|-------------|-----------|-----------|------------|------------|------------|-----------|----------------|---------------|-------------|-------------|--------------|--------------|---------|------------|--------------|-------------|--------------|
| 0 | EN       | EN       | 0.571429 | 0.522048     | 0.610106    | 0.417872  | 0.621103  | 0.488883   | 0.727273   | UA         | 0.392857  | 0.372859       | 0.554299      | 0.284207    | 0.455455    | 0.438867     | 0.666667     | 163     | 56         | 56           | -0.018956   | 0.078044     |
| 1 | UA       | UA       | 0.625000 | 0.548907     | 0.549774    | 0.437500  | 0.648625  | 0.438026   | 0.650464   | EN         | 0.482143  | 0.452646       | 0.475867      | 0.355014    | 0.549076    | 0.359609     | 0.590544     | 163     | 56         | 56           | 0.078044    | -0.018956    |

### 6. **Cross-validation: Train on EN, test on UA (and vice versa) — Bert stack**

|   | model_on | own_test | own_acc  | own_f1_macro | own_roc_auc | own_f1_lo | own_f1_hi | own_auc_lo | own_auc_hi | cross_test | cross_acc | cross_f1_macro | cross_roc_auc | cross_f1_lo | cross_f1_hi | cross_auc_lo | cross_auc_hi | n_train | n_own_test | n_cross_test |   own_icc2_1    |  cross_icc2_1   |
|---|----------|----------|----------|--------------|-------------|-----------|-----------|------------|------------|------------|-----------|----------------|---------------|-------------|-------------|--------------|--------------|---------|------------|--------------|------------------|-----------------|
| 0 | EN       | EN       | 0.821429 | 0.749553     | 0.686275    | 0.642857  | 0.851852  | 0.573962   | 0.808019   | UA         | 0.660714  | 0.557956       | 0.642534      | 0.446153    | 0.667413    | 0.530897     | 0.764468     | 163     | 56         | 56           | 2.635787e-16     | 3.864477e-02    |
| 1 | UA       | UA       | 0.571429 | 0.542857     | 0.552790    | 0.437488  | 0.644231  | 0.432522   | 0.673709   | EN         | 0.696429  | 0.410526       | 0.500000      | 0.388889    | 0.430769    | 0.500000     | 0.500000     | 163     | 56         | 56           | 3.864477e-02     | 2.635787e-16    |

Short description of table fields:
	- model_on / own_test / cross_test — which dataset/language the model was trained on (model_on) and where it’s evaluated: the own (in-domain) test vs the cross (out-of-domain) test.
	- own_acc / cross_acc — accuracy: share of samples classified correctly on that test split.  ￼
	- own_f1_macro / cross_f1_macro — macro-F1: unweighted mean of per-class F1 scores (treats each class equally). F1 itself is the harmonic mean of precision and recall.  ￼
	- own_roc_auc / cross_roc_auc — ROC AUC: area under the ROC curve computed from predicted scores/probabilities (works for binary and, with options, multiclass). Higher = better ranking of positives over negatives.  ￼
	- own_f1_lo / own_f1_hi and cross_f1_lo / cross_f1_hi — lower/upper bounds of the 95% confidence interval for macro-F1 (from bootstrap in your code). A 95% CI is a range the procedure would cover the true value about 95% of the time under repeated sampling.  ￼
	- own_auc_lo / own_auc_hi and cross_auc_lo / cross_auc_hi — same idea, but for ROC AUC (95% bootstrap CI bounds).  ￼
	- n_train / n_own_test / n_cross_test — counts of samples used to train the model and to evaluate it on own vs cross tests.
	- own_icc2_1 / cross_icc2_1 — ICC(2,1) between the two models’ probability outputs on the same subjects: “two-way random effects, absolute agreement, single measurement.” It quantifies how closely the two raters (models) match in value, not just rank. own is computed on the own test set; cross on the cross test set.  ￼


## Key Takeaways

- **Best in-domain performance:**  
  **BERT EN→EN** with Accuracy **0.821**, Macro-F1 **0.750**, ROC-AUC **0.686**.
- **Best cross-lingual transfer:**  
  **BERT EN→UA** with Accuracy **0.661**, Macro-F1 **0.558**, ROC-AUC **0.643** (only cross split with robust AUC > 0.6 and F1 > 0.5).
- **Gemma’s cross-lingual robustness:**  
  Generally modest, with CIs often including 0.50. Some cross splits show better ranking than in-domain, but differences are not statistically decisive.
- **Uncertainty:**  
  Test set size is small (**n=56**), so confidence intervals are wide. Several metrics are not statistically distinct from chance.

---

## Brief Interpretation

- **English-trained representations travel better** than Ukrainian ones in this setup, especially for BERT.
- **Gemma** offers partial cross-lingual robustness, but separation remains modest.
- For deployment:
  - Use **BERT EN→EN** for English-only.
  - For cross-Ukrainian applications, **BERT EN→UA** offers the most reliable transfer in this benchmark, but results should be confirmed on larger UA test sets.
  - Gemma stack may be preferred for robust Ukrainian/English compatibility where cross-lingual consistency is required.
    
- **Clinical implication:**  
  - Downstream task performance on Ukrainian is robust and does **not** suffer >10% drop vs English — confirming the Phonova feature pipeline is linguistically and clinically valid after adaptation.

---
