# Third-Party Data, Model, And Resource Citations

Checked on 2026-05-19. This note records the citation and license/access status for data and external resources referenced by the paper scripts. The repository should not redistribute third-party raw data, model weights, Praat scripts, or controlled-access participant-level artifacts unless the upstream license or data-use agreement explicitly permits that redistribution.

## Summary

| Resource | Used by | Access / license status | Citation / attribution to include |
| --- | --- | --- | --- |
| DAIC-WOZ / Extended DAIC | Track B interview modeling | Controlled access through the USC ICT DAIC-WOZ / Extended DAIC site. The site states that, due to consent constraints, distribution is limited to academics and other non-profit researchers using academic email addresses. Do not redistribute audio, transcripts, labels, demographics, translated transcripts, or participant-level features. | Gratch et al. (2014), DeVault et al. (2014) when describing the virtual interviewer system, and Ringeval et al. (2019) for Extended DAIC / AVEC 2019 usage. |
| COSMUS | `track_a_evaluate_sentiment.py` | Hugging Face dataset card lists license `mit`, one train split, and text-classification columns including `document_content` and `annotator_sentiment`. This repository does not vendor the dataset. | Cite the `YShynkarov/COSMUS` Hugging Face dataset card. No separate formal paper citation was visible on the current dataset card. |
| UkrRoBERTa COSMUS sentiment model | `track_a_evaluate_sentiment.py` | Hugging Face model card lists license `mit` and states that `YShynkarov/COSMUS` was used to train the model. This repository does not vendor weights. | Cite the `YShynkarov/ukr-roberta-cosmus-sentiment` Hugging Face model card and the COSMUS dataset card. |
| CardiffNLP twitter-XLM-R sentiment model | `track_a_evaluate_sentiment.py` | Downloaded from Hugging Face at runtime. The current model card exposes the paper citation but does not show a license field in the visible page. Do not redistribute weights from this repository. | Barbieri, Espinosa-Anke, and Camacho-Collados (2022), XLM-T. |
| Universal Dependencies Ukrainian ParlaMint | `track_a_validate_pos_tense.py` | Repository license is CC BY-SA 4.0. The script reads raw GitHub train/dev/test CoNLL-U URLs or caller-provided local copies. No CoNLL-U files are committed here. | Cite UD_Ukrainian-ParlaMint, Kopp et al. (2023), and Erjavec et al. (2022). Preserve CC BY-SA attribution if derived tables quote or redistribute corpus content. |
| Praat Vocal Toolkit Syllable Nuclei v3 | `track_a_validate_syllables.py` | Public Praat plugin page says the scripts are included with author consent and gives the validation paper. This repository does not vendor `SyllableNucleiv3.praat`; users pass it with `--praat-script`. | De Jong, Pacilly, and Heeren (2021). |
| VADER / vaderSentiment | VADER sentiment features and `track_a_evaluate_sentiment.py` | `vaderSentiment` documentation states that VADER is open sourced under the MIT License. VADER is installed as a dependency, not vendored in this paper folder. | Hutto and Gilbert (2014). |
| UCU Audio Processing Course 2025 / Telebachennia Toronto clip dataset | `track_a_validate_syllables.py` | Public GitHub course repository with a README citation. The notebook used a local course-derived archive unpacking to `labels.jsonl` and `toronto_*/*.wav`. This paper folder does not vendor raw clips or transcripts. No explicit repository license file was visible when checked, so users should follow the repository instructions, any dataset-hosting terms, and original media-source terms. | Sydorskyi et al. (2025), UCU Audio Processing Course 2025. |

## Source Notes

- DAIC-WOZ / Extended DAIC official site: `https://dcapswoz.ict.usc.edu/`
- COSMUS dataset card: `https://huggingface.co/datasets/YShynkarov/COSMUS`
- UkrRoBERTa COSMUS model card: `https://huggingface.co/YShynkarov/ukr-roberta-cosmus-sentiment`
- CardiffNLP XLM-T sentiment model card: `https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment`
- UD Ukrainian ParlaMint repository: `https://github.com/UniversalDependencies/UD_Ukrainian-ParlaMint`
- UD Ukrainian ParlaMint license file: `https://raw.githubusercontent.com/UniversalDependencies/UD_Ukrainian-ParlaMint/master/LICENSE.txt`
- UCU Audio Processing Course repository: `https://github.com/VSydorskyy/ucu_audio_processing_course`
- Praat Vocal Toolkit Syllable Nuclei v3 page: `https://www.praatvocaltoolkit.com/syllable-nuclei-v3.html`
- VADER documentation: `https://vadersentiment.readthedocs.io/en/latest/pages/introduction.html`

## Bibliographic Entries

- Barbieri, Francesco, Luis Espinosa-Anke, and Jose Camacho-Collados. 2022. "XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis and Beyond." In Proceedings of the Thirteenth Language Resources and Evaluation Conference, 258-266. `https://aclanthology.org/2022.lrec-1.27`
- De Jong, N. H., J. J. A. Pacilly, and W. Heeren. 2021. "PRAAT scripts to measure speed fluency and breakdown fluency in speech automatically." Assessment in Education: Principles, Policy & Practice 28(4):456-476. `https://doi.org/10.1080/0969594X.2021.1951162`
- DeVault, D., R. Artstein, G. Benn, T. Dey, E. Fast, A. Gainer, K. Georgila, J. Gratch, A. Hartholt, M. Lhommet, G. Lucas, S. Marsella, F. Morbini, A. Nazarian, S. Scherer, G. Stratou, A. Suri, D. Traum, R. Wood, Y. Xu, A. Rizzo, and L.-P. Morency. 2014. "SimSensei Kiosk: A virtual human interviewer for healthcare decision support." In Proceedings of AAMAS 2014.
- Erjavec, Tomaž, Maciej Ogrodniczuk, Petya Osenova, Nikola Ljubešić, Kiril Simov, Andrej Pančur, Michał Rudolf, Matyáš Kopp, et al. 2022. "The ParlaMint corpora of parliamentary proceedings." Language Resources and Evaluation. `https://doi.org/10.1007/s10579-021-09574-0`
- Gratch, Jonathan, Ron Artstein, Gale Lucas, Giota Stratou, Stefan Scherer, Angela Nazarian, Rachel Wood, Jill Boberg, David DeVault, Stacy Marsella, David Traum, Skip Rizzo, and Louis-Philippe Morency. 2014. "The Distress Analysis Interview Corpus of Human and Computer Interviews." In Proceedings of LREC 2014, 3123-3128. `https://aclanthology.org/L14-1421/`
- Hutto, C. J., and Eric Gilbert. 2014. "VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text." Proceedings of the International AAAI Conference on Web and Social Media 8(1):216-225. `https://doi.org/10.1609/icwsm.v8i1.14550`
- Kopp, Matyas, Anna Kryvenko, and Andriana Rii. 2023. "Ukrainian parliamentary corpus ParlaMint-UA 4.0.1." Slovenian language resource repository CLARIN.SI. `http://hdl.handle.net/11356/1900`
- Ringeval, Fabien, Björn Schuller, Michel Valstar, Nicholas Cummins, Roddy Cowie, Leili Tavabi, Maximilian Schmitt, et al. 2019. "AVEC 2019 Workshop and Challenge: State-of-Mind, Detecting Depression with AI, and Cross-Cultural Affect Recognition." In Proceedings of the 9th International on Audio/Visual Emotion Challenge and Workshop, 3-12. ACM.
- Sydorskyi, Volodymyr, Anton Bazdyrev, Oles Dobosevych, Yurii Laba, Andrii Shevtsov, Ostap Viniavskyi, Yurii Yelisieiev, Yurii Paniv, Andrii Zhuravlov, and Yevhenii Azarov. 2025. "UCU Audio Processing Course 2025." GitHub repository. `https://github.com/VSydorskyy/ucu_audio_processing_course`

## BibTeX For Telebachennia Toronto Clip Dataset Source

```bibtex
@misc{ucu_audio_processing_course_2025,
  author = {Volodymyr Sydorskyi, Anton Bazdyrev, Oles Dobosevych, Yurii Laba, Andrii Shevtsov, Ostap Viniavskyi, Yurii Yelisieiev, Yurii Paniv, Andrii Zhuravlov, Yevhenii Azarov},
  title = {UCU Audio Processing Course 2025},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/VSydorskyy/ucu_audio_processing_course}},
}
```
