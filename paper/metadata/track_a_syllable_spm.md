# Track A Syllable And SPM Data

Script: `paper/scripts/track_a_validate_syllables.py`

## Where To Get Inputs

The syllable/SPM validation audio and transcript package comes from the public
UCU Audio Processing Course repository:

- `https://github.com/VSydorskyy/ucu_audio_processing_course`

The course repository README provides the citation below. The notebook used a
local course-derived package that unpacked into:

- `labels.jsonl`
- `toronto_*/*.wav`

The raw audio clips and transcripts are not vendored in this paper folder. Users
should obtain them through the course repository or its referenced dataset
distribution route and follow the repository, dataset-hosting, and original
media-source terms.

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

The Praat reference script is public and must be supplied locally as `--praat-script`:

- `SyllableNucleiv3.praat`, from the Praat Vocal Toolkit Syllable Nuclei v3 page:
  `https://www.praatvocaltoolkit.com/syllable-nuclei-v3.html`

## Expected Local Layout

After receiving the restricted archive, create a validation CSV with at least:

- `path`: relative audio path such as `toronto_001/example.wav`, or an absolute path
- `transcript`: Ukrainian transcript text
- `audio_dur_sec`: audio duration in seconds

The notebook created this CSV as `meta_toronto.csv`; this repository uses the generic name `syllable_validation_items.csv` in examples.

## Command

```bash
python paper/scripts/track_a_validate_syllables.py \
  --input path/to/syllable_validation_items.csv \
  --audio-root path/to/unpacked_audio_archive \
  --praat-script path/to/SyllableNucleiv3.praat \
  --output-dir outputs/track_a_syllables
```

Use `--skip-audio` for text-only Pyphen, NLTK, and spaCy checks when the restricted audio files are unavailable.

Citation and license audit: `paper/metadata/third_party_citations_and_licenses.md`.
