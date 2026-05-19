# Track A Syllable And SPM Data

Script: `paper/scripts/track_a_validate_syllables.py`

## Where To Get Inputs

The syllable/SPM validation audio and transcript package is not public in this repository. The notebook used an archived project package that unpacked into:

- `labels.jsonl`
- `toronto_*/*.wav`

Because the archive contains speech recordings and transcripts, it should be obtained from the manuscript data steward or corresponding author under the approved sharing terms. It is separate from E-DAIC Track B data.
The public release should not describe this archive as openly available unless
the steward supplies a public landing page or license.

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
