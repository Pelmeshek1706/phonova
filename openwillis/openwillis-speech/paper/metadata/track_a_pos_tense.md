# Track A POS And Verb-Tense Data

Script: `paper/scripts/track_a_validate_pos_tense.py`

## Where To Get Inputs

The POS/tense validation uses Universal Dependencies Ukrainian ParlaMint CoNLL-U files:

- Repository: `https://github.com/UniversalDependencies/UD_Ukrainian-ParlaMint`
- Train: `https://raw.githubusercontent.com/UniversalDependencies/UD_Ukrainian-ParlaMint/master/uk_parlamint-ud-train.conllu`
- Dev: `https://raw.githubusercontent.com/UniversalDependencies/UD_Ukrainian-ParlaMint/master/uk_parlamint-ud-dev.conllu`
- Test: `https://raw.githubusercontent.com/UniversalDependencies/UD_Ukrainian-ParlaMint/master/uk_parlamint-ud-test.conllu`

The script accepts these URLs directly, so the files do not need to be committed.
The treebank license is CC BY-SA 4.0; preserve attribution and share-alike terms
if redistributing corpus text or derived corpus annotations.

Model assets are downloaded separately:

- spaCy Ukrainian model: `python -m spacy download uk_core_news_md`
- Stanza Ukrainian model: `python -c "import stanza; stanza.download('uk')"` or pass `--stanza-download`

## Command

```bash
python paper/scripts/track_a_validate_pos_tense.py \
  --dataset train=https://raw.githubusercontent.com/UniversalDependencies/UD_Ukrainian-ParlaMint/master/uk_parlamint-ud-train.conllu \
  --dataset dev=https://raw.githubusercontent.com/UniversalDependencies/UD_Ukrainian-ParlaMint/master/uk_parlamint-ud-dev.conllu \
  --dataset test=https://raw.githubusercontent.com/UniversalDependencies/UD_Ukrainian-ParlaMint/master/uk_parlamint-ud-test.conllu \
  --output-dir outputs/track_a_pos_tense \
  --stanza-download
```

Use `--skip-spacy` or `--skip-stanza` when validating only one tagger.

Citation and license audit: `paper/metadata/third_party_citations_and_licenses.md`.
