# Data Availability

This paper branch contains only supplementary code, deterministic configuration,
metadata, synthetic example transcripts, and expected outputs for smoke testing.

## Public Materials In This Repository

- `paper/scripts/`: paper pipeline and metric-verification scripts.
- `paper/paper_reproducibility_config.json`: fixed feature set, random seed,
  model hyperparameters, preprocessing, thresholds, and metric definitions.
- `paper/examples/transcripts/`: sanitized synthetic Whisper-like JSON examples.
- `paper/examples/labels.csv`: synthetic labels for the examples.
- `paper/examples/expected_outputs/`: expected word-, turn-, and summary-level
  outputs for the sanitized examples.
- `paper/metadata/`: label definitions, split assignments, sample manifest, and
  checksums.

## Restricted Materials

The original study data are not included in this repository. Restricted materials
include interview audio/video, raw transcripts, cleaned role-labeled transcripts,
translation artifacts, clinical labels, demographics, and frozen full paper
feature matrices. These materials can contain participant information or derived
clinical data and must stay under the approved study governance and consent
terms.

## E-DAIC Controlled Access And License

Track B uses DAIC-WOZ/E-DAIC interview data under the official provider's
controlled-access terms. The manuscript's data-availability statement notes that
E-DAIC is distributed under a signed controlled-access license / end-user license
agreement and cannot be redistributed by the authors. Consistent with those
terms, this repository does not redistribute E-DAIC audio, transcripts,
machine-translated transcripts, or derived participant-level features.

The current official access route is the DAIC-WOZ / Extended DAIC website:
https://dcapswoz.ict.usc.edu/. Use the "APPLY NOW Extended DAIC" route or the
site's contact form for access questions. The official site states that access is
available upon request and, due to consent constraints, distribution is limited
to academics and other non-profit researchers using an academic email address.

The Extended DAIC EULA requires a signed agreement before use. The EULA states
that the database is available for research purposes only, commercial use is
forbidden, redistribution of the database or parts of it to third parties is not
allowed, illustrative publication use must protect subject identity, and
publications must acknowledge the DAIC-WOZ and AVEC 2019 source papers.

## Access Requests

Researchers who need restricted study materials should contact the corresponding
author or study data steward listed with the manuscript and apply through the
official DAIC-WOZ / Extended DAIC provider. Requests should describe the research
purpose, requested variables or artifacts, data-security plan, ethics or IRB
status, institutional affiliation, and whether a data-use agreement is already in
place.

## Metric Verification Paths

The public examples support smoke testing and schema validation only. Exact paper
metrics require the restricted frozen paper feature tables listed in
`paper/metadata/frozen_artifact_checksums.csv`.

For a compact E-DAIC access summary, see
`paper/metadata/e_daic_access.md`.

For third-party dataset, model, and resource citations and license notes, see
`paper/metadata/third_party_citations_and_licenses.md`.
