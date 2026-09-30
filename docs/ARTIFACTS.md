# Artifact inventory

The manuscript is the scientific account of the study; this repository holds the
code and the preserved records behind it. The public release
(<https://github.com/af-ramos/multispace-jaundice>) is built from the `public`
profile in `release/profiles.json` and omits `paper/` until publication.

| Location | Purpose and scope |
| --- | --- |
| `paper/` | Self-contained LaTeX manuscript, bibliography, Springer class/style, eight figure assets and compiled PDF. No appendix or separate supplement. Not included in the public release until publication. |
| `src/` | Training, preprocessing, models, splits and campaign utilities. Public entry points use `python -m src.<module>`. |
| `analysis/` | Current numerical verification and optional diagnostic/figure generation. `pre_submission_revision.py --check` recomputes the paired bootstrap, selection and numerical prose facts from preserved predictions and, when `paper/` is present, checks the five computed tables and nine numerical prose statements against the manuscript. `crop_card_fraction.py` quantifies calibration-card residue in the NeoJaundice crop (needs the original images). |
| `evidence/v7/aggregate.csv` | Preserved campaign record. The loader selects exactly 450 factorial cells by explicit regime and `f1_` tag. Other campaign rows are not additional factorial replicates. |
| `evidence/v7/A1_factorial_accf1.csv` | Historical accuracy/F1 contrast record, used by coverage diagnostics. Not a replacement for current manuscript tables. |
| `evidence/v7/markers/` | 550 original JSON execution records recovered from V2, including per-seed metrics and gate summaries. |
| `evidence/v7/preds/` | Preserved validation/test predictions; the canonical loader selects 88 cells (44 per dataset), not all 450 configurations. Includes original outputs and later replicas. |
| `evidence/v7/prediction_provenance.json` | Replica filename list preserved from the former staging area, so canonical provenance does not depend on local caches. |
| `evidence/v7/best_params/` | 30 saved RGB HPO winners, reused across colour sets. |
| `evidence/v7/stats/` | Preserved normalization statistics. |
| `evidence/v7_reseed/` | Separate six-pair HPO sensitivity experiment; JSON records, predictions and winning configurations. Not part of the original factorial. |
| `splits/frozen/` | Complete, original training/validation/test manifests for both datasets, usable by the training loader. |
| `splits/*.csv` | Previously reconstructed audit manifests; NeoJaundice here covers validation/test only. |
| `splits/njn_phash_d5.json` | Original mapping needed to reconstruct the NJN audit manifest; retained outside the image cache. |
| `results/` | Ten selected analysis outputs used by the manuscript or related diagnostics, including the crop audit (`crop_audit.csv`, `crop_audit_summary.json`). |
| `release/` | Explicit packaging profiles and SHA-256 provenance of imported records. Generated ZIPs go to ignored `release/dist/`. |

## What the checks establish

The CPU verification reproduces the paired bootstrap, restricted validation
selection and nine numerical prose facts from preserved artifacts, and, when the
manuscript is present, its five generated tables. `scripts/verify_artifacts.py` checks imported file integrity,
split membership and group separation. It does not certify ethics, consent,
chronology of experimental decisions or clinical performance.

NJN groups are inferred near-duplicate groups, not verified infant identities.
The complete NeoJaundice manifest adds the previously missing explicit training
membership without regenerating any partition. Original split metadata describe
the environment at split creation, which differs from the later training environment.

Gate weights averaged over images can be recalculated from the imported campaign
markers. The per-image gate and attribution summaries in `results/` concern
retrained replicas. Regenerating those maps needs original images and model
checkpoints from the replica runs, which are not included in the ZIP or Git.

## Not included

Some operational inputs are not distributed: replica model checkpoints, gate dumps
and run records; the six attribution-map arrays behind `results/G2_attribution_shift.csv`;
the HPO study database of the reseed experiment; and original attention dumps and
phase-2 selection inputs used only by historical diagnostics. They are large,
regenerable from the code, or superseded, and none is needed to check the reported
numbers. Regenerating maps or retraining also requires the original images.

## Data and licensing

Source images are obtained separately from the [NJN release](https://zenodo.org/records/7825810)
(CC BY 4.0) and the [NeoJaundice release](https://springernature.figshare.com/articles/dataset/NeoJaundice_Neonatal_Jaundice_Evaluation_in_Demographic_Images/22302559)
(CC0 1.0); the complete image datasets are not redistributed. Prediction dumps and
split manifests retain source image/group identifiers and labels; they are not
represented as anonymized data, and image paths are stored relative to the dataset
root. The project's derived records in `evidence/`, `splits/` and `results/` are
released under CC BY 4.0 and code under the MIT licence, as specified in `LICENSE` and `LICENSE-DATA`;
neither applies to the Springer template files or to third-party images. Reuse must
also honour the source releases' terms, including the NJN attribution requirement and
the NeoJaundice usage notes.
