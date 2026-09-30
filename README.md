# Multi-Space Chromatic Early Fusion for Neonatal Jaundice Classification

Code and preserved research records for the study *Evaluating Multi-Space Chromatic
Early Fusion for Neonatal Jaundice Classification* (A. Ramos, B. Bernal, W. Casaca;
UNESP/IBILCE), prepared for *Health Information Science and Systems*. The manuscript
sources live in `paper/`, which is not part of the public release until publication;
the public repository at <https://github.com/af-ramos/multispace-jaundice> carries
the code, the campaign records and the analysis that reproduces the reported numbers.

The factorial covers 15 colour sets × 15 backbones × 2 datasets (450 configurations,
five training seeds each). Full-factorial summaries use preserved aggregate records.
The illustrative ensembles use a restricted subset with preserved predictions;
those two forms of evidence must not be interchanged.

## Contents

| Directory | Contents |
| --- | --- |
| `src/`, `analysis/`, `scripts/`, `tests/` | Training code, analysis, packaging and checks |
| `evidence/` | Campaign records, predictions, hyperparameters and separate HPO sensitivity records |
| `splits/` | Complete original partitions and reconstructed audit manifests |
| `results/` | Selected numerical summaries supporting the paper |
| `docs/` | Artifact inventory and reproduction instructions |
| `release/` | Export profiles and imported-file provenance |
| `paper/` (not in the public release) | Manuscript, bibliography, Springer template files and figures |
| `local/` (ignored) | Operational replica checkpoints, extra-run records and inputs for historical diagnostic scripts |

## Check the reported numbers

In a Python 3.11 analysis environment:

```bash
python -m pip install -r requirements.txt
python scripts/verify_artifacts.py
python analysis/pre_submission_revision.py --check
```

These commands need neither a GPU nor the original images. The second recomputes the
paired bootstrap, the restricted validation selection and the numerical prose facts
from the preserved predictions and compares them with `results/`. When `paper/` is
present, it also checks the five generated manuscript tables and nine prose statements
against the LaTeX source. It does not verify every scientific statement in the paper.
See [reproduction instructions](docs/REPRODUCIBILITY.md) for the training environment,
dataset layout and tests.

## Packaging

```bash
python scripts/package_release.py public           # code and records, no manuscript
python scripts/package_release.py reproducibility  # the same plus the manuscript
python scripts/package_release.py manuscript       # manuscript only
```

Outputs go to ignored `release/dist/`. Each ZIP includes a SHA-256 manifest.
Use `--list` to inspect the explicit file selection, or `--output <new-path.zip>`
for a new version. The commands do not publish or upload anything. The public
repository is initialised from the `public` profile, never from the working history.

The [artifact inventory](docs/ARTIFACTS.md) explains provenance and what needs
external images or replica checkpoints; files imported from the earlier project are
listed with their SHA-256 hashes in `release/source_provenance.json`.

## Interpretation and reuse

`backbone_mode=lora` is a historical campaign label: the `HCSFModel` used here
does not instantiate LoRA. The manuscript describes the actual partial unfreezing.
The `f1_` tags identify factorial records; later runs are not additional independent
replicates. NJN pHash groups are inferred near-duplicate groups, not verified infant
identities.

Source datasets remain under their own terms and are acquired separately:
[NJN](https://zenodo.org/records/7825810) (CC BY 4.0) and
[NeoJaundice](https://springernature.figshare.com/articles/dataset/NeoJaundice_Neonatal_Jaundice_Evaluation_in_Demographic_Images/22302559)
(CC0 1.0; its usage notes state that the data are not intended for developing
diagnosis-oriented models). Code is released under the MIT licence and the derived
records in `evidence/`, `splits/` and `results/` under CC BY 4.0; see [LICENSE](LICENSE) and
[LICENSE-DATA](LICENSE-DATA).
