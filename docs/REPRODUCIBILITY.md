# Reproduction and optional release

Run commands from the repository root or from an extracted release ZIP. No V2/V3
checkout or author-specific home path is needed for the checks below.

## CPU analysis

Use Python 3.11 in an isolated environment:

```bash
python3.11 -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_artifacts.py
python analysis/pre_submission_revision.py --check
```

The last command takes time for 10,000 paired bootstrap draws per dataset. It
checks preserved predictions and results without training, downloads or GPU use.
Without `paper/` (as in the public release) it stops after the numerical
reproduction and reports that the manuscript comparison was skipped.
`--render` prints additional computed statistical detail; it does not create a
supplement or edit the paper. `--write` updates `results/pre_submission_statistics.json`.

## Manuscript

This step needs `paper/`, which is not in the public release until publication.
With TeX Live (`latexmk`, `pdflatex`, `bibtex`) on PATH:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
cd ..
python analysis/revalidate_review.py
```

The textual checker also requires Poppler's `pdftotext` and the generated `.bbl`.
It is not an editorial approval or a scientific validation of every claim.

## Training environment and original images

`requirements.txt` is analysis-only. `requirements-train.lock` preserves the
captured Python 3.10/PyTorch 2.6.0+cu124 environment; its commented `pip freeze`
is an audit record, not a fully installable lockfile. In particular, the recorded
OpenCV `__version__` is not its distribution version. The freeze records
`opencv-python-headless==4.13.0.92`. Rebuilding training also needs timm, Optuna,
ImageHash, scikit-learn and the other dependencies in that record. Do not install
both environments into one virtualenv. Fresh dependency installation is not certified
by the checks.

Acquire the two original image releases separately. Their expected layout is:

```text
<data-root>/
  NJN/jaundice/...
  NJN/normal/...
  NeoJaundice/images/...
  NeoJaundice/chd_jaundice_published_2.csv
```

Set `JAUNDICE_DATA_ROOT` to this directory for the analysis and orchestration
scripts. They default to `dataset/` in this repository. Direct `src.train`/`src.hpo`
commands accept `--data-root`; they do not implicitly consume this environment
variable. Wrappers accept `JAUNDICE_PYTHON`; otherwise they use the active `python`.

For `--frozen-split`, training first reads `<data-root>/<dataset>/splits/`; if absent,
it reads the complete preserved `splits/frozen/<dataset>/` manifest bundled here.
Do not regenerate those partitions to reproduce the reported campaign. Normal
generated training outputs/cache now default to ignored `local/runs/` and
`local/cache/roi/`. Historical utilities in `src/` target generated run records;
they are not the current manuscript verifier.

Preview commands without launching training:

```bash
python scripts/run_explain_cells.py --dry-run
python scripts/run_hpo_reseed.py --dry-run
```

The first prepares replica attribution runs; the second is a separate HPO
sensitivity analysis. Both use the active Python interpreter. Heavy replica outputs
go to `local/experiments/explain/`. Reseed records retain their separate
`evidence/v7_reseed/` directory. Models pretrained through torchvision/timm may need
network downloads or an existing cache. These commands are optional, not prerequisites
for checking the preserved manuscript numbers.

In the training environment, the CPU tests use:

```bash
python -m pip install -r requirements-test.txt
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 python -m pytest tests -q
```

Some model tests depend on pretrained weights. Their skips must be reported rather
than counted as completed model evaluations.

Dataset tests also skip when the source images are absent. Set
`JAUNDICE_DATA_ROOT` to enable them. The packaging and split-fallback checks can
also run in the CPU analysis environment with
`python -m unittest tests.test_release tests.test_portability`.

## Local ZIP preparation

```bash
python scripts/package_release.py public
python scripts/package_release.py reproducibility
python scripts/package_release.py manuscript
```

`public` contains code, tests, selected results, configuration and preserved evidence,
without the manuscript; the public repository is initialised from it. `reproducibility`
adds the manuscript, and `manuscript` contains only the 13 manuscript files plus
`MANIFEST.json`. Both use
`release/profiles.json`, include SHA-256 hashes, and omit `.git` and internal history.
`--list` previews contents; `--output <new-path.zip>` selects another destination.
Existing ZIPs are never overwritten. Neither command uploads or publishes anything.

Before handing off a new version, compile and check that version, then generate a
new ZIP. For independent verification, extract the reproducibility ZIP into an
empty directory and repeat the CPU and manuscript checks there.
