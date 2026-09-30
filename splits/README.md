# Preserved split manifests

The training seeds change initialization, augmentation and sample order, not the
frozen partition (`split_seed=42`).

| Files | Coverage | Format |
| --- | --- | --- |
| `frozen/NJN/njn_split.csv` | All 760 images: 532/76/152 train/validation/test | Original semicolon-delimited `image;patient_id;split` |
| `frozen/NeoJaundice/neojaundice_split.csv` | All 2,235 images: 1,563/225/447 train/validation/test | Original semicolon-delimited `image;patient_id;split` |
| `njn_split.csv` | All 760 images, including labels | Reconstructed comma-delimited audit manifest |
| `neojaundice_split.csv` | 672 validation/test images, including labels | Reconstructed comma-delimited audit manifest |
| `split_meta.json` | Metadata and checksums for the reconstructed CSVs | JSON |
| `njn_phash_d5.json` | NJN mapping: 760 images in 755 inferred groups | Original pHash cache metadata, retained without image arrays |

The complete originals and their metadata were recovered byte-for-byte from the
V2 campaign's `dataset/<dataset>/splits/` directories on 2026-09-27. Source paths
and SHA-256 checksums are in `release/source_provenance.json`. They were not
regenerated. Split-creation metadata record an earlier environment than training.

Run `python scripts/verify_artifacts.py` to verify checksums, counts, disjoint
recorded groups, and agreement with the reconstructed audit manifests.

For training with `--frozen-split`, `src.splits.load_frozen_split` uses a manifest
under the supplied dataset directory when present, otherwise these preserved
`frozen/` copies. No source-image data are required to audit membership.
Source images are required to load samples and train.

NJN grouping is based on pHash near-duplicates (Hamming distance at most 5), not
verified infant identities. NeoJaundice uses the source patient identifiers. Group
separation is therefore not evidence of verified infant separation on NJN.

The older reconstructed manifests can still be exported with
`python src/export_splits.py`, using `evidence/v7/preds/` and
`splits/njn_phash_d5.json`. That command rewrites only the audit manifests and
`split_meta.json`, not the complete original partitions in `frozen/`.
