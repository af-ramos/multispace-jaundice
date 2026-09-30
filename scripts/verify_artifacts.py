#!/usr/bin/env python3
"""Check imported provenance and frozen membership without images or GPU."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(root: Path) -> None:
    provenance = json.loads((root / "release/source_provenance.json").read_text())
    for row in provenance["files"]:
        path = root / row["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != row["sha256"]:
            raise ValueError(f"Imported artifact changed: {row['path']}")
    print(f"PASS: {len(provenance['files'])} imported artifact SHA-256 checks")

    expected = {"NJN": (532, 76, 152), "NeoJaundice": (1563, 225, 447)}
    expected_groups = {"NJN": (529, 75, 151), "NeoJaundice": (521, 75, 149)}
    for dataset, counts in expected.items():
        path = root / "splits/frozen" / dataset / f"{dataset.lower()}_split.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter=";"))
        roles, groups, by_image = Counter(), defaultdict(set), {}
        for row in rows:
            key = "/".join(row["image"].split("/")[-2:])
            if key in by_image or row["split"] not in ("train", "val", "test"):
                raise ValueError(f"Duplicate image or invalid split: {dataset}/{key}")
            by_image[key] = row
            roles[row["split"]] += 1
            groups[row["split"]].add(row["patient_id"])
        if tuple(roles[r] for r in ("train", "val", "test")) != counts:
            raise ValueError(f"Unexpected image counts: {dataset}: {roles}")
        if tuple(len(groups[r]) for r in ("train", "val", "test")) != expected_groups[dataset]:
            raise ValueError(f"Unexpected group counts: {dataset}")
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if groups[a] & groups[b]:
                raise ValueError(f"Overlapping groups: {dataset}/{a}/{b}")
        with (root / "splits" / f"{dataset.lower()}_split.csv").open(newline="") as handle:
            audit = list(csv.DictReader(handle))
        for row in audit:
            # Reconstructed manifests store basenames; full manifests retain class/dir.
            matches = [r for k, r in by_image.items() if k.split("/")[-1] == row["image"].split("/")[-1]]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous or missing image: {dataset}/{row['image']}")
            original = matches[0]
            group = row["pseudo_patient"] if dataset == "NJN" else row["patient_id"]
            if (original["split"], original["patient_id"]) != (row["split"], group):
                raise ValueError(f"Frozen/audit membership differs: {dataset}/{row['image']}")
        print(f"PASS: {dataset}: {len(rows)} frozen images, disjoint recorded groups; "
              f"{len(audit)} reconstructed audit rows agree")
    print("NJN checks concern inferred pHash groups, not verified infant identities.")


if __name__ == "__main__":
    verify(ROOT)
