"""Checks of portable paths and the packaged split fallback without source images."""
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src import paths, splits

ROOT = Path(__file__).resolve().parents[1]


class PortabilityTests(unittest.TestCase):
    def test_historical_image_paths_are_rebased(self):
        with patch.object(paths, "DATA_ROOT", Path("/new/location")):
            self.assertEqual(paths.image_path("/old/project/dataset/NJN/normal/a.jpg"),
                             Path("/new/location/NJN/normal/a.jpg"))
            self.assertEqual(paths.image_path("dataset/NeoJaundice/images/a.jpg"),
                             Path("/new/location/NeoJaundice/images/a.jpg"))
            with self.assertRaises(ValueError):
                paths.image_path("unidentified/a.jpg")

    def test_complete_packaged_splits_load_without_external_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            for dataset, expected in (("NJN", (532, 76, 152)),
                                      ("NeoJaundice", (1563, 225, 447))):
                with self.subTest(dataset=dataset):
                    path = ROOT / "splits/frozen" / dataset / f"{dataset.lower()}_split.csv"
                    with path.open(encoding="utf-8-sig") as handle:
                        rows = list(csv.DictReader(handle, delimiter=";"))
                    samples = [splits.Sample(Path(tmp) / r["image"], r["patient_id"], 0)
                               for r in rows]
                    cfg = SimpleNamespace(data_root=tmp, dataset=dataset)
                    indices = splits.load_frozen_split(cfg, samples)
                    self.assertEqual(tuple(map(len, indices)), expected)
                    for role, selected in zip(("train", "val", "test"), indices):
                        self.assertTrue(all(rows[i]["split"] == role for i in selected))


if __name__ == "__main__":
    unittest.main()
