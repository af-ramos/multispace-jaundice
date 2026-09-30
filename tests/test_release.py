"""Offline packaging checks, runnable without the training environment."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.package_release import build, selected_files


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "release").mkdir()
        (self.root / "paper").mkdir()
        (self.root / "paper/paper.tex").write_text("manuscript")
        self.profiles({"manuscript": ["paper/paper.tex"]})

    def profiles(self, profiles):
        (self.root / "release/profiles.json").write_text(json.dumps(profiles))

    def test_archive_excludes_internal_material_and_hashes_every_payload(self):
        (self.root / "private.txt").write_text("internal editorial notes")
        (self.root / "paper/paper.log").write_text("generated log")
        output = self.root / "package.zip"
        build(self.root, "manuscript", output)
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(set(archive.namelist()), {"paper/paper.tex", "MANIFEST.json"})
            manifest = json.loads(archive.read("MANIFEST.json"))
            for row in manifest["files"]:
                data = archive.read(row["path"])
                self.assertEqual(len(data), row["bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), row["sha256"])

    def test_existing_package_is_not_overwritten(self):
        output = self.root / "package.zip"
        build(self.root, "manuscript", output)
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            build(self.root, "manuscript", output)
        self.assertEqual(output.read_bytes(), before)

    def test_missing_required_input_fails_before_archive_creation(self):
        self.profiles({"manuscript": ["missing.pdf"]})
        output = self.root / "package.zip"
        with self.assertRaises(FileNotFoundError):
            build(self.root, "manuscript", output)
        self.assertFalse(output.exists())

    def test_symlink_input_is_rejected(self):
        (self.root / "paper/linked.tex").symlink_to(self.root / "paper/paper.tex")
        self.profiles({"manuscript": ["paper/*.tex"]})
        with self.assertRaises(ValueError):
            selected_files(self.root, "manuscript")

    def test_unsafe_patterns_and_cycles_are_rejected(self):
        for pattern in ("../outside", "/outside", "@manuscript"):
            with self.subTest(pattern=pattern):
                self.profiles({"manuscript": [pattern]})
                with self.assertRaises(ValueError):
                    selected_files(self.root, "manuscript")


if __name__ == "__main__":
    unittest.main()
