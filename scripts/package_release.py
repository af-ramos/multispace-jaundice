#!/usr/bin/env python3
"""Build a curated ZIP from an explicit allowlist, never from Git history."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def selected_files(root: Path, profile: str) -> list[Path]:
    profiles = json.loads((root / "release/profiles.json").read_text())

    def expand(name: str, parents: tuple = ()) -> set[Path]:
        if name in parents:
            raise ValueError(f"Circular profile: {name}")
        paths = set()
        for pattern in profiles[name]:
            if pattern.startswith("@"):
                paths.update(expand(pattern[1:], parents + (name,)))
                continue
            if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
                raise ValueError(f"Unsafe release pattern: {pattern}")
            matches = sorted(root.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"Required release input missing: {pattern}")
            for path in matches:
                if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
                    raise ValueError(f"Release inputs must be regular files: {path}")
                path.resolve().relative_to(root.resolve())
                paths.add(path)
        return paths

    return sorted(expand(profile))


def build(root: Path, profile: str, output: Path) -> tuple[int, int]:
    files = selected_files(root, profile)
    manifest = {"profile": profile, "algorithm": "sha256", "files": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents replacing a previously prepared submission.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            data = path.read_bytes()
            name = path.relative_to(root).as_posix()
            entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data)
            manifest["files"].append({"path": name, "bytes": len(data),
                                      "sha256": hashlib.sha256(data).hexdigest()})
        entry = zipfile.ZipInfo("MANIFEST.json", date_time=(2026, 1, 1, 0, 0, 0))
        entry.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(entry, json.dumps(manifest, indent=2) + "\n")
    return len(files), output.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("manuscript", "reproducibility", "public"))
    parser.add_argument("--list", action="store_true", help="List inputs without exporting")
    parser.add_argument("--output", type=Path, help="New ZIP path (must not exist)")
    args = parser.parse_args()
    if args.list:
        for path in selected_files(ROOT, args.profile):
            print(path.relative_to(ROOT))
        return
    output = args.output or ROOT / "release/dist" / f"{args.profile}.zip"
    count, size = build(ROOT, args.profile, output)
    print(f"{output}: {count} files + SHA-256 manifest, {size / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
