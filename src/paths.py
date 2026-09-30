"""Portable locations for optional image-based analyses and retraining."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("JAUNDICE_DATA_ROOT", ROOT / "dataset")).expanduser().resolve()


def image_path(recorded_path: str) -> Path:
    """Rebase historical absolute/relative image names onto the selected dataset."""
    parts = str(recorded_path).replace("\\", "/").split("/")
    for dataset in ("NJN", "NeoJaundice"):
        if dataset in parts:
            return DATA_ROOT.joinpath(*parts[parts.index(dataset):])
    raise ValueError(f"Image path does not identify NJN or NeoJaundice: {recorded_path}")
