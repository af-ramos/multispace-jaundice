#!/usr/bin/env python3
"""Auditoria do bloco de fusão em t=0, sem treino: parâmetros por K e paridade entre braços.

Responde a duas perguntas do parecer de 8 de Setembro sobre a §3.4.2:

* quantos parâmetros o bloco acrescenta para cada número K de espaços (BN afim,
  gate SE com largura escondida max(8, 3K//2), convolução 1x1 com bias);
* em que condições foi medida a diferença entre o tensor que entra no backbone no
  braço de quatro espaços e no braço RGB, e quanto vale.

A medição usa um minibatch real: as primeiras 32 imagens de treino do NJN na ordem
do manifesto ``splits/njn_split.csv``, redimensionadas a 256 (a resolução do
MobileNetV3-L na campanha), com as estatísticas de normalização dos espaços não-RGB
calculadas sobre a partição de treino inteira, como na campanha. O bloco corre em
``train()`` (BatchNorm com estatísticas de batch) e em fp32, com ``torch.manual_seed``
fixada antes de cada instanciação, de modo que o ruído sigma=1e-4 das colunas não-RGB
seja reprodutível. Nenhum backbone é carregado.

Uso (no venv de treino, sem GPU):
    python analysis/fusion_block_audit.py  # JAUNDICE_DATA_ROOT selects the image data
Saída: results/fusion_block_audit.json
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.colorspaces import ColorSpaceStack, compute_dataset_colorspace_stats  # noqa: E402
from src.models import ColorFusion  # noqa: E402

from src.paths import DATA_ROOT

NJN_DIR = DATA_ROOT / "NJN"
SPLIT = ROOT / "splits/njn_split.csv"
OUT = ROOT / "results/fusion_block_audit.json"
RESIZE, BATCH, SEED = 256, 32, 42
SPACES = ("RGB", "LAB", "YCrCb", "HSV")
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _path(name: str) -> Path:
    for sub in ("jaundice", "normal"):
        p = NJN_DIR / sub / name
        if p.exists():
            return p
    raise FileNotFoundError(name)


def n_params(fusion: ColorFusion) -> dict:
    parts = {"batchnorm": sum(p.numel() for p in fusion.bn.parameters()),
             "gate": sum(p.numel() for p in fusion.gate.parameters()),
             "conv1x1": sum(p.numel() for p in fusion.proj.parameters())}
    parts["total"] = sum(parts.values())
    return parts


def main() -> int:
    rows = list(csv.DictReader(SPLIT.open()))
    train = [r["image"] for r in rows if r["split"] == "train"]
    train_paths = [_path(n) for n in train]

    # estatísticas por espaço sobre a partição de treino (como a campanha)
    with tempfile.TemporaryDirectory() as tmp:
        link_dir = Path(tmp) / "all"
        link_dir.mkdir()
        for p in train_paths:
            (link_dir / p.name).symlink_to(p)
        stats = compute_dataset_colorspace_stats(tmp, resize=RESIZE, spaces=SPACES[1:])
    stats = {k: (list(v[0]), list(v[1])) for k, v in stats.items()}
    stats["RGB"] = IMAGENET

    batch_paths = train_paths[:BATCH]
    result = {"resize": RESIZE, "batch": BATCH, "seed": SEED, "mode": "train", "dtype": "float32",
              "batch_images": [p.name for p in batch_paths], "stats": stats, "params": {}, "parity": {}}

    def stack(spaces):
        st = ColorSpaceStack(list(spaces), resize=RESIZE, stats=stats, hue_circular=False)
        return torch.stack([st(Image.open(p).convert("RGB")) for p in batch_paths])

    x_rgb = stack(["RGB"])
    for k in range(1, 5):
        spaces = SPACES[:k]
        torch.manual_seed(SEED)
        fusion = ColorFusion([3] * k, init="identity", gated=True, rgb_group=0).train()
        result["params"][k] = n_params(fusion)
        x = stack(spaces)
        with torch.no_grad():
            torch.manual_seed(SEED)
            base = ColorFusion([3], init="identity", gated=True, rgb_group=0).train()
            y_base = base(x_rgb)
            y = fusion(x)
        d = (y - y_base).abs()
        result["parity"][k] = {"max_abs_diff_vs_rgb_arm": float(d.max()),
                              "mean_abs_diff_vs_rgb_arm": float(d.mean()),
                              "max_abs_diff_vs_normalised_rgb": float((y - x_rgb).abs().max()),
                              "channel_std_of_output": [float(s) for s in y.std(dim=(0, 2, 3))]}
    OUT.write_text(json.dumps(result, indent=2))
    for k in range(1, 5):
        print(f"K={k}: params={result['params'][k]['total']:4d}  "
              f"max|multi-rgb|={result['parity'][k]['max_abs_diff_vs_rgb_arm']:.2e}  "
              f"mean={result['parity'][k]['mean_abs_diff_vs_rgb_arm']:.2e}  "
              f"max|fused-rgb_norm|={result['parity'][k]['max_abs_diff_vs_normalised_rgb']:.3f}")
    print(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
