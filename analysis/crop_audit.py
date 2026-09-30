#!/usr/bin/env python3
"""Folha de contacto para auditar o recorte central [0,30; 0,70] da NeoJaundice.

O parecer de 8 de Setembro pede evidência de que uma janela fixa retém pele e exclui
o cartão de calibração. Este script amostra N imagens da colecção com semente fixa,
desenha o rectângulo do recorte sobre cada uma e grava uma folha de contacto para
inspecção visual; não treina nem altera nada. A contagem de falhas é feita a olho e
registada no manuscrito. Miniaturas de 150 px da imagem inteira escondem as bordas finas
do cartão dentro do recorte; a contagem quantitativa está em `analysis/crop_card_fraction.py`.

Uso:
    python analysis/crop_audit.py --n 120 --seed 20260913 --out /tmp/crop_audit.png
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import DATA_ROOT

NEO = DATA_ROOT / "NeoJaundice"
ROI = (0.30, 0.70)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--thumb", type=int, default=150)
    ap.add_argument("--cols", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader((NEO / "chd_jaundice_published_2.csv").open()))
    names = sorted(r["image_idx"] for r in rows)
    rng = random.Random(args.seed)
    sample = rng.sample(names, args.n)
    t, cols = args.thumb, args.cols
    rows_n = -(-args.n // cols)
    sheet = Image.new("RGB", (cols * t, rows_n * t), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, name in enumerate(sample):
        im = Image.open(NEO / "images" / name).convert("RGB")
        w, h = im.size
        thumb = im.resize((t, t))
        d = ImageDraw.Draw(thumb)
        d.rectangle([ROI[0] * t, ROI[0] * t, ROI[1] * t, ROI[1] * t], outline=(255, 0, 0), width=2)
        d.text((3, 3), name.replace(".jpg", ""), fill=(255, 255, 0), font=font)
        sheet.paste(thumb, ((i % cols) * t, (i // cols) * t))
    sheet.save(args.out)
    (Path(args.out).with_suffix(".txt")).write_text("\n".join(sample))
    print(f"{args.n} imagens, semente {args.seed}: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
