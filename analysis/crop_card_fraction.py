#!/usr/bin/env python3
"""Quantifica o cartão de calibração que entra no recorte central da NeoJaundice.

Complementa `analysis/crop_audit.py`, cuja contagem a olho sobre miniaturas de 150 px
da imagem inteira subestimava os resíduos: uma borda de 1–5 % do recorte vira 1–2 px.
Aqui mede-se, no recorte [0,30; 0,70] em resolução original, a fração de pixels com cor
de cartão — azul/verde/ciano saturados, amarelo e magenta muito saturados, preto e
branco —, que a pele não produz. As sombras de pregas profundas podem somar ~1 %.

Saídas, ambas em ``results/``:
  * ``crop_audit.csv``: a amostra auditada (mesmo sorteio de ``crop_audit.py``:
    120 imagens, semente 20260913), uma linha por imagem, com a fração em %.
  * ``crop_audit_summary.json``: contagens por limiar na amostra e, na colecção
    inteira, a AUC da fração de cartão como preditor do rótulo (TSB >= 12,9 mg/dL).

Requer OpenCV (ambiente de treino) e as imagens originais em ``JAUNDICE_DATA_ROOT``.

Uso:
    JAUNDICE_DATA_ROOT=/caminho/dataset python analysis/crop_card_fraction.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import TSB_BINARY_THRESHOLD
from src.paths import DATA_ROOT

NEO = DATA_ROOT / "NeoJaundice"
ROI = (0.30, 0.70)
SAMPLE_N, SAMPLE_SEED = 120, 20260913
THRESHOLDS = (0.5, 1, 3, 5, 10, 25, 50)


def card_fraction(path: Path) -> float:
    """Percentagem do recorte central com cor de cartão (limiares HSV do OpenCV)."""
    image = cv2.imread(str(path))
    h, w = image.shape[:2]
    crop = image[int(ROI[0] * h):int(ROI[1] * h), int(ROI[0] * w):int(ROI[1] * w)]
    hue, sat, val = (c.astype(int) for c in cv2.split(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)))
    card = (((hue >= 35) & (hue <= 135) & (sat > 90) & (val > 60))            # azul/verde/ciano
            | ((hue >= 22) & (hue < 35) & (sat > 150) & (val > 150))          # amarelo
            | (((hue >= 140) | (hue <= 4)) & (sat > 170) & (val > 120))       # magenta/vermelho
            | (val < 45)                                                      # preto
            | ((sat < 25) & (val > 215)))                                     # branco
    return 100.0 * float(card.mean())


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC pela estatística de Mann–Whitney, empates por posto médio."""
    order = scores.argsort()
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):
        tied = scores == value
        ranks[tied] = ranks[tied].mean()
    pos, neg = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main() -> int:
    rows = list(csv.DictReader((NEO / "chd_jaundice_published_2.csv").open(encoding="utf-8-sig")))
    names = sorted(r["image_idx"] for r in rows)
    sample = random.Random(SAMPLE_SEED).sample(names, SAMPLE_N)   # o sorteio de crop_audit.py

    fraction = {r["image_idx"]: card_fraction(NEO / "images" / r["image_idx"]) for r in rows}
    labels = np.array([float(r["blood(mg/dL)"]) >= TSB_BINARY_THRESHOLD for r in rows])
    scores = np.array([fraction[r["image_idx"]] for r in rows])

    out = ROOT / "results"
    with (out / "crop_audit.csv").open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter=";", lineterminator="\n")
        writer.writerow(["order", "image", "card_pct"])
        for i, name in enumerate(sample, 1):
            writer.writerow([i, name, f"{fraction[name]:.2f}"])

    in_sample = np.array([fraction[n] for n in sample])
    summary = {
        "sample": {"n": SAMPLE_N, "seed": SAMPLE_SEED,
                   "above_pct": {str(t): int((in_sample > t).sum()) for t in THRESHOLDS},
                   "max_pct": round(float(in_sample.max()), 2)},
        "collection": {"n": len(rows),
                       "share_above_1pct": round(float((scores > 1).mean()), 4),
                       "share_above_5pct": round(float((scores > 5).mean()), 4),
                       "auc_card_fraction_vs_label": round(auc(scores, labels), 4)},
    }
    (out / "crop_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
