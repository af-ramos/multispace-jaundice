"""
skin_roi.py (NOVO)
==================

**Extração de ROI de pele para a NJN** (corpo do bebê na incubadora, com fundo de
berço/lençol). Substitui o uso da imagem inteira (com fundo) da v1.

Algoritmo (OpenCV)
------------------
1. :func:`skin_mask_ycrcb_hsv` — máscara binária de pele combinando dois espaços:
   * **YCrCb**: ``Cr ∈ [133,173]`` e ``Cb ∈ [77,127]`` (regra clássica de pele,
     robusta à iluminação por separar luma de croma);
   * **HSV**: matiz de pele ``H ∈ [0,25] ∪ [160,179]`` (faixa ESTENDIDA p/ o
     amarelo/laranja da pele ictérica, que limiares calibrados em pele saudável
     descartariam), com saturação/brilho moderados.
   As duas máscaras são combinadas por AND (pixels que ambos consideram pele),
   seguidas de morfologia OPEN+CLOSE para remover ruído e fechar buracos.

2. :func:`extract_skin_patches` — ``cv2.findContours`` na máscara, ordena por
   ``cv2.contourArea`` e fica com os ``n`` maiores contornos (os patches de pele),
   **descartando o fundo da incubadora**. Recorta o bounding-box que engloba esses
   contornos e zera (preto) os pixels fora da máscara dentro do recorte. Sem pele
   detectada -> fallback documentado (imagem inteira) + contador inspecionável.

A *versão do algoritmo* (:data:`SKIN_ALGO_VERSION`) entra na chave do cache de disco
(:mod:`cache`): ao ajustar os limiares, basta incrementá-la para invalidar máscaras
antigas automaticamente.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("colorspace_final.skin_roi")

# Versão do algoritmo de skin-mask. INCREMENTE ao mexer nos limiares abaixo
# (ex.: "1.1" -> "1.2") para invalidar o cache de disco antigo automaticamente.
SKIN_ALGO_VERSION = "1.1"

# --- Limiares YCrCb (regra clássica de pele) ---
_CR_LO, _CR_HI = 133, 173
_CB_LO, _CB_HI = 77, 127
# --- Limiares HSV (faixa de matiz ESTENDIDA p/ pele ictérica amarela/laranja) ---
_HSV_LOW_1 = np.array([0, 30, 40], dtype=np.uint8)
_HSV_HIGH_1 = np.array([25, 180, 255], dtype=np.uint8)
_HSV_LOW_2 = np.array([160, 30, 40], dtype=np.uint8)
_HSV_HIGH_2 = np.array([179, 180, 255], dtype=np.uint8)

_MIN_CONTOUR_AREA = 200  # px; abaixo disso é ruído, não patch de pele

# Contadores de sucesso/fallback (zerados por processo; inspecionáveis nos logs).
SKIN_STATS = {"ok": 0, "fallback": 0}


def reset_stats() -> None:
    SKIN_STATS["ok"] = 0
    SKIN_STATS["fallback"] = 0


def skin_mask_ycrcb_hsv(bgr: np.ndarray) -> np.ndarray:
    """Máscara binária (uint8 0/255) de pele combinando YCrCb e HSV."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    mask_ycrcb = ((cr >= _CR_LO) & (cr <= _CR_HI) &
                  (cb >= _CB_LO) & (cb <= _CB_HI)).astype(np.uint8) * 255

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.bitwise_or(cv2.inRange(hsv, _HSV_LOW_1, _HSV_HIGH_1),
                              cv2.inRange(hsv, _HSV_LOW_2, _HSV_HIGH_2))

    mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _largest_contours(mask: np.ndarray, n: int) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= _MIN_CONTOUR_AREA]
    contours.sort(key=cv2.contourArea, reverse=True)
    return contours[:n]


def extract_skin_patches(bgr: np.ndarray, n: int = 3,
                         return_bbox: bool = False
                         ) -> "np.ndarray | Tuple[np.ndarray, Tuple[int, int, int, int]]":
    """Recorta a região de pele (n maiores contornos), descartando o fundo.

    Retorna o crop BGR (fundo fora da máscara zerado). Com ``return_bbox=True``,
    devolve ``(crop, (x0, y0, x1, y1))`` — a bbox no sistema de coordenadas da
    imagem ORIGINAL, usada para reprojetar heatmaps sobre a foto inteira.

    Fallback (sem pele detectada): devolve a imagem inteira (+ contador), bbox = full.
    """
    h, w = bgr.shape[:2]
    full_bbox = (0, 0, w, h)
    mask = skin_mask_ycrcb_hsv(bgr)
    contours = _largest_contours(mask, n)
    if not contours:
        SKIN_STATS["fallback"] += 1
        return (bgr, full_bbox) if return_bbox else bgr

    SKIN_STATS["ok"] += 1
    keep = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(keep, contours, -1, color=255, thickness=cv2.FILLED)

    xs, ys = np.where(keep > 0)
    y0, y1 = int(xs.min()), int(xs.max()) + 1
    x0, x1 = int(ys.min()), int(ys.max()) + 1

    masked = cv2.bitwise_and(bgr, bgr, mask=keep)  # zera fundo fora dos patches
    crop = masked[y0:y1, x0:x1]
    if crop.size == 0:
        SKIN_STATS["fallback"] += 1
        return (bgr, full_bbox) if return_bbox else bgr
    bbox = (x0, y0, x1, y1)
    return (crop, bbox) if return_bbox else crop


def fallback_rate() -> float:
    """Taxa de fallback (0..1) acumulada nos contadores."""
    total = SKIN_STATS["ok"] + SKIN_STATS["fallback"]
    return (SKIN_STATS["fallback"] / total) if total else 0.0


# --------------------------------------------------------------------------- #
# Sanity check visual (mosaico de máscaras) — conferência humana antes do treino
# --------------------------------------------------------------------------- #
def save_sanity_mosaic(image_paths: List[str], out_path: str | Path, n: int = 20,
                       n_patches: int = 3, seed: int = 0, thumb: int = 160) -> Path:
    """Salva um mosaico de ``n`` amostras (original | máscara | ROI extraída).

    Permite bater o olho e garantir que a rede vai treinar com pele de verdade —
    não com pedaços do berço/lençol. Retorna o caminho do PNG salvo.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    chosen = list(image_paths)
    rng.shuffle(chosen)
    chosen = chosen[:n]

    rows: List[np.ndarray] = []
    for p in chosen:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        mask = skin_mask_ycrcb_hsv(bgr)
        roi = extract_skin_patches(bgr, n=n_patches)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        trio = [cv2.resize(bgr, (thumb, thumb)),
                cv2.resize(mask_bgr, (thumb, thumb)),
                cv2.resize(roi, (thumb, thumb))]
        rows.append(np.hstack(trio))
    if not rows:
        raise RuntimeError("Nenhuma imagem válida para o mosaico de sanity check.")
    mosaic = np.vstack(rows)
    cv2.imwrite(str(out_path), mosaic)
    logger.info("Mosaico de sanity check salvo em %s (%d amostras)", out_path, len(rows))
    return out_path
