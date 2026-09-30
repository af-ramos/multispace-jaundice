"""
preprocessing.py (v2)
=====================

Pre-processamento **live** das imagens do NeoJaundice (aplicado em
``Dataset.__getitem__``, antes da conversao de espacos de cor):

1. **White-balance pela carta de calibracao** — remove o vies de iluminacao usando
   o ColorChecker presente na borda da imagem. Metodos disponiveis:

   * ``graypatch`` (DEFAULT, recomendado) — von Kries pelos *patches cinza neutros*
     do ColorChecker: mede a cor media das regioes de baixa saturacao e aplica um
     ganho por canal que as neutraliza, **preservando a cromaticidade amarela da
     pele** (o sinal da ictericia).
   * ``grayworld`` — gray-world classico (nao usa a carta); baseline robusto.
   * ``lab`` — fiel ao paper de Makhloughi: neutraliza a/b do maior patch amarelo;
     tende a *suprimir* o amarelo (deixa a pele arroxeada) — disponivel p/ ablacao.
   * ``gain`` — von Kries pelo patch amarelo de referencia.
   * ``off`` — sem white-balance.

2. **Crop geometrico central estrito** (`extract_roi`) — recorta a janela central
   ``[lo, hi]`` (fracao de cada dimensao), **descartando o cartao de calibracao**
   na borda antes que o tensor 2D chegue a rede. Para 567x567 e ``(0.30, 0.70)`` o
   crop e ~227x227, centrado na pele.

Trabalha sempre com arrays **BGR uint8** (convencao OpenCV). Falhas de deteccao da
carta caem em fallback documentado (imagem intacta) + contador inspecionavel.

Portado/condensado de makhloughi_experiments_v2/src/preprocessing.py e
colorspace_experiments/src/make_wb_dataset.py.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("colorspace_v2.preprocessing")

# Faixa de amarelo em HSV (OpenCV: H em [0,179]). Amarelo ~ H 20-40.
_YELLOW_LOW = np.array([20, 80, 80], dtype=np.uint8)
_YELLOW_HIGH = np.array([40, 255, 255], dtype=np.uint8)
_LAB_NEUTRAL = 128.0  # a/b centrados em 128 = sem desvio cromatico (8 bits)
_REF_YELLOW_BGR = np.array([60.0, 200.0, 230.0], dtype=np.float32)  # B,G,R de referencia

# Contadores de fallback (zerados por processo; inspecionaveis nos logs/scripts).
WB_STATS = {"ok": 0, "no_card": 0, "no_neutral": 0}


# --------------------------------------------------------------------------- #
# Deteccao do patch amarelo (metodos baseados na carta)
# --------------------------------------------------------------------------- #
def _largest_yellow_patch_mask(bgr: np.ndarray) -> Optional[np.ndarray]:
    """Mascara binaria do maior contorno amarelo, ou None se nao achar cartao."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _YELLOW_LOW, _YELLOW_HIGH)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 25:
        return None
    patch_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(patch_mask, [largest], -1, color=255, thickness=cv2.FILLED)
    return patch_mask


# --------------------------------------------------------------------------- #
# Metodos de white-balance
# --------------------------------------------------------------------------- #
def white_balance_graypatch(bgr: np.ndarray, s_thresh: int = 35,
                            v_lo: int = 60, v_hi: int = 240) -> np.ndarray:
    """von Kries pelos patches CINZA NEUTROS do ColorChecker (DEFAULT).

    Mede a cor media das regioes de baixa saturacao (cinzas + branco do xadrez,
    excluindo patches coloridos e a pele) e aplica um ganho por canal que as torna
    neutras — removendo o vies de iluminacao e **preservando a cromaticidade da
    pele**. Sem regiao neutra suficiente, devolve a imagem intacta.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    mask = (S < s_thresh) & (V > v_lo) & (V < v_hi)
    if int(mask.sum()) < 50:
        WB_STATS["no_neutral"] += 1
        return bgr
    WB_STATS["ok"] += 1
    means = bgr[mask].reshape(-1, 3).mean(0)          # B,G,R nos neutros
    gain = means.mean() / np.clip(means, 1.0, None)   # iguala os canais
    return np.clip(bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def white_balance_grayworld(bgr: np.ndarray) -> np.ndarray:
    """Gray-world classico (nao usa a carta). Baseline robusto."""
    means = bgr.reshape(-1, 3).mean(axis=0).astype(np.float32)
    means = np.clip(means, 1.0, None)
    gain = np.clip(means.mean() / means, 0.5, 2.0)
    out = bgr.astype(np.float32) * gain.reshape(1, 1, 3)
    WB_STATS["ok"] += 1
    return np.clip(out, 0, 255).astype(np.uint8)


def white_balance_lab(bgr: np.ndarray) -> np.ndarray:
    """FIEL ao paper: neutraliza a/b do patch amarelo p/ cinza (a=b=128), preserva L.

    Tende a suprimir o proprio amarelo da ictericia (pele arroxeada) — usar so p/
    ablacao. Fallback (carta nao detectada): imagem intacta + contador.
    """
    pm = _largest_yellow_patch_mask(bgr)
    if pm is None:
        WB_STATS["no_card"] += 1
        return bgr
    WB_STATS["ok"] += 1
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    _, a_ref, b_ref, _ = cv2.mean(lab, mask=pm)
    lab[:, :, 1] += (_LAB_NEUTRAL - a_ref)
    lab[:, :, 2] += (_LAB_NEUTRAL - b_ref)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def white_balance_gain(bgr: np.ndarray) -> np.ndarray:
    """von Kries pelo patch amarelo de referencia (preserva a direcao amarela)."""
    pm = _largest_yellow_patch_mask(bgr)
    if pm is None:
        WB_STATS["no_card"] += 1
        return bgr
    WB_STATS["ok"] += 1
    obs = np.array(cv2.mean(bgr, mask=pm)[:3], dtype=np.float32)
    obs = np.clip(obs, 1.0, None)
    gain = np.clip(_REF_YELLOW_BGR / obs, 0.5, 2.0)
    out = bgr.astype(np.float32) * gain.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


_WB = {
    "graypatch": white_balance_graypatch,
    "grayworld": white_balance_grayworld,
    "lab": white_balance_lab,
    "gain": white_balance_gain,
    "off": lambda bgr: bgr,
}


def apply_white_balance(bgr: np.ndarray, method: str) -> np.ndarray:
    fn = _WB.get(method)
    if fn is None:
        raise ValueError(f"wb_method desconhecido: {method!r} (use {list(_WB)}).")
    return fn(bgr)


# --------------------------------------------------------------------------- #
# ROI (crop central estrito — descarta o cartao)
# --------------------------------------------------------------------------- #
def extract_roi(bgr: np.ndarray, roi: Tuple[float, float]) -> np.ndarray:
    """Recorta a janela central da pele (fracao ``[lo, hi]`` de cada dimensao).

    Mecanismo geometrico: dada a imagem ``H x W`` e a fracao ``(lo, hi)``,
    ``x0 = round(lo*W), y0 = round(lo*H), x1 = round(hi*W), y1 = round(hi*H)`` e o
    crop e ``bgr[y0:y1, x0:x1]`` — a moldura externa (onde fica o ColorChecker) e
    descartada. Fracoes degeneradas caem para a imagem inteira (+ log).
    """
    h, w = bgr.shape[:2]
    lo, hi = roi
    x0, y0 = int(round(lo * w)), int(round(lo * h))
    x1, y1 = int(round(hi * w)), int(round(hi * h))
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        logger.warning("ROI degenerada (%s); usando imagem inteira.", roi)
        return bgr
    return crop


def preprocess(bgr: np.ndarray, wb_method: str, roi: Tuple[float, float]) -> np.ndarray:
    """Pipeline completo: white-balance (opcional) -> crop ROI central."""
    bgr = apply_white_balance(bgr, wb_method)
    return extract_roi(bgr, roi)
