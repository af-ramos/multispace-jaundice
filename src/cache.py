"""
cache.py (NOVO)
===============

**Cache de disco do pré-processamento** (white-balance + ROI central / skin-ROI).

O passo caro do pipeline é ``cv2.imread`` + white-balance + (NJN) extração de
contornos de pele. Sem cache, isso é refeito a cada época (~50 min/treino na v1).
Aqui o resultado — a imagem RGB uint8 já pré-processada — é salvo como ``.npy`` no
**primeiro acesso**; nas épocas seguintes carrega-se direto do disco
(``np.load``), pulando imread/WB/contornos. A augmentação e a conversão de espaços
(``ColorSpaceStack``) seguem aplicadas on-the-fly sobre o array cacheado (a
aleatoriedade da augmentação é preservada).

Chave do cache
--------------
Cada ``(dataset, wb_method, roi, njn_mode, resize, SKIN_ALGO_VERSION)`` é uma
subpasta própria (``cache_tag`` em :class:`config.ExperimentConfig`), e dentro dela
o arquivo é o hash do caminho absoluto da imagem. Como ``SKIN_ALGO_VERSION`` faz
parte da tag, ajustar os limiares de skin invalida o cache antigo automaticamente.

Gestão de disco: ~150 KB por imagem 224x224x3 uint8. Use ``clear_cache.sh`` (ou
``run.sh --clear-cache``) entre grandes rodadas de ablação se o espaço for limitado.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Callable, Optional

import numpy as np


def _key(path: str) -> str:
    return hashlib.sha1(str(Path(path).resolve()).encode("utf-8")).hexdigest()


def cached_npy_path(cache_dir: str | Path, tag: str, src_path: str) -> Path:
    return Path(cache_dir) / tag / f"{_key(src_path)}.npy"


def load_rgb_cached(src_path: str, loader: Callable[[str], np.ndarray],
                    cache_dir: Optional[str | Path], tag: str,
                    use_cache: bool = True) -> np.ndarray:
    """Devolve a imagem RGB uint8 pré-processada, usando/alimentando o cache.

    ``loader(src_path)`` computa a imagem do zero (imread + WB + ROI/skin). Quando
    ``use_cache`` e o ``.npy`` existe, lê do disco; senão computa e grava.
    """
    if not use_cache or cache_dir is None:
        return loader(src_path)
    npy = cached_npy_path(cache_dir, tag, src_path)
    if npy.is_file():
        try:
            return np.load(npy)
        except Exception:  # arquivo corrompido -> recomputa e sobrescreve
            pass
    rgb = loader(src_path)
    npy.parent.mkdir(parents=True, exist_ok=True)
    # Escrita atômica via file-handle (evita que np.save reanexe '.npy' ao nome tmp
    # e impede .npy parcial com workers concorrentes).
    tmp = npy.with_name(npy.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.save(fh, rgb)
    tmp.replace(npy)
    return rgb


def clear_cache(cache_dir: str | Path, tag: Optional[str] = None) -> None:
    """Remove o cache inteiro (``tag=None``) ou apenas a subpasta da ``tag``."""
    cache_dir = Path(cache_dir)
    target = cache_dir if tag is None else cache_dir / tag
    if target.exists():
        shutil.rmtree(target)
