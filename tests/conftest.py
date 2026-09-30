"""Fixtures e utilitários compartilhados dos testes da Fase 0 (offline, sem GPU).

Todos os testes rodam em CPU. Backbones pré-treinados são carregados do cache local
do timm/torchvision; quando um peso não está em cache e não há rede, o helper
:func:`skip_if_no_weights` pula o teste (mantém o gate offline-limpo)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import BACKBONES, space_nchannels
from src.paths import DATA_ROOT


@pytest.fixture
def dataset_root():
    """Real-image tests are optional in the image-free release."""
    required = (DATA_ROOT / "NJN", DATA_ROOT / "NeoJaundice/images",
                DATA_ROOT / "NeoJaundice/chd_jaundice_published_2.csv")
    if not all(path.exists() for path in required):
        pytest.skip("original images absent; set JAUNDICE_DATA_ROOT for dataset tests")
    return DATA_ROOT

# Backbone CNN pequeno (pesos em cache) + resolução baixa = testes rápidos.
SPEC_CNN = BACKBONES["resnet18"]
RES_CNN = 64
CS_RGB = ["RGB"]
CS_COL = ["RGB", "YCrCb", "HSV"]


def nch(colorspaces, hue_circular: bool = False) -> int:
    return sum(space_nchannels(s, hue_circular) for s in colorspaces)


def rand_stack(colorspaces, batch: int = 2, res: int = RES_CNN, hue_circular: bool = False):
    """Tensor sintético [B, N, res, res] no formato de um Color-Space Stack."""
    return torch.randn(batch, nch(colorspaces, hue_circular), res, res)


def rand_pil(res: int = 32, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(res, res, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def skip_if_no_weights(fn):
    """Executa ``fn()`` e devolve o resultado; pula o teste se faltar peso/rede."""
    try:
        return fn()
    except Exception as e:  # pragma: no cover - depende de rede/cache
        msg = repr(e).lower()
        if any(k in msg for k in ("download", "connection", "url", "http",
                                   "offline", "no such file", "certificate")):
            pytest.skip(f"peso pré-treinado indisponível offline: {e}")
        raise


@pytest.fixture(scope="module")
def model_rgb():
    from src import models
    torch.manual_seed(0)
    return skip_if_no_weights(lambda: models.build_model(
        SPEC_CNN, nch(CS_RGB), fusion="adapter_v2", colorspaces=CS_RGB, num_classes=2))


@pytest.fixture(scope="module")
def model_col():
    from src import models
    torch.manual_seed(0)
    return skip_if_no_weights(lambda: models.build_model(
        SPEC_CNN, nch(CS_COL), fusion="adapter_v2", colorspaces=CS_COL, num_classes=2))
