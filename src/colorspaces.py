"""
colorspaces.py
==============

Componente central da metodologia: o **Color-Space Stack**.

Ideia
-----
A cor da pele e o melhor indicador clinico de icterícia neonatal. O espaço RGB
mistura luminância e cromância nos três canais, o que pode dificultar a separação
do tom amarelado. Espaços perceptuais/luminância-cromância isolam a cor:

* **LAB**   — L (luminância) + a* (verde↔vermelho) + b* (azul↔**amarelo**);
              o canal b* é diretamente sensível ao amarelado da icterícia.
* **YCrCb** — Y (luma) + Cr (croma vermelho) + Cb (croma azul); separa luma de croma.
* **HSV**   — Hue (matiz) + Saturation + Value; o matiz captura a cor independente
              do brilho/iluminação.

A hipótese do artigo é que fornecer ao backbone uma **pilha de múltiplos espaços**
(ou um espaço isolado) altera — positiva ou negativamente — a performance. Por isso
este módulo converte uma imagem RGB para qualquer subconjunto de
{RGB, LAB, YCrCb, HSV} e os empilha em um tensor ``[N, H, W]`` com
``N = 3 × nº de espaços``.

Normalização
------------
Cada espaço é normalizado com **suas próprias estatísticas**:

* **RGB** usa as estatísticas do ImageNet (mean/std padrão), preservando a
  compatibilidade com os pesos pré-treinados do backbone.
* **LAB / YCrCb / HSV** usam média/desvio calculados no *split de treino* do
  dataset (função :func:`compute_dataset_colorspace_stats`), pois suas faixas
  numéricas (ex.: Hue do OpenCV vai de 0–179) diferem do RGB. As estatísticas são
  cacheadas em JSON e reaproveitadas por qualquer subconjunto.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile

from .config import canonical_colorset, colorset_id, space_nchannels

# OpenCV usado dentro dos workers `fork` do DataLoader: o pool de threads nativo do
# OpenCV (OpenMP/TBB) conflita com o fork do multiprocessing e causa segfaults
# nativos intermitentes (worker morto por SIGSEGV -> unidade pulada ou deadlock do
# loader). Desligar o paralelismo interno do OpenCV elimina o conflito; cada worker
# ja roda em seu proprio processo, entao nao ha perda de throughput relevante.
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# Imagens truncadas/parcialmente corrompidas nao derrubam o decodificador.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Estatisticas do ImageNet (usadas para o RGB).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Conversoes OpenCV para os espacos nao-RGB (a entrada e sempre RGB uint8).
CV2_CONVERSIONS = {
    "LAB": cv2.COLOR_RGB2LAB,
    "YCrCb": cv2.COLOR_RGB2YCrCb,
    "HSV": cv2.COLOR_RGB2HSV,
}


# ---------------------------------------------------------------------------
# Banco de cor fisiologico (v6): features de cor AUTO-NORMALIZADAS (saida ~[-1,1]),
# motivadas pela clinica da ictericia e por invariancia a iluminacao. Sao "stat-free"
# — NAO usam media/desvio do dataset (como o path sin/cos do Hue circular), pois ja
# nascem numa faixa comparavel. Cada funcao recebe RGB uint8 (H,W,3) e devolve
# ``np.ndarray`` [C,H,W] float32. O nº de canais e declarado em config.SPACE_CHANNELS.
# ---------------------------------------------------------------------------
_LAB_L_SCALE = 100.0 / 255.0   # OpenCV LAB: L em [0,255] mapeia [0,100]


def _lab_star(np_rgb_uint8: np.ndarray):
    """(L*, a*, b*) fisicos a partir do LAB do OpenCV (L*∈[0,100]; a*,b*∈~[-128,127])."""
    lab = cv2.cvtColor(np_rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0] * _LAB_L_SCALE
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    return L, a, b


def _space_bstar(np_rgb_uint8: np.ndarray) -> np.ndarray:
    """b* do CIELAB (eixo azul↔amarelo) normalizado; amarelo (ictericia) > 0. [1,H,W]."""
    _, _, b = _lab_star(np_rgb_uint8)
    return (b / 128.0).astype(np.float32)[None]


def _space_ita(np_rgb_uint8: np.ndarray) -> np.ndarray:
    """Individual Typology Angle atan2(L*−50, b*) normalizado por π. [1,H,W].

    Medida dermatologica de tom de pele; o amarelamento da ictericia aumenta b* e
    rotaciona o angulo, isolando o desvio relativo ao tom de pele basal."""
    L, _, b = _lab_star(np_rgb_uint8)
    ita = np.arctan2(L - 50.0, b) / np.pi
    return ita.astype(np.float32)[None]


def _space_opp(np_rgb_uint8: np.ndarray) -> np.ndarray:
    """Cromaticidade oponente normalizada (rg, yb) — robusta a intensidade. [2,H,W]."""
    rgb = np_rgb_uint8.astype(np.float32) / 255.0
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    denom = R + G + B + 1e-6
    rg = (R - G) / denom
    yb = 0.5 * (R + G - 2.0 * B) / denom      # amarelo↔azul; amarelo > 0
    return np.stack([rg, yb], axis=0).astype(np.float32)


def _space_logchroma(np_rgb_uint8: np.ndarray) -> np.ndarray:
    """log(R/G), log(B/G) normalizados — aprox. invariante ao iluminante. [2,H,W].

    Alvo direto do cenario NJN (nunca calibrado): remove o ganho multiplicativo da
    iluminacao, deixando a cromaticidade intrinseca da pele."""
    rgb = np_rgb_uint8.astype(np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    scale = float(np.log(256.0))
    lr = np.log((R + 1.0) / (G + 1.0)) / scale
    lb = np.log((B + 1.0) / (G + 1.0)) / scale
    return np.stack([lr, lb], axis=0).astype(np.float32)


# Registro nome -> funcao (self-normalized). Nº de canais vem de config.SPACE_CHANNELS.
CUSTOM_SPACES = {
    "BSTAR": _space_bstar,
    "ITA": _space_ita,
    "OPP": _space_opp,
    "LOGCHROMA": _space_logchroma,
}


def _space_to_tensor_raw(np_rgb_uint8: np.ndarray, space: str) -> torch.Tensor:
    """Converte uma imagem RGB (H,W,3 uint8) para ``space`` -> tensor [3,H,W] em [0,1].

    Todos os espacos sao escalados por 1/255 para uma faixa comparavel; a
    padronizacao fina (mean/std) e aplicada depois, por espaco.
    """
    if space == "RGB":
        arr = np_rgb_uint8.astype(np.float32) / 255.0
    else:
        conv = cv2.cvtColor(np_rgb_uint8, CV2_CONVERSIONS[space]).astype(np.float32) / 255.0
        arr = conv
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class ColorSpaceStack:
    """Transform: PIL RGB -> tensor ``[N,H,W]`` com os espacos pedidos, normalizados.

    Parameters
    ----------
    spaces : sequencia de espacos (ex.: ("RGB", "LAB")). Ordenada canonicamente.
    resize : lado da imagem quadrada de saida.
    stats  : dict {space: (mean3, std3)} com estatisticas de normalizacao. RGB usa
             ImageNet por padrao se ausente.
    hue_circular : se True, o canal H do HSV (circular, 0-179 no OpenCV) e
             codificado como (sin H, cos H) — remove a descontinuidade artificial
             no wrap 0<->180 (regiao de tons de pele). O HSV passa a contribuir
             com 4 canais: (sinH, cosH, S, V). sin/cos ja sao zero-centrados em
             [-1,1] e nao usam estatisticas do dataset; S e V seguem normalizados
             com as estatisticas de treino (indices 1 e 2 do HSV).
    """

    def __init__(self, spaces: Sequence[str], resize: int,
                 stats: Dict[str, Tuple[Sequence[float], Sequence[float]]] | None = None,
                 hue_circular: bool = False):
        self.spaces: Tuple[str, ...] = canonical_colorset(spaces)
        self.resize = resize
        self.hue_circular = hue_circular
        stats = dict(stats or {})
        stats.setdefault("RGB", (IMAGENET_MEAN, IMAGENET_STD))
        # Pre-monta tensores de mean/std por espaco (só para espacos padrao 3-canais;
        # os do banco fisiologico em CUSTOM_SPACES sao auto-normalizados, sem stats).
        self._mean: Dict[str, torch.Tensor] = {}
        self._std: Dict[str, torch.Tensor] = {}
        for sp in self.spaces:
            if sp in CUSTOM_SPACES:
                continue
            mean, std = stats.get(sp, ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            self._mean[sp] = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
            self._std[sp] = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    @property
    def n_channels(self) -> int:
        return sum(space_nchannels(sp, self.hue_circular) for sp in self.spaces)

    def _hsv_circular(self, np_rgb_uint8: np.ndarray) -> torch.Tensor:
        """HSV com Hue circular -> tensor [4,H,W]: (sinH, cosH, S_norm, V_norm)."""
        hsv = cv2.cvtColor(np_rgb_uint8, cv2.COLOR_RGB2HSV)  # H em [0,179] uint8
        theta = hsv[..., 0].astype(np.float32) * (2.0 * np.pi / 180.0)
        sv = hsv[..., 1:].astype(np.float32) / 255.0          # S, V em [0,1]
        mean, std = self._mean["HSV"], self._std["HSV"]
        sv_t = torch.from_numpy(sv).permute(2, 0, 1).contiguous()
        sv_t = (sv_t - mean[1:]) / std[1:]
        sincos = torch.from_numpy(np.stack([np.sin(theta), np.cos(theta)], axis=0))
        return torch.cat([sincos, sv_t], dim=0)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = TF.resize(img, [self.resize, self.resize])
        if img.mode != "RGB":
            img = img.convert("RGB")
        np_rgb = np.asarray(img, dtype=np.uint8)
        channels: List[torch.Tensor] = []
        for sp in self.spaces:
            if sp == "HSV" and self.hue_circular:
                channels.append(self._hsv_circular(np_rgb))
                continue
            if sp in CUSTOM_SPACES:                          # banco fisiologico (self-normalized)
                arr = CUSTOM_SPACES[sp](np_rgb)              # [C,H,W] float32, ~[-1,1]
                channels.append(torch.from_numpy(np.ascontiguousarray(arr)))
                continue
            t = _space_to_tensor_raw(np_rgb, sp)
            t = (t - self._mean[sp]) / self._std[sp]
            channels.append(t)
        return torch.cat(channels, dim=0)  # [N,H,W]


# ---------------------------------------------------------------------------
# Estatisticas de normalizacao por espaco (calculadas no split de treino)
# ---------------------------------------------------------------------------
def compute_dataset_colorspace_stats(train_dir: str | Path, resize: int = 224,
                                     spaces: Sequence[str] = ("LAB", "YCrCb", "HSV"),
                                     max_images: int | None = None
                                     ) -> Dict[str, Tuple[List[float], List[float]]]:
    """Media/desvio por canal de cada espaco, sobre as imagens do treino.

    RGB nao e calculado aqui (usa ImageNet). Percorre ``train_dir/<classe>/*``.
    """
    train_dir = Path(train_dir)
    paths: List[Path] = []
    for cls_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        for f in sorted(os.listdir(cls_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(cls_dir / f)
    if max_images is not None and len(paths) > max_images:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(paths), size=max_images, replace=False)
        paths = [paths[i] for i in sorted(idx)]

    # Só espacos com conversao OpenCV precisam de stats de dataset; RGB usa ImageNet e
    # o banco fisiologico (CUSTOM_SPACES) e auto-normalizado.
    spaces = [s for s in spaces if s in CV2_CONVERSIONS]
    # Acumuladores de soma e soma dos quadrados por canal.
    n_pixels = 0
    ssum = {s: np.zeros(3, dtype=np.float64) for s in spaces}
    ssq = {s: np.zeros(3, dtype=np.float64) for s in spaces}

    for p in paths:
        img = Image.open(p).convert("RGB")
        img = TF.resize(img, [resize, resize])
        np_rgb = np.asarray(img, dtype=np.uint8)
        for s in spaces:
            arr = cv2.cvtColor(np_rgb, CV2_CONVERSIONS[s]).astype(np.float64) / 255.0
            ssum[s] += arr.reshape(-1, 3).sum(axis=0)
            ssq[s] += (arr.reshape(-1, 3) ** 2).sum(axis=0)
        n_pixels += np_rgb.shape[0] * np_rgb.shape[1]

    stats: Dict[str, Tuple[List[float], List[float]]] = {}
    for s in spaces:
        mean = ssum[s] / n_pixels
        var = np.maximum(ssq[s] / n_pixels - mean ** 2, 1e-8)
        stats[s] = (mean.tolist(), np.sqrt(var).tolist())
    return stats


def get_colorspace_stats(dataset: str, train_dir: str | Path, resize: int,
                         cache_dir: str | Path = "local/runs/stats"
                         ) -> Dict[str, Tuple[List[float], List[float]]]:
    """Carrega (ou calcula+cacheia) as estatisticas LAB/YCrCb/HSV de um dataset.

    Inclui sempre o RGB (ImageNet). A cache e por (dataset, resize), valida para
    qualquer subconjunto de espacos.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{dataset}_r{resize}.json"
    if cache.exists():
        data = json.loads(cache.read_text())
        stats = {k: (tuple(v["mean"]), tuple(v["std"])) for k, v in data.items()}
    else:
        computed = compute_dataset_colorspace_stats(train_dir, resize=resize,
                                                    spaces=("LAB", "YCrCb", "HSV"))
        data = {k: {"mean": m, "std": s} for k, (m, s) in computed.items()}
        cache.write_text(json.dumps(data, indent=2))
        stats = {k: (tuple(v["mean"]), tuple(v["std"])) for k, v in data.items()}
    stats["RGB"] = (IMAGENET_MEAN, IMAGENET_STD)
    return stats


def describe_colorset(spaces: Sequence[str]) -> str:
    """Texto curto descrevendo o subconjunto (para logs/relatorio)."""
    spaces = canonical_colorset(spaces)
    return f"{colorset_id(spaces)} ({3 * len(spaces)} canais)"


# ---------------------------------------------------------------------------
# Estatisticas a partir de um ITERAVEL de imagens RGB ja pre-processadas
# (necessario no pipeline 'live': WB+ROI/skin sao aplicados antes de medir).
# ---------------------------------------------------------------------------
def compute_stats_from_iter(iter_rgb, resize: int = 224,
                            spaces: Sequence[str] = ("LAB", "YCrCb", "HSV")
                            ) -> Dict[str, Tuple[List[float], List[float]]]:
    """Media/desvio por canal de cada espaco, sobre um iteravel de RGB uint8.

    Cada imagem e redimensionada para ``resize`` (coerente com o treino) antes da
    conversao. RGB (ImageNet) e o banco fisiologico (auto-normalizado) nao sao medidos."""
    spaces = [s for s in spaces if s in CV2_CONVERSIONS]
    n_pixels = 0
    ssum = {s: np.zeros(3, dtype=np.float64) for s in spaces}
    ssq = {s: np.zeros(3, dtype=np.float64) for s in spaces}
    for rgb in iter_rgb:
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.shape[0] != resize or rgb.shape[1] != resize:
            rgb = cv2.resize(rgb, (resize, resize))
        for s in spaces:
            arr = cv2.cvtColor(rgb, CV2_CONVERSIONS[s]).astype(np.float64) / 255.0
            ssum[s] += arr.reshape(-1, 3).sum(axis=0)
            ssq[s] += (arr.reshape(-1, 3) ** 2).sum(axis=0)
        n_pixels += rgb.shape[0] * rgb.shape[1]
    if n_pixels == 0:
        raise ValueError("compute_stats_from_iter: iteravel vazio.")
    stats: Dict[str, Tuple[List[float], List[float]]] = {}
    for s in spaces:
        mean = ssum[s] / n_pixels
        var = np.maximum(ssq[s] / n_pixels - mean ** 2, 1e-8)
        stats[s] = (mean.tolist(), np.sqrt(var).tolist())
    stats["RGB"] = (list(IMAGENET_MEAN), list(IMAGENET_STD))
    return stats


def get_stats(cache_path: str | Path, iter_fn, resize: int,
              spaces: Sequence[str] = ("LAB", "YCrCb", "HSV")
              ) -> Dict[str, Tuple[Sequence[float], Sequence[float]]]:
    """Carrega (ou calcula+cacheia) as estatisticas a partir de ``iter_fn()``.

    ``cache_path`` ja inclui dataset/wb/roi/njn/resize (config.stats_path). Valido
    para qualquer subconjunto de espacos (todos os nao-RGB sao medidos)."""
    cache_path = Path(cache_path)
    if cache_path.is_file():
        data = json.loads(cache_path.read_text())
        stats = {k: (tuple(v["mean"]), tuple(v["std"])) for k, v in data.items()}
    else:
        computed = compute_stats_from_iter(iter_fn(), resize=resize, spaces=spaces)
        data = {k: {"mean": list(m), "std": list(s)} for k, (m, s) in computed.items()}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, indent=2))
        stats = {k: (tuple(v["mean"]), tuple(v["std"])) for k, v in data.items()}
    stats["RGB"] = (IMAGENET_MEAN, IMAGENET_STD)
    return stats
