"""
data.py
=======

Construção dos ``DataLoader`` de treino/validação/teste, unindo:

* **subconjuntos de espaço de cor** (v1) via :class:`colorspaces.ColorSpaceStack`;
* **pré-processamento live** (v2): white-balance pela carta + crop ROI central
  (NeoJaundice) — e a **extração de ROI de pele** (NJN, novo);
* **cache de disco** do RGB pré-processado (:mod:`cache`) — 1ª época grava, demais
  carregam direto;
* **split agrupado por paciente** in-memory (:mod:`splits`), com ``patient_id`` real
  (NeoJaundice) ou pseudo (NJN, pHash) anexado a cada amostra para as métricas
  nível-paciente.

A augmentação geométrica/fotométrica é aplicada no domínio RGB (PIL->PIL) ANTES da
conversão de espaços, garantindo coerência entre os canais empilhados.

O ``loader`` de imagem é uma ``functools.partial`` de função de módulo (picklável)
para funcionar com os workers ``spawn`` do DataLoader.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from . import cache as cache_mod
from . import skin_roi, splits
from .colorspaces import ColorSpaceStack, IMAGENET_MEAN, IMAGENET_STD, compute_stats_from_iter, get_stats
from .config import ExperimentConfig
from .preprocessing import preprocess as neo_preprocess

# OpenCV com thread-pool/OpenCL ATIVO num processo que depois forka workers é uma
# fonte clássica de segfault (o thread-pool nativo é clonado em estado inconsistente).
# Desligamos já no IMPORT do módulo — que roda tanto no processo pai quanto no
# servidor 'forkserver'/worker 'spawn' — para que NENHUM fork herde threads do OpenCV.
# (O _worker_init reforça, mas ali já é tarde para o fork inicial.)
cv2.setNumThreads(0)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


def _worker_init(worker_id: int) -> None:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    # O worker e o PRODUTOR dos tensores compartilhados; sua estrategia decide como
    # eles sao passados ao processo principal. 'file_system' evita o acumulo de
    # descritores que estoura em "Too many open files" ao longo do HPO.
    try:
        import torch.multiprocessing as _mp
        _mp.set_sharing_strategy("file_system")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Loader de imagem (picklável p/ workers spawn): imread -> WB+ROI / skin -> RGB
# --------------------------------------------------------------------------- #
def _read_bgr(path: str) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Falha ao ler {path}")
    return bgr


def _load_rgb(path: str, *, kind: str, wb_method: str, roi: Tuple[float, float],
              njn_mode: str, n_patches: int) -> np.ndarray:
    """Carrega + pré-processa uma imagem, devolvendo RGB uint8.

    * NeoJaundice (``kind='neojaundice_live'``): white-balance + crop ROI central.
    * NJN ``skin_roi``: extração de ROI de pele (n maiores contornos).
    * NJN ``full_image``: imagem inteira (sem máscara).
    """
    bgr = _read_bgr(path)
    if kind == "neojaundice_live":
        bgr = neo_preprocess(bgr, wb_method=wb_method, roi=roi)
    elif njn_mode == "skin_roi":
        bgr = skin_roi.extract_skin_patches(bgr, n=n_patches)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class _StackDataset(Dataset):
    """Entrega ``(tensor[N,H,W], label)`` com cache de disco e patient ids.

    ``items`` = lista ``(path, label, patient_id)``. Expõe ``.samples`` (=[(path,
    label)]) e ``.patient_ids`` (alinhado), p/ sampler e métricas nível-paciente.
    """

    def __init__(self, items: List[Tuple[str, int, str]], classes: Sequence[str],
                 transform, loader: Callable[[str], np.ndarray],
                 cache_dir: Optional[str], cache_tag: str, use_cache: bool,
                 tsb_norm: Optional[Sequence[float]] = None, return_tsb: bool = False):
        self.items = list(items)
        self.classes = tuple(classes)
        self.transform = transform
        self.loader = loader
        self.cache_dir = cache_dir
        self.cache_tag = cache_tag
        self.use_cache = use_cache
        # Multitarefa: alvo TSB ja NORMALIZADO (com stats do TREINO do fold). O cv.py
        # des-normaliza as predicoes com as mesmas stats antes de MAE/R² (simetria).
        self.tsb_norm = list(tsb_norm) if tsb_norm is not None else None
        self.return_tsb = bool(return_tsb and tsb_norm is not None)

    @property
    def samples(self) -> List[Tuple[str, int]]:
        return [(p, l) for p, l, _ in self.items]

    @property
    def patient_ids(self) -> List[str]:
        return [pid for _, _, pid in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def _rgb(self, path: str) -> np.ndarray:
        return cache_mod.load_rgb_cached(path, self.loader, self.cache_dir,
                                         self.cache_tag, self.use_cache)

    def __getitem__(self, idx: int):
        path, label, _ = self.items[idx]
        rgb = self._rgb(path)
        img = self.transform(Image.fromarray(rgb))
        if self.return_tsb:
            return img, label, float(self.tsb_norm[idx])
        return img, label

    def iter_rgb(self) -> Iterable[np.ndarray]:
        for path, _, _ in self.items:
            yield self._rgb(path)


# --------------------------------------------------------------------------- #
# Augmentação / transforms / sampler
# --------------------------------------------------------------------------- #
def build_augment_pipeline(resize: int, strength: float = 1.0) -> List[object]:
    s = max(0.0, strength)
    return [
        transforms.RandomResizedCrop(resize, scale=(1.0 - 0.2 * s, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15 * s),
        transforms.ColorJitter(brightness=0.2 * s, contrast=0.2 * s,
                               saturation=0.2 * s, hue=0.1 * s),
    ]


def build_transforms(cfg: ExperimentConfig, resize: int, augment: bool,
                     da_strength: float, stats):
    stack = ColorSpaceStack(cfg.colorspaces, resize=resize, stats=stats,
                            hue_circular=cfg.hue_circular)
    eval_tf = transforms.Compose([stack])
    if augment:
        train_tf = transforms.Compose(build_augment_pipeline(resize, da_strength) + [stack])
    else:
        train_tf = eval_tf
    return train_tf, eval_tf


def _balanced_sampler(ds: _StackDataset, generator) -> WeightedRandomSampler:
    counts = torch.zeros(len(ds.classes), dtype=torch.float64)
    for _, label in ds.samples:
        counts[label] += 1
    counts = torch.clamp(counts, min=1.0)
    class_w = 1.0 / counts
    sample_w = torch.tensor([class_w[label] for _, label in ds.samples], dtype=torch.double)
    return WeightedRandomSampler(sample_w, num_samples=len(ds.samples),
                                 replacement=True, generator=generator)


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    n_channels: int
    classes: Tuple[str, ...]
    sizes: Tuple[int, int, int]
    path_to_pid: Dict[str, str]
    # Multitarefa (TSB): stats do TREINO do fold p/ des-normalizar predicoes.
    tsb_mean: float = 0.0
    tsb_std: float = 1.0


def _limit(items, n):
    return items if not n else items[:n]


def _make_loader(cfg: ExperimentConfig) -> Callable[[str], np.ndarray]:
    return partial(_load_rgb, kind=cfg.dataset_spec.kind, wb_method=cfg.wb_method,
                   roi=cfg.roi, njn_mode=cfg.njn_mode, n_patches=cfg.skin_n_patches)


def prepare_njn_skin(cfg: ExperimentConfig, train_paths: Sequence[str]) -> None:
    """Sanity check visual do skin-mask (NJN skin_roi) — mosaico de 20, gerado 1x.

    Roda no processo PRINCIPAL. A taxa de FALLBACK e medida sem leitura extra:
    :func:`report_skin_fallback` aproveita a passada de estatisticas (que ja decodifica
    + extrai pele de cada imagem de treino), em vez de um scan dedicado.
    """
    if cfg.is_neojaundice or cfg.njn_mode != "skin_roi":
        return
    sanity_png = Path(cfg.results_dir) / cfg.dataset / "skin_sanity" / "skin_mask_mosaic.png"
    if not sanity_png.exists():
        try:
            skin_roi.save_sanity_mosaic(list(train_paths), sanity_png, n=20,
                                        n_patches=cfg.skin_n_patches, seed=cfg.seed)
            print(f"[data] Sanity check do skin-mask salvo em {sanity_png} "
                  f"(inspecione antes de confiar na ablacao).")
        except Exception as e:  # nao bloquear o treino por causa do mosaico
            print(f"[data] WARN: falha ao gerar mosaico de sanity check: {e}")


def _fallback_json(cfg: ExperimentConfig) -> Path:
    return Path(cfg.cache_dir) / f"{cfg.cache_tag()}__fallback.json"


def report_skin_fallback(cfg: ExperimentConfig, train_paths: Sequence[str]) -> None:
    """Reporta a taxa de fallback do skin-mask (NJN skin_roi), sem leitura redundante.

    Preferencia: usar os contadores ``SKIN_STATS`` ja acumulados pela passada de
    estatisticas (mesma decodificacao + extracao de pele que aquece o cache). Se a
    passada nao ocorreu (stats em cache), le o JSON salvo numa rodada anterior; em
    ultimo caso (nada disponivel) faz um scan leve.
    """
    if cfg.is_neojaundice or cfg.njn_mode != "skin_roi":
        return
    import json
    scan_json = _fallback_json(cfg)
    total = skin_roi.SKIN_STATS["ok"] + skin_roi.SKIN_STATS["fallback"]
    if total > 0:
        rate = skin_roi.fallback_rate()
        scan_json.parent.mkdir(parents=True, exist_ok=True)
        scan_json.write_text(json.dumps({"fallback_rate": rate, **skin_roi.SKIN_STATS}))
    elif scan_json.is_file():
        try:
            rate = json.loads(scan_json.read_text()).get("fallback_rate", 0.0)
        except Exception:
            rate = None
    else:
        rate = None
    if rate is None:  # ultimo recurso: scan leve (stats em cache e sem JSON previo)
        skin_roi.reset_stats()
        for p in train_paths:
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is not None:
                skin_roi.extract_skin_patches(bgr, n=cfg.skin_n_patches)
        rate = skin_roi.fallback_rate()
        scan_json.parent.mkdir(parents=True, exist_ok=True)
        scan_json.write_text(json.dumps({"fallback_rate": rate, **skin_roi.SKIN_STATS}))
    _print_fallback(rate)


def _print_fallback(rate: float) -> None:
    pct = 100.0 * rate
    print(f"[data] NJN Skin ROI: {100.0 - pct:.1f}% sucesso, {pct:.1f}% fallback.")
    if rate > 0.10:
        import logging
        logging.getLogger("colorspace_final.data").warning(
            "Skin-mask fallback %.1f%% > 10%% — ajuste fino dos limiares HSV/YCrCb em "
            "skin_roi.py (e incremente SKIN_ALGO_VERSION) ANTES da ablacao completa; "
            "senao o modelo recebe muito ruido de fundo.", pct)


def build_dataloaders(cfg: ExperimentConfig, augment: bool, da_strength: float = 1.0,
                      batch_size: int | None = None, resize: int | None = None,
                      limit_per_split: Optional[int] = None) -> DataBundle:
    """Split único 70/10/20 (legado v5, usado no HPO). Delega ao core por-split."""
    samples, (tr, va, te), classes = splits.make_split(cfg)
    return build_dataloaders_from_split(cfg, samples, tr, va, te, classes, augment,
                                        da_strength=da_strength, batch_size=batch_size,
                                        resize=resize, limit_per_split=limit_per_split)


def build_dataloaders_from_split(cfg: ExperimentConfig, samples, tr, va, te, classes,
                                 augment: bool, da_strength: float = 1.0,
                                 batch_size: int | None = None, resize: int | None = None,
                                 limit_per_split: Optional[int] = None,
                                 multitask: bool = False) -> DataBundle:
    """Core: constrói os 3 loaders a partir de índices EXPLÍCITOS (usado pelo cv.py).

    ``multitask`` liga o alvo de regressão TSB (só NeoJaundice); as stats de
    normalização do TSB são calculadas **só no treino do fold** e devolvidas no bundle
    (des-normalização simétrica no cv.py)."""
    resize = resize or cfg.spec.input_size
    batch_size = batch_size or cfg.batch_size

    def items(idx):
        return [(str(samples[i].path), samples[i].label, samples[i].patient_id) for i in idx]

    tr_items = _limit(items(tr), limit_per_split)
    va_items = _limit(items(va), limit_per_split)
    te_items = _limit(items(te), limit_per_split)
    path_to_pid = {str(s.path): s.patient_id for s in samples}

    # Alvos TSB normalizados com stats do TREINO do fold (train-only; sem vazamento).
    tsb_mean, tsb_std = 0.0, 1.0
    tr_tsb = va_tsb = te_tsb = None
    if multitask:
        tr_raw = np.asarray([samples[i].tsb for i in tr], dtype=np.float64)
        tr_raw = tr_raw[:len(tr_items)] if limit_per_split else tr_raw
        tsb_mean = float(np.nanmean(tr_raw)); tsb_std = float(np.nanstd(tr_raw)) or 1.0
        def norm(idx, n):
            raw = np.asarray([samples[i].tsb for i in idx], dtype=np.float64)
            raw = raw[:n] if limit_per_split else raw
            return ((raw - tsb_mean) / tsb_std).tolist()
        tr_tsb = norm(tr, len(tr_items)); va_tsb = norm(va, len(va_items)); te_tsb = norm(te, len(te_items))

    loader = _make_loader(cfg)
    use_cache = cfg.use_cache and not limit_per_split
    cache_dir = cfg.cache_dir if use_cache else None
    tag = cfg.cache_tag()

    train_paths = [p for p, _, _ in tr_items]
    prepare_njn_skin(cfg, train_paths)

    skin_roi.reset_stats()
    stats_src = _StackDataset(tr_items, classes, transform=None, loader=loader,
                              cache_dir=cache_dir, cache_tag=tag, use_cache=use_cache)
    if limit_per_split:
        stats = compute_stats_from_iter(stats_src.iter_rgb(), resize=resize)
    else:
        stats = get_stats(cfg.stats_path(), lambda: stats_src.iter_rgb(), resize)
    report_skin_fallback(cfg, train_paths)

    train_tf, eval_tf = build_transforms(cfg, resize, augment, da_strength, stats)
    common = dict(loader=loader, cache_dir=cache_dir, cache_tag=tag, use_cache=use_cache)
    # Só o TREINO entrega o alvo TSB (usado na loss). Val/test avaliam classificação
    # com o _collect padrão (2-tuplas); as predições de regressão vêm de model.last_tsb.
    train_ds = _StackDataset(tr_items, classes, transform=train_tf, tsb_norm=tr_tsb,
                             return_tsb=multitask, **common)
    val_ds = _StackDataset(va_items, classes, transform=eval_tf, tsb_norm=va_tsb,
                           return_tsb=False, **common)
    test_ds = _StackDataset(te_items, classes, transform=eval_tf, tsb_norm=te_tsb,
                            return_tsb=False, **common)

    g = torch.Generator(); g.manual_seed(cfg.seed)
    sampler = _balanced_sampler(train_ds, g)

    kw = dict(num_workers=cfg.num_workers, pin_memory=True)
    if cfg.num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4, worker_init_fn=_worker_init,
                  multiprocessing_context=cfg.mp_context, timeout=300)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              drop_last=False, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw)

    return DataBundle(
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        n_channels=ColorSpaceStack(cfg.colorspaces, resize, stats,
                                   hue_circular=cfg.hue_circular).n_channels,
        classes=tuple(classes),
        sizes=(len(train_ds), len(val_ds), len(test_ds)),
        path_to_pid=path_to_pid, tsb_mean=tsb_mean, tsb_std=tsb_std,
    )


# --------------------------------------------------------------------------- #
# CLI: v6.data --precompute / --verify-cache  (Fase 0)
# --------------------------------------------------------------------------- #
def precompute_cache(cfg: ExperimentConfig) -> int:
    """Aquece o cache .npy (imread + WB/ROI/skin) de TODAS as amostras do dataset.

    Reprocessa do zero se a chave (cache_tag) mudou vs. a v5 — decisão consciente
    (ver plano §5). Colorspace-agnóstico: 1 cache serve qualquer subconjunto de cor."""
    from . import splits as _splits
    samples, _ = _splits.load_samples(cfg)
    items = [(str(s.path), s.label, s.patient_id) for s in samples]
    loader = _make_loader(cfg)
    tag = cfg.cache_tag()
    ds = _StackDataset(items, ("healthy", "jaundice"), transform=None, loader=loader,
                       cache_dir=cfg.cache_dir, cache_tag=tag, use_cache=True)
    n = 0
    for _ in ds.iter_rgb():
        n += 1
        if n % 200 == 0:
            print(f"  [precompute] {n}/{len(items)} …")
    print(f"[precompute] {n} imagens em {Path(cfg.cache_dir) / tag}")
    return n


def verify_cache(cfg: ExperimentConfig) -> None:
    tag = cfg.cache_tag()
    d = Path(cfg.cache_dir) / tag
    if not d.is_dir():
        print(f"[verify] AUSENTE: {d} (rode --precompute).")
        return
    npys = list(d.glob("*.npy"))
    size_mb = sum(p.stat().st_size for p in npys) / 1e6
    print(f"[verify] {tag}: {len(npys)} .npy, {size_mb:.1f} MB em {d}")


def main(argv=None):
    import argparse
    import sys
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description="v6.data — cache (precompute/verify).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--precompute", action="store_true")
    ap.add_argument("--verify-cache", action="store_true")
    ap.add_argument("--wb", default="on", choices=["on", "off"])
    ap.add_argument("--njn-mode", default="skin_roi", choices=["skin_roi", "full_image"])
    ap.add_argument("--data-root", default="dataset")
    ap.add_argument("--cache-dir", default="local/cache/roi")
    ap.add_argument("--results-dir", default="local/runs")
    args = ap.parse_args(argv)
    ds = {"neojaundice": "NeoJaundice", "njn": "NJN"}.get(args.dataset, args.dataset)
    cfg = ExperimentConfig(
        backbone="efficientnet_b4", dataset=ds, colorspaces=("RGB",),
        wb_method=("graypatch" if args.wb == "on" else "off"), njn_mode=args.njn_mode,
        data_root=args.data_root, cache_dir=args.cache_dir, results_dir=args.results_dir)
    if args.precompute:
        precompute_cache(cfg)
    if args.verify_cache or not args.precompute:
        verify_cache(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
