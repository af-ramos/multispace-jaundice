"""
splits.py
=========

Divisão **agrupada por paciente, in-memory e sem vazamento** para ambos os datasets.

* **NeoJaundice** — IDs reais de paciente: nomes ``NNNN-p.jpg`` (``0005-1.jpg`` e
  ``0005-3.jpg`` são ROIs do mesmo bebê 0005). Rótulos vêm do CSV (TSB), quantizados
  em 2 ou 3 classes. ``StratifiedGroupKFold`` 70/10/20 estratificado por classe e
  agrupado por ``patient_id``.

* **NJN** — pasta plana ``jaundice/normal`` SEM IDs (nomes são índices sequenciais).
  Para blindar contra leakage de fotos quase-duplicadas do mesmo bebê, reconstruímos
  **pseudo-IDs de paciente por similaridade perceptual** (:func:`phash_pseudo_patients`,
  ``imagehash.phash`` + union-find com distância ``<= phash_dist``) e aplicamos o
  MESMO ``StratifiedGroupKFold``. Sem ``imagehash`` instalado, cai no fallback
  :func:`stratified_image_split` (cada imagem = um pseudo-paciente) com WARNING.

Em ambos, nenhum paciente aparece em mais de um split (``assert_no_patient_leak``).
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import (CLASS_NAMES_2, CLASS_NAMES_3, TSB_3CLASS_BINS,
                     TSB_BINARY_THRESHOLD)

logger = logging.getLogger("colorspace_final.splits")

# Mensagens de diagnostico do split (fallback pHash, resumo de clusterizacao) sao
# idempotentes para a mesma config, mas load_samples e chamado a CADA trial do HPO.
# Este guard garante que cada mensagem distinta e emitida UMA vez por processo
# (ou seja, uma vez por seed/run), em vez de poluir o log a cada trial.
_EMITTED_ONCE: set = set()


def _emit_once(key: str, emit) -> None:
    """Executa ``emit()`` (print/warning) apenas na 1a vez que ``key`` aparece."""
    if key not in _EMITTED_ONCE:
        _EMITTED_ONCE.add(key)
        emit()

CSV_IMAGE_COL = "image_idx"      # ex.: '0003-1.jpg'
CSV_TSB_COL = "blood(mg/dL)"     # bilirrubina sérica total
PATIENT_RE = re.compile(r"^(\d+)")
_IMG_EXTS = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class Sample:
    path: Path
    patient_id: str   # real (NeoJaundice) ou pseudo (NJN, via pHash)
    label: int
    tsb: float = float("nan")


# --------------------------------------------------------------------------- #
# Rótulos / CSV (NeoJaundice)
# --------------------------------------------------------------------------- #
def patient_id_of(filename: str) -> str:
    m = PATIENT_RE.match(Path(filename).name)
    if not m:
        raise ValueError(f"ID de paciente nao extraido de {filename!r}")
    return m.group(1)


def tsb_to_class(tsb: float, num_classes: int) -> int:
    if num_classes == 2:
        return 1 if tsb >= TSB_BINARY_THRESHOLD else 0
    if num_classes == 3:
        for idx, (lo, hi) in enumerate(TSB_3CLASS_BINS):
            if lo <= tsb < hi:
                return idx
        return len(TSB_3CLASS_BINS) - 1
    raise ValueError(f"num_classes invalido: {num_classes}")


def class_names(num_classes: int) -> Tuple[str, ...]:
    return CLASS_NAMES_3 if num_classes == 3 else CLASS_NAMES_2


def _read_tsb_map(csv_path: Path) -> Dict[str, float]:
    text = csv_path.read_text(encoding="utf-8-sig")
    first = text.splitlines()[0]
    sep = ";" if first.count(";") >= first.count(",") else ","
    out: Dict[str, float] = {}
    for row in csv.DictReader(text.splitlines(), delimiter=sep):
        img = (row.get(CSV_IMAGE_COL) or "").strip()
        try:
            out[img] = float(row[CSV_TSB_COL])
        except (KeyError, ValueError, TypeError):
            continue
    if not out:
        raise ValueError(f"Nenhum TSB lido de {csv_path} (colunas esperadas: "
                         f"{CSV_IMAGE_COL!r}, {CSV_TSB_COL!r}).")
    return out


def load_neojaundice_samples(data_root: str | Path, num_classes: int) -> List[Sample]:
    base = Path(data_root) / "NeoJaundice"
    images_dir = base / "images"
    csv_path = base / "chd_jaundice_published_2.csv"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Pasta de imagens nao encontrada: {images_dir}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV nao encontrado: {csv_path}")

    tsb_map = _read_tsb_map(csv_path)
    samples: List[Sample] = []
    for f in sorted(images_dir.iterdir()):
        if f.suffix.lower() not in _IMG_EXTS:
            continue
        tsb = tsb_map.get(f.name)
        if tsb is None:
            continue
        samples.append(Sample(f, patient_id_of(f.name), tsb_to_class(float(tsb), num_classes),
                              float(tsb)))
    if not samples:
        raise FileNotFoundError(f"Sem imagens rotuladas em {images_dir}.")
    return samples


# --------------------------------------------------------------------------- #
# NJN — pseudo-IDs de paciente por similaridade perceptual (pHash)
# --------------------------------------------------------------------------- #
class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def phash_pseudo_patients(paths: Sequence[Path], dist: int = 5,
                          cache_path: Optional[Path] = None
                          ) -> Tuple[Dict[str, str], bool]:
    """Agrupa imagens quase-duplicadas em pseudo-pacientes via ``imagehash.phash``.

    Devolve ``(map {abs_path -> pseudo_id}, used_phash)``. ``used_phash=False``
    indica fallback (imagehash ausente): cada imagem vira seu próprio pseudo-paciente.
    O resultado é cacheado em JSON por ``dist`` (o pHash é determinístico).
    """
    paths = [Path(p) for p in paths]
    abs_paths = [str(p.resolve()) for p in paths]

    if cache_path is not None and cache_path.is_file():
        try:
            data = json.loads(cache_path.read_text())
            if data.get("dist") == dist and set(data.get("map", {})) == set(abs_paths):
                return data["map"], bool(data.get("used_phash", True))
        except Exception:
            pass

    try:
        import imagehash
        from PIL import Image
    except Exception:
        _emit_once("phash_fallback", lambda: logger.warning(
            "imagehash indisponivel -> fallback: cada imagem NJN vira um "
            "pseudo-paciente (sem agrupamento perceptual)."))
        mapping = {ap: f"img{i:05d}" for i, ap in enumerate(abs_paths)}
        return mapping, False

    hashes = []
    for p in paths:
        with Image.open(p) as im:
            hashes.append(imagehash.phash(im.convert("RGB")))

    uf = _UnionFind(len(paths))
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if (hashes[i] - hashes[j]) <= dist:
                uf.union(i, j)

    root_to_pid: Dict[int, str] = {}
    mapping: Dict[str, str] = {}
    for i, ap in enumerate(abs_paths):
        r = uf.find(i)
        if r not in root_to_pid:
            root_to_pid[r] = f"pp{len(root_to_pid):05d}"
        mapping[ap] = root_to_pid[r]

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(
            {"dist": dist, "used_phash": True, "map": mapping}, indent=2))
    return mapping, True


def load_njn_samples(data_root: str | Path, phash_dist: int = 5,
                     cache_dir: Optional[str | Path] = None) -> List[Sample]:
    """Carrega a NJN (pasta plana ``jaundice``/``normal``) com pseudo-IDs de paciente.

    Imprime o resumo da clusterização pHash e emite WARNING se o maior cluster for
    desproporcional (distância frouxa demais).
    """
    root = Path(data_root) / "NJN"
    # NJN: 'normal'/'healthy' = healthy (0), 'jaundice' = jaundice (1).
    folder_to_label = {"normal": 0, "healthy": 0, "jaundice": 1}
    paths: List[Path] = []
    labels: List[int] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        lab = folder_to_label.get(sub.name.lower())
        if lab is None:
            continue
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() in _IMG_EXTS:
                paths.append(f)
                labels.append(lab)
    if not paths:
        raise FileNotFoundError(f"Sem imagens em {root}/(normal|jaundice).")

    cache_path = (Path(cache_dir) / f"njn_phash_d{phash_dist}.json") if cache_dir else None
    pid_map, used = phash_pseudo_patients(paths, dist=phash_dist, cache_path=cache_path)

    samples = [Sample(p, pid_map[str(p.resolve())], lab) for p, lab in zip(paths, labels)]
    _print_phash_summary(samples, total=len(paths), phash_dist=phash_dist, used_phash=used)
    return samples


def _print_phash_summary(samples: Sequence[Sample], total: int, phash_dist: int,
                         used_phash: bool) -> None:
    from collections import Counter
    counts = Counter(s.patient_id for s in samples)
    n_pp = len(counts)
    largest = max(counts.values()) if counts else 0
    tag = "pHash" if used_phash else "fallback(sem pHash)"
    summary = (f"[splits] NJN {tag}: {total} imagens -> {n_pp} pseudo-pacientes "
               f"(maior cluster: {largest} imgs, dist<={phash_dist}).")
    _emit_once(f"phash_summary:{tag}:{total}:{n_pp}:{largest}:{phash_dist}",
               lambda: print(summary))
    if used_phash and (largest > 20 or largest > 0.05 * total):
        _emit_once(f"phash_big_cluster:{largest}", lambda: logger.warning(
            "NJN: maior cluster pHash = %d imgs (>%d ou >5%% do total) — distancia "
            "provavelmente frouxa demais (fundo/pose sendo fundidos). Considere "
            "--phash-dist 3 (mais estrito) antes da ablacao completa.", largest, 20))


# --------------------------------------------------------------------------- #
# Splits (agrupado por paciente; fallback estratificado por imagem)
# --------------------------------------------------------------------------- #
def patient_group_split(labels: Sequence[int], groups: Sequence[str], seed: int = 42
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Índices train/val/test (~70/10/20), estratificados por classe e agrupados.

    Dois estágios com ``StratifiedGroupKFold``: (1) n_splits=5 separa ~20% teste;
    (2) nos 80% restantes, n_splits=8 separa ~10% como val. Nenhum grupo (paciente)
    aparece em mais de um split.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    y = np.asarray(labels)
    g = np.asarray(groups)
    X = np.zeros((len(y), 1))

    sgkf_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(iter(sgkf_test.split(X, y, g)))

    sgkf_val = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed)
    sub_train, sub_val = next(iter(
        sgkf_val.split(X[trainval_idx], y[trainval_idx], g[trainval_idx])))
    train_idx = trainval_idx[sub_train]
    val_idx = trainval_idx[sub_val]
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def stratified_image_split(labels: Sequence[int], seed: int = 42
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback sem grupos: split estratificado por imagem ~70/10/20 (semeado)."""
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(labels)
    X = np.zeros((len(y), 1))
    skf_test = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(iter(skf_test.split(X, y)))
    skf_val = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)
    sub_train, sub_val = next(iter(skf_val.split(X[trainval_idx], y[trainval_idx])))
    return (np.sort(trainval_idx[sub_train]), np.sort(trainval_idx[sub_val]),
            np.sort(test_idx))


def assert_no_patient_leak(samples: Sequence[Sample], train_idx, val_idx, test_idx) -> None:
    def pats(idx):
        return {samples[i].patient_id for i in idx}
    tr, va, te = pats(train_idx), pats(val_idx), pats(test_idx)
    assert not (tr & te), f"VAZAMENTO train∩test: {len(tr & te)} pacientes"
    assert not (tr & va), f"VAZAMENTO train∩val: {len(tr & va)} pacientes"
    assert not (va & te), f"VAZAMENTO val∩test: {len(va & te)} pacientes"


def load_samples(cfg) -> Tuple[List[Sample], Tuple[str, ...]]:
    """Dispatcher: carrega ``Sample`` do dataset da config + nomes das classes."""
    if cfg.is_neojaundice:
        samples = load_neojaundice_samples(cfg.data_root, cfg.num_classes)
    else:
        samples = load_njn_samples(cfg.data_root, phash_dist=cfg.phash_dist,
                                   cache_dir=cfg.cache_dir)
    return samples, class_names(cfg.num_classes)


def make_split(cfg) -> Tuple[List[Sample], Tuple[np.ndarray, np.ndarray, np.ndarray],
                             Tuple[str, ...]]:
    """Carrega samples e devolve (samples, (train,val,test) idx, classes), sem leak.

    ``split_source='frozen'`` -> LÊ o split canônico congelado (dataset/<DS>/splits/),
    idêntico entre pesquisadores; caso contrário gera o split 70/10/20 semeado por cfg.seed.
    """
    samples, classes = load_samples(cfg)
    if getattr(cfg, "split_source", "kfold") == "frozen":
        tr, va, te = load_frozen_split(cfg, samples)
        return samples, (tr, va, te), classes
    labels = [s.label for s in samples]
    groups = [s.patient_id for s in samples]
    # NJN sem pHash (fallback) tem pseudo-id único por imagem -> grouped == estratificado.
    tr, va, te = patient_group_split(labels, groups, seed=cfg.seed)
    assert_no_patient_leak(samples, tr, va, te)
    return samples, (tr, va, te), classes


# --------------------------------------------------------------------------- #
# v6 — Cross-validation k-fold (5-fold × 5-seed no cv.py)
# --------------------------------------------------------------------------- #
def kfold_indices(labels: Sequence[int], groups: Sequence[str], seed: int = 42,
                  k: int = 5, mode: str = "group"
                  ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Lista de ``(train_idx, val_idx, test_idx)`` para os ``k`` folds.

    Para cada fold: ``test`` = a partição retida (1/k); do restante (trainval) carve-se
    um ``val`` (split interno), garantindo **val e test disjuntos DENTRO do fold** — o
    limiar (pooled_oof) é escolhido só no ``val`` e aplicado ao ``test`` (nunca no test).

    ``mode='group'`` = StratifiedGroupKFold por ``patient_id`` (honesto); ``mode='random'``
    = StratifiedKFold por imagem (Track A, expõe o vazamento na tabela-delta 2e).
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    y = np.asarray(labels)
    g = np.asarray(groups)
    X = np.zeros((len(y), 1))
    if mode == "group":
        outer = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        outer_splits = list(outer.split(X, y, g))
    elif mode == "random":
        outer = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        outer_splits = list(outer.split(X, y))
    else:
        raise ValueError(f"kfold mode invalido: {mode} (group|random).")

    folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for trainval_idx, test_idx in outer_splits:
        yv, gv = y[trainval_idx], g[trainval_idx]
        Xv = X[trainval_idx]
        if mode == "group":
            inner = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed + 1)
            sub_tr, sub_va = next(iter(inner.split(Xv, yv, gv)))
        else:
            inner = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed + 1)
            sub_tr, sub_va = next(iter(inner.split(Xv, yv)))
        tr = trainval_idx[sub_tr]
        va = trainval_idx[sub_va]
        folds.append((np.sort(tr), np.sort(va), np.sort(test_idx)))
    return folds


def make_kfold(cfg, seed: int, k: int = 5, mode: str = "group"
               ) -> Tuple[List[Sample], List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                          Tuple[str, ...]]:
    """Carrega samples e devolve (samples, folds, classes); valida ausência de leak
    em cada fold (só no modo 'group'; 'random' vaza por construção — é o controle)."""
    samples, classes = load_samples(cfg)
    labels = [s.label for s in samples]
    groups = [s.patient_id for s in samples]
    folds = kfold_indices(labels, groups, seed=seed, k=k, mode=mode)
    if mode == "group":
        for tr, va, te in folds:
            assert_no_patient_leak(samples, tr, va, te)
    return samples, folds, classes


# --------------------------------------------------------------------------- #
# Split canônico CONGELADO (70/10/20, agrupado por paciente) — compartilhável
# --------------------------------------------------------------------------- #
# Um único split train/val/test por dataset, materializado em
# ``dataset/<DS>/splits/<ds>_split.csv`` (colunas ``image;patient_id;split``), para
# GARANTIR o MESMO split entre pesquisadores/pipelines, independente da versão de
# ``sklearn``/``imagehash`` ou da ordem de descoberta dos arquivos. Gerado 1× por
# ``make_splits.py`` (semente SPLIT_SEED); o pipeline apenas LÊ (fail-closed).
SPLIT_SEED = 42                          # semente única do split canônico (registrada no meta.json)
SPLIT_CSV_HEADER = ("image", "patient_id", "split")
_SPLIT_ROLES = ("train", "val", "test")


def frozen_split_path(data_root: str | Path, dataset: str) -> Path:
    """Caminho canônico do CSV do split congelado de ``dataset``."""
    return Path(data_root) / dataset / "splits" / f"{dataset.lower()}_split.csv"


def _relpath(path: str | Path, data_root: str | Path) -> str:
    """Caminho da imagem relativo à raiz de dados (chave estável no CSV, ex.:
    ``NeoJaundice/images/0001-1.jpg`` / ``NJN/jaundice/xxx.jpg``). Usa '/' sempre."""
    rp = Path(path).resolve().relative_to(Path(data_root).resolve())
    return rp.as_posix()


def build_canonical_split(cfg, seed: int = SPLIT_SEED
                          ) -> Tuple[List[Sample], Tuple[np.ndarray, np.ndarray, np.ndarray],
                                     Tuple[str, ...]]:
    """Gera (1×) o split canônico 70/10/20 agrupado por paciente, sem leak.

    Reutiliza :func:`patient_group_split` (mesmo procedimento honesto do resto do
    projeto). É a fonte de verdade que ``make_splits.py`` serializa no CSV."""
    samples, classes = load_samples(cfg)
    labels = [s.label for s in samples]
    groups = [s.patient_id for s in samples]
    tr, va, te = patient_group_split(labels, groups, seed=seed)
    assert_no_patient_leak(samples, tr, va, te)
    return samples, (tr, va, te), classes


def load_frozen_split(cfg, samples: Sequence[Sample]
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lê o split canônico congelado e devolve (train_idx, val_idx, test_idx) alinhados
    a ``samples``. Fail-closed: se o CSV faltar, ou alguma amostra não estiver mapeada,
    ergue erro (dataset divergiu -> regenerar com make_splits). O leak-check usa o
    ``patient_id`` DO CSV, então independe do pHash/imagehash do ambiente atual."""
    path = frozen_split_path(cfg.data_root, cfg.dataset)
    if not path.is_file():
        path = (Path(__file__).resolve().parents[1] / "splits" / "frozen"
                / cfg.dataset / f"{cfg.dataset.lower()}_split.csv")
    if not path.is_file():
        raise FileNotFoundError(
            f"Split congelado ausente: {path}. Gere-o 1x com:\n"
            f"  python -m src.make_splits --dataset {cfg.dataset}")

    role: Dict[str, str] = {}
    pat: Dict[str, str] = {}
    text = path.read_text(encoding="utf-8-sig")
    for row in csv.DictReader(text.splitlines(), delimiter=";"):
        img = (row.get("image") or "").strip()
        if not img:
            continue
        r = (row.get("split") or "").strip()
        if r not in _SPLIT_ROLES:
            raise ValueError(f"Split inválido {r!r} para {img!r} em {path} (use {_SPLIT_ROLES}).")
        role[img] = r
        pat[img] = (row.get("patient_id") or "").strip()

    buckets: Dict[str, List[int]] = {r: [] for r in _SPLIT_ROLES}
    groups_by_split: Dict[str, set] = {r: set() for r in _SPLIT_ROLES}
    missing: List[str] = []
    for i, s in enumerate(samples):
        rp = _relpath(s.path, cfg.data_root)
        r = role.get(rp)
        if r is None:
            missing.append(rp)
            continue
        buckets[r].append(i)
        groups_by_split[r].add(pat[rp])
    if missing:
        raise ValueError(
            f"{len(missing)} amostra(s) sem entrada no split congelado {path} "
            f"(ex.: {missing[:3]}). O dataset mudou desde a geração — regenere com make_splits.")

    tr, va, te = groups_by_split["train"], groups_by_split["val"], groups_by_split["test"]
    assert not (tr & te), f"VAZAMENTO train∩test no split congelado: {len(tr & te)} pacientes"
    assert not (tr & va), f"VAZAMENTO train∩val no split congelado: {len(tr & va)} pacientes"
    assert not (va & te), f"VAZAMENTO val∩test no split congelado: {len(va & te)} pacientes"

    return (np.sort(np.asarray(buckets["train"], dtype=int)),
            np.sort(np.asarray(buckets["val"], dtype=int)),
            np.sort(np.asarray(buckets["test"], dtype=int)))
