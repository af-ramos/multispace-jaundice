#!/usr/bin/env python3
"""Leitura canónica dos dumps de predições da campanha v7.

Existe porque identificar "que célula é este `.npz`" tem duas armadilhas, e ambas
morderam:

1. **O dataset não se lê do directório.** A cópia de `local/experiments/extra` para
   `evidence/v7/preds/<dataset>/` já colocou os 28 dumps de madrugada nos dois
   directórios ao mesmo tempo, e um relatório que confiasse no nome da pasta contava
   cada célula do NeoJaundice como cobertura do NJN. O dataset lê-se do campo `paths`
   do próprio dump, que é a única fonte que não mente.
2. **Nem todo o `.npz` é a mesma arquitectura.** A campanha guardou dumps de fases
   com fusão diferente (`stemcnn`, `ccat`), sem TTA (`notta`) e multitarefa (`mt`).
   Comparar qualquer um deles com o factorial A1 — que é fase 1, `adapter_v2` — dá
   divergências de até 8 p.p. que não são irreprodutibilidade nenhuma, são outro
   modelo. A célula canónica é `f1_*` + `adapter_v2` + `fixedsplit_x5`, na condição
   canónica de cada dataset (NJN `full_image`, NeoJaundice `wb-off`).

Saída: `canonical_dumps()`, que devolve uma célula por `(dataset, backbone, colorset)`
com a acurácia de teste repontuada, nível imagem, limiar 0,5, média sobre as 5
sementes — a mesma quantidade que o A1 regista em `accuracy_cor` — a **AUC-ROC** e o
**F1-macro** de teste pela mesma receita (um por semente, média das cinco).
"""

from __future__ import annotations

import functools
import json
import os
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREDS = ROOT / "evidence/v7/preds"
EXTRA = ROOT / "local/experiments/extra"

Key = tuple[str, str, str]


def _dataset_of(npz: "np.lib.npyio.NpzFile") -> str:
    """O dataset segundo os caminhos das imagens, não segundo a pasta."""
    example = str(npz["paths"][0])
    return "NJN" if "/NJN/" in example else "NeoJaundice"


def _is_canonical(stem: str, dataset: str) -> bool:
    if not stem.startswith("f1_") or "__adapter_v2__" not in stem:
        return False
    if "fixedsplit_x5" not in stem or "split-random" in stem or "__mt__" in stem:
        return False
    return ("full_image" in stem) if dataset == "NJN" else ("wb-off" in stem)


def _auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """AUC-ROC pela estatística de Mann–Whitney, com empates por posto médio.

    Escrita aqui em vez de importada do `sklearn` para que a leitura dos dumps —
    de que dependem as tabelas — não ganhe uma dependência a mais. A identidade
    AUC = (R₁ − n₁(n₁+1)/2) / (n₁n₀) é exacta, não uma aproximação trapezoidal.
    """
    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(y_prob, kind="mergesort")
    ranks = np.empty(len(y_prob), dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1, dtype=float)
    ordered = y_prob[order]
    start = 0
    for i in range(1, len(ordered) + 1):  # empates recebem o posto médio do bloco
        if i == len(ordered) or ordered[i] != ordered[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[y_true == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _retrained_names() -> set[str]:
    """Replica classification independent of the local-only staging directory."""
    manifest = ROOT / "evidence/v7/prediction_provenance.json"
    return set(json.loads(manifest.read_text())["retrained_names"])


def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1-macro: média não-ponderada do F1 das duas classes.

    Escrita aqui, e não importada do `sklearn`, pela mesma razão que o `_auc`: as
    tabelas do artigo não devem ganhar uma dependência a mais. Uma classe ausente das
    predições E do rótulo contribui F1=0, que é a convenção do `sklearn` com
    `zero_division=0` — o caso não ocorre nestes dumps, mas fica definido.
    """
    scores = []
    for cls in (0, 1):
        tp = float(((y_pred == cls) & (y_true == cls)).sum())
        fp = float(((y_pred == cls) & (y_true != cls)).sum())
        fn = float(((y_pred != cls) & (y_true == cls)).sum())
        denom = 2.0 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2.0 * tp / denom)
    return statistics.fmean(scores)


@functools.lru_cache(maxsize=1)
def canonical_dumps() -> dict[Key, dict]:
    """`(dataset, backbone, colorset)` -> acurácia repontuada e proveniência."""
    retrained = _retrained_names()
    out: dict[Key, dict] = {}
    for path in sorted(PREDS.glob("*/*.npz")):
        data = np.load(path, allow_pickle=True)
        dataset = _dataset_of(data)
        if not _is_canonical(path.stem, dataset):
            continue
        parts = path.stem.split("__")
        if len(parts) < 3:
            continue
        key = (dataset, parts[1], parts[2])
        test = data["splits"] == "test"
        per_seed: dict[int, float] = {}
        auc_seed: dict[int, float] = {}
        f1_seed: dict[int, float] = {}
        for seed in np.unique(data["seeds"]):
            mask = test & (data["seeds"] == seed)
            if not mask.any():
                continue
            pred = (data["y_prob"][mask] >= 0.5).astype(int)
            hit = pred == data["y_true"][mask]
            per_seed[int(seed)] = 100.0 * float(hit.mean())
            f1_seed[int(seed)] = _f1_macro(data["y_true"][mask], pred)
            auc = _auc(data["y_true"][mask], data["y_prob"][mask])
            if auc is not None:
                auc_seed[int(seed)] = auc
        if not per_seed:
            continue
        if key in out:  # não deve acontecer; se acontecer, é ambiguidade a resolver
            raise RuntimeError(f"dois dumps canónicos para {key}: {path.name}")
        out[key] = {
            "acc": statistics.fmean(per_seed.values()),
            "per_seed": per_seed,
            "auc": statistics.fmean(auc_seed.values()) if auc_seed else None,
            "auc_per_seed": auc_seed,
            "f1": statistics.fmean(f1_seed.values()) if f1_seed else None,
            "f1_per_seed": f1_seed,
            "retrained": path.name in retrained,
            "file": path.name,
        }
    return out


def misplaced() -> list[tuple[Path, str, str]]:
    """Dumps cuja pasta contradiz o dataset que os seus `paths` declaram."""
    bad = []
    for path in sorted(PREDS.glob("*/*.npz")):
        real = _dataset_of(np.load(path, allow_pickle=True))
        if real != path.parent.name:
            bad.append((path, path.parent.name, real))
    return bad


def install_from_extra(dry_run: bool = True) -> list[tuple[Path, Path]]:
    """Copia os dumps de `v7_extra` para a pasta do dataset a que pertencem.

    Substitui o `cp local/experiments/extra/*/*/preds/*.npz evidence/v7/preds/<dataset>/`
    que o `run_missing_dumps.sh` sugeria: aquele glob apanha os dois datasets de uma
    vez, e correndo-o uma vez por pasta acaba com cada dump nas duas — foi o que
    aconteceu a 2026-08-07 e o que inflou a cobertura de 88 para 115. Aqui o destino
    vem do conteúdo do ficheiro, portanto não há como enganá-lo.
    """
    planned: list[tuple[Path, Path]] = []
    for src in sorted(EXTRA.glob("*/*/preds/*.npz")):
        dataset = _dataset_of(np.load(src, allow_pickle=True))
        dest = PREDS / dataset / src.name
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            continue
        planned.append((src, dest))
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())
    return planned


if __name__ == "__main__":
    import sys

    if "--install" in sys.argv:
        dry = "--apply" not in sys.argv
        todo = install_from_extra(dry_run=dry)
        verb = "a copiar" if dry else "copiados"
        print(f"{verb}: {len(todo)} dumps" + (" (use --apply para escrever)" if dry else ""))
        for src, dest in todo:
            print(f"  {src.name[:64]} -> {dest.parent.name}/")
        raise SystemExit(0)

    cells = canonical_dumps()
    n_re = sum(1 for c in cells.values() if c["retrained"])
    print(f"células canónicas com dump: {len(cells)}  ({n_re} retreinadas na madrugada)")
    for dataset in ("NJN", "NeoJaundice"):
        print(f"  {dataset:12} {sum(1 for k in cells if k[0] == dataset):>3}")
    bad = misplaced()
    print(f"dumps na pasta errada: {len(bad)}")
    for path, folder, real in bad:
        print(f"  {path.name[:70]}  em {folder}/ mas é {real}")
