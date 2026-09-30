"""
make_splits.py
==============

Gera (1×, offline, SEM GPU) o **split canônico congelado** 70/10/20 agrupado por
paciente para cada dataset e o materializa em::

    dataset/<DS>/splits/<ds>_split.csv        # image;patient_id;split   (fonte de verdade)
    dataset/<DS>/splits/<ds>_split_meta.json  # semente, versões, contagens, commit

Objetivo: **garantir o MESMO split entre pesquisadores e pipelines**. Uma vez
congelado e commitado (a pasta ``dataset/`` é versionada), qualquer pessoa —
inclusive um pipeline externo que só leia o CSV — usa exatamente as mesmas
partições, independente da versão de ``sklearn``/``imagehash`` ou da ordem de
descoberta dos arquivos.

O split reutiliza ``splits.patient_group_split`` (mesmo procedimento honesto do
projeto: ``StratifiedGroupKFold`` estratificado por classe, agrupado por paciente;
NeoJaundice = IDs reais; NJN = pseudo-pacientes pHash). Depois de gerado, o treino
o consome com ``--frozen-split`` (ver EXPERIMENT_MATRIX.md).

Uso::

    python -m src.make_splits --dataset both
    python -m src.make_splits --dataset NJN --phash-dist 5 --force

NÃO usa GPU nem toca o cluster (regra #8): pode rodar localmente.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List

from .config import ExperimentConfig
from . import splits as S


def _git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo_root),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _sklearn_version() -> str:
    try:
        import sklearn
        return sklearn.__version__
    except Exception:
        return "unknown"


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8-sig")
    os.replace(tmp, path)


def _build_cfg(dataset: str, data_root: str, num_classes: int, phash_dist: int) -> ExperimentConfig:
    # cfg mínimo só p/ carregar samples: backbone/colorset/fusão são irrelevantes ao split.
    return ExperimentConfig(
        backbone="resnet18", dataset=dataset, colorspaces=("RGB",),
        num_classes=num_classes, data_root=data_root, phash_dist=phash_dist,
        njn_mode="full_image",
    )


def _write_split(dataset: str, args, repo_root: Path) -> None:
    out_csv = S.frozen_split_path(args.data_root, dataset)
    if out_csv.is_file() and not args.force:
        print(f"[skip] {out_csv} já existe — use --force para regerar.")
        return

    cfg = _build_cfg(dataset, args.data_root, args.num_classes, args.phash_dist)
    samples, (tr, va, te), classes = S.build_canonical_split(cfg, seed=args.seed)

    role = {}
    for idx_arr, name in ((tr, "train"), (va, "val"), (te, "test")):
        for i in idx_arr:
            role[i] = name

    # Linhas ordenadas pelo caminho relativo (determinístico, diff-friendly).
    rows = []
    for i, s in enumerate(samples):
        rows.append((S._relpath(s.path, args.data_root), s.patient_id, role[i]))
    rows.sort(key=lambda r: r[0])

    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";", lineterminator="\n")
    w.writerow(list(S.SPLIT_CSV_HEADER))
    w.writerows(rows)
    _atomic_write(out_csv, buf.getvalue())

    # --- meta.json (procedência do congelamento) ---
    def _stat(idx_arr):
        pats = {samples[i].patient_id for i in idx_arr}
        cls = Counter(classes[samples[i].label] for i in idx_arr)
        return {"imagens": int(len(idx_arr)), "pacientes": int(len(pats)),
                "por_classe": dict(cls)}

    meta = {
        "dataset": dataset,
        "split_seed": args.seed,
        "procedimento": "patient_group_split (StratifiedGroupKFold ~70/10/20, agrupado por paciente)",
        "num_classes_estratificacao": args.num_classes,
        "phash_dist": (args.phash_dist if dataset == "NJN" else None),
        "n_total_imagens": int(len(samples)),
        "n_total_pacientes": int(len({s.patient_id for s in samples})),
        "classes": list(classes),
        "train": _stat(tr), "val": _stat(va), "test": _stat(te),
        "sklearn_version": _sklearn_version(),
        "python_version": sys.version.split()[0],
        "git_commit": _git_commit(repo_root),
        "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S"),
        "csv": out_csv.name,
    }
    _atomic_write(out_csv.with_name(f"{dataset.lower()}_split_meta.json"),
                  json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"[ok] {dataset}: {len(samples)} imgs -> train {meta['train']['imagens']} / "
          f"val {meta['val']['imagens']} / test {meta['test']['imagens']}  "
          f"(pacientes {meta['train']['pacientes']}/{meta['val']['pacientes']}/"
          f"{meta['test']['pacientes']})")
    print(f"     CSV : {out_csv}")
    print(f"     meta: {out_csv.with_name(dataset.lower() + '_split_meta.json')}")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Congela o split canônico 70/10/20 por dataset.")
    ap.add_argument("--dataset", default="both", choices=["NJN", "NeoJaundice", "both"])
    ap.add_argument("--data-root", default="dataset")
    ap.add_argument("--num-classes", type=int, default=2,
                    help="Só p/ ESTRATIFICAR a geração (o split é por paciente, aplica-se a 2 e 3 classes).")
    ap.add_argument("--seed", type=int, default=S.SPLIT_SEED)
    ap.add_argument("--phash-dist", type=int, default=5, help="distância pHash p/ pseudo-pacientes NJN.")
    ap.add_argument("--force", action="store_true", help="regera mesmo se o CSV já existir.")
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    datasets = ["NJN", "NeoJaundice"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        _write_split(ds, args, repo_root)
    print("\n[pronto] Split(s) congelado(s). Commite dataset/<DS>/splits/ e rode o treino com --frozen-split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
