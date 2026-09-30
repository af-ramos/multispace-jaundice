#!/usr/bin/env python3
"""Retreina doze células com persistência de pesos, para as figuras de atribuição.

## Porque é preciso

A campanha nunca chamou ``torch.save``: os pesos viviam em RAM e morriam com o
processo. As predições e o α médio do gate sobreviveram (ver `analysis/gate_weights.py`),
mas Grad-CAM e attention rollout precisam do **modelo**, não das suas saídas. Não há
atalho — ou se retreina, ou não há atribuição espacial nenhuma.

## O que corre, e porquê estas células

Seis pares (dataset, backbone), cada um com o seu braço cromático **e** o seu baseline
RGB, porque a figura é uma comparação e não um retrato:

* os dois sistemas em destaque da Tabela 4, que são os que o artigo reporta;
* os quatro extremos da Tabela 5 --- DenseNet121 (+3,01) e DINOv3-ViT-S/16 (−1,88) no
  NJN, DeiT-Base (+1,64) e DeiT-Small (−1,59) no NeoJaundice --- que são os backbones
  cuja resposta o artigo diz ser oposta.

O colour set de cada par é o que a figura correspondente já usa, para as imagens serem
as mesmas: `RGB+LAB+HSV` e `RGB+LAB+YCrCb+HSV` nos destaques, `RGB+YCrCb+HSV` no par da
Figura 6, e o conjunto de quatro espaços nos dois DeiT.

## Configuração

**Idêntica à da campanha, sem re-optimizar nada.** Os hiperparâmetros vêm de
`evidence/v7/best_params/<backbone>_<dataset>.json`; re-optimizar tornaria estas células
incomparáveis com as 450 originais, que é precisamente o que a figura precisa de poder
afirmar. A única diferença é `--save-explain`, que é aditiva e corre depois do treino.

**Os pesos novos não são os que produziram a Tabela 4.** O treino é estocástico e a
campanha não guardou estado de RNG suficiente para o reproduzir bit a bit. A figura tem
de dizer que é uma RÉPLICA sob configuração idêntica, e o `--report` compara a acurácia
da réplica com a do registo original para que a divergência seja medida e não suposta.

## Custo

Estimado dos `elapsed_min` da campanha: ~70 min no total para as doze células, das
quais o `vit_l_16` no NeoJaundice é metade. Os `.pt` ocupam de ~9 MB (MobileNetV3) a
~1,2 GB (cinco seeds do ViT-L/16); escreve numa árvore separada e não toca em nada.

Uso:
    python scripts/run_explain_cells.py --dry-run     # imprime os comandos
    nohup bash scripts/run_explain_cells.sh &         # a sério
    python scripts/run_explain_cells.py --report      # réplica contra o registo
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import DATA_ROOT
RESULTS_DIR = ROOT / "local/experiments/explain"
BEST_PARAMS = ROOT / "evidence/v7/best_params"
SEEDS = "42 123 456 7 99"
PYTHON = sys.executable

#: (dataset, backbone, colour set cromático). O baseline RGB é acrescentado a cada par.
CELLS = [
    ("NJN",         "mobilenetv3_large", "RGB+LAB+HSV"),        # destaque, Tabela 4
    ("NeoJaundice", "vit_l_16",          "RGB+LAB+YCrCb+HSV"),  # destaque, Tabela 4
    ("NJN",         "densenet121",       "RGB+YCrCb+HSV"),      # +3,01, Figura 6
    ("NJN",         "dinov3_vits16",     "RGB+YCrCb+HSV"),      # −1,88, Figura 6
    ("NeoJaundice", "deit_base",         "RGB+LAB+YCrCb+HSV"),  # +1,64
    ("NeoJaundice", "deit_small",        "RGB+LAB+YCrCb+HSV"),  # −1,59
]

PARAM_FLAGS = {
    "lr": "--lr", "optimizer": "--optimizer", "n_unfreeze": "--n-unfreeze",
    "batch_size": "--batch-size", "da_strength": "--da-strength", "fusion": "--fusion",
}


def best_params(backbone: str, dataset: str) -> dict:
    path = BEST_PARAMS / f"{backbone}_{dataset}.json"
    if not path.is_file():
        raise SystemExit(f"Sem HPO para {backbone}/{dataset}: {path}")
    return json.loads(path.read_text(encoding="utf-8")).get("best_params", {})


def build_command(dataset: str, backbone: str, colorset: str, gpu: int) -> list[str]:
    params = best_params(backbone, dataset)
    tag = f"f1_{backbone}_njn" if dataset == "NJN" else f"f1_{backbone}_neo_wboff"
    cmd = [
        PYTHON, "-m", "src.train",
        "--backbone", backbone, "--dataset", dataset, "--colors", colorset,
        "--backbone-mode", "lora", "--frozen-split", "--save-explain",
        "--seeds", SEEDS, "--tag", tag, "--gpu", str(gpu),
        "--data-root", str(DATA_ROOT), "--results-dir", str(RESULTS_DIR),
    ]
    cmd += ["--njn-mode", "full_image"] if dataset == "NJN" else ["--wb", "off"]
    for key, flag in PARAM_FLAGS.items():
        if key in params and params[key] is not None:
            cmd += [flag, str(params[key])]
    if params.get("augment"):
        cmd.append("--augment")
    if "--fusion" not in cmd:
        cmd += ["--fusion", "adapter_v2"]
    return cmd


def jobs(gpu: int) -> list[tuple[str, str, str, list[str]]]:
    """Cada par cromático seguido do seu baseline RGB, sem repetir baselines."""
    out, seen = [], set()
    for dataset, backbone, colorset in CELLS:
        for cs in (colorset, "RGB"):
            key = (dataset, backbone, cs)
            if key in seen:
                continue
            seen.add(key)
            out.append((dataset, backbone, cs, build_command(dataset, backbone, cs, gpu)))
    return out


def done(dataset: str, backbone: str, colorset: str) -> bool:
    """Idempotência: uma célula com gate gravado não volta a correr."""
    d = RESULTS_DIR / dataset / backbone / "explain"
    return d.is_dir() and any(f"__{colorset}__" in p.name for p in d.glob("*__gate.npz"))


def report() -> int:
    """A acurácia da réplica contra o registo da campanha, célula a célula."""
    sys.path.insert(0, str(ROOT))
    import csv

    original = {}
    with (ROOT / "evidence/v7/aggregate.csv").open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            if r["tag"].startswith("f1_"):
                original[(r["dataset"], r["backbone"], r["colorset_id"])] = r["acc_mean"]

    print(f"{'dataset':12s} {'backbone':18s} {'colour set':20s} {'réplica':>8s} {'registo':>8s} {'Δ':>7s}")
    worst = 0.0
    for dataset, backbone, colorset, _ in jobs(0):
        marker = None
        d = RESULTS_DIR / dataset / backbone
        if d.is_dir():
            hits = [p for p in d.glob("*.json") if f"__{colorset}__" in p.name]
            marker = hits[0] if hits else None
        if marker is None:
            print(f"{dataset:12s} {backbone:18s} {colorset:20s} {'—':>8s} "
                  f"{'—':>8s} {'por correr':>7s}")
            continue
        new = json.loads(marker.read_text())["test_metrics_mean"]["accuracy"]
        old = original.get((dataset, backbone, colorset))
        old_f = float(str(old).replace(",", ".")) if old else float("nan")
        delta = new - old_f
        worst = max(worst, abs(delta) if delta == delta else 0.0)
        print(f"{dataset:12s} {backbone:18s} {colorset:20s} {new:8.2f} {old_f:8.2f} {delta:+7.2f}")
    print(f"\nmaior divergência réplica-registo: {worst:.2f} p.p. "
          "(declarar na legenda da figura se passar de ~1 p.p.)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    if args.report:
        return report()

    todo = jobs(args.gpu)
    if args.dry_run:
        for dataset, backbone, colorset, cmd in todo:
            flag = "JÁ FEITA" if done(dataset, backbone, colorset) else "a correr"
            print(f"# [{flag}] {dataset}/{backbone}/{colorset}\n{shlex.join(cmd)}\n")
        print(f"# {len(todo)} células, {sum(not done(d, b, c) for d, b, c, _ in todo)} por correr")
        return 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    failed = []
    for i, (dataset, backbone, colorset, cmd) in enumerate(todo, 1):
        label = f"{dataset}/{backbone}/{colorset}"
        if done(dataset, backbone, colorset):
            print(f"[{i}/{len(todo)}] {label}: já feita, saltada", flush=True)
            continue
        print(f"[{i}/{len(todo)}] {datetime.now():%H:%M:%S} {label}", flush=True)
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            failed.append(label)
            print(f"    FALHOU (rc={rc})", flush=True)
    mins = (time.time() - start) / 60
    print(f"\nfim: {mins:.1f} min | {len(failed)} falhas" +
          ("".join(f"\n  - {f}" for f in failed) if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
