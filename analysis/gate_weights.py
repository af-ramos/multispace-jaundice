#!/usr/bin/env python3
"""Pesos por espaço do gate do ``adapter_v2``, colhidos dos markers da campanha.

## Porque este módulo existe

O manuscrito afirmava, em dois sítios, que os pesos do gate "were not persisted during
the campaign". **Era falso, e por uma confusão entre duas coisas diferentes:**

* os **pesos do modelo** (``*.pt``) não existem, de facto — a v7 nunca chamou
  ``torch.save``, e é por isso que Grad-CAM ou rollout exigem retreinar;
* o **α por espaço** que o gate produz existe: ``cv._collect_interpretability`` lia
  ``fusion.gate.last_weights`` no fim de cada fold e gravava-o no marker JSON como
  ``alpha_space`` / ``alpha_space_names``.

O ``docs/PENDENCIAS.md`` item 14 dava o experimento como "bloqueado" com base na mesma
confusão. Este módulo lê o que sempre lá esteve.

## O que o número é, e o que não é

``alpha_space`` é a **média sobre o test loader** do vector $w \\in \\mathbb{R}^K$ que o
``_SpaceGate`` prevê por imagem, na escala em que **1,0 é a neutralidade** (a init
zera os logits e o softmax é escalado por K, portanto w = 1 em todos os espaços à
partida). É diagnóstico **descritivo**: não entra em selecção nenhuma, não decide
limiar, arm nem hiperparâmetro, e nenhuma métrica reportada depende dele.

E não é importância de feature. A convolução 1x1 a jusante pode reescalar, cancelar ou
recombinar qualquer canal que o gate já ponderou, portanto um peso alto diz que o
modelo **amplifica** aquele espaço à entrada, não que a saída dependa dele.

## Cobertura

A colheita não corria em todas as células: dos 550 markers, 86 têm ``alpha_space``, e
o subconjunto canónico (``f1_*`` + ``adapter_v2`` + agrupado, na condição canónica de
cada dataset) dá **27 células com K >= 2**, uma por par (dataset, backbone). É amostra
ilustrativa, larga em backbones e rasa em colour sets — **não é o painel dos 210**, e
nenhuma afirmação aqui pode ser generalizada a ele.

## O α por imagem (emenda de 2026-08-18)

O retreino de `scripts/run_explain_cells.py` gravou, em seis pares, o vector do gate
**por imagem** em vez da sua média. Isso responde a uma pergunta que a média não pode
responder de todo: um α médio de 1,01 tanto pode vir de um gate parado como de um gate
que oscila muito e cujas oscilações se cancelam. São coisas diferentes e distinguem-se.

``--per-image`` lê os ``*__gate.npz`` de `local/experiments/explain/` e escreve, ao lado do
spread médio, o spread **dentro de cada imagem** — mediana e percentil 95.

Uso:  python analysis/gate_weights.py [--markers DIR]
      python analysis/gate_weights.py --per-image
Saídas: results/G1_gate_weights.csv
        results/G3_gate_per_image.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
# Preserved campaign markers, copied byte-for-byte with provenance checksums.
MARKERS = ROOT / "evidence/v7/markers"


def canonical(d: dict) -> bool:
    """A mesma célula canónica que `analysis/dumps.py` reconhece, pelos campos do marker.

    Lida pelos campos e não pelo nome do ficheiro: o nome já enganou duas vezes neste
    projecto, e o marker traz `dataset`, `fusion` e `njn_mode`/`wb` explícitos.
    """
    if not str(d.get("tag", "")).startswith("f1_"):
        return False
    if d.get("fusion") != "adapter_v2" or d.get("multitask"):
        return False
    if d.get("split_mode") != "group":
        return False
    if d.get("dataset") == "NJN":
        return d.get("njn_mode") == "full_image"
    return d.get("wb") == "off"


def load(markers: Path) -> list[dict]:
    rows = []
    for path in sorted(markers.glob("*/*/*.json")):
        d = json.loads(path.read_text())
        weights, names = d.get("alpha_space"), d.get("alpha_space_names")
        if not weights or not names or len(weights) != len(names):
            continue
        if not canonical(d):
            continue
        rows.append({
            "dataset": d["dataset"],
            "backbone": d["backbone"],
            "colorset": d["colorset_id"],
            "K": len(weights),
            "names": list(names),
            "weights": [float(w) for w in weights],
        })
    return rows


def poisson_binomial_upper(probs: list[float], k: int) -> float:
    """P(X >= k) para somas de Bernoulli com probabilidades distintas.

    Exacta, por convolução. Serve o único teste feito aqui: sob indiferença o gate
    escolheria cada um dos K espaços com probabilidade 1/K, e essa probabilidade
    difere entre células porque o K difere.
    """
    dist = [1.0]
    for p in probs:
        nxt = [0.0] * (len(dist) + 1)
        for i, v in enumerate(dist):
            nxt[i] += v * (1 - p)
            nxt[i + 1] += v * p
        dist = nxt
    return sum(dist[k:])


#: Onde o retreino com ``--save-explain`` deixou o gate por imagem.
EXPLAIN = ROOT / "local/experiments/explain"


def per_image(explain: Path) -> int:
    """Spread do gate DENTRO de cada imagem, contra o spread da média.

    A distinção importa: um gate cujo α médio é ~1 em todos os espaços pode estar parado
    ou pode estar a oscilar por imagem de forma que se cancela na média. A primeira
    leitura diz que o gate não faz nada; a segunda diz que faz algo que a agregação
    apaga. Só o α por imagem separa as duas.
    """
    import numpy as np

    files = sorted(explain.glob("*/*/explain/*__gate.npz"))
    if not files:
        print(f"sem gate por imagem em {explain}", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "G3_gate_per_image.csv"
    rows = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        names = [str(x) for x in d["space_names"]]
        if len(names) < 2:
            continue                                   # o braço RGB não tem gate a medir
        w = np.asarray(d["w"], dtype=np.float64)
        mean = w.mean(0)
        spread_img = w.max(1) - w.min(1)
        rows.append({
            "dataset": f.parts[-4], "backbone": f.parts[-3],
            "colorset": "+".join(names), "K": len(names),
            "n_images": len(set(d["paths"].tolist())), "n_rows": len(w),
            "spread_mean": mean.max() - mean.min(),
            "spread_img_median": float(np.median(spread_img)),
            "spread_img_p95": float(np.percentile(spread_img, 95)),
            "spread_img_max": float(spread_img.max()),
            "ratio": float(np.median(spread_img)) / max(mean.max() - mean.min(), 1e-12),
            "weights": "+".join(f"{v:.4f}" for v in mean),
        })

    cols = ["dataset", "backbone", "colorset", "K", "n_images", "n_rows",
            "spread_mean", "spread_img_median", "spread_img_p95", "spread_img_max",
            "ratio", "weights"]
    with dest.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, delimiter=";")
        wr.writeheader()
        for r in sorted(rows, key=lambda r: -r["spread_img_median"]):
            wr.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k])
                         for k in cols})

    print(f"escrito: {dest.relative_to(ROOT)}   ({len(rows)} células com K >= 2)")
    print(f"{'dataset':12s} {'backbone':18s} {'spread médio':>12s} "
          f"{'mediana/imagem':>15s} {'p95/imagem':>11s} {'razão':>7s}")
    for r in sorted(rows, key=lambda r: -r["spread_img_median"]):
        print(f"{r['dataset']:12s} {r['backbone']:18s} {r['spread_mean']:12.4f} "
              f"{r['spread_img_median']:15.4f} {r['spread_img_p95']:11.4f} "
              f"{r['ratio']:7.1f}x")
    med_ratio = st.median([r["ratio"] for r in rows])
    print(f"\nA média subestima o movimento do gate por um factor mediano de "
          f"{med_ratio:.1f}x nestas {len(rows)} células.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markers", type=Path, default=MARKERS)
    ap.add_argument("--per-image", action="store_true",
                    help="lê o gate POR IMAGEM de local/experiments/explain/ (seis pares)")
    ap.add_argument("--explain", type=Path, default=EXPLAIN)
    args = ap.parse_args(argv)

    if args.per_image:
        return per_image(args.explain)

    if not args.markers.is_dir():
        print(f"markers não encontrados em {args.markers}", file=sys.stderr)
        return 1

    rows = load(args.markers)
    multi = [r for r in rows if r["K"] >= 2]
    if not multi:
        print("nenhuma célula canónica com K >= 2 e alpha_space", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "G1_gate_weights.csv"
    with dest.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["dataset", "backbone", "colorset", "K", "spread",
                    "argmax_space", "w_rgb", "spaces", "weights"])
        for r in sorted(multi, key=lambda r: -(max(r["weights"]) - min(r["weights"]))):
            spread = max(r["weights"]) - min(r["weights"])
            argmax = r["names"][r["weights"].index(max(r["weights"]))]
            w_rgb = r["weights"][r["names"].index("RGB")] if "RGB" in r["names"] else ""
            w.writerow([r["dataset"], r["backbone"], r["colorset"], r["K"],
                        f"{spread:.4f}", argmax,
                        f"{w_rgb:.4f}" if w_rgb != "" else "",
                        "+".join(r["names"]),
                        "+".join(f"{v:.4f}" for v in r["weights"])])

    spreads = [max(r["weights"]) - min(r["weights"]) for r in multi]
    with_rgb = [r for r in multi if "RGB" in r["names"]]
    top = [r for r in with_rgb if r["names"][r["weights"].index(max(r["weights"]))] == "RGB"]
    p = poisson_binomial_upper([1.0 / r["K"] for r in with_rgb], len(top))

    print(f"escrito: {dest.relative_to(ROOT)}")
    print(f"células canónicas com K >= 2: {len(multi)} "
          f"({sum(1 for r in multi if r['dataset'] == 'NJN')} NJN, "
          f"{sum(1 for r in multi if r['dataset'] != 'NJN')} NeoJaundice), "
          f"{len({(r['dataset'], r['backbone']) for r in multi})} pares (dataset, backbone)")
    print(f"spread do gate: mediana {st.median(spreads):.3f} | "
          f"máximo {max(spreads):.3f} | mínimo {min(spreads):.4f}   [1,0 = neutralidade]")
    print(f"RGB é o espaço de maior peso em {len(top)}/{len(with_rgb)} células que o contêm; "
          f"esperado sob indiferença {sum(1.0 / r['K'] for r in with_rgb):.2f}; "
          f"P(X >= {len(top)}) = {p:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
