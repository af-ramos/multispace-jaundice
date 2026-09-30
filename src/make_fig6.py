"""
make_fig6.py — Figura 6 do paper: H4, efeito da cor estratificado por tom de pele (ITA°).

Duas linhas, porque o resultado de H4 tem duas metades e a de cima explica a de baixo:

  linha 1  DISTRIBUIÇÃO de ITA° por dataset, com as faixas de Fitzpatrick sombreadas.
           É aqui que se vê POR QUE H4 fica inconclusiva: os dois datasets se concentram
           em `tan`/`brown` e quase não têm pele muito clara nem muito escura.
  linha 2  Δ(ARM − RGB) por faixa, com IC95 pareado por seed. Faixas com n < 25 saem em
           CINZA e hachuradas — presentes, porém marcadas como não interpretáveis. Omiti-las
           esconderia a lacuna de cobertura, que é justamente o achado.

Local, sem GPU. Uso:  python -m src.make_fig6
Saída: paper/figs/fig6_ita_estratificacao.{pdf,png}
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .ita_strat import ITA_BANDS, MIN_N

# Paleta de referência (dataviz): slots categóricos 1 e 2 = um dataset cada.
DS_COLOR = {"NJN": "#2a78d6", "NeoJaundice": "#eb6834"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e3e2dd"
INCONC = "#b9b8b2"                      # cinza para faixa não interpretável
BAND_ORDER = [lbl for _, lbl in ITA_BANDS][::-1]      # escura -> clara no eixo
f = lambda s: float(str(s).replace(",", "."))


def load():
    strat = defaultdict(dict)
    for r in csv.DictReader(open("local/runs/fase5_ita_estratificacao.csv",
                                 encoding="utf-8-sig"), delimiter=";"):
        strat[r["dataset"]][r["faixa_ita"]] = r
    per_img = defaultdict(list)
    for r in csv.DictReader(open("local/runs/fase5_ita_por_imagem.csv",
                                 encoding="utf-8-sig"), delimiter=";"):
        per_img[r["dataset"]].append(f(r["ita_graus"]))
    return strat, per_img


def main() -> int:
    strat, per_img = load()
    datasets = ["NJN", "NeoJaundice"]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8),
                             gridspec_kw={"height_ratios": [1.0, 1.25]})
    fig.patch.set_facecolor("white")

    # ---------- linha 1: distribuição de ITA° ----------
    edges = [b[0] for b in ITA_BANDS if b[0] > -1e8]          # 55, 41, 28, 10, -30
    for col, ds in enumerate(datasets):
        ax = axes[0][col]
        v = np.asarray(per_img[ds])
        ax.hist(v, bins=36, color=DS_COLOR[ds], alpha=0.85, edgecolor="white", linewidth=0.6)
        for e in edges:
            ax.axvline(e, color=MUTED, lw=0.9, ls=(0, (3, 3)), zorder=3)
        ax.set_title(f"{ds}  —  n={len(v)} imagens de test", fontsize=10, color=INK, pad=8)
        ax.set_xlabel("ITA°  (maior = pele mais clara)", fontsize=8.5, color=INK2)
        if col == 0:
            ax.set_ylabel("imagens", fontsize=9, color=INK)
        ax.tick_params(labelsize=8, colors=INK2, length=0)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.98, 0.93, f"mediana {np.median(v):.1f}°", transform=ax.transAxes,
                ha="right", fontsize=8.5, color=INK2)

    # ---------- linha 2: Δ por faixa ----------
    for col, ds in enumerate(datasets):
        ax = axes[1][col]
        labels, vals, los, his, ns, ok = [], [], [], [], [], []
        for lbl in BAND_ORDER:
            r = strat[ds].get(lbl)
            if not r:
                continue
            n = int(r["n_imagens"])
            labels.append(f"{lbl}\n(n={n})")
            vals.append(f(r["delta"]))
            has_ci = r["ic95_lo"] not in ("", None)
            los.append(f(r["ic95_lo"]) if has_ci else f(r["delta"]))
            his.append(f(r["ic95_hi"]) if has_ci else f(r["delta"]))
            ns.append(n)
            ok.append(n >= MIN_N and "INCONCLUSIVA" not in r["veredito"])

        y = np.arange(len(labels))
        for i in range(len(labels)):
            c = DS_COLOR[ds] if ok[i] else INCONC
            ax.errorbar(vals[i], y[i], xerr=[[vals[i] - los[i]], [his[i] - vals[i]]],
                        fmt="o", color=c, markersize=7.5, capsize=3.5, capthick=1.1,
                        elinewidth=1.4, markeredgecolor="white", markeredgewidth=1.2,
                        zorder=3)
            if not ok[i]:
                ax.text(his[i] + 0.6, y[i], "inconclusiva", va="center", fontsize=7.2,
                        color=MUTED, style="italic")

        ax.axvline(0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8, color=INK2)
        ax.set_xlabel("Δ acurácia (p.p.),  braço de cor − RGB", fontsize=9, color=INK)
        ax.tick_params(labelsize=8, colors=INK2, length=0)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        arm = next(iter(strat[ds].values()))["arm"]
        ax.set_title(f"{ds}  ·  braço {arm}", fontsize=9.5, color=INK, pad=8)

    fig.suptitle("H4 — efeito da cor por tom de pele (ITA°), backbone headline deit_tiny",
                 fontsize=12.5, color=INK, y=1.02)
    fig.text(0.5, 0.972,
             f"ITA° medido sobre os pixels de PELE. Barras = IC95 pareado por seed (n=5). "
             f"Faixas com n < {MIN_N} em cinza: não interpretadas.",
             ha="center", va="top", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.955])

    out = Path("local/generated/figs")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out / f"fig6_ita_estratificacao.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[fig6] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
