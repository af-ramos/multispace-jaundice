"""
make_fig4.py — Figura 4 do paper: dose-resposta de calibração (H3).

Lê os marcadores da Fase 2 (`f2_*.json`) e desenha, por dataset e por métrica, a curva
Δ(ARM−RGB) contra a dose de calibração. H3 prevê curva DECRESCENTE (o ganho da cor
encolhe conforme a calibração melhora — efeito compensatório).

Forma: slopegraph com barras de IC95 por ponto. A pergunta é "esta linha desce?", então
o que precisa ser legível é a INCLINAÇÃO de cada backbone e a referência Δ=0 — não os
valores absolutos. Uma linha por backbone; painéis por dataset (doses diferentes, escalas
diferentes: nunca um eixo duplo).

Cores: slots 1–5 da paleta categórica de referência do skill `dataviz`, em ORDEM FIXA
(nunca cicladas). Cor identifica o backbone e nada mais; todo texto usa tokens de tinta.

Local, sem GPU. Uso:
    python -m src.make_fig4
Saída: paper/figs/fig4_dose_resposta.{pdf,png}
"""
from __future__ import annotations

import glob
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats as sstats


def t95(n: int) -> float:
    """Valor crítico bilateral 95% com df = n−1.

    Calculado, não tabelado: uma tabela hard-coded chaveada por df e indexada por n
    já produziu aqui um IC ~7% estreito demais (t=2,571 em vez de 2,776 para n=5)."""
    return float(sstats.t.ppf(0.975, n - 1)) if n > 1 else 0.0

# Paleta categórica de referência (dataviz/references/palette.md), slots 1-5, ordem fixa.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e3e2dd"

LEVELS = {"NeoJaundice": ["off", "on"], "NJN": ["full_image", "skin_roi"]}
XLBL = {"NeoJaundice": ["sem WB", "WB graypatch"],
        "NJN": ["imagem inteira", "ROI de pele"]}
DSLBL = {"NJN": "NJN  ·  dose de ROI", "NeoJaundice": "NeoJaundice  ·  dose de white balance"}
METRICS = [("roc_auc", "Δ AUC  (braço de cor − RGB)", 1.0),
           ("accuracy", "Δ acurácia  (p.p.)", 1.0)]


def load():
    M = {}
    for f in glob.glob("local/runs/*/*/f2_*.json"):
        d = json.load(open(f))
        lvl = d["wb"] if d["dataset"] == "NeoJaundice" else d["njn_mode"]
        M[(d["dataset"], d["backbone"], lvl, d["colorset_id"])] = d
    arms = json.load(open("local/runs/fase2_selected_arms.json"))["selection"]
    return M, arms


def delta_ci(M, ds, bb, lvl, arm, metric):
    """Δ(arm−RGB) pareado por seed + meia-largura do IC95 (t de Student, n=5)."""
    r, a = M[(ds, bb, lvl, "RGB")], M[(ds, bb, lvl, arm)]
    sr = {int(s): v[metric] for s, v in r["per_seed_metrics"].items()}
    sa = {int(s): v[metric] for s, v in a["per_seed_metrics"].items()}
    dl = [sa[s] - sr[s] for s in sorted(set(sr) & set(sa))]
    n = len(dl)
    m = st.mean(dl)
    hw = 0.0 if n < 2 else t95(n) * st.stdev(dl) / math.sqrt(n)
    return m, hw


def main() -> int:
    M, arms = load()
    backbones = sorted({k[1] for k in M})
    color = {bb: SERIES[i % len(SERIES)] for i, bb in enumerate(backbones)}

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4))
    fig.patch.set_facecolor("white")

    for row, (metric, ylab, _) in enumerate(METRICS):
        for col, ds in enumerate(["NJN", "NeoJaundice"]):
            ax = axes[row][col]
            lv = LEVELS[ds]
            slopes = []
            for bb in backbones:
                arm = arms[ds][bb]["arm"]
                if (ds, bb, lv[0], arm) not in M:
                    continue
                ys, hs = zip(*(delta_ci(M, ds, bb, l, arm, metric) for l in lv))
                slopes.append(ys[1] - ys[0])
                ax.errorbar([0, 1], ys, yerr=hs, color=color[bb], lw=1.8,
                            marker="o", markersize=6.5, capsize=3, capthick=1.1,
                            elinewidth=1.1, zorder=3,
                            markeredgecolor="white", markeredgewidth=1.2)

            # Δ=0: a referência que decide o sinal do efeito de cor
            ax.axhline(0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
            ax.set_xlim(-0.35, 1.35)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(XLBL[ds], color=INK2, fontsize=9)
            ax.tick_params(axis="y", labelsize=8.5, colors=INK2, length=0)
            ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right", "bottom", "left"):
                ax.spines[side].set_visible(False)
            if col == 0:
                ax.set_ylabel(ylab, color=INK, fontsize=9.5)
            if row == 0:
                ax.set_title(DSLBL[ds], color=INK, fontsize=10, pad=10)

            # inclinação média do painel: o teste de H3 agregado
            if slopes:
                n = len(slopes)
                m = st.mean(slopes)
                hw = t95(n) * st.stdev(slopes) / math.sqrt(n) if n > 1 else 0.0
                sig = (m - hw) * (m + hw) > 0
                dec = "%+.4f" % m if metric == "roc_auc" else "%+.2f p.p." % m
                # No TOPO: embaixo colidia com os xticklabels e com as barras de IC.
                # Negrito só quando o IC exclui zero — é a única leitura acionável.
                ax.text(0.5, 0.99,
                        f"inclinação média {dec}  ·  IC95 "
                        f"{'exclui' if sig else 'inclui'} 0",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=8, color=INK if sig else MUTED,
                        fontweight="bold" if sig else "normal")

    handles = [Line2D([], [], color=color[bb], lw=1.8, marker="o", markersize=6,
                      markeredgecolor="white", markeredgewidth=1.0, label=bb)
               for bb in backbones]
    fig.legend(handles=handles, loc="lower center", ncol=len(backbones), frameon=False,
               fontsize=8.5, labelcolor=INK2, bbox_to_anchor=(0.5, -0.005),
               handletextpad=0.5, columnspacing=1.6)

    fig.suptitle("Dose-resposta da calibração de cor (H3)", color=INK, fontsize=12.5,
                 x=0.5, y=0.985)
    fig.text(0.5, 0.935,
             "H3 prevê curvas DECRESCENTES: o ganho da cor encolhe conforme a calibração melhora. "
             "Barras = IC95 pareado por seed (n=5).",
             ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.045, 1, 0.925])

    out = Path("local/generated/figs")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out / f"fig4_dose_resposta.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[fig4] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
