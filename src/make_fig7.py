"""
make_fig7.py — Heatmap backbone × colorset em **Δ ACURÁCIA** (p.p.).

Contraparte da Fig 3 no eixo que o paper passa a reportar. A Fig 3 mostra Δ AUC, que é
métrica de RANQUEAMENTO; esta mostra Δ acurácia, que é o PONTO DE OPERAÇÃO — e as duas
não coincidem. Em `densenet121`×NJN a AUC sobe +0,010 (nada) e a acurácia sobe até
+4,5 p.p.: a cromância não reordena os casos, ela os afasta do limiar.

FORMA: heatmap divergente. Δ tem POLARIDADE e o zero é a referência ⇒ paleta divergente
azul↔vermelho com MIDPOINT CINZA NEUTRO (nunca arco-íris, nunca um matiz no meio — o meio
tem de ler como "nada"). Mesma paleta da Fig 3, de propósito: as duas figuras são lidas
lado a lado e uma troca de paleta seria lida como troca de significado.

LINHAS ORDENADAS PELA ACURÁCIA DO BASELINE RGB (mais fraco em cima), COLUNAS = os 7
colorsets que contêm RGB. Os chroma-only ficam de fora: descartar o RGB é outra
intervenção, e a perda que ela causa dominaria a escala e esconderia o efeito de interesse.

A coluna de rótulos traz `k/7` = em quantos colorsets o Δ é positivo. É a estatística que
sustenta o argumento: uma linha 7/7 é um efeito do backbone, não um colorset sorteado.

Local, sem GPU. Uso:
    python -m src.make_fig7
Saída: paper/figs/fig7_heatmap_delta_acuracia.{pdf,png}
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
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import stats as sstats

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
# Paleta DIVERGENTE (idêntica à da Fig 3): azul ↔ vermelho, midpoint cinza neutro.
DIVERGING = LinearSegmentedColormap.from_list("bl_gy_rd", [
    "#184f95", "#2a78d6", "#86b6ef", "#cde2fb",
    "#f0efec",                                    # midpoint neutro = Δ 0
    "#f7d3d3", "#e34948", "#d03b3b", "#8f1f1f",
])

#: Ordem das colunas: por número de espaços somados ao RGB, depois alfabética. Assim a
#: figura responde de relance "adicionar MAIS espaços ajuda?" — a resposta é: não.
COLORSETS = ["RGB+HSV", "RGB+LAB", "RGB+YCrCb",
             "RGB+LAB+HSV", "RGB+LAB+YCrCb", "RGB+YCrCb+HSV",
             "RGB+LAB+YCrCb+HSV"]
COLLBL = ["+HSV", "+LAB", "+YCrCb", "+LAB\n+HSV", "+LAB\n+YCrCb", "+YCrCb\n+HSV",
          "+LAB+YCrCb\n+HSV"]
METRICA = "accuracy"


def t95(n: int) -> float:
    """Crítico bilateral 95%, df = n−1. Calculado, nunca tabelado."""
    return float(sstats.t.ppf(0.975, n - 1)) if n > 1 else 0.0


def load():
    """Marcadores da Fase 1 (regime não calibrado), por (dataset, backbone, colorset)."""
    M = {}
    for f in glob.glob("local/runs/*/*/f1_*.json"):
        d = json.load(open(f))
        if d.get("status") != "completed":
            continue
        M[(d["dataset"], d["backbone"], d["colorset_id"])] = d
    return M


def ps(d, metric=METRICA):
    return {int(s): v[metric] for s, v in d["per_seed_metrics"].items()}


def delta(M, ds, bb, cs):
    """Δ(cs − RGB) pareado por seed + meia-largura do IC95. None se a célula falta."""
    r, c = M.get((ds, bb, "RGB")), M.get((ds, bb, cs))
    if not r or not c:
        return None
    a, b = ps(r), ps(c)
    dl = [b[s] - a[s] for s in sorted(set(a) & set(b))]
    if not dl:
        return None
    n, m = len(dl), st.mean(dl)
    return m, (t95(n) * st.stdev(dl) / math.sqrt(n) if n > 1 else 0.0)


def main() -> int:
    M = load()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.8))
    fig.patch.set_facecolor("white")
    # Cada painel tem a sua própria ordem de linhas (ordenada pela força NAQUELE dataset),
    # então cada um precisa dos seus rótulos — daí a folga horizontal.
    fig.subplots_adjust(wspace=0.62)

    for col, ds in enumerate(["NJN", "NeoJaundice"]):
        ax = axes[col]
        backbones = sorted({k[1] for k in M if k[0] == ds and k[2] == "RGB"})
        # ordenar por força do backbone = acurácia do baseline RGB (fraco em cima)
        base_acc = {bb: st.mean(ps(M[(ds, bb, "RGB")]).values()) for bb in backbones}
        order = sorted(backbones, key=lambda b: base_acc[b])

        grid = np.full((len(order), len(COLORSETS)), np.nan)
        ann = [["" for _ in COLORSETS] for _ in order]
        for i, bb in enumerate(order):
            for j, cs in enumerate(COLORSETS):
                r = delta(M, ds, bb, cs)
                if r is None:
                    continue
                m, hw = r
                grid[i, j] = m
                ann[i][j] = f"{m:+.1f}{'*' if (m - hw) * (m + hw) > 0 else ''}"

        lim = float(np.nanmax(np.abs(grid)))
        ax.imshow(grid, cmap=DIVERGING, norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                  aspect="auto")

        for i in range(len(order)):
            for j in range(len(COLORSETS)):
                if np.isnan(grid[i, j]):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=8, color=MUTED)
                    continue
                escuro = abs(grid[i, j]) > 0.62 * lim   # tinta clara só sobre fundo escuro
                ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=8,
                        color=("white" if escuro else INK),
                        fontweight="bold" if ann[i][j].endswith("*") else "normal")

        # Rótulo de linha carrega o baseline e a contagem de sinal — é o que torna a
        # figura auto-suficiente sem consultar a tabela A2.
        rotulos = []
        for i, bb in enumerate(order):
            k = int(np.nansum(grid[i] > 0))
            tot = int(np.sum(~np.isnan(grid[i])))
            rotulos.append(f"{bb}   RGB {base_acc[bb]:.1f}%   {k}/{tot}")

        ax.set_xticks(range(len(COLORSETS)))
        ax.set_xticklabels(COLLBL, fontsize=8, color=INK2)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(rotulos, fontsize=7.6, color=INK2)
        if col == 1:
            ax.yaxis.tick_right()      # senão os rótulos caem sobre o painel da esquerda
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        # 2px de superfície entre células (separador, não borda)
        ax.set_xticks(np.arange(-.5, len(COLORSETS), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
        ax.grid(which="minor", color="white", lw=2)
        ax.tick_params(which="minor", length=0)
        ax.set_title(f"{ds}   (escala ±{lim:.1f} p.p.)", fontsize=11, color=INK, pad=10)
        if col == 0:
            ax.set_ylabel("backbone · acurácia do baseline RGB · colorsets com Δ>0"
                          "   —   mais forte para baixo ↓", fontsize=9, color=INK)

    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.04, pad=0.10,
                      ticks=[-1, 0, 1])
    cb.ax.set_xticklabels(["cor PIORA", "Δ = 0", "cor AJUDA"], fontsize=8.5, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)

    fig.suptitle("Efeito da cromância sobre a ACURÁCIA, por backbone e colorset",
                 fontsize=12.5, color=INK, y=1.045)
    fig.text(0.5, 0.995,
             "Célula = Δ acurácia em pontos percentuais (colorset − RGB), pareado por seed "
             "(n=5).  * = IC95 exclui zero.  Escala normalizada por painel.\n"
             "O efeito não é do espaço de cor (colunas homogêneas) e sim do backbone "
             "(linhas 7/7 ou 0/7) — comparar com a Fig 3, em AUC, onde ele é invisível.",
             ha="center", va="top", fontsize=8.5, color=MUTED)

    out = Path("local/generated/figs")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out / f"fig7_heatmap_delta_acuracia.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[fig7] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
