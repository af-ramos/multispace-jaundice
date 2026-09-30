"""
make_fig3.py — Figura 3 do paper: heatmap backbone × calibração (H2).

Junta as Fases 1 (não-calibrado, 15 backbones), 2 (calibrado, 5 backbones) e 3 (calibrado,
DINOv3 = controle forte×calibrado). Célula = Δ(cor−RGB) em AUC.

FORMA: heatmap divergente. Δ tem POLARIDADE (a cor pode ajudar ou atrapalhar) e o zero é
o valor de referência — logo, paleta divergente azul↔vermelho com MIDPOINT CINZA NEUTRO
(nunca arco-íris, nunca um matiz no meio: o meio precisa ler como "nada").

LINHAS ORDENADAS POR FORÇA DO BACKBONE (AUC do baseline RGB não-calibrado, fraco em cima).
Isso faz a figura responder H2 diretamente: se o efeito da cor é COMPENSATÓRIO, o |Δ| deve
encolher para BAIXO (backbone mais forte) e para a DIREITA (mais calibrado).

COMPARABILIDADE: dentro de uma linha, as duas colunas usam o MESMO colorset — senão o par
não é uma dose. O colorset varia ENTRE linhas (Fase 2 usa o braço selecionado por
validação; Fase 3 usa o canônico RGB+YCrCb+HSV) e por isso vai anotado em cada linha.
Linhas sem célula calibrada (backbones fora do roster da Fase 2) usam o canônico e ficam
com a coluna calibrada VAZIA — ausência é mostrada como ausência, nunca interpolada.

Local, sem GPU. Uso:
    python -m src.make_fig3
Saída: paper/figs/fig3_heatmap_backbone_calibracao.{pdf,png}
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

CANON = "RGB+YCrCb+HSV"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
# Paleta DIVERGENTE de referência (dataviz): azul ↔ vermelho, midpoint cinza neutro.
DIVERGING = LinearSegmentedColormap.from_list("bl_gy_rd", [
    "#184f95", "#2a78d6", "#86b6ef", "#cde2fb",
    "#f0efec",                                    # midpoint neutro = Δ 0
    "#f7d3d3", "#e34948", "#d03b3b", "#8f1f1f",
])
CAL = {"NeoJaundice": "on", "NJN": "skin_roi"}
COLLBL = {"NJN": ["não calibrado\n(imagem inteira)", "calibrado\n(ROI de pele)"],
          "NeoJaundice": ["não calibrado\n(sem WB)", "calibrado\n(WB graypatch)"]}


def t95(n: int) -> float:
    """Crítico bilateral 95%, df = n−1. Calculado, nunca tabelado (ver make_fig4)."""
    return float(sstats.t.ppf(0.975, n - 1)) if n > 1 else 0.0


def load():
    """Marcadores das 3 fases, indexados por (dataset, backbone, calibrado?, colorset)."""
    M = {}
    for f in glob.glob("local/runs/*/*/f[123]_*.json"):
        d = json.load(open(f))
        cal = (d["wb"] == "on") if d["dataset"] == "NeoJaundice" else (d.get("njn_mode") == "skin_roi")
        M[(d["dataset"], d["backbone"], cal, d["colorset_id"])] = d
    arms = json.load(open("local/runs/fase2_selected_arms.json"))["selection"]
    return M, arms


def ps(d, metric="roc_auc"):
    return {int(s): v[metric] for s, v in d["per_seed_metrics"].items()}


def delta(M, ds, bb, cal, cs):
    """Δ(cs − RGB) pareado por seed + meia-largura do IC95. None se a célula não existe."""
    r, c = M.get((ds, bb, cal, "RGB")), M.get((ds, bb, cal, cs))
    if not r or not c:
        return None
    a, b = ps(r), ps(c)
    dl = [b[s] - a[s] for s in sorted(set(a) & set(b))]
    n = len(dl)
    m = st.mean(dl)
    return m, (t95(n) * st.stdev(dl) / math.sqrt(n) if n > 1 else 0.0)


def main() -> int:
    M, arms = load()
    backbones = sorted({k[1] for k in M if k[2] is False})

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.6))
    fig.patch.set_facecolor("white")
    # Os dois painéis têm ORDENS DE LINHA DIFERENTES (cada dataset ordenado pela sua
    # própria força de backbone), então cada um precisa dos seus rótulos. O da direita vai
    # para o lado direito, senão invade o painel da esquerda.
    fig.subplots_adjust(wspace=0.55)

    for col, ds in enumerate(["NJN", "NeoJaundice"]):
        ax = axes[col]
        # colorset por linha: o do braço calibrado quando existe; senão o canônico
        rowcs = {}
        for bb in backbones:
            if (ds, bb, True, "RGB") in M:
                cand = arms[ds][bb]["arm"] if (ds, bb, True, arms[ds][bb]["arm"]) in M else CANON
                rowcs[bb] = cand if (ds, bb, True, cand) in M else CANON
            else:
                rowcs[bb] = CANON
        # ordenar por força do backbone: AUC do RGB não-calibrado (fraco em cima)
        strength = {bb: st.mean(ps(M[(ds, bb, False, "RGB")]).values())
                    if False else sum(ps(M[(ds, bb, False, "RGB")]).values()) / 5
                    for bb in backbones}
        order = sorted(backbones, key=lambda b: strength[b])

        grid = np.full((len(order), 2), np.nan)
        ann = [["", ""] for _ in order]
        for i, bb in enumerate(order):
            for j, cal in enumerate([False, True]):
                r = delta(M, ds, bb, cal, rowcs[bb])
                if r is None:
                    continue
                m, hw = r
                grid[i, j] = m
                star = "*" if (m - hw) * (m + hw) > 0 else ""
                ann[i][j] = f"{m:+.3f}{star}"

        lim = float(np.nanmax(np.abs(grid)))
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        ax.imshow(grid, cmap=DIVERGING, norm=norm, aspect="auto")

        for i in range(len(order)):
            for j in range(2):
                if np.isnan(grid[i, j]):
                    ax.text(j, i, "não rodado", ha="center", va="center",
                            fontsize=7, color=MUTED, style="italic")
                    continue
                # tinta clara só quando o fundo é escuro o bastante
                dark = abs(grid[i, j]) > 0.62 * lim
                ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=8.5,
                        color=("white" if dark else INK),
                        fontweight="bold" if ann[i][j].endswith("*") else "normal")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(COLLBL[ds], fontsize=8.5, color=INK2)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f"{bb}  ({rowcs[bb]})" for bb in order], fontsize=7.8, color=INK2)
        if col == 1:
            ax.yaxis.tick_right()          # senão os rótulos caem sobre o painel esquerdo
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        # separadores finos entre células (2px de superfície entre preenchimentos)
        ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
        ax.grid(which="minor", color="white", lw=2)
        ax.tick_params(which="minor", length=0)
        ax.set_title(ds, fontsize=11, color=INK, pad=10)
        if col == 0:
            ax.set_ylabel("backbone  (colorset usado)   —   mais forte para baixo ↓",
                          fontsize=9, color=INK)

    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.04, pad=0.09,
                      ticks=[-1, 0, 1])
    cb.ax.set_xticklabels(["cor PIORA", "Δ = 0", "cor AJUDA"], fontsize=8.5, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)

    fig.suptitle("Efeito da cor por backbone e nível de calibração (H2)",
                 fontsize=12.5, color=INK, y=1.045)
    fig.text(0.5, 0.995,
             "Célula = Δ AUC (cor − RGB), pareado por seed.  * = IC95 exclui zero.  "
             "Escala normalizada por painel.\n"
             "H2 prevê |Δ| encolhendo para baixo (backbone mais forte) e para a direita "
             "(mais calibrado).",
             ha="center", va="top", fontsize=8.5, color=MUTED)

    out = Path("local/generated/figs")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out / f"fig3_heatmap_backbone_calibracao.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[fig3] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
