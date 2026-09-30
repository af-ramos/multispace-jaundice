#!/usr/bin/env python3
"""Figuras de dados do artigo, regeneráveis a partir da evidência da v7.

## Escolha de forma, por figura

* **Fig 4 e 5 — heatmaps de delta.** O dado tem duas dimensões categóricas
  (backbone × colorset) e uma grandeza com **polaridade** (melhor ou pior que o
  RGB). Matriz com escala **divergente** centrada em zero é a forma certa. Os
  colorsets são ordenados em dois blocos, *com RGB* e *sem RGB*, para a tese ficar
  legível de relance em vez de exigir leitura célula a célula.
* **Fig 6 — complementaridade.** Quatro estimativas de efeito médio com intervalo.
  Dot plot com barras de erro, não barras: barras partem de zero e sugerem
  contagem, quando aqui o zero é referência e a grandeza é uma diferença.
* **Fig 7 — fusão tardia com controlo.** Efeitos pareados por backbone com IC.
  Forest plot é a forma convencional e a que um revisor de medicina espera.

## Cor

Valores tirados **verbatim** da paleta documentada, nunca escolhidos à mão:
divergente azul `#2a78d6` ↔ cinza neutro `#f0efec` ↔ vermelho `#e34948`; categórico
slot 1 `#2a78d6` e slot 2 `#eb6834`. O validador da paleta não pôde correr nesta
máquina (sem `node`), o que torna o uso de valores documentados mais importante e
não menos — a paleta de origem já está validada; escolher hexes novos aqui seria
introduzir valores por validar.

Como o destino é impressão, a cor nunca carrega sentido sozinha: cada célula do
heatmap leva o seu valor impresso, e os pontos do forest plot levam forma além de
cor. Assim a figura sobrevive a preto-e-branco e a daltonismo.

Saída: `figures/fig4_heatmap_NJN.{pdf,png}` e restantes.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EVIDENCE = ROOT / "evidence/v7"
OUT = ROOT / "local/generated/figures"

from analysis.register_b import (  # noqa: E402
    BACKBONES, CHROMA_ONLY, RGB_PRESERVING, Cell, Key, deltas_against_rgb,
    load_cells,
)

# Paleta documentada — ver o cabeçalho.
BLUE, ORANGE, RED, GRAY = "#2a78d6", "#eb6834", "#e34948", "#f0efec"
INK, INK_SOFT = "#0b0b0b", "#52514e"
DIVERGING = LinearSegmentedColormap.from_list("delta", [RED, GRAY, BLUE])

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.edgecolor": INK_SOFT,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    # TrueType (42) em vez de Type 3: o sistema de submissão da Springer sinaliza Type 3.
    "pdf.fonttype": 42,
})


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=";"))


def num(value: str | None) -> float | None:
    text = (value or "").strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


#: Figuras que o manuscrito inclui, e o nome com que la entram. O `paper.tex` le de
#: `paper/figs/`, nao de `figures/`, e sem esta tabela regenerar uma figura
#: escrevia em `figures/` e o PDF continuava com a versao antiga — foi o que aconteceu
#: quando as figuras foram renomeadas a mao.
IN_PAPER = {"fig4_heatmap_delta": "heatmap_delta"}
FIGS = ROOT / "paper/figs"


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    note = ""
    if name in IN_PAPER:
        FIGS.mkdir(parents=True, exist_ok=True)
        target = FIGS / f"{IN_PAPER[name]}.pdf"
        target.write_bytes((OUT / f"{name}.pdf").read_bytes())
        note = f"  ->  figs/{target.name}"
    print(f"  {name}.pdf / .png{note}")


# --------------------------------------------------------------------------- #

DISPLAY = {"convnextv2_tiny": "ConvNeXtV2-T", "densenet121": "DenseNet121",
           "efficientnet_b0": "EfficientNetB0", "efficientnet_b4": "EfficientNetB4",
           "inception_v3": "Inception-v3", "mobilenetv3_large": "MobileNetV3-L",
           "resnet18": "ResNet18", "resnet50": "ResNet50",
           "deit_base": "DeiT-Base", "deit_small": "DeiT-Small", "deit_tiny": "DeiT-Tiny",
           "dinov3_vits16": "DINOv3-ViT-S/16", "vit_b_16": "ViT-B/16",
           "vit_b_32": "ViT-B/32", "vit_l_16": "ViT-L/16"}


def _delta_grid(cells: dict[Key, Cell], dataset: str
                ) -> tuple[np.ndarray, list[str], list[str]]:
    """Grelha Δacurácia backbone × colorset do campaign record do Register B."""
    backbones = list(BACKBONES)
    colorsets = list(RGB_PRESERVING + CHROMA_ONLY)
    deltas = deltas_against_rgb(cells)
    grid = np.full((len(backbones), len(colorsets)), np.nan)
    for i, backbone in enumerate(backbones):
        for j, colorset in enumerate(colorsets):
            grid[i, j] = deltas[(dataset, backbone, colorset)]
    return grid, backbones, colorsets


def _panel(ax, grid, backbones, colorsets, norm, label: str, show_x: bool) -> object:
    im = ax.imshow(grid, cmap=DIVERGING, norm=norm, aspect="auto")
    limit = max(abs(norm.vmin), abs(norm.vmax))
    for i in range(len(backbones)):
        for j in range(len(colorsets)):
            if np.isnan(grid[i, j]):
                continue
            shade = INK if abs(grid[i, j]) < limit * 0.55 else "white"
            text = f"{grid[i, j]:+.1f}"
            # |delta| < 0.05 arredonda para "+0.0"/"-0.0"; o sinal aí não carrega informação.
            text = "0.0" if text in ("+0.0", "-0.0") else text
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=5.0, color=shade)
    split = sum(1 for c in colorsets if c.startswith("RGB"))
    ax.axvline(split - 0.5, color=INK, linewidth=1.3)
    ax.set_yticks(range(len(backbones)))
    ax.set_yticklabels([DISPLAY.get(b, b) for b in backbones], fontsize=6.2)
    ax.set_xticks(range(len(colorsets)))
    if show_x:
        ax.set_xticklabels([c.replace("+", "+\u200b") for c in colorsets],
                           rotation=45, ha="right", fontsize=6.0)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(label, fontsize=8)
    # a fronteira entre os dois regimes é a tese da figura, e vai rotulada só no topo
    if not show_x:
        ax.text((split - 1) / 2, -1.05, "RGB retained", ha="center", fontsize=7, color=INK)
        ax.text((split + len(colorsets) - 1) / 2, -1.05, "RGB removed",
                ha="center", fontsize=7, color=INK)
    return im


def heatmaps(cells: dict[Key, Cell], individual: bool = True) -> None:
    """Δ acurácia face ao RGB do próprio backbone, os dois datasets num só painel.

    Substitui no artigo a tabela de 115 células: mostra o mesmo painel completo, sem
    filtrar nada, e torna imediatamente visível o que a tabela obrigava a contar — o
    bloco esquerdo (mantém RGB) centrado em zero, o direito (RGB removido) todo
    negativo. A escala de cor é comum aos dois datasets, de propósito: é o que permite
    comparar a imagem global da NJN com o recorte de pele da NeoJaundice.
    """
    panels = []
    for dataset in ("NJN", "NeoJaundice"):
        panels.append((dataset, *_delta_grid(cells, dataset)))

    limit = max(np.nanmax(np.abs(g)) for _, g, _, _ in panels)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.4),
                             gridspec_kw={"hspace": 0.06})
    for ax, (dataset, grid, backbones, colorsets), last in zip(
            axes, panels, (False, True)):
        im = _panel(ax, grid, backbones, colorsets, norm,
                    f"{dataset}\n({'whole-body' if dataset == 'NJN' else 'skin region'})",
                    last)
    bar = fig.colorbar(im, ax=axes, fraction=0.021, pad=0.012)
    bar.set_label("$\\Delta$ accuracy against the same backbone's RGB baseline (p.p.)",
                  fontsize=7)
    bar.ax.tick_params(labelsize=6)
    save(fig, "fig4_heatmap_delta")

    if individual:
        # painéis individuais, para quem os quiser fora do artigo
        for dataset, grid, backbones, colorsets in panels:
            fig, ax = plt.subplots(figsize=(7.2, 4.0))
            im = _panel(ax, grid, backbones, colorsets, norm, dataset, True)
            bar = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.015)
            bar.set_label("$\\Delta$ accuracy (p.p.)", fontsize=7)
            bar.ax.tick_params(labelsize=6)
            save(fig, f"fig4_heatmap_{dataset}" if dataset == "NJN"
                      else f"fig5_heatmap_{dataset}")


def complementarity(cells: dict[Key, Cell]) -> None:
    """O achado central: manter RGB custa ~zero, removê-lo custa muito."""
    fig, ax = plt.subplots(figsize=(6.2, 2.0))
    labels, means, los, his, colors = [], [], [], [], []
    deltas = deltas_against_rgb(cells)
    for dataset in ("NJN", "NeoJaundice"):
        for group, sets, colour in (
            ("RGB + chroma", RGB_PRESERVING, BLUE),
            ("chroma only (RGB removed)", CHROMA_ONLY, ORANGE),
        ):
            values = [deltas[(dataset, backbone, colorset)]
                      for backbone in BACKBONES for colorset in sets]
            mean = statistics.fmean(values)
            half = 1.96 * statistics.stdev(values) / (len(values) ** 0.5)
            labels.append(f"{dataset}\n{group}")
            means.append(mean)
            los.append(mean - half)
            his.append(mean + half)
            colors.append(colour)

    y = np.arange(len(labels))
    ax.axvline(0, color=INK_SOFT, linewidth=1.0, zorder=1)
    for i, (m, lo, hi, c) in enumerate(zip(means, los, his, colors)):
        ax.plot([lo, hi], [i, i], color=c, linewidth=2.4, solid_capstyle="round", zorder=2)
        ax.plot(m, i, "o", color=c, markersize=8, zorder=3,
                markeredgecolor="white", markeredgewidth=1.4)
        ax.text(hi + 0.25, i, f"{m:+.2f}", va="center", fontsize=7, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Δ acurácia face ao RGB (p.p.) — média sobre as células, com IC95",
                  fontsize=7.5)
    ax.set_title("Acrescentar croma ao RGB é gratuito; removê-lo é caro",
                 fontsize=9, pad=8)
    ax.tick_params(labelsize=7)
    save(fig, "fig6_complementaridade")


def forest(control: list[dict[str, str]]) -> None:
    """Fusão tardia: contra um modelo RGB único e contra um ensemble de custo igual."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.4), sharex=True)
    for ax, dataset in zip(axes, ("NJN", "NeoJaundice")):
        rows = [r for r in control if r["dataset"] == dataset]
        rows.sort(key=lambda r: num(r["delta_vs_custo_igual"]) or 0.0)
        y = np.arange(len(rows))
        ax.axvline(0, color=INK_SOFT, linewidth=1.0, zorder=1)
        for i, r in enumerate(rows):
            for key, colour, marker, offset in (
                ("vs_rgb_unico", ORANGE, "s", +0.18),
                ("vs_custo_igual", BLUE, "o", -0.18),
            ):
                m = num(r[f"delta_{key}"])
                lo, hi = num(r[f"ic95_lo_{key}"]), num(r[f"ic95_hi_{key}"])
                ax.plot([lo, hi], [i + offset] * 2, color=colour, linewidth=1.5,
                        solid_capstyle="round", zorder=2)
                ax.plot(m, i + offset, marker, color=colour, markersize=4.6, zorder=3,
                        markeredgecolor="white", markeredgewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([r["backbone"] for r in rows], fontsize=6.4)
        ax.invert_yaxis()
        ax.set_title(dataset, fontsize=8.5)
        ax.set_xlabel("Δ acurácia (p.p.)", fontsize=7.5)
        ax.tick_params(labelsize=6.6)

    # forma além de cor, para a figura sobreviver a preto-e-branco
    handles = [
        plt.Line2D([], [], color=ORANGE, marker="s", linewidth=1.5, markersize=4.6,
                   label="vs. modelo RGB único"),
        plt.Line2D([], [], color=BLUE, marker="o", linewidth=1.5, markersize=4.6,
                   label="vs. ensemble RGB de custo igual"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Fusão tardia de espaços de cor, com e sem controlo de custo",
                 fontsize=9.5, y=0.99)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, "fig7_fusao_tardia")


def main() -> int:
    cells = load_cells()
    paper_figure_only = "--paper-figure4-only" in sys.argv[1:]
    print("figuras escritas em figures/:")
    heatmaps(cells, individual=not paper_figure_only)
    if paper_figure_only:
        return 0
    complementarity(cells)
    out = ROOT / "results"
    if (out / "equal_cost_control.csv").is_file():
        forest(read(out / "equal_cost_control.csv"))
    else:
        print("  (fig7 saltada: corra analysis/equal_cost_control.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
