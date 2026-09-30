"""Figura da atribuição: o que a cromância move, e contra que piso de ruído.

Desenho, e a razão dele. Um par de mapas "RGB" e "multi-espaço" lado a lado não é
interpretável sozinho, porque o leitor não tem escala: não sabe quanto dois mapas do
MESMO modelo já diferem quando só muda a semente. O painel (a) põe essa escala na
figura --- duas sementes do braço RGB antes do braço multi-espaço --- de modo que a
comparação que interessa se faz com os olhos e não com a fé. O painel (b) generaliza-a
aos seis pares.

Só o NeoJaundice entra com fotografia: o recorte [0,30, 0,70] do protocolo é pele, sem
rosto. O NJN entra apenas no painel (b), em número.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
MAPS = OUT / "attr_maps"
FIGDIR = ROOT / "local/generated/figures"
FIGS = ROOT / "paper/figs"
sys.path.insert(0, str(ROOT))
from src.paths import image_path

INK, MUTED, EDGE = "#0b0b0b", "#52514e", "#d8d7d3"
BLUE, RED = "#2a78d6", "#e34948"
ROI_LO, ROI_HI = 0.30, 0.70
NAME = "fig10_atribuicao"
IN_PAPER = "qualitative_attribution"

DISPLAY = {
    "mobilenetv3_large": "MobileNetV3-L", "densenet121": "DenseNet121",
    "dinov3_vits16": "DINOv3-ViT-S/16", "vit_l_16": "ViT-L/16",
    "deit_base": "DeiT-Base", "deit_small": "DeiT-Small",
}


def _crop(path: str) -> np.ndarray:
    from PIL import Image
    img = Image.open(image_path(path)).convert("RGB")
    w, h = img.size
    return np.asarray(img.crop((int(ROI_LO * w), int(ROI_LO * h),
                                int(ROI_HI * w), int(ROI_HI * h))))


def _unit(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float64)
    lo, hi = m.min(), m.max()
    return (m - lo) / max(hi - lo, 1e-12)


def _rows() -> list[dict]:
    with (OUT / "G2_attribution_shift.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter=";"))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def build(install: bool = False) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    rows = _rows()
    z = np.load(MAPS / "NeoJaundice__vit_l_16.npz", allow_pickle=True)
    multi, rgb, paths = z["multi"].astype(np.float32), z["rgb"].astype(np.float32), z["paths"]

    # Escolha das imagens SEM cereja: a mediana da semelhança entre braços, e o
    # percentil 5 --- onde os mapas mais diferem. Se o caso extremo já cabe dentro do
    # ruído de semente, o caso típico cabe por maioria de razão.
    r_between = np.array([_pearson(_unit(multi[:, i].mean(0)), _unit(rgb[:, i].mean(0)))
                          for i in range(multi.shape[1])])
    order = np.argsort(np.nan_to_num(r_between, nan=1.0))
    picks = [int(order[max(0, int(0.05 * len(order)))]), int(order[len(order) // 2])]
    labels = ["most divergent (P5)", "typical case (median)"]

    fig = plt.figure(figsize=(7.1, 5.5))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.0, 1.0, 0.95],
                  hspace=0.30, wspace=0.06)

    cols = ["model input", "RGB, seed 42", "RGB, seed 123", "4 spaces, mean"]
    for r, (idx, tag) in enumerate(zip(picks, labels)):
        img = _crop(str(paths[idx]))
        panels = [None, _unit(rgb[0, idx]), _unit(rgb[1, idx]), _unit(multi[:, idx].mean(0))]
        for c, m in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(img)
            if m is not None:
                ax.imshow(np.kron(m, np.ones((16, 16)))[:img.shape[0], :img.shape[1]],
                          cmap="inferno", alpha=0.55, extent=(0, img.shape[1], img.shape[0], 0))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(EDGE)
            if r == 0:
                ax.set_title(cols[c], fontsize=7.5, color=INK, pad=4)
            if c == 0:
                ax.set_ylabel(tag, fontsize=7, color=MUTED)

    # ---- painel (b): r_between contra a banda de ruído dentro de cada braço
    ax = fig.add_subplot(gs[2, :])
    rows = sorted(rows, key=lambda r: float(r["r_within_rgb"]))
    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        lo, hi = sorted((float(r["r_within_rgb"]), float(r["r_within_multi"])))
        ax.plot([lo, hi], [i, i], color=EDGE, lw=6, solid_capstyle="butt", zorder=1)
        ax.plot([lo, hi], [i, i], color=MUTED, lw=0.8, zorder=2)
        # Uma cor só. Marcar "dentro/fora" da banda a vermelho inventaria um limiar que
        # a análise não tem: o DINOv3 cai 0,005 abaixo do seu piso, o que é posição e não
        # categoria. O leitor vê a posição do ponto e tira a conclusão dela.
        ax.plot(float(r["r_between"]), i, "o", ms=6, color=BLUE, zorder=3,
                mec="white", mew=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{DISPLAY.get(r['backbone'], r['backbone'])} · {r['dataset']}"
                        for r in rows], fontsize=7, color=INK)
    ax.set_xlabel("correlation between attribution maps of the same image", fontsize=7.5,
                  color=INK)
    ax.tick_params(axis="x", labelsize=7, colors=MUTED)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(EDGE)
    ax.grid(axis="x", color=EDGE, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.text(0.01, 1.06, "bar: two seeds of the SAME arm (noise floor)   "
            "$\\bullet$ between the RGB and multi-space arms",
            transform=ax.transAxes, fontsize=6.8, color=MUTED)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{NAME}.{ext}", bbox_inches="tight", facecolor="white", dpi=300)
    if install:
        FIGS.mkdir(parents=True, exist_ok=True)
        (FIGS / f"{IN_PAPER}.pdf").write_bytes((FIGDIR / f"{NAME}.pdf").read_bytes())
        print(f"  instalada: paper/figs/{IN_PAPER}.pdf")
    print(f"  {NAME}.pdf / .png")
    print(f"  imagens escolhidas: P5 r={r_between[picks[0]]:.3f}, "
          f"mediana r={r_between[picks[1]]:.3f}")
    return 0
