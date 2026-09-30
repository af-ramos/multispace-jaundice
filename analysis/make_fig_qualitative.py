#!/usr/bin/env python3
"""Figura qualitativa: os casos que a entrada multi-espaço muda, e para quem.

## Porque existem

O manuscrito responde à pergunta "quanto muda" com painéis de 450 células. Não
responde a "muda o quê" — que imagens deixam de ser erradas e quais passam a sê-lo.
A figura do manuscrito responde sem retreinar nada: sai dos 188 dumps de predição,
que trazem ``path``, ``patient_id``, ``y_true`` e ``y_prob`` por imagem e por semente.
O módulo também preserva, como derivado exploratório fora do artigo, a dispersão que
compara a resposta de dois backbones.

## Figura do manuscrito e derivado exploratório

* **Casos (NeoJaundice, ViT-L/16; Figura 4).** Os pacientes que o baseline RGB
  classifica mal e o braço de quatro espaços classifica bem, e os que perde no sentido
  inverso. Os rótulos explicitam que o ganho de +4,02 p.p. vem de corrigir **falsos
  positivos** ao custo de **falsos negativos** novos
  (sensibilidade 88,9 -> 77,8, especificidade 73,3 -> 88,4), e uma galeria só de
  acertos contradiria o texto que ela ilustra.

  As imagens aparecem **como o modelo as vê**: recorte central [0,30, 0,70] de cada
  eixo, que é a §3.2 item 3. Isso descarta o cartão de calibração e o fundo, e o que
  resta é pele — nenhuma face, nenhum traço identificável. Não é anonimização
  aplicada à figura; é o pré-processamento do protocolo.

* **Dependência (NJN, RGB+YCrCb+HSV; fora do manuscrito).** O mesmo input cromático
  em dois backbones com resposta oposta na Tabela 5: DenseNet121, que ganha, e
  DINOv3-ViT-S/16, que perde.
  Dispersão de P(icterícia) do braço RGB contra o multi-espaço, **as 152 imagens de
  teste**, nada filtrado. Os quadrantes fora da diagonal são as decisões que mudam.

  Aqui não há fotografias: o NJN é corpo inteiro e mostrar essas imagens exigiria
  mascarar faces caso a caso, o que não é reprodutível por script. A dispersão diz o
  que a figura precisa de dizer sem essa dívida.

## Registo

Tudo aqui é **Register A** da §3.3: média das probabilidades das cinco sementes,
agregação por ``patient_id`` no NeoJaundice, limiar fixo em 0,5. É a mesma receita da
Tabela 8, e ``--verify`` confirma que reproduz os seus números exactos. Nenhuma
quantidade de teste escolhe nada: as células e os braços são os que a §4.1 e a
Tabela 5 já fixaram.

Uso:  python analysis/make_fig_qualitative.py [--verify]
Saída: figures/fig8_casos_neojaundice.{pdf,png}
       figures/fig9_dependencia_njn.{pdf,png}
       e instala em paper/figs/ apenas a figura de casos usada no manuscrito.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

PREDS = ROOT / "evidence/v7/preds"
OUT = ROOT / "local/generated/figures"
FIGS = ROOT / "paper/figs"
sys.path.insert(0, str(ROOT))
from src.paths import image_path

#: Instalação no manuscrito: nome gerado -> nome que o \includegraphics espera.
IN_PAPER = {
    "fig8_casos_neojaundice": "qualitative_cases",
}

# Paleta documentada em docs/FIGURES.md. Não escolher hexes novos aqui.
BLUE, RED, INK, MUTED, EDGE = "#2a78d6", "#e34948", "#0b0b0b", "#52514e", "#d8d7d3"

#: Recorte do NeoJaundice, §3.2 item 3 do manuscrito.
ROI_LO, ROI_HI = 0.30, 0.70


# --------------------------------------------------------------- leitura
def cell(dataset: str, backbone: str, colorset: str) -> Path:
    """O dump canónico de uma célula, pela mesma regra de `analysis/dumps.py`."""
    from dumps import _dataset_of, _is_canonical

    hits = []
    for path in sorted((PREDS / dataset).glob("*.npz")):
        if f"__{backbone}__{colorset}__" not in path.name:
            continue
        with np.load(path, allow_pickle=True) as npz:
            if _dataset_of(npz) != dataset:
                continue
        if _is_canonical(path.stem, dataset):
            hits.append(path)
    if not hits:
        raise SystemExit(f"sem dump canónico para {dataset}/{backbone}/{colorset}")
    return hits[0]


def ensemble(path: Path, by_patient: bool) -> dict[str, dict]:
    """Register A: média das 5 sementes por imagem, depois por paciente se pedido.

    A chave é o *basename* do caminho porque os dumps não concordam no prefixo --- uns
    guardam `dataset/NJN/...`, outros o caminho absoluto da máquina que treinou.
    """
    with np.load(path, allow_pickle=True) as npz:
        keep = npz["splits"] == "test"
        paths = [str(x) for x in npz["paths"][keep]]
        y = npz["y_true"][keep].astype(int)
        prob = npz["y_prob"][keep].astype(float)
        pid = [str(x) for x in npz["patient_ids"][keep]]

    per_image: dict[str, dict] = {}
    acc = defaultdict(list)
    for p, yy, pp, pi in zip(paths, y, prob, pid):
        key = os.path.basename(p)
        acc[key].append(pp)
        per_image[key] = {"y": int(yy), "pid": pi, "file": p}
    for key, probs in acc.items():
        per_image[key]["p"] = float(np.mean(probs))
        per_image[key]["n_seeds"] = len(probs)

    if not by_patient:
        return per_image

    grouped = defaultdict(list)
    for key, rec in per_image.items():
        grouped[rec["pid"]].append(rec)
    return {
        pid: {"y": recs[0]["y"], "pid": pid,
              "p": float(np.mean([r["p"] for r in recs])),
              "file": sorted(r["file"] for r in recs)[0],
              "n_images": len(recs)}
        for pid, recs in grouped.items()
    }


def accuracy(units: dict[str, dict]) -> float:
    return 100.0 * np.mean([(u["p"] >= 0.5) == (u["y"] == 1) for u in units.values()])


def flips(base: dict, arm: dict) -> tuple[list, list]:
    """(corrigidos, quebrados), em ordem de chave --- determinística, sem selecção."""
    keys = sorted(set(base) & set(arm))
    ok = lambda u: (u["p"] >= 0.5) == (u["y"] == 1)
    return ([k for k in keys if not ok(base[k]) and ok(arm[k])],
            [k for k in keys if ok(base[k]) and not ok(arm[k])])


def model_input(relpath: str) -> "np.ndarray":
    """A imagem como o modelo a vê no NeoJaundice: recorte central do protocolo."""
    from PIL import Image

    img = Image.open(image_path(relpath)).convert("RGB")
    w, h = img.size
    return np.asarray(img.crop((int(ROI_LO * w), int(ROI_LO * h),
                                int(ROI_HI * w), int(ROI_HI * h))))


# --------------------------------------------------------------- figuras
def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    import matplotlib

    # TrueType (42) em vez de Type 3: o sistema de submissão da Springer sinaliza Type 3.
    with matplotlib.rc_context({"pdf.fonttype": 42}):
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white", dpi=300)
    if name in IN_PAPER:
        FIGS.mkdir(parents=True, exist_ok=True)
        (FIGS / f"{IN_PAPER[name]}.pdf").write_bytes((OUT / f"{name}.pdf").read_bytes())
        print(f"  instalado: paper/figs/{IN_PAPER[name]}.pdf")
    print(f"  {name}.pdf / .png")


def fig_cases(n_each: int = 4) -> None:
    import matplotlib.pyplot as plt

    base = ensemble(cell("NeoJaundice", "vit_l_16", "RGB"), by_patient=True)
    arm = ensemble(cell("NeoJaundice", "vit_l_16", "RGB+LAB+YCrCb+HSV"), by_patient=True)
    fixed, broken = flips(base, arm)
    print(f"  NeoJaundice ViT-L/16 paciente: RGB {accuracy(base):.2f}% -> "
          f"4 espaços {accuracy(arm):.2f}%  ({len(fixed)} corrigidos, {len(broken)} quebrados)")

    rows = [("False positives corrected · negative", fixed[:n_each], BLUE, "FP", "TN"),
            ("New false negatives · positive", broken[:n_each], RED, "TP", "FN")]
    fig, axes = plt.subplots(2, n_each, figsize=(6.3, 3.9))
    for r, (title, keys, colour, before, after) in enumerate(rows):
        for c in range(n_each):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for side in ax.spines.values():
                side.set_color(EDGE); side.set_linewidth(0.6)
            if c >= len(keys):
                ax.axis("off"); continue
            rec = arm[keys[c]]
            ax.imshow(model_input(rec["file"]))
            truth = "positive" if rec["y"] == 1 else "negative"
            ax.set_title(f"patient {keys[c]} · {truth}", fontsize=6.2, color=INK, pad=3)
            ax.set_xlabel(
                f"RGB {base[keys[c]]['p']:.2f} ({before})  →\n"
                f"4-space {rec['p']:.2f} ({after})",
                fontsize=6.0, color=colour, labelpad=1,
            )
        axes[r, 0].text(-0.06, 0.5, title, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=7, color=INK)
    fig.subplots_adjust(hspace=0.52, wspace=0.06)
    save(fig, "fig8_casos_neojaundice")
    plt.close(fig)


def fig_dependency() -> None:
    import matplotlib.pyplot as plt

    panels = [("densenet121", "DenseNet121"), ("dinov3_vits16", "DINOv3-ViT-S/16")]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.25), sharex=True, sharey=True)
    for ax, (backbone, label) in zip(axes, panels):
        base = ensemble(cell("NJN", backbone, "RGB"), by_patient=False)
        arm = ensemble(cell("NJN", backbone, "RGB+YCrCb+HSV"), by_patient=False)
        keys = sorted(set(base) & set(arm))
        fixed, broken = flips(base, arm)
        delta = accuracy(arm) - accuracy(base)
        print(f"  NJN {label}: RGB {accuracy(base):.2f}% -> RGB+YCrCb+HSV "
              f"{accuracy(arm):.2f}%  ({delta:+.2f} p.p.; {len(fixed)} corrigidas, "
              f"{len(broken)} quebradas)")

        ax.axhspan(0.5, 1.0, xmin=0, xmax=0.5, color=BLUE, alpha=0.05, lw=0)
        ax.axvspan(0.5, 1.0, ymin=0, ymax=0.5, color=RED, alpha=0.05, lw=0)
        ax.plot([0, 1], [0, 1], color=EDGE, lw=0.7, zorder=1)
        ax.axhline(0.5, color=MUTED, lw=0.6, ls=":", zorder=1)
        ax.axvline(0.5, color=MUTED, lw=0.6, ls=":", zorder=1)
        for key in keys:
            jaundiced = base[key]["y"] == 1
            ax.scatter(base[key]["p"], arm[key]["p"], s=13, zorder=3,
                       marker="o" if jaundiced else "s",
                       facecolor=(INK if jaundiced else "none"),
                       edgecolor=INK, linewidths=0.6, alpha=0.75)
        ax.set_title(f"{label}   {delta:+.2f} p.p.", fontsize=8, color=INK, pad=6)
        ax.set_xlabel("P(jaundice), RGB baseline", fontsize=7.5, color=INK)
        ax.text(0.03, 0.97, f"{len(fixed)} corrected", fontsize=6.8, color=BLUE,
                va="top", ha="left", transform=ax.transAxes)
        ax.text(0.97, 0.03, f"{len(broken)} broken", fontsize=6.8, color=RED,
                va="bottom", ha="right", transform=ax.transAxes)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.tick_params(labelsize=6.8, colors=MUTED, length=2)
        for side in ax.spines.values():
            side.set_color(EDGE); side.set_linewidth(0.6)
    axes[0].set_ylabel("P(jaundice), RGB+YCrCb+HSV", fontsize=7.5, color=INK)
    handles = [plt.Line2D([], [], marker="o", color=INK, ls="", ms=4, label="jaundiced"),
               plt.Line2D([], [], marker="s", color=INK, ls="", ms=4,
                          markerfacecolor="none", label="healthy")]
    axes[1].legend(handles=handles, fontsize=6.8, frameon=False, loc="lower right",
                   bbox_to_anchor=(1.0, 0.10))
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig9_dependencia_njn")
    plt.close(fig)


def verify() -> int:
    """Confirma que a receita do Register A reproduz a Tabela 8 do manuscrito."""
    checks = [
        ("NJN MobileNetV3-L, RGB", ("NJN", "mobilenetv3_large", "RGB"), False, 94.08),
        ("NJN MobileNetV3-L, RGB+L+H", ("NJN", "mobilenetv3_large", "RGB+LAB+HSV"), False, 95.39),
        ("Neo ViT-L/16, RGB (patient)", ("NeoJaundice", "vit_l_16", "RGB"), True, 79.87),
        ("Neo ViT-L/16, 4-space (patient)", ("NeoJaundice", "vit_l_16", "RGB+LAB+YCrCb+HSV"), True, 83.89),
        ("Neo ViT-L/16, RGB (image)", ("NeoJaundice", "vit_l_16", "RGB"), False, 75.84),
        ("Neo ViT-L/16, 4-space (image)", ("NeoJaundice", "vit_l_16", "RGB+LAB+YCrCb+HSV"), False, 78.97),
    ]
    bad = 0
    for label, key, by_patient, expected in checks:
        got = accuracy(ensemble(cell(*key), by_patient=by_patient))
        ok = abs(got - expected) < 0.01
        bad += not ok
        print(f"  {'OK ' if ok else 'ERRO'} {label:34s} {got:6.2f}%  (Tabela 8: {expected:.2f}%)")
    print("verificação: Register A reproduz a Tabela 8" if not bad
          else f"verificação FALHOU em {bad} linhas")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="só confere os números contra a Tabela 8, não desenha")
    args = ap.parse_args(argv)
    if args.verify:
        return verify()
    fig_cases()
    fig_dependency()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
