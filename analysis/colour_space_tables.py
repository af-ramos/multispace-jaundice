"""Tabelas centradas no espaco de cor, geradas a partir do Register B canónico.

Revisão de setembro de 2026: tex_t11 apresenta amplitude observada, não IC.
Os campos históricos lo/hi/separated permanecem nos CSV para rastreabilidade,
mas não são usados como inferência na versão pré-submissão. Para verificar a
versão atual completa, usar analysis/pre_submission_revision.py --check.

Duas tabelas que o manuscrito nao usava e que respondem a pergunta do titulo no
registo do espaco de cor, sem seleccao:

  T10  efeito marginal de ACRESCENTAR um espaco, sobre todos os contextos em que
       o resto do conjunto fica fixo. Contraste emparelhado por celula.
  T11  efeito medio de acrescentar croma ao RGB, por backbone, sobre os 7
       conjuntos que retem RGB. Media, nao maximo: nao ha seleccao.

Uso:  python analysis/colour_space_tables.py
Saida: results/T10_espaco_marginal.csv
       results/T11_por_backbone_medio.csv
       results/T10_T11.tex
"""

import csv
import itertools
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)

from analysis.register_b import (  # noqa: E402
    BACKBONES, CNN, RGB_PRESERVING, TRANSFORMERS, load_cells,
)

SPACES = ["RGB", "LAB", "YCrCb", "HSV"]
T6 = 2.447  # t_{0.975, 6}: os 7 conjuntos que retem RGB
Z = 1.96    # painel grande na T10
EPS = 1e-9  # tolerancia de virgula flutuante: um delta de 1e-15 p.p. e zero, nao um ganho

ABBR = {"RGB": "RGB", "LAB": "L", "YCrCb": "Y", "HSV": "H"}
PRETTY = {
    "convnextv2_tiny": "ConvNeXtV2-T", "densenet121": "DenseNet121",
    "efficientnet_b0": "EfficientNetB0", "efficientnet_b4": "EfficientNetB4",
    "inception_v3": "Inception-v3", "mobilenetv3_large": "MobileNetV3-L",
    "resnet18": "ResNet18", "resnet50": "ResNet50",
    "deit_base": "DeiT-Base", "deit_small": "DeiT-Small", "deit_tiny": "DeiT-Tiny",
    "dinov3_vits16": "DINOv3-ViT-S/16", "vit_b_16": "ViT-B/16",
    "vit_b_32": "ViT-B/32", "vit_l_16": "ViT-L/16",
}
TRF = TRANSFORMERS


def num(x):
    return float(x.replace(",", ".")) if x else None


def load():
    """As 450 células validadas pelo loader comum do Register B."""
    cells = load_cells()
    acc = {key: cell.accuracy for key, cell in cells.items()}
    auc = {key: cell.auc for key, cell in cells.items()}
    assert len(acc) == 450, f"esperadas 450 celulas, obtidas {len(acc)}"
    return acc, auc


def setname(s):
    return "+".join(x for x in SPACES if x in s)


def ci(d, tcrit):
    m = st.mean(d)
    se = st.stdev(d) / len(d) ** 0.5
    return m, m - tcrit * se, m + tcrit * se


def marginal(acc, auc):
    """Para cada espaco, o delta de acrescenta-lo a todo o conjunto-base possivel."""
    out = []
    for ds in ["NJN", "NeoJaundice"]:
        bbs = sorted({k[1] for k in acc if k[0] == ds})
        for sp in SPACES:
            others = [x for x in SPACES if x != sp]
            for ctx in ["with RGB", "without RGB"]:
                if sp == "RGB" and ctx == "with RGB":
                    continue
                da, du = [], []
                for bb in bbs:
                    for k in range(1, 4):
                        for rest in itertools.combinations(others, k):
                            base = set(rest)
                            if ("RGB" in base) != (ctx == "with RGB"):
                                continue
                            a = acc.get((ds, bb, setname(base)))
                            b = acc.get((ds, bb, setname(base | {sp})))
                            ua = auc.get((ds, bb, setname(base)))
                            ub = auc.get((ds, bb, setname(base | {sp})))
                            if None in (a, b, ua, ub):
                                continue
                            da.append(b - a)
                            du.append(ub - ua)
                if not da:
                    continue
                m, lo, hi = ci(da, Z)
                mu, ulo, uhi = ci(du, Z)
                out.append(dict(
                    dataset=ds, space=sp, context=ctx, n=len(da),
                    mean_acc=m, lo_acc=lo, hi_acc=hi,
                    positive=sum(1 for x in da if x > EPS),
                    mean_auc=mu, lo_auc=ulo, hi_auc=uhi,
                    separated="yes" if (lo > 0 or hi < 0) else "no"))
    return out


def per_backbone(acc):
    """Media sobre os 7 conjuntos que retem RGB, por backbone. Sem seleccao."""
    out = []
    for ds in ["NJN", "NeoJaundice"]:
        for bb in BACKBONES:
            base = acc.get((ds, bb, "RGB"))
            d = [acc[(ds, bb, c)] - base for c in RGB_PRESERVING
                 if (ds, bb, c) in acc]
            if base is None or len(d) != 7:
                continue
            m, lo, hi = ci(d, T6)
            out.append(dict(
                dataset=ds, backbone=bb, family="CNN" if bb in CNN else "Transformer",
                rgb=base, mean=m, lo=lo, hi=hi,
                positive=sum(1 for x in d if x > EPS),
                best=max(d), worst=min(d),
                separated="yes" if (lo > 0 or hi < 0) else "no"))
    return out


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(rows)


def fmt(x, n=2, sign=True):
    s = f"{x:+.{n}f}" if sign else f"{x:.{n}f}"
    return s.replace("-", "$-$")


def tex_t10(rows):
    """A tabela marginal: uma linha por (espaco, contexto), os dois datasets lado a lado."""
    order = [("RGB", "without RGB"), ("LAB", "with RGB"), ("YCrCb", "with RGB"),
             ("HSV", "with RGB"), ("LAB", "without RGB"), ("YCrCb", "without RGB"),
             ("HSV", "without RGB")]
    idx = {(r["dataset"], r["space"], r["context"]): r for r in rows}
    L = []
    for ctx, header in [("with RGB", "\\textit{Adding a chromatic space to a set that already contains RGB}"),
                        ("without RGB", "\\textit{Adding a space to a set that does not contain RGB}")]:
        L.append(f"  \\multicolumn{{7}}{{@{{}}l}}{{{header}}} \\\\")
        for sp, c in order:
            if c != ctx:
                continue
            cells = []
            for ds in ["NJN", "NeoJaundice"]:
                r = idx[(ds, sp, c)]
                mark = "\\highcell" if r["separated"] == "yes" else ""
                v = fmt(r["mean_acc"])
                v = f"{mark}{{{v}}}" if mark else v
                cells += [v, f"[{fmt(r['lo_acc'])}, {fmt(r['hi_acc'])}]", f"{r['positive']}/{r['n']}"]
            L.append(f"  \\quad {ABBR[sp] if sp != 'RGB' else 'RGB'} & " + " & ".join(cells) + " \\\\")
    return "\n".join(L)


def tex_t11(rows):
    idx = {(r["dataset"], r["backbone"]): r for r in rows}
    L = []
    for fam, bbs in [("CNN", CNN), ("Transformer", TRF)]:
        L.append(f"  \\multicolumn{{7}}{{@{{}}l}}{{\\textbf{{{fam}}}}} \\\\")
        for bb in bbs:
            cells = []
            for ds in ["NJN", "NeoJaundice"]:
                r = idx[(ds, bb)]
                v = fmt(r["mean"])
                # The fixed seven treatments are not independent replicates.
                # Show their observed range, without significance highlighting.
                cells += [v, f"[{fmt(r['worst'])}, {fmt(r['best'])}]", f"{r['positive']}/7"]
            L.append(f"  {PRETTY[bb]} & " + " & ".join(cells) + " \\\\")
    return "\n".join(L)


def main():
    os.makedirs(OUT, exist_ok=True)
    acc, auc = load()
    assert len(acc) == 450, f"esperava 450 celulas, obtive {len(acc)}"

    t10 = marginal(acc, auc)
    t11 = per_backbone(acc)
    write_csv(os.path.join(OUT, "T10_espaco_marginal.csv"), t10)
    write_csv(os.path.join(OUT, "T11_por_backbone_medio.csv"), t11)

    with open(os.path.join(OUT, "T10_T11.tex"), "w", encoding="utf-8") as fh:
        fh.write("%% T10\n" + tex_t10(t10) + "\n\n%% T11\n" + tex_t11(t11) + "\n")

    # resumo para o texto
    sep = [r for r in t11 if r["separated"] == "yes"]
    for ds in ["NJN", "NeoJaundice"]:
        s = [r for r in sep if r["dataset"] == ds]
        print(f"{ds}: {len(s)}/15 separados de zero "
              f"({sum(1 for r in s if r['mean'] > 0)} positivos, "
              f"{sum(1 for r in s if r['mean'] < 0)} negativos)")
    print(f"total: {len(sep)}/30")
    for r in t10:
        print(f"  {r['dataset']:12s} {r['space']:6s} {r['context']:12s} n={r['n']:3d} "
              f"acc {r['mean_acc']:+.2f} [{r['lo_acc']:+.2f},{r['hi_acc']:+.2f}] "
              f"auc {r['mean_auc']:+.4f} [{r['lo_auc']:+.4f},{r['hi_auc']:+.4f}] {r['separated']}")


if __name__ == "__main__":
    main()
