"""Sistemas de manchete sob a regra de pontuação declarada do protocolo.

O manuscrito reportava ao nível da imagem com limiar fixo 0,5 e uma métrica por
semente. Isso deixa em cima da mesa as duas alavancas que o `docs/PROTOCOL.md` já
declara — a unidade de avaliação por paciente no NeoJaundice (§4) e o limiar escolhido
na validação (§3) — e trata as cinco sementes como incerteza a confessar em vez de as
combinar num sistema único.

Este módulo aplica **uma regra, declarada à partida, igual para todas as células**:

  1. média das probabilidades das 5 sementes (ensemble);
  2. no NeoJaundice, média das probabilidades dentro de cada `patient_id`;
  3. limiar fixo em 0,50;
  4. nada é escolhido no teste.

Esclarecimento dos autores (revisão pré-submissão de setembro de 2026): o teste
não foi usado para decisões. O limiar fixo de 0,50 define o Register A; a variante
`--val-threshold` é uma análise distinta, não o critério de escolha dessa regra.
A interpretação causal anteriormente registrada neste cabeçalho foi corrigida.

A seleção por AUC de validação do ensemble é restrita aos braços com dumps
canônicos preservados. Não representa uma busca sobre os 14 braços do factorial:
os dois backbones destacados têm dois candidatos cromáticos cada. Para as tabelas
da revisão e a incerteza agrupada, usar `analysis/pre_submission_revision.py`.

Não é a melhor combinação por célula: é a mesma receita em toda a parte, o que é o que
a torna reportável. As colunas combinadas do `evidence/v7/A5_alavancas.csv` não são
usadas — para ResNet18/`RGB+LAB+YCrCb+HSV` elas dão ensemble+paciente ABAIXO de só
paciente, o que é incompatível com a média de +1,45 p.p. em 80/86 células e aponta erro
de ordem de operações. Tudo aqui é recomputado dos dumps.

Uso:  python analysis/headline_systems.py
Saída: results/H1_sistemas.csv        (todas as células com dump, regra nova)
       results/H2_manchete.csv        (as linhas de manchete, com métricas clínicas)
       results/H_delong.csv           (DeLong pareado RGB vs multi-espaço)
"""

from __future__ import annotations

import csv
import os
import statistics as st
from collections import defaultdict

import numpy as np
from scipy.stats import beta, norm

from dumps import PREDS, _auc, _dataset_of, _f1_macro, _is_canonical

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
GRID = np.round(np.arange(0.05, 1.00, 0.05), 2)
PATIENT_LEVEL = {"NeoJaundice"}  # NJN colapsa 760 imagens em 755 pseudo-pacientes


# ----------------------------------------------------------------- intervalos
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (100 * (c - h) / d, 100 * (c + h) / d)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exacto. É o registo que o PROTOCOL.md §5 pede para sensibilidade/especificidade."""
    if n == 0:
        return (float("nan"),) * 2
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (100 * lo, 100 * hi)


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """AUC-PR pela soma de Riemann à esquerda, como o `sklearn.average_precision_score`."""
    order = np.argsort(-p, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    total = y.sum()
    return float((prec * y).sum() / total) if total else float("nan")


# ----------------------------------------------------------------- DeLong
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    s = x[order]
    n = len(x)
    r = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and s[j + 1] == s[i]:
            j += 1
        r[i : j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = r
    return out


def delong(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict:
    """Teste de DeLong para duas AUC correlacionadas nas MESMAS amostras."""
    pos, neg = p1[y == 1], p1[y == 0]
    m, n = len(pos), len(neg)
    preds = np.vstack([p1, p2])
    aucs, v01, v10 = [], [], []
    for row in preds:
        x, yv = row[y == 1], row[y == 0]
        tx, ty, tz = _midrank(x), _midrank(yv), _midrank(row)
        aucs.append((tz[y == 1].sum() - m * (m + 1) / 2) / (m * n))
        v01.append((tz[y == 1] - tx) / n)
        v10.append(1 - (tz[y == 0] - ty) / m)
    v01, v10 = np.array(v01), np.array(v10)
    s = np.cov(v01) / m + np.cov(v10) / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    d = aucs[0] - aucs[1]
    z = d / (var ** 0.5) if var > 0 else 0.0
    return {"auc1": aucs[0], "auc2": aucs[1], "diff": d,
            "se": var ** 0.5, "z": z, "p": 2 * norm.sf(abs(z))}


# ----------------------------------------------------------------- carregamento
def load_cells() -> dict:
    """`(dataset, backbone, colorset)` -> probabilidades já ensembladas, val e test."""
    cells: dict = {}
    for fname in sorted(os.listdir(PREDS)):
        pass  # PREDS tem subpastas por dataset
    for sub in ("NJN", "NeoJaundice"):
        d = os.path.join(PREDS, sub)
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".npz"):
                continue
            data = np.load(os.path.join(d, fname), allow_pickle=True)
            dataset = _dataset_of(data)
            stem = fname[:-4]
            if not _is_canonical(stem, dataset) or dataset != sub:
                continue
            parts = stem.split("__")
            key = (dataset, parts[1], parts[2])
            if key in cells:
                raise RuntimeError(f"dois dumps canónicos para {key}")

            arms = {}
            for split in ("val", "test"):
                sel = data["splits"] == split
                if not sel.any():
                    continue
                paths, pids = data["paths"][sel], data["patient_ids"][sel]
                prob, true, seed = data["y_prob"][sel], data["y_true"][sel], data["seeds"][sel]
                # ensemble: média sobre as sementes, imagem a imagem
                acc_p, acc_y, acc_pid = defaultdict(list), {}, {}
                for i in range(len(paths)):
                    acc_p[paths[i]].append(prob[i])
                    acc_y[paths[i]] = true[i]
                    acc_pid[paths[i]] = pids[i]
                order = sorted(acc_p)
                img_prob = np.array([st.fmean(acc_p[k]) for k in order])
                img_true = np.array([acc_y[k] for k in order])
                img_pid = np.array([acc_pid[k] for k in order])
                n_seeds = len(np.unique(seed))
                # por semente: o registo que o manuscrito reporta hoje, e a base de
                # comparação para medir quanto o ensemble acrescenta
                per_seed_auc, per_seed_acc = [], []
                for s_ in np.unique(seed):
                    m_ = seed == s_
                    a = _auc(true[m_], prob[m_])
                    if a is not None:
                        per_seed_auc.append(a)
                    per_seed_acc.append(100.0 * float(((prob[m_] >= 0.5).astype(int) == true[m_]).mean()))
                unit_prob, unit_true = img_prob, img_true
                if dataset in PATIENT_LEVEL:
                    grp_p, grp_y = defaultdict(list), {}
                    for i in range(len(img_pid)):
                        grp_p[img_pid[i]].append(img_prob[i])
                        grp_y[img_pid[i]] = img_true[i]
                    gk = sorted(grp_p)
                    unit_prob = np.array([st.fmean(grp_p[k]) for k in gk])
                    unit_true = np.array([grp_y[k] for k in gk])
                arms[split] = {"prob": unit_prob, "true": unit_true,
                               "unit_ids": np.array(gk) if dataset in PATIENT_LEVEL else np.array(order),
                               "img_paths": np.array(order), "img_patient_ids": img_pid,
                               "img_prob": img_prob, "img_true": img_true,
                               "n_seeds": n_seeds, "seed_auc": per_seed_auc,
                               "seed_acc": per_seed_acc}
            if "test" not in arms:
                continue
            cells[key] = arms
    return cells


# ----------------------------------------------------------------- pontuação
def pick_threshold(val: dict, tune: bool = False) -> float:
    """0,50 por omissão. Com `tune`, grelha 0,05 na validação por F1-macro.

    O desempate favorece 0,50 para que a variante sintonizada só se afaste do ponto de
    operação por omissão quando a validação realmente o prefere.
    """
    if val is None or not tune:
        return 0.50
    best, best_thr = -1.0, 0.50
    for t in GRID:
        f1 = _f1_macro(val["true"], (val["prob"] >= t).astype(int))
        if f1 > best + 1e-12 or (abs(f1 - best) <= 1e-12 and abs(t - 0.5) < abs(best_thr - 0.5)):
            best, best_thr = f1, float(t)
    return best_thr


def score(test: dict, thr: float) -> dict:
    y, p = test["true"], test["prob"]
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n = tp + tn + fp + fn
    acc_lo, acc_hi = wilson(tp + tn, n)
    sen_lo, sen_hi = clopper_pearson(tp, tp + fn)
    spe_lo, spe_hi = clopper_pearson(tn, tn + fp)
    auc = _auc(y, p)
    sd = st.stdev(test["seed_auc"]) if len(test["seed_auc"]) > 1 else 0.0
    return {
        "n": n, "threshold": thr, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": 100.0 * (tp + tn) / n, "acc_lo": acc_lo, "acc_hi": acc_hi,
        "f1_macro": _f1_macro(y, pred),
        "sensitivity": 100.0 * tp / (tp + fn) if tp + fn else float("nan"),
        "sen_lo": sen_lo, "sen_hi": sen_hi,
        "specificity": 100.0 * tn / (tn + fp) if tn + fp else float("nan"),
        "spe_lo": spe_lo, "spe_hi": spe_hi,
        "auc_roc": auc, "auc_seed_sd": sd, "auc_pr": average_precision(y, p),
    }


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(rows)


def levers(cells: dict) -> list[dict]:
    """Quanto vale cada alavanca, em média sobre as células com dump.

    A linha de partida é o registo do manuscrito: nível imagem, limiar 0,50, média das
    acurácias das cinco sementes. Cada linha seguinte acrescenta uma alavanca à anterior.
    """
    rows = []
    for ds in ("NJN", "NeoJaundice"):
        sub = {k: v for k, v in cells.items() if k[0] == ds}
        base = st.fmean(st.fmean(v["test"]["seed_acc"]) for v in sub.values())
        variants = {
            "per-seed mean, image, 0.50 (manuscript)": lambda t, v: None,
            "+ 5-seed ensemble": lambda t, v: (t["img_prob"], t["img_true"], 0.50),
            "+ protocol unit": lambda t, v: (t["prob"], t["true"], 0.50),
            "+ validation threshold": lambda t, v: (
                t["prob"], t["true"], pick_threshold(v, tune=True)),
        }
        for name, fn in variants.items():
            accs, sens = [], []
            for k, arms in sub.items():
                got = fn(arms["test"], arms.get("val"))
                if got is None:
                    accs.append(st.fmean(arms["test"]["seed_acc"]))
                    sens.append(float("nan"))
                    continue
                p, y, thr = got
                pred = (p >= thr).astype(int)
                accs.append(100.0 * float((pred == y).mean()))
                tp = int(((pred == 1) & (y == 1)).sum())
                fn_ = int(((pred == 0) & (y == 1)).sum())
                sens.append(100.0 * tp / (tp + fn_) if tp + fn_ else float("nan"))
            m = st.fmean(accs)
            s = [x for x in sens if x == x]
            rows.append({"dataset": ds, "lever": name, "n_cells": len(sub),
                         "mean_acc": m, "vs_manuscript": m - base,
                         "mean_sensitivity": st.fmean(s) if s else float("nan")})
    return rows


def main() -> None:
    tune = "--val-threshold" in os.sys.argv
    os.makedirs(OUT, exist_ok=True)
    cells = load_cells()
    print(f"células com dump canónico: {len(cells)}")
    for ds in ("NJN", "NeoJaundice"):
        ks = [k for k in cells if k[0] == ds]
        nb = len({k[1] for k in ks})
        print(f"  {ds:12s} {len(ks):3d} células, {nb} backbones, "
              f"unidade = {'paciente' if ds in PATIENT_LEVEL else 'imagem'}")

    lev = levers(cells)
    write_csv(os.path.join(OUT, "H0_alavancas.csv"), lev)
    print(f"\n=== alavancas de pontuação (média sobre as células com dump) ===")
    print(f"{'dataset':12s} {'lever':42s} {'acc':>7s} {'vs manuscrito':>14s} {'sens':>6s}")
    for r in lev:
        s = f"{r['mean_sensitivity']:6.1f}" if r["mean_sensitivity"] == r["mean_sensitivity"] else "     —"
        print(f"{r['dataset']:12s} {r['lever']:42s} {r['mean_acc']:7.2f} "
              f"{r['vs_manuscript']:+14.2f} {s}")

    rows = []
    for (ds, bb, cs), arms in sorted(cells.items()):
        thr = pick_threshold(arms.get("val"), tune=tune)
        m = score(arms["test"], thr)
        rows.append({"dataset": ds, "backbone": bb, "colorset": cs,
                     "n_seeds": arms["test"]["n_seeds"], **m})
    write_csv(os.path.join(OUT, "H1_sistemas.csv"), rows)

    # ---- delta contra o baseline RGB do mesmo backbone, sob a mesma regra
    idx = {(r["dataset"], r["backbone"], r["colorset"]): r for r in rows}
    print(f"\n{'dataset':12s} {'backbone':18s} {'colorset':20s} {'acc':>7s} {'ΔRGB':>7s} "
          f"{'sens':>6s} {'spec':>6s} {'AUC':>6s} {'thr':>5s}")
    for r in sorted(rows, key=lambda r: (r["dataset"], r["backbone"], -r["accuracy"])):
        base = idx.get((r["dataset"], r["backbone"], "RGB"))
        d = r["accuracy"] - base["accuracy"] if base else float("nan")
        print(f"{r['dataset']:12s} {r['backbone']:18s} {r['colorset']:20s} "
              f"{r['accuracy']:7.2f} {d:+7.2f} {r['sensitivity']:6.1f} "
              f"{r['specificity']:6.1f} {r['auc_roc']:6.3f} {r['threshold']:5.2f}")

    # ---- por backbone: braço cromático escolhido na VALIDAÇÃO contra o próprio RGB.
    # O par é o mesmo backbone, logo é emparelhado; a escolha do braço nunca vê o teste.
    per_bb = []
    for ds in ("NJN", "NeoJaundice"):
        for bb in sorted({k[1] for k in cells if k[0] == ds}):
            rgb = cells.get((ds, bb, "RGB"))
            arms = [(c, v) for (d, b, c), v in cells.items()
                    if d == ds and b == bb and c != "RGB"]
            if rgb is None or not arms:
                continue
            pick = max(arms, key=lambda cv: _auc(cv[1]["val"]["true"], cv[1]["val"]["prob"]))
            r, a = score(rgb["test"], 0.50), score(pick[1]["test"], 0.50)
            per_bb.append({"dataset": ds, "backbone": bb, "arm": pick[0],
                           "n_arms_with_dump": len(arms),
                           "rgb_acc": r["accuracy"], "arm_acc": a["accuracy"],
                           "delta_acc": a["accuracy"] - r["accuracy"],
                           "rgb_auc": r["auc_roc"], "arm_auc": a["auc_roc"],
                           "delta_auc": a["auc_roc"] - r["auc_roc"],
                           "rgb_sens": r["sensitivity"], "arm_sens": a["sensitivity"]})
    write_csv(os.path.join(OUT, "H3_por_backbone.csv"), per_bb)

    # ---- manchete: o sistema com a maior AUC de VALIDAÇÃO em cada dataset, e o
    # melhor RGB-only pelo mesmo critério. Nada é escolhido no teste.
    head, delongs = [], []
    for ds in ("NJN", "NeoJaundice"):
        pool = [(k, v) for k, v in cells.items() if k[0] == ds and "val" in v]
        vauc = {k: _auc(v["val"]["true"], v["val"]["prob"]) for k, v in pool}
        rgb_k = max((k for k, _ in pool if k[2] == "RGB"), key=lambda k: vauc[k])
        multi_k = max((k for k, _ in pool if k[2] != "RGB"), key=lambda k: vauc[k])
        for tag, k in (("best RGB-only", rgb_k), ("best multi-space", multi_k)):
            t = cells[k]["test"]
            head.append({"system": tag, "dataset": ds, "backbone": k[1], "colorset": k[2],
                         "unit": "patient" if ds in PATIENT_LEVEL else "image",
                         "val_auc": vauc[k], **score(t, 0.50)})
            if ds in PATIENT_LEVEL:  # a unidade em que o benchmark publicado reporta
                img = dict(t, prob=t["img_prob"], true=t["img_true"])
                head.append({"system": tag, "dataset": ds, "backbone": k[1], "colorset": k[2],
                             "unit": "image", "val_auc": vauc[k], **score(img, 0.50)})
        # o par em destaque no artigo, e o par melhor-contra-melhor pela validacao
        feat = FEATURED[ds]
        feat_arm = next((k for k in cells if k[0] == ds and k[1] == feat
                         and k[2] != "RGB" and vauc.get(k) == max(
                             vauc[j] for j in cells if j[0] == ds and j[1] == feat and j[2] != "RGB")), None)
        pairs = [("featured", (ds, feat, "RGB"), feat_arm)] if feat_arm else []
        pairs.append(("val-best", rgb_k, multi_k))
        for tag, rk, mk in pairs:
            a, b = cells[rk]["test"], cells[mk]["test"]
            if len(a["true"]) == len(b["true"]) and (a["true"] == b["true"]).all():
                delongs.append({"pair": tag, "dataset": ds, "multi": f"{mk[1]}/{mk[2]}",
                                "rgb": f"{rk[1]}/{rk[2]}",
                                **delong(a["true"], b["prob"], a["prob"])})
    write_csv(os.path.join(OUT, "H2_manchete.csv"), head)
    if delongs:
        write_csv(os.path.join(OUT, "H_delong.csv"), delongs)

    print("\n=== por backbone: braço escolhido na validação contra o próprio RGB ===")
    for ds in ("NJN", "NeoJaundice"):
        sub = [r for r in per_bb if r["dataset"] == ds]
        d = [r["delta_acc"] for r in sub]
        print(f"  {ds}: média {st.fmean(d):+.2f} p.p., positivos "
              f"{sum(1 for x in d if x > 0)}/{len(d)}, maior {max(d):+.2f} "
              f"({max(sub, key=lambda r: r['delta_acc'])['backbone']})")

    print("\n=== MANCHETE (ensemble 5 seeds + unidade do protocolo + limiar 0,50;"
          " sistema escolhido pela AUC de validação) ===")
    for r in head:
        print(f"{r['dataset']:12s} {r['system']:17s} {r['backbone']:18s} {r['colorset']:20s} "
              f"acc {r['accuracy']:6.2f} [{r['acc_lo']:.2f}, {r['acc_hi']:.2f}]  "
              f"F1 {r['f1_macro']:.4f}  sens {r['sensitivity']:5.1f}  "
              f"spec {r['specificity']:5.1f}  AUC {r['auc_roc']:.3f}  AP {r['auc_pr']:.3f}  N={r['n']}")
    print("\n=== DeLong (multi-espaço − RGB) ===")
    for d in delongs:
        print(f"{d['dataset']:12s} {d['multi']:34s} vs {d['rgb']:24s} "
              f"ΔAUC {d['diff']:+.4f}  z {d['z']:+.2f}  p {d['p']:.3f}")

    emit_tex(cells, per_bb)


# ----------------------------------------------------------------- LaTeX
FEATURED = {"NJN": "mobilenetv3_large", "NeoJaundice": "vit_l_16"}
PRETTY = {"mobilenetv3_large": "MobileNetV3-L", "vit_l_16": "ViT-L/16",
          "densenet121": "DenseNet121", "convnextv2_tiny": "ConvNeXtV2-T",
          "efficientnet_b0": "EfficientNetB0", "efficientnet_b4": "EfficientNetB4",
          "inception_v3": "Inception-v3", "resnet18": "ResNet18", "resnet50": "ResNet50",
          "deit_base": "DeiT-Base", "deit_small": "DeiT-Small", "deit_tiny": "DeiT-Tiny",
          "dinov3_vits16": "DINOv3-ViT-S/16", "vit_b_16": "ViT-B/16", "vit_b_32": "ViT-B/32"}
CNN = ["convnextv2_tiny", "densenet121", "efficientnet_b0", "efficientnet_b4",
       "inception_v3", "mobilenetv3_large", "resnet18", "resnet50"]
TRF = ["deit_base", "deit_small", "deit_tiny", "dinov3_vits16",
       "vit_b_16", "vit_b_32", "vit_l_16"]
ABBR = [("RGB+LAB+YCrCb+HSV", "RGB+L+Y+H"), ("RGB+LAB+YCrCb", "RGB+L+Y"),
        ("RGB+YCrCb+HSV", "RGB+Y+H"), ("RGB+LAB+HSV", "RGB+L+H"), ("RGB+LAB", "RGB+L"),
        ("RGB+YCrCb", "RGB+Y"), ("RGB+HSV", "RGB+H"), ("LAB+YCrCb", "L+Y"),
        ("YCrCb+HSV", "Y+H"), ("LAB+HSV", "L+H"), ("YCrCb", "Y"), ("LAB", "L"), ("HSV", "H")]


def short(cs: str) -> str:
    for long_, s in ABBR:
        if cs == long_:
            return s
    return cs


def neg(x: str) -> str:
    return x.replace("-", "$-$")


def emit_tex(cells: dict, per_bb: list[dict]) -> None:
    """Duas tabelas: os sistemas em destaque, e o painel dos 15 backbones."""
    lines = ["%% H2 — sistemas em destaque (linhas do artigo)"]
    for ds, bb in FEATURED.items():
        rgb = score(cells[(ds, bb, "RGB")]["test"], 0.50)
        armname = next(r["arm"] for r in per_bb if r["dataset"] == ds and r["backbone"] == bb)
        arm = score(cells[(ds, bb, armname)]["test"], 0.50)
        lines.append(f"  \\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{ds}}} "
                     f"($N_{{\\mathrm{{test}}}} = {rgb['n']}$)}} \\\\")
        for tag, cs, m, hl in (("RGB baseline", "RGB", rgb, "\\second"),
                               ("multi-space", short(armname), arm, "\\highcell")):
            lines.append(
                f"  \\quad {PRETTY[bb]}, {cs} & {hl}{{{m['accuracy']:.2f}}} & "
                f"[{m['acc_lo']:.2f}, {m['acc_hi']:.2f}] & {m['f1_macro']:.4f} & "
                f"{m['sensitivity']:.1f} & {m['specificity']:.1f} & {m['auc_roc']:.3f} \\\\")
    lines.append("")
    lines.append("%% H3 — painel dos 15 backbones, braço escolhido na validacao")
    idx = {(r["dataset"], r["backbone"]): r for r in per_bb}
    for fam, bbs in (("CNN", CNN), ("Transformer", TRF)):
        lines.append(f"  \\multicolumn{{7}}{{@{{}}l}}{{\\textbf{{{fam}}}}} \\\\")
        for bb in bbs:
            cs = []
            for ds in ("NJN", "NeoJaundice"):
                r = idx.get((ds, bb))
                if r is None:
                    cs += ["---", "---", "---"]
                    continue
                d = f"{r['delta_acc']:+.2f}"
                d = f"\\highcell{{{d}}}" if r["delta_acc"] > 0 else neg(d)
                cs += [short(r["arm"]), f"{r['rgb_acc']:.2f}", d]
            lines.append(f"  {PRETTY[bb]} & " + " & ".join(cs) + " \\\\")
    with open(os.path.join(OUT, "H_tables.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nescrito: {os.path.join(OUT, 'H_tables.tex')}")


if __name__ == "__main__":
    main()
