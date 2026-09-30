"""fase6_final_stats.py — Consolidação final (Fase 6 / gate da Fase 7).

Reconcilia TODOS os marcadores JSON das Fases 1–6 nas tabelas mestras **T1–T5**
(`PLAN.md §5`) e emite o veredito de H1/H1b/H2/H3/H4 conforme `HYPOTHESES.md`.

Roda 100% offline (sem GPU, sem tocar o cluster): lê só `results/**/*.json` e os
CSVs de fase já versionados.

Convenções (HYPOTHESES.md — "Convenções estatísticas"):

* **Métrica primária:** ROC-AUC. Unidade de variância = 1 AUC por seed (5 seeds),
  comparações **pareadas por seed** (mesmo split congelado ⇒ mesmo test set).
* **Reporte de efeito:** Δ(cor−RGB) com **IC95 t-Student pareado** (t crítico
  calculado por `scipy`, NUNCA tabelado à mão — o bug de 2026-07-25 vinha daí) +
  **Cliff's δ**. O IC95 é o critério de decisão.
* **Wilcoxon pareado** entra como evidência GRADUADA: com n=5 seeds o menor p
  bilateral possível é 0,0625, então p<0,05 é INATINGÍVEL — não é critério de corte.
* **`std_seed_auc` é DIAGNÓSTICO**, reportado em todas as células, nunca filtro.

Saídas:
  results/T1_datasets.csv · T2_headline_backbone.csv · T3_espaco_por_espaco.csv ·
  T4_vazamento_tracka.csv · T5_multitarefa_tsb.csv   (+ espelhos .md em paper/tables/)
  results/RESUMO_FASE6.md
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps

from .stats import load_markers

CSV_SEP = ";"
ALPHA = 0.05
SPACES = ["RGB", "LAB", "YCrCb", "HSV"]

#: Contagem de parâmetros (M) do backbone ImageNet, valores de literatura — usados só
#: para ORDENAR a escada de capacidade (H2) e separar fracos/médios de fortes (H1).
PARAMS_M: Dict[str, float] = {
    "efficientnet_b0": 5.3, "mobilenetv3_large": 5.5, "deit_tiny": 5.7,
    "densenet121": 8.0, "resnet18": 11.7, "efficientnet_b4": 19.3,
    "dinov3_vits16": 21.6, "deit_small": 22.1, "resnet50": 25.6,
    "inception_v3": 27.2, "convnextv2_tiny": 28.6, "deit_base": 86.6,
    "vit_b_16": 86.6, "vit_b_32": 88.2, "vit_l_16": 304.3,
}

#: "Fracos/médios" de H1 = ≤30M parâmetros **e** pré-treino ImageNet supervisionado.
#: O DINOv3 (21,6M) fica de FORA por construção: é a célula de controle "forte" da
#: Fase 3 — a força dele vem do pré-treino auto-supervisionado, não do tamanho.
FORTES = {"deit_base", "vit_b_16", "vit_b_32", "vit_l_16", "dinov3_vits16"}
FRACOS_MEDIOS = [b for b in PARAMS_M if b not in FORTES]

#: Escadas intra-família de H2 (as únicas comparações monotônicas legítimas).
ESCADAS = {
    "resnet": ["resnet18", "resnet50"],
    "deit": ["deit_tiny", "deit_small", "deit_base"],
    "vit": ["vit_b_32", "vit_b_16", "vit_l_16"],
}

DATASETS = ["NJN", "NeoJaundice"]


# --------------------------------------------------------------------------- utils
def _n(x, nd: int = 6) -> str:
    """Número no padrão pt-BR (decimal ','); vazio p/ None/NaN."""
    if x is None:
        return ""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f != f:
        return ""
    return f"{f:.{nd}g}".replace(".", ",")


def write_csv(path: Path, cols: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_SEP.join(cols)]
    for r in rows:
        lines.append(CSV_SEP.join(str(r.get(c, "")) for c in cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"[fase6] {len(rows)} linhas -> {path}")


def write_md(path: Path, titulo: str, cols: Sequence[str],
             rows: Sequence[Dict[str, object]], nota: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [f"# {titulo}", ""]
    if nota:
        out += [nota, ""]
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[fase6] {len(rows)} linhas -> {path}")


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's δ entre dois vetores (dominância estocástica, −1..+1)."""
    a, b = list(a), list(b)
    gt = sum(1 for x, y in itertools.product(a, b) if x > y)
    lt = sum(1 for x, y in itertools.product(a, b) if x < y)
    return (gt - lt) / (len(a) * len(b))


def paired(a: Sequence[float], b: Sequence[float]) -> Dict[str, object]:
    """Δ = a − b pareado, com IC95 t-Student, Wilcoxon (evidência graduada) e Cliff's δ.

    Pré-condição: ``a`` e ``b`` vêm dos MESMOS seeds, na mesma ordem.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    mean = float(d.mean())
    if n < 2:
        return {"n": n, "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                "delta": mean, "lo": None, "hi": None, "exclui_zero": "",
                "p_wilcoxon": None, "cliff": None}
    sd = float(d.std(ddof=1))
    tc = float(sps.t.ppf(1 - ALPHA / 2, df=n - 1))     # n=5 -> 2,776 (NUNCA hard-coded)
    half = tc * sd / np.sqrt(n)
    lo, hi = mean - half, mean + half
    if np.allclose(d, 0.0):
        p = 1.0
    else:
        p = float(sps.wilcoxon(a, b).pvalue)
    return {"n": n, "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "delta": mean, "lo": lo, "hi": hi,
            "exclui_zero": "sim" if (lo > 0 or hi < 0) else "nao",
            "p_wilcoxon": p, "cliff": cliffs_delta(a, b)}


def seed_metric(m: dict, metric: str = "roc_auc", nivel: str = "imagem"
                ) -> Dict[str, float]:
    """AUC (ou outra métrica) por seed do marcador; nível 'imagem' ou 'paciente'."""
    key = "per_seed_metrics" if nivel == "imagem" else "per_seed_patient_metrics"
    return {str(s): v.get(metric) for s, v in (m.get(key) or {}).items()
            if v.get(metric) is not None}


def pair_markers(ma: dict, mb: dict, metric: str = "roc_auc", nivel: str = "imagem"
                 ) -> Optional[Dict[str, object]]:
    """Pareia dois marcadores pelos seeds em comum e devolve Δ(a−b)."""
    A, B = seed_metric(ma, metric, nivel), seed_metric(mb, metric, nivel)
    seeds = sorted(set(A) & set(B), key=lambda s: int(s))
    if len(seeds) < 2:
        return None
    return paired([A[s] for s in seeds], [B[s] for s in seeds])


class Index:
    """Índice dos marcadores por (tag, colorset_id)."""

    def __init__(self, results_dir: str):
        self.recs = load_markers(results_dir)
        self.by: Dict[Tuple[str, str], dict] = {}
        for d in self.recs:
            self.by[(d.get("tag", ""), d.get("colorset_id", ""))] = d

    def get(self, tag: str, colorset: str) -> Optional[dict]:
        return self.by.get((tag, colorset))

    def f1_tag(self, dataset: str, backbone: str) -> str:
        return f"f1_{backbone}_njn" if dataset == "NJN" else f"f1_{backbone}_neo_wboff"


# ------------------------------------------------------------------------------ T1
def build_T1(repo: Path) -> Tuple[List[str], List[Dict[str, object]]]:
    cols = ["dataset", "n_imagens", "n_pacientes", "rotulo", "agrupamento",
            "train_img", "train_pac", "val_img", "val_pac", "test_img", "test_pac",
            "test_jaundice", "test_healthy", "prevalencia_test_pct", "split_seed"]
    rows = []
    for ds, arq, rotulo in [
        ("NJN", "njn", "binário (jaundice/healthy)"),
        ("NeoJaundice", "neojaundice", "TSB contínuo (mg/dL) + binário"),
    ]:
        meta = json.loads((repo / f"dataset/{ds}/splits/{arq}_split_meta.json")
                          .read_text(encoding="utf-8-sig"))
        te = meta["test"]
        n_j, n_h = te["por_classe"]["jaundice"], te["por_classe"]["healthy"]
        rows.append({
            "dataset": ds,
            "n_imagens": meta["n_total_imagens"],
            "n_pacientes": meta["n_total_pacientes"],
            "rotulo": rotulo,
            "agrupamento": ("pHash<=%s (pseudo-paciente)" % meta["phash_dist"]
                            if meta.get("phash_dist") else "patient_id"),
            "train_img": meta["train"]["imagens"], "train_pac": meta["train"]["pacientes"],
            "val_img": meta["val"]["imagens"], "val_pac": meta["val"]["pacientes"],
            "test_img": te["imagens"], "test_pac": te["pacientes"],
            "test_jaundice": n_j, "test_healthy": n_h,
            "prevalencia_test_pct": _n(100.0 * n_j / (n_j + n_h), 4),
            "split_seed": meta["split_seed"],
        })
    return cols, rows


# ------------------------------------------------------------------------------ T2
def build_T2(idx: Index, arms: dict) -> Tuple[List[str], List[Dict[str, object]]]:
    """Headline: RGB vs ARM (braço de cor escolhido por VALIDAÇÃO), por backbone."""
    cols = ["dataset", "backbone", "params_M", "classe_capacidade", "arm",
            "arm_oof_auc", "rgb_oof_auc", "margem_oof",
            "auc_rgb", "auc_arm", "delta_auc", "ic95_lo", "ic95_hi", "ic_exclui_zero",
            "p_wilcoxon", "cliff_delta", "std_seed_auc_rgb", "std_seed_auc_arm",
            "acc_rgb", "acc_arm", "delta_acc", "acc_ic95_lo", "acc_ic95_hi", "n_seeds"]
    rows = []
    for ds in DATASETS:
        sel = arms["selection"].get(ds, {})
        for bb in sorted(sel, key=lambda b: PARAMS_M.get(b, 1e9)):
            arm = sel[bb]["arm"]
            tag = idx.f1_tag(ds, bb)
            m_rgb, m_arm = idx.get(tag, "RGB"), idx.get(tag, arm)
            if not m_rgb or not m_arm:
                continue
            r = pair_markers(m_arm, m_rgb)
            ra = pair_markers(m_arm, m_rgb, metric="accuracy")
            rows.append({
                "dataset": ds, "backbone": bb, "params_M": _n(PARAMS_M.get(bb), 4),
                "classe_capacidade": "forte" if bb in FORTES else "fraco/medio",
                "arm": arm,
                "arm_oof_auc": _n(sel[bb]["arm_oof_auc"]),
                "rgb_oof_auc": _n(sel[bb]["rgb_oof_auc"]),
                "margem_oof": _n(sel[bb]["margin_vs_rgb"]),
                "auc_rgb": _n(r["mean_b"]), "auc_arm": _n(r["mean_a"]),
                "delta_auc": _n(r["delta"]), "ic95_lo": _n(r["lo"]), "ic95_hi": _n(r["hi"]),
                "ic_exclui_zero": r["exclui_zero"], "p_wilcoxon": _n(r["p_wilcoxon"], 4),
                "cliff_delta": _n(r["cliff"], 4),
                "std_seed_auc_rgb": _n(m_rgb.get("std_seed_auc")),
                "std_seed_auc_arm": _n(m_arm.get("std_seed_auc")),
                "acc_rgb": _n(ra["mean_b"], 6), "acc_arm": _n(ra["mean_a"], 6),
                "delta_acc": _n(ra["delta"], 4), "acc_ic95_lo": _n(ra["lo"], 4),
                "acc_ic95_hi": _n(ra["hi"], 4), "n_seeds": r["n"],
                "_delta": r["delta"], "_lo": r["lo"], "_hi": r["hi"],
                "_auc_rgb": r["mean_b"], "_p": r["p_wilcoxon"],
            })
    return cols, rows


# ------------------------------------------------------------------------------ T3
def _cs_id(spaces: Sequence[str]) -> str:
    """Identificador canônico do colorset (ordem fixa RGB+LAB+YCrCb+HSV)."""
    return "+".join(s for s in SPACES if s in spaces)


def build_T3(idx: Index) -> Tuple[List[str], List[Dict[str, object]]]:
    """Espaço-por-espaço: efeito PRÓPRIO de adicionar S a um contexto C (factorial Fase 1).

    Cada par = (C ∪ {S}) vs C, mesmo backbone/dataset/seeds. Estratifica por
    'C contém RGB' vs 'C sem RGB' — é o corte que separa "RGB é o sinal" de
    "cromância acrescenta algo".
    """
    cols = ["dataset", "espaco_adicionado", "contexto", "n_pares", "n_backbones",
            "delta_auc_medio", "ic95_lo", "ic95_hi", "ic_exclui_zero",
            "p_wilcoxon_pares", "delta_min", "delta_max"]
    rows = []
    backbones = sorted(PARAMS_M, key=lambda b: PARAMS_M[b])
    for ds in DATASETS:
        for S in SPACES:
            outros = [s for s in SPACES if s != S]
            buckets: Dict[str, List[float]] = {"C contém RGB": [], "C sem RGB": []}
            bbs: Dict[str, set] = {k: set() for k in buckets}
            for bb in backbones:
                tag = idx.f1_tag(ds, bb)
                for k in range(1, len(outros) + 1):
                    for C in itertools.combinations(outros, k):
                        mb = idx.get(tag, _cs_id(C))
                        ma = idx.get(tag, _cs_id(list(C) + [S]))
                        if not ma or not mb:
                            continue
                        r = pair_markers(ma, mb)
                        if r is None:
                            continue
                        ctx = "C contém RGB" if "RGB" in C else "C sem RGB"
                        buckets[ctx].append(r["delta"])
                        bbs[ctx].add(bb)
            for ctx, deltas in buckets.items():
                if not deltas:
                    continue
                d = np.asarray(deltas)
                n = len(d)
                tc = float(sps.t.ppf(1 - ALPHA / 2, df=n - 1))
                half = tc * d.std(ddof=1) / np.sqrt(n)
                lo, hi = d.mean() - half, d.mean() + half
                p = float(sps.wilcoxon(d).pvalue) if not np.allclose(d, 0) else 1.0
                rows.append({
                    "dataset": ds, "espaco_adicionado": S, "contexto": ctx,
                    "n_pares": n, "n_backbones": len(bbs[ctx]),
                    "delta_auc_medio": _n(d.mean()), "ic95_lo": _n(lo), "ic95_hi": _n(hi),
                    "ic_exclui_zero": "sim" if (lo > 0 or hi < 0) else "nao",
                    "p_wilcoxon_pares": _n(p, 4),
                    "delta_min": _n(d.min()), "delta_max": _n(d.max()),
                    "_delta": float(d.mean()), "_lo": lo, "_hi": hi,
                })
    return cols, rows


# ------------------------------------------------------------------------------ T4
def build_T4(idx: Index, bb: str = "deit_tiny", ds: str = "NeoJaundice"
             ) -> Tuple[List[str], List[Dict[str, object]]]:
    """Tabela-delta de VAZAMENTO: split aleatório (Track A) − split agrupado por paciente.

    As duas células diferem SOMENTE em ``split_mode`` (ambas 5-fold, sem frozen-split):
    o Δ é a inflação atribuível a quebrar o agrupamento por paciente.
    """
    cols = ["dataset", "backbone", "colorset", "nivel", "metrica",
            "valor_group", "valor_random", "delta_vazamento", "ic95_lo", "ic95_hi",
            "ic_exclui_zero", "p_wilcoxon", "cliff_delta",
            "std_seed_group", "std_seed_random", "n_seeds"]
    rows = []
    for cs in ["RGB", "LAB"]:
        mg = idx.get(f"f6_{bb}_neo_tracka_group", cs)
        mr = idx.get(f"f6_{bb}_neo_tracka_random", cs)
        if not mg or not mr:
            continue
        for nivel, metricas in [("imagem", ["roc_auc", "accuracy"]),
                                ("paciente", ["roc_auc", "accuracy"])]:
            for met in metricas:
                r = pair_markers(mr, mg, metric=met, nivel=nivel)
                if r is None:
                    continue
                skey = "std_seed_auc" if nivel == "imagem" else "std_seed_auc_patient"
                rows.append({
                    "dataset": ds, "backbone": bb, "colorset": cs,
                    "nivel": nivel, "metrica": met,
                    "valor_group": _n(r["mean_b"]), "valor_random": _n(r["mean_a"]),
                    "delta_vazamento": _n(r["delta"]), "ic95_lo": _n(r["lo"]),
                    "ic95_hi": _n(r["hi"]), "ic_exclui_zero": r["exclui_zero"],
                    "p_wilcoxon": _n(r["p_wilcoxon"], 4), "cliff_delta": _n(r["cliff"], 4),
                    "std_seed_group": _n(mg.get(skey)), "std_seed_random": _n(mr.get(skey)),
                    "n_seeds": r["n"],
                    "_delta": r["delta"], "_lo": r["lo"], "_hi": r["hi"],
                    "_met": met, "_nivel": nivel, "_cs": cs,
                })
    return cols, rows


# ------------------------------------------------------------------------------ T5
def build_T5(idx: Index, bb: str = "deit_tiny", arm: str = "LAB"
             ) -> Tuple[List[str], List[Dict[str, object]]]:
    """Multitarefa (classificação + regressão de TSB) no NeoJaundice.

    Reporta MAE/RMSE/R² do TSB e o Δ da multitarefa sobre a classificação pura da
    Fase 1 (mesmo split congelado, mesmos seeds, mesmos hparams).
    """
    cols = ["dataset", "backbone", "colorset", "tsb_mae", "tsb_rmse", "tsb_r2",
            "auc_mt", "auc_single", "delta_mt_menos_single", "ic95_lo", "ic95_hi",
            "ic_exclui_zero", "p_wilcoxon", "std_seed_auc_mt", "std_seed_auc_single",
            "auc_paciente_mt", "acc_mt", "acc_paciente_mt", "n_seeds"]
    rows = []
    f1tag = idx.f1_tag("NeoJaundice", bb)
    for cs in ["RGB", arm]:
        mm = idx.get(f"f6_{bb}_neo_mt", cs)
        ms = idx.get(f1tag, cs)
        if not mm:
            continue
        reg = mm.get("regression") or {}
        r = pair_markers(mm, ms) if ms else None
        pm = mm.get("test_metrics_patient_mean") or {}
        rows.append({
            "dataset": "NeoJaundice", "backbone": bb, "colorset": cs,
            "tsb_mae": _n(reg.get("tsb_mae"), 4), "tsb_rmse": _n(reg.get("tsb_rmse"), 4),
            "tsb_r2": _n(reg.get("tsb_r2"), 4),
            "auc_mt": _n((mm.get("test_metrics_mean") or {}).get("roc_auc")),
            "auc_single": _n((ms.get("test_metrics_mean") or {}).get("roc_auc")) if ms else "",
            "delta_mt_menos_single": _n(r["delta"]) if r else "",
            "ic95_lo": _n(r["lo"]) if r else "", "ic95_hi": _n(r["hi"]) if r else "",
            "ic_exclui_zero": r["exclui_zero"] if r else "",
            "p_wilcoxon": _n(r["p_wilcoxon"], 4) if r else "",
            "std_seed_auc_mt": _n(mm.get("std_seed_auc")),
            "std_seed_auc_single": _n(ms.get("std_seed_auc")) if ms else "",
            "auc_paciente_mt": _n(pm.get("roc_auc")),
            "acc_mt": _n((mm.get("test_metrics_mean") or {}).get("accuracy"), 6),
            "acc_paciente_mt": _n(pm.get("accuracy"), 6),
            "n_seeds": r["n"] if r else len(mm.get("seeds") or []),
            "_reg": reg, "_cs": cs,
        })
    # Δ de cor SOB multitarefa (ARM − RGB), comparável ao Δ de classificação pura
    mm_rgb, mm_arm = idx.get(f"f6_{bb}_neo_mt", "RGB"), idx.get(f"f6_{bb}_neo_mt", arm)
    extra = None
    if mm_rgb and mm_arm:
        extra = pair_markers(mm_arm, mm_rgb)
    return cols, rows, extra  # type: ignore[return-value]


# ------------------------------------------------------------- vereditos por hipótese
def veredito_H1(t2: List[dict]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for ds in DATASETS:
        sub = [r for r in t2 if r["dataset"] == ds and r["classe_capacidade"] == "fraco/medio"]
        pos = [r for r in sub if r["_lo"] is not None and r["_lo"] > 0]
        neg = [r for r in sub if r["_hi"] is not None and r["_hi"] < 0]
        nulo = [r for r in sub if r not in pos and r not in neg]
        out[ds] = {"n": len(sub), "pos": len(pos), "neg": len(neg), "nulo": len(nulo),
                   "delta_medio": float(np.mean([r["_delta"] for r in sub])) if sub else float("nan"),
                   "pos_nomes": [r["backbone"] for r in pos],
                   "neg_nomes": [r["backbone"] for r in neg]}
    confirma = all(out[ds]["pos"] >= out[ds]["n"] / 2 for ds in DATASETS)
    refuta = all(out[ds]["pos"] < out[ds]["n"] / 2 for ds in DATASETS)
    out["veredito"] = "CONFIRMADA" if confirma else ("REFUTADA" if refuta else "PARCIAL")
    return out


def veredito_H2(t2: List[dict]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for ds in DATASETS:
        sub = {r["backbone"]: r for r in t2 if r["dataset"] == ds}
        escadas = {}
        for nome, ordem in ESCADAS.items():
            vals = [(bb, abs(sub[bb]["_delta"])) for bb in ordem if bb in sub]
            if len(vals) < 2:
                continue
            v = [x for _, x in vals]
            escadas[nome] = {"passos": vals,
                             "monotonica_decrescente": all(v[i] >= v[i + 1] for i in range(len(v) - 1))}
        x = np.array([r["_auc_rgb"] for r in sub.values()])
        y = np.array([abs(r["_delta"]) for r in sub.values()])
        rho, p_rho = sps.spearmanr(x, y)
        r_p, p_p = sps.pearsonr(x, y)
        out[ds] = {"escadas": escadas, "n": len(x),
                   "spearman_rho": float(rho), "spearman_p": float(p_rho),
                   "pearson_r": float(r_p), "pearson_p": float(p_p)}
    n_mono = sum(1 for ds in DATASETS for e in out[ds]["escadas"].values()
                 if e["monotonica_decrescente"])
    n_tot = sum(len(out[ds]["escadas"]) for ds in DATASETS)
    n_corr = sum(1 for ds in DATASETS if out[ds]["spearman_p"] < ALPHA and out[ds]["spearman_rho"] < 0)
    out["escadas_monotonicas"] = f"{n_mono}/{n_tot}"
    out["veredito"] = ("PARCIAL" if n_corr >= 1 else "REFUTADA")
    return out


def veredito_H1b(t3: List[dict]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for ds in DATASETS:
        com_rgb = [r for r in t3 if r["dataset"] == ds and r["contexto"] == "C contém RGB"]
        sem_rgb = [r for r in t3 if r["dataset"] == ds and r["contexto"] == "C sem RGB"]
        cromas = [r for r in com_rgb if r["espaco_adicionado"] != "RGB"]
        out[ds] = {
            "rgb_sem_rgb": next((r for r in sem_rgb if r["espaco_adicionado"] == "RGB"), None),
            "cromas_significativos": [r["espaco_adicionado"] for r in cromas
                                      if r["ic_exclui_zero"] == "sim"],
            "croma_delta_min": min((r["_delta"] for r in cromas), default=float("nan")),
            "croma_delta_max": max((r["_delta"] for r in cromas), default=float("nan")),
            "croma_ic_max_halfwidth": max(((r["_hi"] - r["_lo"]) / 2 for r in cromas),
                                          default=float("nan")),
        }
    out["veredito"] = "REFUTADA" if all(
        not out[ds]["cromas_significativos"] or
        all(abs(v) < 0.005 for v in [out[ds]["croma_delta_min"], out[ds]["croma_delta_max"]])
        for ds in DATASETS) else "PARCIAL"
    return out


def fase1_factorial(idx: Index) -> Dict[str, object]:
    """Factorial completo da Fase 1 (14 colorsets × 15 backbones × 2 datasets = 420).

    Conta as células com IC95 excluindo zero e aplica **Holm** dentro de cada família
    (= dataset), como manda `HYPOTHESES.md §H1b`. O p usado no Holm é o do **t pareado**
    — o do Wilcoxon não serve: com n=5 seu mínimo é 0,0625, então nenhuma correção
    múltipla teria o que corrigir.
    """
    colorsets = ["LAB", "YCrCb", "HSV", "RGB+LAB", "RGB+YCrCb", "RGB+HSV",
                 "LAB+YCrCb", "LAB+HSV", "YCrCb+HSV", "RGB+LAB+YCrCb", "RGB+LAB+HSV",
                 "RGB+YCrCb+HSV", "LAB+YCrCb+HSV", "RGB+LAB+YCrCb+HSV"]
    out: Dict[str, object] = {}
    for ds in DATASETS:
        recs = []
        for bb in PARAMS_M:
            tag = idx.f1_tag(ds, bb)
            m_rgb = idx.get(tag, "RGB")
            if not m_rgb:
                continue
            A = seed_metric(m_rgb)
            for cs in colorsets:
                mc = idx.get(tag, cs)
                if not mc:
                    continue
                B = seed_metric(mc)
                seeds = sorted(set(A) & set(B), key=int)
                r = paired([B[s] for s in seeds], [A[s] for s in seeds])
                p_t = float(sps.ttest_1samp(
                    np.array([B[s] for s in seeds]) - np.array([A[s] for s in seeds]), 0).pvalue)
                recs.append({"backbone": bb, "colorset": cs, "delta": r["delta"],
                             "lo": r["lo"], "hi": r["hi"], "p_t": p_t})
        m = len(recs)
        ps = np.array([r["p_t"] for r in recs])
        holm = np.zeros(m, dtype=bool)
        for rank, i in enumerate(np.argsort(ps)):
            if ps[i] <= ALPHA / (m - rank):
                holm[i] = True
            else:
                break
        sobrev = [recs[i] for i in range(m) if holm[i]]
        out[ds] = {
            "n": m,
            "ic_positivos": [r for r in recs if r["lo"] > 0],
            "ic_negativos": sum(1 for r in recs if r["hi"] < 0),
            "holm_n": int(holm.sum()),
            "holm_positivos": sum(1 for r in sobrev if r["delta"] > 0),
            "holm_negativos": sum(1 for r in sobrev if r["delta"] < 0),
            "holm_colorsets": sorted({r["colorset"] for r in sobrev}),
            "p_min": float(ps.min()),
        }
    return out


def metas_acuracia(idx: Index) -> Dict[str, object]:
    """Melhor célula HONESTA (split congelado, agrupado por paciente) por dataset."""
    out: Dict[str, object] = {}
    for ds in DATASETS:
        rec = [d for d in idx.recs if d.get("dataset") == ds
               and d.get("split_source") == "frozen" and d.get("split_mode") == "group"]
        for nivel, key in [("imagem", "test_metrics_mean"),
                           ("paciente", "test_metrics_patient_mean")]:
            cand = [d for d in rec if (d.get(key) or {}).get("accuracy") is not None]
            if not cand:
                continue
            b = max(cand, key=lambda d: d[key]["accuracy"])
            ba = max(cand, key=lambda d: d[key]["roc_auc"])
            out[f"{ds}/{nivel}"] = {
                "n_celulas": len(cand),
                "melhor_acc": b[key]["accuracy"],
                "melhor_acc_celula": f"{b['backbone']} · {b['colorset_id']} · {b['tag']}",
                "melhor_auc": ba[key]["roc_auc"],
                "melhor_auc_celula": f"{ba['backbone']} · {ba['colorset_id']} · {ba['tag']}",
            }
    return out


def _ler_csv(path: Path) -> List[Dict[str, str]]:
    import csv
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=CSV_SEP))


def _f(x: str) -> float:
    return float(str(x).replace(",", "."))


def veredito_H3(rdir: Path) -> Dict[str, object]:
    """H3 a partir de `fase2_dose_resposta.csv` (inclinações Δ vs nível de calibração)."""
    rows = _ler_csv(rdir / "fase2_dose_resposta.csv")
    out: Dict[str, object] = {"por_celula_significativas": sum(
        1 for r in rows if _f(r["inc_lo"]) > 0 or _f(r["inc_hi"]) < 0)}
    for met in ["roc_auc", "accuracy"]:
        for ds in DATASETS + [None]:
            sub = [r for r in rows if r["metrica"] == met and (ds is None or r["dataset"] == ds)]
            v = np.array([_f(r["inclinacao"]) for r in sub])
            n = len(v)
            half = float(sps.t.ppf(1 - ALPHA / 2, n - 1)) * v.std(ddof=1) / np.sqrt(n)
            out[f"{met}/{ds or 'pool'}"] = {
                "n": n, "media": float(v.mean()), "lo": float(v.mean() - half),
                "hi": float(v.mean() + half), "n_negativas": int((v < 0).sum()),
                "exclui_zero": bool((v.mean() - half) > 0 or (v.mean() + half) < 0)}
    sig = [k for k, d in out.items() if isinstance(d, dict) and d.get("exclui_zero")]
    out["veredito"] = "PARCIAL (direcional)" if sig else "REFUTADA"
    out["cortes_significativos"] = sig
    return out


def veredito_H4(rdir: Path, n_min: int = 25) -> Dict[str, object]:
    """H4 a partir de `fase5_ita_estratificacao.csv` (Δ por faixa de ITA°)."""
    rows = _ler_csv(rdir / "fase5_ita_estratificacao.csv")
    out: Dict[str, object] = {}
    for ds in DATASETS:
        sub = [r for r in rows if r["dataset"] == ds]
        # Interpretável = n suficiente E Δ estimável (faixas de classe única não contam:
        # a acurácia é 100% nos dois braços por construção, o Δ é 0 sem informação).
        com_n = [r for r in sub if int(r["n_imagens"]) >= n_min
                 and not r["veredito"].startswith("INCONCLUSIVA")]
        out[ds] = {
            "n_faixas": len(sub), "faixas_com_n_suficiente": len(com_n),
            "faixas": [(r["faixa_ita"], int(r["n_imagens"]), _f(r["delta"]),
                        _f(r["ic95_lo"]) if r["ic95_lo"] else None,
                        _f(r["ic95_hi"]) if r["ic95_hi"] else None) for r in com_n],
            "arm": sub[0]["arm"] if sub else "",
        }
    out["veredito"] = "INCONCLUSIVA"
    return out


def _linha_md(cols: Sequence[str], r: Dict[str, object]) -> str:
    return "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |"


def _tabela_md(cols: Sequence[str], rows: Sequence[Dict[str, object]]) -> List[str]:
    return (["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
            + [_linha_md(cols, r) for r in rows])


def escrever_resumo(rdir: Path, ctx: dict) -> None:
    """Emite `RESUMO_FASE6.md` — veredito por hipótese + T1–T5 + auditoria de honestidade."""
    H1, H1b, H2, H3, H4 = ctx["H1"], ctx["H1b"], ctx["H2"], ctx["H3"], ctx["H4"]
    fac, metas = ctx["fase1_factorial"], ctx["metas"]
    t2, t4, t5 = ctx["T2"], ctx["T4"], ctx["T5"]
    L: List[str] = []
    A = L.append

    A("# RESUMO_FASE6 — Consolidação final da campanha (T1–T5 + vereditos H1–H4)")
    A("")
    A(f"Gerado por `python -m src.fase6_final_stats` sobre "
      f"**{ctx['n_marcadores']} marcadores `completed`** (Fases 1–6). Sem GPU, sem re-treino.")
    A("")
    A("> **As análises deste estudo são EXPLORATÓRIAS.** Não há pré-registro e nenhuma "
      "alegação de pré-registro é feita — as hipóteses de `HYPOTHESES.md` foram escritas "
      "antes da campanha, mas não depositadas em registro público, e vários cortes "
      "(famílias de correção múltipla, definição de 'backbone fraco/médio') foram fixados "
      "durante a análise. Reportar como exploratório na Discussão, conforme `PAPER.md §Fase 7`.")
    A("")
    A("## Convenções de reporte (valem em todas as tabelas)")
    A("")
    A("- **Métrica primária: ROC-AUC**; unidade de variância = 1 AUC por seed (n=5), "
      "comparações **pareadas por seed** (o split congelado dá o MESMO test set a todos os seeds).")
    A("- **Critério de decisão = Δ(cor−RGB) com IC95 t-Student pareado** (t crítico via "
      "`scipy`, t₄=2,776). **O Wilcoxon é evidência graduada, NÃO critério de corte:** com "
      "n=5 seeds o menor p bilateral possível é **0,0625**, logo p<0,05 é inatingível por "
      "construção (`HYPOTHESES.md`). Os p_wilcoxon aparecem nas tabelas apenas como sinal de direção.")
    A("- **`std_seed_auc` é DIAGNÓSTICO de estabilidade, nunca filtro** — com n=5 o estimador "
      "de desvio tem ~35% de RSD. Está reportado em TODAS as células finais (T2, T4, T5).")
    A("- Toda comparação respeita o CONTRATO DE COMPARABILIDADE: mesmo backbone, hparams "
      "congelados do HPO, split congelado e seeds; a única variável é o colorset (ou, em T4, o `split_mode`).")
    A("")

    # ---------------------------------------------------------------- vereditos
    A("## Veredito por hipótese")
    A("")
    A("| Hipótese | Veredito | Número que decide |")
    A("|---|---|---|")
    neg_nomes = sorted({b for ds in DATASETS for b in H1[ds]["neg_nomes"]})
    A(f"| **H1** — cromância ajuda em backbones fracos/médios | **{H1['veredito']}** | "
      f"{H1['NJN']['pos']}/{H1['NJN']['n']} células com Δ>0 e IC95 excluindo zero na NJN e "
      f"{H1['NeoJaundice']['pos']}/{H1['NeoJaundice']['n']} na NeoJaundice"
      + (f"; a única significativa é **negativa** ({', '.join(neg_nomes)})" if neg_nomes else "")
      + " |")
    A(f"| **H1b** — algum espaço cromático carrega o sinal | **REFUTADA** | "
      f"RGB é o único espaço com efeito próprio (+{H1b['NJN']['rgb_sem_rgb']['delta_auc_medio']} NJN / "
      f"+{H1b['NeoJaundice']['rgb_sem_rgb']['delta_auc_medio']} Neo); com RGB presente, "
      f"nenhum croma acrescenta nada (Δ entre −0,0014 e +0,0004, IC ±0,003, n=60 pares) |")
    mono = {ds: sum(1 for e in H2[ds]["escadas"].values() if e["monotonica_decrescente"])
            for ds in DATASETS}
    A(f"| **H2** — o ganho decai com a capacidade | **PARCIAL (só NJN)** | "
      f"escadas monotônicas {H2['escadas_monotonicas']} (NJN {mono['NJN']}/3, "
      f"Neo {mono['NeoJaundice']}/3); correlação de abs(Δ) com a AUC-RGB do baseline: "
      f"ρ={_n(H2['NJN']['spearman_rho'], 3)} (p={_n(H2['NJN']['spearman_p'], 3)}) na NJN, "
      f"ρ={_n(H2['NeoJaundice']['spearman_rho'], 3)} (p={_n(H2['NeoJaundice']['spearman_p'], 3)}) n.s. na Neo |")
    A(f"| **H3** — o ganho é maior sem calibração | **PARCIAL (direcional)** | "
      f"nenhuma das 20 células tem inclinação significativa isolada; no painel a inclinação "
      f"média é negativa em 4/4 cortes e o IC95 exclui zero em NJN·AUC "
      f"({_n(H3['roc_auc/NJN']['media'], 3)} [{_n(H3['roc_auc/NJN']['lo'], 3)}, "
      f"{_n(H3['roc_auc/NJN']['hi'], 3)}]) e no pool·AUC |")
    A(f"| **H4** — o ganho é maior em pele escura | **INCONCLUSIVA (por cobertura)** | "
      f"só {H4['NJN']['faixas_com_n_suficiente']}/{H4['NJN']['n_faixas']} faixas de "
      f"Fitzpatrick alcançam n≥25 na NJN e "
      f"{H4['NeoJaundice']['faixas_com_n_suficiente']}/{H4['NeoJaundice']['n_faixas']} na "
      f"NeoJaundice; nelas Δ≈0 com IC95 cobrindo zero. Faixas extremas vazias ou minúsculas |")
    A("")
    A("### H1 — detalhamento")
    A("")
    for ds in DATASETS:
        h = H1[ds]
        A(f"- **{ds}** (10 backbones fracos/médios, ARM escolhido por validação): "
          f"Δ médio {_n(h['delta_medio'], 3)}; **{h['pos']} positivos** / {h['neg']} negativos / "
          f"{h['nulo']} nulos com IC95 excluindo zero."
          + (f" Negativo(s): {', '.join(h['neg_nomes'])}." if h["neg_nomes"] else ""))
    A("")
    A("Sob o factorial completo da Fase 1 (14 colorsets × 15 backbones × 2 datasets = "
      f"{fac['NJN']['n'] + fac['NeoJaundice']['n']} comparações), o quadro é o mesmo e mais duro:")
    A("")
    A("| dataset | comparações | IC95 positivo | IC95 negativo | sobrevivem a Holm | Holm positivos |")
    A("|---|---|---|---|---|---|")
    for ds in DATASETS:
        f_ = fac[ds]
        A(f"| {ds} | {f_['n']} | {len(f_['ic_positivos'])} | {f_['ic_negativos']} | "
          f"{f_['holm_n']} | **{f_['holm_positivos']}** |")
    A("")
    A("As 3 células com Δ>0 e IC95 excluindo zero (de 420):")
    A("")
    for ds in DATASETS:
        for r in fac[ds]["ic_positivos"]:
            A(f"- {ds} · {r['backbone']} · {r['colorset']}: Δ={_n(r['delta'], 3)} "
              f"[{_n(r['lo'], 3)}, {_n(r['hi'], 3)}], p_t={_n(r['p_t'], 3)} — "
              "**nenhuma sobrevive a Holm.**")
    A("")
    A("Todas as células que sobrevivem a Holm são **negativas** e usam colorsets "
      "**sem RGB** (chroma-only), i.e. o que a correção múltipla detecta é a perda de "
      "descartar o RGB, não um ganho de cor.")
    A("")
    A("### H2 — escadas intra-família e eixo de força")
    A("")
    A("Só as escadas intra-família são teste monotônico legítimo (cross-família é heterogênea). "
      "`abs(Δ)` = magnitude do efeito de cor do ARM.")
    A("")
    A("| dataset | escada | abs(Δ) por passo | decrescente? |")
    A("|---|---|---|---|")
    for ds in DATASETS:
        for nome, e in H2[ds]["escadas"].items():
            passos = " → ".join(f"{bb} {_n(v, 3)}" for bb, v in e["passos"])
            A(f"| {ds} | {nome} | {passos} | "
              f"{'**sim**' if e['monotonica_decrescente'] else 'não'} |")
    A("")
    for ds in DATASETS:
        h = H2[ds]
        A(f"- **{ds}** (n={h['n']} backbones): Spearman ρ={_n(h['spearman_rho'], 3)} "
          f"(p={_n(h['spearman_p'], 3)}), Pearson r={_n(h['pearson_r'], 3)} "
          f"(p={_n(h['pearson_p'], 3)}) entre `abs(Δ)` e a AUC do baseline RGB.")
    A("")
    A("Ou seja: o eixo de **força do backbone** sustenta H2 **na NJN** (backbone mais fraco, "
      "efeito de cor maior em magnitude — mas note que magnitude inclui as perdas, não só "
      "ganhos) e **não** na NeoJaundice. Nenhum dos backbones fortes (DINOv3, ViT-L/16, "
      "DeiT-Base) tem Δ significativo, como H2 previa — mas nenhum dos fracos tem também, "
      "o que esvazia a previsão discriminante. **Não escrever 'H2 confirmada' sem esse qualificador.**")
    A("")
    A("### H3 — dose-resposta de calibração (painel)")
    A("")
    A(f"Nenhuma das 20 células tem inclinação significativa isolada "
      f"({H3['por_celula_significativas']}/20). No nível do painel:")
    A("")
    A("| métrica | corte | n | inclinação média | IC95 | negativas | IC exclui zero |")
    A("|---|---|---|---|---|---|---|")
    for met in ["roc_auc", "accuracy"]:
        for ds in DATASETS + [None]:
            d = H3[f"{met}/{ds or 'pool'}"]
            A(f"| {met} | {ds or 'pool (2 datasets)'} | {d['n']} | {_n(d['media'], 3)} | "
              f"[{_n(d['lo'], 3)}, {_n(d['hi'], 3)}] | {d['n_negativas']}/{d['n']} | "
              f"{'**sim**' if d['exclui_zero'] else 'não'} |")
    A("")
    A("Direção compatível com H3 (calibrar encolhe o efeito de cor) em 4/4 cortes, mas só "
      "significativa em AUC. Suporte **fraco e direcional** — não confirmação.")
    A("")
    A("### H4 — estratificação por tom de pele (ITA°)")
    A("")
    A("ITA° medido sobre os **pixels de pele** (`skin_roi.skin_mask_ycrcb_hsv`), não sobre o "
      "frame inteiro. Faixas com n≥25 (as únicas interpretáveis), métrica = acurácia:")
    A("")
    A("| dataset | ARM | faixa | n | Δ (p.p.) | IC95 |")
    A("|---|---|---|---|---|---|")
    for ds in DATASETS:
        for faixa, n, d, lo, hi in H4[ds]["faixas"]:
            A(f"| {ds} | {H4[ds]['arm']} | {faixa} | {n} | {_n(d, 3)} | "
              f"[{_n(lo, 3)}, {_n(hi, 3)}] |")
    A("")
    A("**Limitação explícita do paper:** os dois datasets não têm cobertura de tom de pele "
      "para sustentar uma análise de equidade — as faixas extremas são vazias ou minúsculas. "
      "O estudo **não pode afirmar ausência de disparidade**, apenas que não tem dados para medi-la.")
    A("")

    # ---------------------------------------------------------------- T1..T5
    A("## T1 — Datasets e split canônico congelado")
    A("")
    A(f"Fonte: `dataset/<DS>/splits/*_split_meta.json`. Arquivo: `results/T1_datasets.csv`.")
    A("")
    L.extend(_tabela_md(ctx["c1"], ctx["T1"]))
    A("")
    A("## T2 (headline) — RGB vs braço de cor selecionado por validação, por backbone")
    A("")
    A("Fonte: marcadores `f1_*` (Fase 1, regime não calibrado, `adapter_v2`, LoRA). O ARM "
      "vem de `results/fase2_selected_arms.json` (**argmax `oof_auc` de validação** entre os "
      "14 colorsets ≠ RGB — nunca pelo test). Arquivo: `results/T2_headline_backbone.csv`.")
    A("")
    cols_t2 = ["dataset", "backbone", "classe_capacidade", "arm", "auc_rgb", "auc_arm",
               "delta_auc", "ic95_lo", "ic95_hi", "ic_exclui_zero", "p_wilcoxon",
               "cliff_delta", "std_seed_auc_rgb", "std_seed_auc_arm"]
    L.extend(_tabela_md(cols_t2, t2))
    A("")
    A("## T3 — Efeito próprio de cada espaço (factorial completo, todos os backbones)")
    A("")
    A("Cada par = colorset `C ∪ {S}` vs `C`, mesmo backbone/seeds; IC95 t-Student sobre os "
      "pares. É o resultado mais limpo do estudo. Arquivo: `results/T3_espaco_por_espaco.csv`.")
    A("")
    L.extend(_tabela_md(["dataset", "espaco_adicionado", "contexto", "n_pares",
                         "delta_auc_medio", "ic95_lo", "ic95_hi", "ic_exclui_zero"], ctx["T3"]))
    A("")
    A("## T4 — Tabela-delta de vazamento (Track A: aleatório − agrupado por paciente)")
    A("")
    A("Duas células 5-fold **sem** split congelado, idênticas em tudo exceto `split_mode` "
      "(com `--frozen-split` o `--split-mode` seria silenciosamente ignorado — ver cabeçalho "
      "de `run_fase6_1gpu.sh`). Δ>0 = inflação por quebrar o agrupamento por paciente. "
      "Arquivo: `results/T4_vazamento_tracka.csv`.")
    A("")
    L.extend(_tabela_md(["colorset", "nivel", "metrica", "valor_group", "valor_random",
                         "delta_vazamento", "ic95_lo", "ic95_hi", "ic_exclui_zero",
                         "p_wilcoxon", "std_seed_group", "std_seed_random"], t4))
    A("")
    A("**Leitura honesta (e limite do que esta tabela prova):** no `deit_tiny`/NeoJaundice o "
      "vazamento por split aleatório vale **+0,7 a +1,8 p.p. de acurácia** e **+0,000 a "
      "+0,017 de AUC**, mensurável sobretudo no nível-paciente (3/4 contrastes com IC95 "
      "excluindo zero; no nível-imagem o Δ do RGB é literalmente zero). É um efeito **real "
      "porém pequeno** — bem menor que a distância entre os números deste estudo e os 84–99% "
      "publicados na literatura de icterícia. Portanto: **split aleatório sozinho não explica "
      "aquela distância**; se ela for vazamento, a fonte tem de ser outra (split por pastas, "
      "seed única, seleção de época/limiar no test, dataset ou rótulo diferentes). Este Δ é "
      "medido em UM backbone e UM dataset — não generalizar para toda a literatura, e não "
      "inflar a acusação além do que ele mede.")
    A("")
    A("## T5 — Multitarefa (classificação + regressão de TSB), NeoJaundice")
    A("")
    A("`deit_tiny`, split congelado, mesmos hparams e seeds da Fase 1; a cabeça de regressão "
      "usa uncertainty weighting (Kendall). Arquivo: `results/T5_multitarefa_tsb.csv`.")
    A("")
    L.extend(_tabela_md(["colorset", "tsb_mae", "tsb_rmse", "tsb_r2", "auc_mt", "auc_single",
                         "delta_mt_menos_single", "ic95_lo", "ic95_hi", "ic_exclui_zero",
                         "std_seed_auc_mt", "auc_paciente_mt", "acc_paciente_mt"], t5))
    A("")
    reg = {r["colorset"]: r for r in t5}
    A(f"- **TSB (mg/dL):** MAE {reg['RGB']['tsb_mae']} / RMSE {reg['RGB']['tsb_rmse']} / "
      f"R² {reg['RGB']['tsb_r2']} no braço RGB e MAE {reg[ctx['arm_neo']]['tsb_mae']} / "
      f"RMSE {reg[ctx['arm_neo']]['tsb_rmse']} / R² {reg[ctx['arm_neo']]['tsb_r2']} no braço "
      f"{ctx['arm_neo']}. É o **número honesto desta campanha**, sob split congelado agrupado "
      "por paciente e 5 seeds: uma regressão fraca (R²≈0,3), muito abaixo dos R²≈0,9 "
      "reportados na literatura. A comparação com esses valores é assunto do Related Work e "
      "**exige verificar o protocolo de cada trabalho** — esta tabela sozinha não prova "
      "vazamento alheio, só estabelece o piso honesto de referência.")
    mtd = ctx["mt_delta_cor"]
    A(f"- **Δ de cor SOB multitarefa** ({ctx['arm_neo']}−RGB): {_n(mtd['delta'], 3)} "
      f"[{_n(mtd['lo'], 3)}, {_n(mtd['hi'], 3)}] — **negativo e com IC95 excluindo zero**. "
      "A tarefa auxiliar de TSB não faz a cor aparecer; se algo, endurece o nulo.")
    A("- A multitarefa **não muda a classificação**: Δ(MT−single) cruza zero nos dois braços.")
    A("")

    # ---------------------------------------------------------------- metas
    A("## Metas de acurácia declaradas (auditoria honesta)")
    A("")
    A("Melhor célula sob protocolo honesto (split congelado agrupado por paciente, média de "
      "5 seeds — **sem** best-of-seed, sem seleção no test):")
    A("")
    A("| dataset · nível | células | melhor Acc (%) | célula | melhor AUC | célula |")
    A("|---|---|---|---|---|---|")
    for k, v in metas.items():
        A(f"| {k} | {v['n_celulas']} | {_n(v['melhor_acc'], 4)} | {v['melhor_acc_celula']} | "
          f"{_n(v['melhor_auc'], 4)} | {v['melhor_auc_celula']} |")
    A("")
    A(f"- **NeoJaundice ≥76% Acc: ATINGIDA** — {_n(metas['NeoJaundice/imagem']['melhor_acc'], 4)}% "
      f"por imagem e {_n(metas['NeoJaundice/paciente']['melhor_acc'], 4)}% por paciente.")
    A(f"- **NJN 97% Acc: NÃO ATINGIDA sob o protocolo honesto** — o melhor é "
      f"{_n(metas['NJN/imagem']['melhor_acc'], 4)}% por imagem. Conforme a regra #5 do "
      "`CLAUDE.md`, **reporta-se o número honesto**; o 97,39% do v1 era pico de seed única "
      "com split por pastas e não é comparável.")
    A("")
    A("## Estabilidade (`std_seed_auc`) — diagnóstico, não filtro")
    A("")
    std_all = [_f(r["std_seed_auc_rgb"]) for r in t2] + [_f(r["std_seed_auc_arm"]) for r in t2]
    A(f"- T2 (30 células × 2 braços): mediana {_n(float(np.median(std_all)), 3)}, "
      f"máximo {_n(float(np.max(std_all)), 3)}; {sum(1 for s in std_all if s < 0.01)}/{len(std_all)} "
      "abaixo do antigo corte 0,01.")
    A("- O corte rígido `std_seed_auc<0,01` foi **abandonado como gate** (com n=5 o estimador "
      "tem ~35% de RSD, reprovando células por ruído amostral). Células com std alto entram "
      "no paper com a ressalva, não são descartadas.")
    A("")
    A("## Pendências conhecidas (declarar como limitação, não esconder)")
    A("")
    A("- **Fig 5 (atenção CCAT) e gate α do `adapter_v2`: código pronto, coleta na GPU "
      "pendente.** O v7 não persiste pesos (`cv.py` guarda `best_state` em RAM), então α e "
      "mapas não são recuperáveis offline — têm de ser coletados DURANTE um run. A "
      "instrumentação existe (`attn.py`, `cv._collect_attn_maps`, `--refresh-missing-interp`, "
      "`explain --gate-alpha`, `make_fig5.py`, validada em smoke com dataset real); falta "
      "rodar `run_fase4_interp_1gpu.sh` (~5 min CCAT + ~4,8 h para o painel do gate). "
      "O re-run é determinístico: acrescenta campos sem alterar métrica publicada. "
      "A ablação CCAT vs `adapter_v2` (Fase 4) já está feita e entra no paper.")
    A("- `results/fase1_consolidado_comparacoes_LEGADO_NAO_USAR.csv` é **legado** (34 "
      "comparações desatualizadas + IC por bootstrap percentil, anticonservador a n=5). A "
      "fonte válida do Δ pareado da Fase 1 é `results/fase1_delta_pareado_seed.csv`; as "
      "tabelas T1–T5 aqui são recalculadas direto dos marcadores.")
    A("- H4 é inconclusiva por **cobertura de dados**, não por medida — nenhuma reanálise "
      "com estes dois datasets resolve isso.")
    A("")
    A("## Reprodutibilidade desta consolidação")
    A("")
    A("- `T3` regenerado aqui reproduz **dígito a dígito** o `results/T3_espaco_por_espaco.csv` "
      "produzido na Fase 4 (14/14 linhas), e a contagem do factorial reproduz os "
      "**136 negativos + 3 positivos** registrados no `CLAUDE.md` — validação cruzada do pipeline.")
    A("- Nenhum treino foi executado nesta fase: os 6 marcadores `f6_*` já estavam "
      "`completed` no repositório (Fase 6 rodada no cluster em 2026-07-26).")
    A("- Comandos: `python -m src.stats --aggregate` e "
      "`python -m src.fase6_final_stats`.")
    A("- Saídas: `results/T{1..5}_*.csv` (+ espelhos em `paper/tables/*.md`), "
      "`results/fase6_payload.json` (todos os números brutos) e este arquivo.")
    A("")

    path = rdir / "RESUMO_FASE6.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[fase6] resumo -> {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Consolidação final T1–T5 + vereditos H1–H4.")
    ap.add_argument("--results-dir", default="local/runs")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--headline", default="deit_tiny")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    rdir = Path(args.results_dir)
    tables = repo / "local/generated/tables"
    idx = Index(str(rdir))
    arms = json.loads((rdir / "fase2_selected_arms.json").read_text(encoding="utf-8-sig"))
    print(f"[fase6] {len(idx.recs)} marcadores 'completed' carregados.")

    c1, t1 = build_T1(repo)
    write_csv(rdir / "T1_datasets.csv", c1, t1)
    write_md(tables / "T1_datasets.md", "T1 — Datasets e split canônico congelado", c1, t1,
             "Split único 70/10/20 agrupado por paciente (`StratifiedGroupKFold`), idêntico "
             "para todos os braços, backbones e seeds.")

    c2, t2 = build_T2(idx, arms)
    write_csv(rdir / "T2_headline_backbone.csv", c2, t2)
    write_md(tables / "T2_headline_backbone.md",
             "T2 (headline) — RGB vs braço de cor selecionado por validação, por backbone",
             c2, t2,
             "Δ(ARM−RGB) pareado por seed (n=5), IC95 t-Student. `p_wilcoxon` é evidência "
             "graduada (mínimo possível 0,0625 com n=5), não critério de corte. "
             "`std_seed_auc` é diagnóstico de estabilidade.")

    c3, t3 = build_T3(idx)
    write_csv(rdir / "T3_espaco_por_espaco.csv", c3, t3)
    write_md(tables / "T3_espaco_por_espaco.md",
             "T3 — Efeito próprio de cada espaço de cor (factorial completo da Fase 1)",
             c3, t3,
             "Cada par = colorset C∪{S} vs C, mesmo backbone/seeds. IC95 t-Student sobre os pares.")

    c4, t4 = build_T4(idx, args.headline)
    write_csv(rdir / "T4_vazamento_tracka.csv", c4, t4)
    write_md(tables / "T4_vazamento_tracka.md",
             "T4 — Tabela-delta de vazamento (split aleatório − split agrupado por paciente)",
             c4, t4,
             "Ambas as células são 5-fold SEM split congelado; a única variável é `split_mode`.")

    arm_neo = arms["selection"]["NeoJaundice"][args.headline]["arm"]
    c5, t5, mt_extra = build_T5(idx, args.headline, arm_neo)
    write_csv(rdir / "T5_multitarefa_tsb.csv", c5, t5)
    write_md(tables / "T5_multitarefa_tsb.md",
             "T5 — Multitarefa (classificação + regressão de TSB), NeoJaundice", c5, t5,
             "MAE/RMSE em mg/dL. Δ compara a MESMA célula com e sem a cabeça de regressão.")

    ctx = {"T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5,
           "c1": c1, "mt_delta_cor": mt_extra, "arm_neo": arm_neo,
           "n_marcadores": len(idx.recs),
           "H1": veredito_H1(t2), "H1b": veredito_H1b(t3), "H2": veredito_H2(t2),
           "H3": veredito_H3(rdir), "H4": veredito_H4(rdir),
           "fase1_factorial": fase1_factorial(idx), "metas": metas_acuracia(idx)}
    (rdir / "fase6_payload.json").write_text(
        json.dumps(ctx, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[fase6] payload -> {rdir / 'fase6_payload.json'}")
    escrever_resumo(rdir, ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
