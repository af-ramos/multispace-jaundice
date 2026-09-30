"""
stats.py (NOVO — v6)
====================

Agregação e estatística sobre os marcadores JSON gravados por :mod:`train` (um por
config, já com média ± IC95%, std entre seeds, AUCs por seed, regressão).

Subcomandos (ver plano §7 e Fase 2f):

* ``--aggregate``            → CSV consolidado (separador ``;`` e decimal ``,``, Excel pt-BR).
* ``--gate``                 → lista configs que passam no GATE (std_seed_AUC < 0.01 +
                               metas honestas) — decisão de avançar da Fase 1.
* ``--select-top-colorspaces N`` → top-N subconjuntos de cor por métrica (média entre
                               backbones), grava ``top_colorspaces.txt``.
* ``--paired wilcoxon --compare A B`` → teste de Wilcoxon pareado (por seed) entre duas
                               tags/fusões (ex.: chromafilm vs adapter_v2; channel vs spatial).

**Unidade da variância do GATE:** o ``std_seed_auc`` já vem calculado sobre as AUCs a
**nível de seed** (1 AUC por seed = test agrupado dos folds), como fixado no cv.py — NÃO
por-fit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

GATE_STD = 0.01
CSV_SEP = ";"


def _num(x) -> str:
    """Formata número no padrão pt-BR (decimal ','); vazio p/ None/NaN."""
    if x is None:
        return ""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f != f:  # NaN
        return ""
    return f"{f:.6g}".replace(".", ",")


def load_markers(results_dir: str, tag_prefix: Optional[str] = None) -> List[dict]:
    """Carrega todos os marcadores 'completed' sob ``results_dir/<dataset>/<backbone>/*.json``."""
    recs: List[dict] = []
    for p in Path(results_dir).rglob("*.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("status") != "completed":
            continue
        if tag_prefix and not str(d.get("tag", "")).startswith(tag_prefix):
            continue
        d["_path"] = str(p)
        recs.append(d)
    return recs


_COLS = ["dataset", "backbone", "colorset_id", "fusion", "backbone_mode", "film_mode",
         "multitask", "wb", "njn_mode", "split_mode", "folds", "tag",
         "auc_mean", "auc_ci95", "std_seed_auc", "gate_stable",
         "acc_mean", "f1_macro_mean", "threshold", "tsb_mae", "tsb_r2", "elapsed_min"]


def _row(d: dict) -> Dict[str, str]:
    m = d.get("test_metrics_mean", {}) or {}
    ci = d.get("test_metrics_ci95", {}) or {}
    reg = d.get("regression") or {}
    return {
        "dataset": d.get("dataset", ""), "backbone": d.get("backbone", ""),
        "colorset_id": d.get("colorset_id", ""), "fusion": d.get("fusion", ""),
        "backbone_mode": d.get("backbone_mode", ""), "film_mode": d.get("film_mode", ""),
        "multitask": d.get("multitask", ""), "wb": d.get("wb", ""),
        "njn_mode": d.get("njn_mode", "") or "", "split_mode": d.get("split_mode", ""),
        "folds": d.get("folds", ""), "tag": d.get("tag", ""),
        "auc_mean": _num(m.get("roc_auc")), "auc_ci95": _num(ci.get("roc_auc")),
        "std_seed_auc": _num(d.get("std_seed_auc")), "gate_stable": d.get("gate_stable", ""),
        "acc_mean": _num(m.get("accuracy")), "f1_macro_mean": _num(m.get("f1_macro")),
        "threshold": _num(d.get("threshold")), "tsb_mae": _num(reg.get("tsb_mae")),
        "tsb_r2": _num(reg.get("tsb_r2")), "elapsed_min": _num(d.get("elapsed_min")),
    }


def aggregate(results_dir: str, out: str) -> None:
    recs = load_markers(results_dir)
    rows = [_row(d) for d in recs]
    rows.sort(key=lambda r: (r["dataset"], r["backbone"], r["colorset_id"], r["fusion"]))
    lines = [CSV_SEP.join(_COLS)]
    for r in rows:
        lines.append(CSV_SEP.join(str(r.get(c, "")) for c in _COLS))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"[stats] {len(rows)} configs -> {out}")


def gate(results_dir: str) -> None:
    recs = load_markers(results_dir)
    print(f"{'CONFIG':<55} {'AUC':>7} {'std_seed':>9} {'GATE'}")
    n_ok = 0
    for d in sorted(recs, key=lambda x: x.get("run_id", "")):
        auc = (d.get("test_metrics_mean") or {}).get("roc_auc", float("nan"))
        std = d.get("std_seed_auc", float("nan"))
        ok = bool(d.get("gate_stable"))
        n_ok += int(ok)
        print(f"{d.get('run_id', '')[:55]:<55} {auc:>7.4f} {std:>9.4f} "
              f"{'OK' if ok else '—'}")
    print(f"\n[gate] {n_ok}/{len(recs)} configs estáveis (std_seed_AUC < {GATE_STD}).")


def select_top_colorspaces(results_dir: str, n: int, metric: str, out: str) -> None:
    recs = load_markers(results_dir)
    by_cs: Dict[str, List[float]] = {}
    for d in recs:
        cs = d.get("colorset_id", "")
        val = (d.get("test_metrics_mean") or {}).get(metric)
        if cs and val is not None and val == val:
            by_cs.setdefault(cs, []).append(float(val))
    ranked = sorted(by_cs.items(), key=lambda kv: np.mean(kv[1]), reverse=True)
    top = [cs for cs, _ in ranked[:n]]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(" ".join(top))
    print(f"[stats] top-{n} colorspaces por {metric}: {top} -> {out}")
    for cs, vals in ranked:
        print(f"  {cs:<22} {metric}={np.mean(vals):.4f}  (n={len(vals)})")


def wilcoxon_paired(a, b) -> Tuple[float, float, int]:
    """Wilcoxon pareado sobre dois vetores alinhados (mesmos folds/seeds).

    Devolve ``(estatística, p, sinal)`` com ``sinal = sign(média(a) - média(b))``.
    Convenção do contrato (``docs/TESTING.md``): entradas idênticas (todas as
    diferenças zero) → ``p = 1.0`` (sem evidência de diferença), em vez do
    ``ValueError`` que o scipy levanta quando ``a - b`` é todo zero. Pré-condição:
    ``len(a) == len(b)`` e os pares vêm dos MESMOS folds/seeds."""
    from scipy.stats import wilcoxon
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"pares desalinhados: {a.shape} != {b.shape} "
                         "(Wilcoxon exige mesmos folds/seeds).")
    diff = a - b
    sign = int(np.sign(a.mean() - b.mean()))
    if np.allclose(diff, 0.0):
        return 0.0, 1.0, 0
    stat, p = wilcoxon(a, b)
    return float(stat), float(p), sign


def paired_wilcoxon(results_dir: str, tag_a: str, tag_b: str, metric: str = "roc_auc") -> None:
    """Wilcoxon pareado por SEED entre duas tags (mesmas seeds/folds)."""
    from scipy.stats import wilcoxon
    recs = load_markers(results_dir)

    def seed_vals(tag):
        out = {}
        for d in recs:
            if d.get("tag") == tag or d.get("run_id", "").startswith(tag):
                for s, m in (d.get("per_seed_metrics") or {}).items():
                    out[str(s)] = m.get(metric)
        return out

    a, b = seed_vals(tag_a), seed_vals(tag_b)
    common = sorted(set(a) & set(b))
    xa = [a[s] for s in common if a[s] is not None and b[s] is not None]
    xb = [b[s] for s in common if a[s] is not None and b[s] is not None]
    if len(xa) < 2:
        print(f"[wilcoxon] seeds pareadas insuficientes ({len(xa)}) entre '{tag_a}' e '{tag_b}'.")
        return
    stat, p = wilcoxon(xa, xb)
    da, db = float(np.mean(xa)), float(np.mean(xb))
    print(f"[wilcoxon] {metric}: {tag_a}={da:.4f} vs {tag_b}={db:.4f} | "
          f"W={stat:.3f} p={p:.4f} | n_seeds={len(xa)} | "
          f"{'SIGNIF.' if p < 0.05 else 'n.s.'} (Δ={da-db:+.4f})")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description="v6 — agregação e estatística dos marcadores.")
    ap.add_argument("--results-dir", default="local/runs")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--select-top-colorspaces", type=int, default=0)
    ap.add_argument("--metric", default="f1_macro")
    ap.add_argument("--paired", choices=["wilcoxon"], default=None)
    ap.add_argument("--compare", nargs=2, default=None, metavar=("A", "B"))
    ap.add_argument("--out", default="local/runs/aggregate.csv")
    args = ap.parse_args(argv)

    did = False
    if args.aggregate:
        aggregate(args.results_dir, args.out); did = True
    if args.gate:
        gate(args.results_dir); did = True
    if args.select_top_colorspaces:
        select_top_colorspaces(args.results_dir, args.select_top_colorspaces,
                               args.metric, "local/runs/top_colorspaces.txt"); did = True
    if args.paired == "wilcoxon" and args.compare:
        paired_wilcoxon(args.results_dir, args.compare[0], args.compare[1],
                        metric="roc_auc"); did = True
    if not did:
        gate(args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
