"""
select_top_colorspaces.py (NOVO)
================================

Seleção dos espaços de cor para a **ablação completa** (Tarefa 6) e relatórios de
análise pós-campanha.

Seleção **por arquitetura, não por média global cega**: ViTs (DeiT) e CNNs
(Inception) reagem de forma muito diferente às cores — a média global esconderia o
melhor espaço de cada arquitetura. Por isso :func:`select_top_colorspaces` devolve a
**união dos Top-3 da melhor CNN com os Top-3 do melhor ViT** (ranqueados por
Macro-F1 nível-paciente *dentro* de cada arquitetura).

Também gera:

* :func:`write_vs_v1` — ``report_final_vs_v1.md`` (delta Macro-F1/AUC paciente vs v1);
* :func:`write_ablation_matrix` — ``report_ablation.md`` (matriz backbone × colorspace).

Uso (run.sh): ``python -m src.select_top_colorspaces topk --results-dir ... --dataset ...``
imprime os colorsets selecionados (um por linha).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

if __package__:
    from .config import BACKBONES, DISPLAY_NAMES, colorset_id, parse_colorset
else:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from src.config import BACKBONES, DISPLAY_NAMES, colorset_id, parse_colorset

# Referência da v1 (números honestos do projeto) p/ a análise pós-campanha.
# Macro-F1 / ROC-AUC nível-paciente das configurações campeãs da v1.
V1_REFERENCE: Dict[str, Dict[str, float]] = {
    "NeoJaundice": {"f1_macro_patient": 0.762, "roc_auc_patient": 0.836,
                    "note": "v1 Inception·LAB ~76,4% / AUC ViT-B/32 0,836"},
    "NJN": {"f1_macro_patient": 0.974, "roc_auc_patient": 0.96,
            "note": "v1 DeiT-Tiny RGB+YCrCb+HSV 97,39%"},
}

_DISPLAY_TO_KEY = {v: k for k, v in DISPLAY_NAMES.items()}
_METRIC = "F1_macro_patient"  # métrica primária (nível-paciente)


def load_results(results_dir: str, dataset: str) -> pd.DataFrame:
    csv = Path(results_dir) / f"results_{dataset}.csv"
    if not csv.is_file():
        raise FileNotFoundError(f"CSV consolidado nao encontrado: {csv} "
                                f"(rode aggregate.py primeiro).")
    return pd.read_csv(csv, sep=";", decimal=",")


def filter_results(df: pd.DataFrame, num_classes: int = 2,
                   njn_mode: str | None = None) -> pd.DataFrame:
    """Restringe a UMA combinacao (num_classes, njn_mode) — nunca misturar 2/3
    classes ou skin_roi/full_image ao ranquear os espacos de cor."""
    out = df[df["Num_Classes"] == num_classes] if "Num_Classes" in df.columns else df
    if njn_mode is not None and "NJN_Mode" in out.columns:
        out = out[out["NJN_Mode"] == njn_mode]
    return out


def _family_of(model_display: str) -> str:
    key = _DISPLAY_TO_KEY.get(model_display, model_display)
    spec = BACKBONES.get(key)
    return spec.family if spec else "cnn"


def per_config_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Média (sobre seeds) do Macro-F1/AUC nível-paciente por (Model, ColorSpace)."""
    g = (df.groupby(["Model", "ColorSpace"], dropna=False)
           .agg(F1_macro_patient=(_METRIC, "mean"),
                ROC_AUC_patient=("ROC_AUC_patient", "mean"))
           .reset_index())
    g["family"] = g["Model"].map(_family_of)
    return g


def select_top_colorspaces(df: pd.DataFrame, per_arch_top: int = 3) -> List[str]:
    """União dos Top-K colorsets da melhor CNN com os da melhor ViT (por Macro-F1
    paciente). Preserva as sinergias arquiteturais (sem média global)."""
    g = per_config_patient(df)
    selected: List[str] = []
    for fam in ("cnn", "vit"):
        sub = g[g["family"] == fam]
        if sub.empty:
            continue
        # melhor modelo da família = o de maior pico de Macro-F1 paciente.
        best_model = sub.loc[sub["F1_macro_patient"].idxmax(), "Model"]
        top = (sub[sub["Model"] == best_model]
               .sort_values("F1_macro_patient", ascending=False)
               .head(per_arch_top))
        for cs in top["ColorSpace"]:
            cid = colorset_id(parse_colorset(str(cs)))
            if cid not in selected:
                selected.append(cid)
    return selected


def write_ablation_matrix(df: pd.DataFrame, out: str | Path) -> Path:
    """Matriz backbone × colorspace (Média±DP Macro-F1 paciente) em markdown."""
    out = Path(out)
    piv_mean = df.pivot_table(index="Model", columns="ColorSpace", values=_METRIC, aggfunc="mean")
    piv_std = df.pivot_table(index="Model", columns="ColorSpace", values=_METRIC, aggfunc="std")
    lines = ["# Ablação completa — Macro-F1 nível-paciente (Média ± DP)", ""]
    cols = list(piv_mean.columns)
    lines.append("| Backbone | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for model in piv_mean.index:
        cells = []
        for c in cols:
            m = piv_mean.loc[model, c]
            s = piv_std.loc[model, c] if c in piv_std.columns else float("nan")
            cells.append(f"{m:.3f} ± {0.0 if pd.isna(s) else s:.3f}" if pd.notna(m) else "—")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[select] matriz de ablacao -> {out}")
    return out


def write_vs_v1(df: pd.DataFrame, dataset: str, out: str | Path) -> Path:
    """Compara a melhor config do `final` contra a referência da v1 (delta paciente)."""
    out = Path(out)
    g = per_config_patient(df)
    ref = V1_REFERENCE.get(dataset, {})
    best = g.loc[g["F1_macro_patient"].idxmax()] if not g.empty else None
    lines = [f"# {dataset}: final vs v1 (nível-paciente)", ""]
    if ref:
        lines.append(f"**Referência v1:** {ref.get('note','')} "
                     f"(Macro-F1 {ref.get('f1_macro_patient', float('nan')):.3f}, "
                     f"AUC {ref.get('roc_auc_patient', float('nan')):.3f})")
        lines.append("")
    if best is not None:
        f1, auc = best["F1_macro_patient"], best["ROC_AUC_patient"]
        lines += [
            f"**Melhor final:** {best['Model']} · {best['ColorSpace']} — "
            f"Macro-F1 {f1:.3f}, AUC {auc:.3f}", "",
        ]
        if ref:
            d_f1 = f1 - ref.get("f1_macro_patient", float("nan"))
            d_auc = auc - ref.get("roc_auc_patient", float("nan"))
            verdict = ("SUPERA" if d_f1 > 0.005 else
                       "EMPATA (~v1)" if abs(d_f1) <= 0.005 else "ABAIXO de")
            lines += [f"**Delta vs v1:** ΔMacro-F1 = {d_f1:+.3f}, ΔAUC = {d_auc:+.3f} "
                      f"→ {verdict} a v1.", ""]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[select] relatorio vs-v1 -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("topk", "report"):
        p = sub.add_parser(name)
        p.add_argument("--results-dir", default="local/runs")
        p.add_argument("--dataset", required=True, choices=["NJN", "NeoJaundice"])
        p.add_argument("--per-arch-top", type=int, default=3)
        p.add_argument("--num-classes", type=int, default=2, choices=[2, 3])
        p.add_argument("--njn-mode", default=None, choices=[None, "skin_roi", "full_image"])
    args = ap.parse_args()

    df = load_results(args.results_dir, args.dataset)
    # NUNCA misturar 2/3 classes ou skin_roi/full_image ao ranquear os espacos.
    njn_mode = args.njn_mode if args.dataset == "NJN" else None
    df = filter_results(df, num_classes=args.num_classes, njn_mode=njn_mode)
    if df.empty:
        raise SystemExit(f"[select] sem linhas para num_classes={args.num_classes}"
                         f"{'' if njn_mode is None else f', njn_mode={njn_mode}'} em {args.dataset}.")
    if args.cmd == "topk":
        for cs in select_top_colorspaces(df, per_arch_top=args.per_arch_top):
            print(cs)
    else:  # report
        base = Path(args.results_dir) / args.dataset
        write_vs_v1(df, args.dataset, base / "report_final_vs_v1.md")
        write_ablation_matrix(df, base / "report_ablation.md")


if __name__ == "__main__":
    main()
