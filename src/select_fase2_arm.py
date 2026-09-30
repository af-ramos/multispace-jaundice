"""select_fase2_arm.py — seleção leakage-free do braço de cor da Fase 2.

Para cada (dataset, backbone), escolhe o colorset com a MAIOR **AUC de validação**
(`oof_auc`, gravada por train.py/cv.py a partir das predições de val do pooled_oof),
entre os 14 colorsets != RGB. NUNCA olha o test — a métrica de test só é reportada
DEPOIS, para o braço já escolhido (evita winner-curse).

Uso:
    python -m src.select_fase2_arm \
        [--results-dir local/runs] \
        [--tag-prefix f1_] [--out .../results/fase2_selected_arms.json]

Saída: marcador JSON VERSIONADO `results/fase2_selected_arms.json` com, por célula:
    arm, arm_oof_auc, rgb_oof_auc, margin_vs_rgb, runner_up, runner_up_oof_auc.

Pré-condição: os marcadores da Fase 1 precisam ter `oof_auc` (campo novo). Marcadores
antigos não têm — rode a Fase 1 com `--refresh-missing-oof` para relogar a validação.
Este utilitário é LOCAL (sem GPU) e apenas lê JSONs já sincronizados.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

RGB = "RGB"


def load_markers(results_dir: str, tag_prefix: Optional[str]) -> List[dict]:
    recs = []
    for p in Path(results_dir).rglob("*.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("status") != "completed":
            continue
        if tag_prefix and not str(d.get("tag", "")).startswith(tag_prefix):
            continue
        recs.append(d)
    return recs


def select(results_dir: str, tag_prefix: Optional[str], out: str) -> int:
    recs = load_markers(results_dir, tag_prefix)
    # index (dataset, backbone, colorset) -> oof_auc
    by_cell: Dict[tuple, dict] = {}
    for d in recs:
        key = (d.get("dataset"), d.get("backbone"), d.get("colorset_id"))
        by_cell[key] = d

    datasets = sorted({k[0] for k in by_cell})
    backbones = sorted({k[1] for k in by_cell})

    missing: List[str] = []
    selection: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for ds in datasets:
        for bb in backbones:
            cands = []  # (colorset, oof_auc)
            for (d_ds, d_bb, cs), d in by_cell.items():
                if d_ds != ds or d_bb != bb or cs == RGB:
                    continue
                oof = d.get("oof_auc")
                if oof is None:
                    missing.append(f"{ds}/{bb}/{cs}")
                    continue
                if oof == oof:  # not NaN
                    cands.append((cs, float(oof)))
            if not cands:
                continue
            cands.sort(key=lambda t: -t[1])
            arm, arm_auc = cands[0]
            runner = cands[1] if len(cands) > 1 else (None, None)
            rgb_d = by_cell.get((ds, bb, RGB))
            rgb_auc = rgb_d.get("oof_auc") if rgb_d else None
            selection[ds][bb] = {
                "arm": arm,
                "arm_oof_auc": round(arm_auc, 6),
                "rgb_oof_auc": (round(float(rgb_auc), 6)
                                if rgb_auc is not None and rgb_auc == rgb_auc else None),
                "margin_vs_rgb": (round(arm_auc - float(rgb_auc), 6)
                                  if rgb_auc is not None and rgb_auc == rgb_auc else None),
                "runner_up": runner[0],
                "runner_up_oof_auc": (round(runner[1], 6) if runner[1] is not None else None),
                "n_candidatos": len(cands),
            }

    if missing:
        print(f"[select] {len(missing)} células SEM oof_auc — seleção incompleta.")
        print("  Rode a Fase 1 com --refresh-missing-oof para relogar a validação. Ex.:")
        for s in missing[:8]:
            print(f"    faltando: {s}")
        if len(missing) > 8:
            print(f"    … (+{len(missing) - 8})")
        if not selection:
            print("[select] nenhuma célula selecionável ainda — abortando sem gravar.")
            return 1

    payload = {
        "kind": "fase2_selected_arms",
        "criterion": "argmax oof_auc (AUC de validação, pooled_oof) entre colorsets != RGB",
        "pool": "todos os 14 colorsets != RGB",
        "leakage_free": True,
        "note": "Seleção NUNCA usa test. Reportar Δ(arm−RGB) no test só após esta escolha.",
        "n_missing_oof": len(missing),
        "selection": selection,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(out).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    os.replace(tmp, out)
    print(f"[select] gravado {out}")
    for ds in selection:
        print(f"\n== {ds} ==")
        print(f"  {'backbone':<18}{'arm (val-selected)':<22}{'oof_auc':>9}{'vs RGB':>9}{'runner-up':>16}")
        for bb, s in sorted(selection[ds].items()):
            mv = f"{s['margin_vs_rgb']:+.4f}" if s["margin_vs_rgb"] is not None else "n/d"
            print(f"  {bb:<18}{s['arm']:<22}{s['arm_oof_auc']:>9.4f}{mv:>9}"
                  f"{(s['runner_up'] or ''):>16}")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description="Seleção do braço de cor da Fase 2 por AUC de validação.")
    ap.add_argument("--results-dir", default="local/runs")
    ap.add_argument("--tag-prefix", default="f1_",
                    help="Só considera marcadores cuja tag começa com isto (default: f1_).")
    ap.add_argument("--out", default="local/runs/fase2_selected_arms.json")
    args = ap.parse_args(argv)
    return select(args.results_dir, args.tag_prefix, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
