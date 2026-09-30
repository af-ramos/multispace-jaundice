"""
audit.py — v7, Fase 0 (bloqueante, offline, sem GPU/treino)
===========================================================

Auditoria que emite ``results/audit_report.md``. **NÃO** prosseguir para a Fase 1
sem um relatório limpo (leak=0). Cobre o ``smoke_audit`` do ``docs/TESTING.md``:

1. **Vazamento de paciente = 0** nos 5 folds (modo ``group``), NeoJaundice + NJN
   (:func:`splits.make_kfold` já valida; aqui recontamos e reportamos).
2. **Contagens** por dataset (NeoJaundice ~2235 imgs / ~745 pacientes; NJN ~760).
3. **Distribuição de TSB** (limiar binário 12,9 mg/dL + bins de 3 classes) — NeoJaundice.
4. **ITA°** computado numa amostra (prova que o pipeline de tom de pele roda).
5. **Chave de cache** separa wb/roi/njn e é colorspace-agnóstica.
6. **Backbones canônicos** registrados; **limiar clínico** e bins sãos.

Uso:
    python -m src.audit               # NeoJaundice + NJN
    python -m src.audit --no-leak-check   # só checagens estáticas
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np

from .config import (BACKBONES, TSB_3CLASS_BINS, TSB_BINARY_THRESHOLD,
                     ExperimentConfig, parse_colorset)


class _Report:
    def __init__(self):
        self.lines: List[str] = []
        self.ok = True

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        mark = "PASS" if passed else "FAIL"
        self.ok = self.ok and passed
        self.lines.append(f"- [{mark}] **{name}** {('— ' + detail) if detail else ''}")

    def note(self, text: str) -> None:
        self.lines.append(f"  - {text}")


def _cfg(dataset: str, wb: str = "graypatch", njn_mode: str = "skin_roi",
         colors: str = "RGB+LAB+YCrCb+HSV", **kw) -> ExperimentConfig:
    ds = {"neojaundice": "NeoJaundice", "njn": "NJN"}.get(dataset, dataset)
    return ExperimentConfig(backbone="efficientnet_b0", dataset=ds,
                            colorspaces=parse_colorset(colors), wb_method=wb,
                            njn_mode=njn_mode, **kw)


# --------------------------------------------------------------------------- #
# Checagens estáticas (não tocam imagens)
# --------------------------------------------------------------------------- #
def check_backbones(rep: _Report) -> None:
    need = ["efficientnet_b0", "resnet18", "deit_tiny", "convnextv2_tiny",
            "resnet50", "densenet121", "inception_v3", "vit_b_16", "swinv2_tiny"]
    missing = [b for b in need if b not in BACKBONES]
    rep.check("Backbones canônicos (CNN+ViT) registrados", not missing,
              f"faltando: {missing}" if missing else f"{len(need)} presentes")


def check_tsb_thresholds(rep: _Report) -> None:
    rep.check("Limiar binário TSB == 12.9 mg/dL",
              abs(TSB_BINARY_THRESHOLD - 12.9) < 1e-9, f"={TSB_BINARY_THRESHOLD}")
    rep.check("Bins de 3 classes definidos (3 faixas)", len(TSB_3CLASS_BINS) == 3,
              str(TSB_3CLASS_BINS))


def check_cache_keys(rep: _Report) -> None:
    neo_on, neo_off = _cfg("neojaundice", wb="graypatch"), _cfg("neojaundice", wb="off")
    t_on, t_off = neo_on.cache_tag(), neo_off.cache_tag()
    rep.check("Chave de cache separa wb-on de wb-off", t_on != t_off, f"'{t_on}' != '{t_off}'")
    rep.check("stats_path separa wb-on de wb-off", neo_on.stats_path() != neo_off.stats_path())
    colors_in_key = any(c in t_on for c in ("LAB", "YCrCb", "HSV", "RGB+"))
    rep.check("Cache é colorspace-agnóstico (colors fora da chave)", not colors_in_key,
              "o .npy guarda RGB pré-processado; conversão de espaços é on-the-fly")
    njn_roi = _cfg("njn", njn_mode="skin_roi").cache_tag()
    njn_full = _cfg("njn", njn_mode="full_image").cache_tag()
    rep.check("Chave de cache separa NJN skin_roi de full_image", njn_roi != njn_full,
              f"'{njn_roi}' != '{njn_full}'")


# --------------------------------------------------------------------------- #
# Checagens sobre os dados reais (contagens, leak, TSB, ITA°)
# --------------------------------------------------------------------------- #
def check_dataset(rep: _Report, dataset: str, k: int = 5, seed: int = 42,
                  ita_sample: int = 40) -> None:
    from . import splits
    cfg = _cfg(dataset)
    try:
        samples = splits.load_samples(cfg)[0]
    except Exception as e:  # pragma: no cover
        rep.check(f"Carregar amostras ({dataset})", False, f"erro: {e}")
        return

    n_imgs = len(samples)
    n_pat = len({s.patient_id for s in samples})
    rep.check(f"Contagem de imagens ({dataset})", n_imgs > 0,
              f"{n_imgs} imagens, {n_pat} pacientes/pseudo-pacientes")
    labels = Counter(s.label for s in samples)
    rep.note(f"distribuição de classes: {dict(sorted(labels.items()))}")

    # --- leak = 0 nos k folds (modo group, agrupado por paciente) ---
    try:
        _, folds, _ = splits.make_kfold(cfg, seed, k=k, mode="group")
    except Exception as e:  # pragma: no cover
        rep.check(f"K-fold sem leak ({dataset})", False, f"erro no split: {e}")
        return
    leaks = 0
    for tr, va, te in folds:
        pats = lambda idx: {samples[i].patient_id for i in idx}
        tr_p, va_p, te_p = pats(tr), pats(va), pats(te)
        if (tr_p & te_p) or (tr_p & va_p) or (va_p & te_p):
            leaks += 1
    rep.check(f"Vazamento de paciente = 0 nos {k} folds ({dataset})", leaks == 0,
              f"leaks={leaks}, {len(folds)} folds")

    # --- distribuição de TSB (só NeoJaundice, tem TSB contínuo) ---
    tsb = np.asarray([s.tsb for s in samples if s.tsb == s.tsb], dtype=np.float64)  # descarta NaN
    if tsb.size:
        pos = int((tsb >= TSB_BINARY_THRESHOLD).sum())
        rep.check(f"TSB contínuo presente ({dataset})", True,
                  f"n={tsb.size}, faixa [{tsb.min():.1f}, {tsb.max():.1f}] mg/dL, "
                  f"≥{TSB_BINARY_THRESHOLD}: {pos} ({100*pos/tsb.size:.1f}%)")
        bins3 = [int(((tsb >= lo) & (tsb < hi)).sum()) for lo, hi in TSB_3CLASS_BINS]
        rep.note(f"3 classes {TSB_3CLASS_BINS}: {bins3}")

    # --- ITA° computado numa amostra (prova do pipeline de tom de pele) ---
    try:
        from .colorspaces import _lab_star
        from .metrics import ita_degrees
        from PIL import Image
        rng = np.random.default_rng(0)
        idx = rng.choice(n_imgs, size=min(ita_sample, n_imgs), replace=False)
        itas = []
        for i in idx:
            im = Image.open(samples[i].path).convert("RGB").resize((96, 96))
            L, _, b = _lab_star(np.asarray(im, dtype=np.uint8))
            itas.append(ita_degrees(float(np.median(L)), float(np.median(b))))
        itas = np.asarray(itas, dtype=np.float64)
        rep.check(f"ITA° computado ({dataset})", np.isfinite(itas).all(),
                  f"amostra n={itas.size}, ITA° médio {itas.mean():.1f}° "
                  f"[{itas.min():.1f}, {itas.max():.1f}]")
    except Exception as e:  # pragma: no cover
        rep.check(f"ITA° computado ({dataset})", False, f"erro: {e}")


def run_audit(datasets: List[str], out_path: Path, static_only: bool = False) -> bool:
    rep = _Report()
    rep.lines.append("# audit_report.md — HCSF v7, Fase 0\n")
    check_backbones(rep)
    check_tsb_thresholds(rep)
    check_cache_keys(rep)
    for ds in datasets:
        rep.lines.append(f"\n## Dataset: {ds}")
        check_dataset(rep, ds)
    rep.lines.append("")
    rep.lines.append(f"## Resultado: {'LIMPO (leak=0) ✅' if rep.ok else 'PENDÊNCIAS ❌'}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # escrita atômica (tmp -> rename)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text("\n".join(rep.lines))
    tmp.replace(out_path)
    print("\n".join(rep.lines))
    print(f"\n[audit] relatório salvo em {out_path}")
    return rep.ok


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description="v7 — auditoria da Fase 0 (offline).")
    ap.add_argument("--all", action="store_true", help="Audita NeoJaundice e NJN (default).")
    ap.add_argument("--dataset", default=None, choices=["neojaundice", "njn"])
    ap.add_argument("--out", default="local/runs/audit_report.md")
    ap.add_argument("--no-leak-check", action="store_true",
                    help="Só checagens estáticas (não carrega imagens/pHash).")
    args = ap.parse_args(argv)

    if args.no_leak_check:
        datasets: List[str] = []
    elif args.dataset and not args.all:
        datasets = [args.dataset]
    else:
        datasets = ["neojaundice", "njn"]
    ok = run_audit(datasets, Path(args.out), static_only=args.no_leak_check)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
