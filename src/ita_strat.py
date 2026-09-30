"""
ita_strat.py — H4: estratificação do efeito da cor por tom de pele (ITA°).

H4 pergunta se o ganho da cromaticidade depende do TOM DE PELE — se a fusão de espaços de
cor ajuda mais (ou menos) em peles mais escuras. É a análise de equidade do estudo.

MÉTODO
  1. ITA° por imagem = ``atan2(L* − 50, b*) · 180/π`` (metrics.ita_degrees), com L* e b*
     medianos **sobre os pixels de PELE** (skin_roi.skin_mask_ycrcb_hsv). Medir sobre a
     imagem inteira — como o audit.py faz na sua amostra de sanidade — contamina o ITA°
     com o fundo; para estratificar tom de pele isso não serve. Sem pele detectada, a
     imagem é DESCARTADA da estratificação (e contada), nunca empurrada para uma faixa.
  2. Faixas de Fitzpatrick por ITA° (Chardon et al.): very light >55, light 41–55,
     intermediate 28–41, tan 10–28, brown −30–10, dark ≤−30.
  3. Δ(ARM − RGB) por faixa, calculado sobre as MESMAS imagens de test nos dois braços,
     a partir dos dumps de predição por imagem (preds.py).

FONTE DAS PREDIÇÕES. Os 450 marcadores da Fase 1 são ANTERIORES ao preds.py e não têm
dump — não há predição por imagem para estratificar. Usa-se o nível BASAL da Fase 2
(NJN full_image · NeoJaundice wb off), que é a MESMA configuração da Fase 1 (mesmos
hparams, split e seeds) e que já foi verificada como reprodução BIT A BIT dela nas 20
células basais. É a Fase 1, com dump.

Local, sem GPU (lê imagens + .npz já sincronizados). Uso:
    python -m src.ita_strat [--backbone deit_tiny]
Saídas: results/fase5_ita_estratificacao.csv · results/fase5_ita_por_imagem.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .metrics import compute_metrics, ita_degrees
from .preds import PredDump

# Faixas de Fitzpatrick por ITA° (limite inferior, rótulo), da mais clara para a mais escura.
ITA_BANDS = [(55.0, "very light"), (41.0, "light"), (28.0, "intermediate"),
             (10.0, "tan"), (-30.0, "brown"), (-1e9, "dark")]
MIN_N = 25   # abaixo disto a faixa é reportada como INCONCLUSIVA, nunca interpretada


def band_of(ita: float) -> str:
    for lo, lbl in ITA_BANDS:
        if ita > lo:
            return lbl
    return ITA_BANDS[-1][1]


def ita_of_image(path: str, res: int = 192) -> Optional[float]:
    """ITA° mediano sobre os pixels de PELE. None se não houver pele detectável."""
    import cv2
    from .colorspaces import _lab_star
    from .skin_roi import skin_mask_ycrcb_hsv

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    if max(h, w) > res:
        s = res / max(h, w)
        bgr = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    mask = skin_mask_ycrcb_hsv(bgr)
    if mask is None or int(np.count_nonzero(mask)) < 50:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    L, _, b = _lab_star(rgb)
    m = mask.astype(bool)
    return float(ita_degrees(float(np.median(L[m])), float(np.median(b[m]))))


def find_dump(results_dir: str, dataset: str, backbone: str, colorset: str) -> Optional[Path]:
    """Dump do nível BASAL da Fase 2 (== configuração da Fase 1) para (ds, bb, colorset)."""
    pat = f"{results_dir}/{dataset}/{backbone}/preds/f2_*.npz"
    want = "full_image" if dataset == "NJN" else "wb-off"
    for p in glob.glob(pat):
        name = Path(p).name
        if want not in name:
            continue
        # o colorset aparece entre '__' no run_id
        if f"__{colorset}__" in name:
            return Path(p)
    return None


def ci(v: List[float]):
    from scipy import stats as sstats
    n = len(v)
    if n == 0:
        return float("nan"), float("nan")
    m = st.mean(v)
    if n < 2:
        return m, 0.0
    return m, float(sstats.t.ppf(0.975, n - 1)) * st.stdev(v) / math.sqrt(n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="H4 — Δ(ARM−RGB) estratificado por ITA°.")
    ap.add_argument("--backbone", default="deit_tiny", help="backbone headline")
    ap.add_argument("--results-dir", default="local/runs")
    ap.add_argument("--metric", default="accuracy", choices=["accuracy", "roc_auc", "f1_macro"])
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    arms = json.load(open(f"{args.results_dir}/fase2_selected_arms.json"))["selection"]
    per_image_rows, strat_rows = [], []

    for ds in ["NJN", "NeoJaundice"]:
        arm = arms[ds][args.backbone]["arm"]
        d_rgb = find_dump(args.results_dir, ds, args.backbone, "RGB")
        d_arm = find_dump(args.results_dir, ds, args.backbone, arm)
        if not d_rgb or not d_arm:
            print(f"[ita] {ds}: dump ausente (RGB={bool(d_rgb)} ARM={bool(d_arm)}) — pulando.")
            continue
        R, A = PredDump.load(d_rgb).subset(split="test"), PredDump.load(d_arm).subset(split="test")
        paths = sorted(set(R.paths) & set(A.paths))
        print(f"\n== {ds} · {args.backbone} · ARM={arm} ==")
        print(f"  imagens de test em ambos os braços: {len(paths)}")

        ita: Dict[str, float] = {}
        sem_pele = 0
        for p in paths:
            v = ita_of_image(p)
            if v is None:
                sem_pele += 1
            else:
                ita[p] = v
        print(f"  ITA° computado: {len(ita)}  |  sem pele detectável (descartadas): {sem_pele}")
        if not ita:
            continue
        vals = np.array(list(ita.values()))
        print(f"  ITA° distribuição: mediana {np.median(vals):.1f}°  "
              f"[{vals.min():.1f}°, {vals.max():.1f}°]")

        # índices por (seed, imagem) — o Δ por faixa é PAREADO POR SEED, como no resto do
        # estudo, para render um IC95 t sobre as 5 seeds. Mediar as seeds antes (ensemble)
        # daria um Δ pontual sem incerteza, e a incerteza é justamente o que decide H4.
        def by_seed_path(D):
            acc: Dict[int, Dict[str, tuple]] = {}
            for sd, pa, yt, yp in zip(D.seeds, D.paths, D.y_true, D.y_prob):
                acc.setdefault(int(sd), {})[str(pa)] = (int(yt), float(yp))
            return acc

        SR, SA = by_seed_path(R), by_seed_path(A)
        seeds_common = sorted(set(SR) & set(SA))

        def by_path(D):
            acc: Dict[str, List] = {}
            for pa, yt, yp in zip(D.paths, D.y_true, D.y_prob):
                acc.setdefault(str(pa), [int(yt), []])[1].append(float(yp))
            return {k: (v[0], float(np.mean(v[1]))) for k, v in acc.items()}

        pr, pa_ = by_path(R), by_path(A)
        thr_r = json.loads(Path(str(d_rgb).replace("/preds/", "/").replace(".npz", ".json")).read_text()
                           )["threshold"] if Path(str(d_rgb).replace("/preds/", "/").replace(".npz", ".json")).is_file() else 0.5
        thr_a = json.loads(Path(str(d_arm).replace("/preds/", "/").replace(".npz", ".json")).read_text()
                           )["threshold"] if Path(str(d_arm).replace("/preds/", "/").replace(".npz", ".json")).is_file() else 0.5

        for p in ita:
            per_image_rows.append([ds, args.backbone, p, round(ita[p], 2), band_of(ita[p]),
                                   pr[p][0], round(pr[p][1], 6), round(pa_[p][1], 6)])

        print(f"  {'faixa ITA°':<16}{'n':>6}{'RGB':>9}{'ARM':>9}{'Δ':>8}{'IC95 (5 seeds)':>22}"
              f"   veredito")
        for _, lbl in ITA_BANDS:
            sel = [p for p in ita if band_of(ita[p]) == lbl]
            if not sel:
                continue
            y = [pr[p][0] for p in sel]
            m_r = compute_metrics(y, y_prob=[pr[p][1] for p in sel], threshold=thr_r)
            m_a = compute_metrics(y, y_prob=[pa_[p][1] for p in sel], threshold=thr_a)
            vr, va = m_r.get(args.metric, float("nan")), m_a.get(args.metric, float("nan"))
            # Δ pareado por seed DENTRO da faixa -> IC95 t sobre as 5 seeds
            per_seed_d = []
            for sd in seeds_common:
                ys = [SR[sd][p][0] for p in sel if p in SR[sd] and p in SA[sd]]
                if len(ys) != len(sel) or len(set(ys)) < 2:
                    continue
                a_s = compute_metrics(ys, y_prob=[SR[sd][p][1] for p in sel], threshold=thr_r)
                b_s = compute_metrics(ys, y_prob=[SA[sd][p][1] for p in sel], threshold=thr_a)
                per_seed_d.append(b_s.get(args.metric, float("nan"))
                                  - a_s.get(args.metric, float("nan")))
            d, hw = ci(per_seed_d) if per_seed_d else (va - vr, float("nan"))
            uni = len(set(y)) < 2
            if len(sel) < MIN_N:
                ver = f"INCONCLUSIVA (n<{MIN_N})"
            elif uni:
                ver = "INCONCLUSIVA (classe única)"
            elif hw == hw and (d - hw) * (d + hw) > 0:
                ver = "Δ significativo"
            else:
                ver = "Δ ≈ 0 (IC inclui zero)"
            ic = f"[{d-hw:+.2f}, {d+hw:+.2f}]" if hw == hw else "(n/d)"
            print(f"  {lbl:<16}{len(sel):>6}{vr:>9.2f}{va:>9.2f}{d:>+8.2f}{ic:>22}   {ver}")
            strat_rows.append([ds, args.backbone, arm, args.metric, lbl, len(sel),
                               round(float(np.median([ita[p] for p in sel])), 2),
                               round(vr, 4), round(va, 4), round(d, 4),
                               round(d - hw, 4) if hw == hw else "",
                               round(d + hw, 4) if hw == hw else "", ver])

    br = lambda x: str(x).replace(".", ",") if isinstance(x, float) else x
    out1 = Path(args.results_dir) / "fase5_ita_estratificacao.csv"
    with open(out1, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["dataset", "backbone", "arm", "metrica", "faixa_ita", "n_imagens",
                    "ita_mediano", "valor_rgb", "valor_arm", "delta", "ic95_lo", "ic95_hi",
                    "veredito"])
        for r in strat_rows:
            w.writerow([br(x) for x in r])
    out2 = Path(args.results_dir) / "fase5_ita_por_imagem.csv"
    with open(out2, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["dataset", "backbone", "path", "ita_graus", "faixa_ita",
                    "y_true", "prob_rgb", "prob_arm"])
        for r in per_image_rows:
            w.writerow([br(x) for x in r])
    print(f"\n-> {out1}\n-> {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
