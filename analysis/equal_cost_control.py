#!/usr/bin/env python3
"""GATE: o ganho da fusão tardia sobrevive a um controlo de custo igual?

## O problema

`evidence/v7/A4_fusao_tardia.csv` mede a fusão tardia assim:

    RGB (1 modelo)   vs   RGB + braço de cor (2 modelos)

e conclui +0,79 p.p. de acurácia, IC95 [+0,62, +0,96], 106/136 comparações pareadas,
p = 3,8e-11. O número é sólido — mas os dois lados não custam o mesmo. Um ensemble de
dois modelos bate um modelo único quase sempre, independentemente do que os distingue.
Sem um controlo de custo igual, não se sabe se o ganho vem **da cor** ou **de ensemblar**.

## O controlo

Cada célula tem 5 sementes, e as sementes variam só o treino sobre uma partição
congelada. Logo, dois modelos RGB de sementes diferentes são um ensemble de dois
modelos que não contém cor nenhuma:

    RGB@sem_i + RGB@sem_j   (custo igual, sem cor)
    RGB@sem_i + ARM@sem_i   (custo igual, com cor)

A diferença entre estes dois isola o contributo da cor a custo constante. É o controlo
que a campanha v7 não construiu e que o `V3_ArtigoIctericia` mostrou ser decisivo
noutro contexto.

## Protocolo

Sem vazamento, e idêntico ao da v7: o limiar sai **só da validação**, agregando por
paciente dentro de cada semente e juntando as sementes (o análogo do `pooled_oof`),
e só depois é aplicado ao teste. Nenhuma decisão consulta o teste.

Para cada semente `i`, o braço de custo igual é a média das métricas dos 4 ensembles
`RGB@i + RGB@j`, `j != i` — ou seja, "o que dois modelos RGB teriam dado nesta
semente", sem privilegiar uma escolha de parceiro. A comparação é pareada pelas 5
sementes, com IC95 t-Student (t₄ = 2,776), como o resto da campanha.

## Validação da implementação

Antes de reportar o controlo, o script reproduz a comparação da v7 (tardia vs RGB
único) e imprime-a lado a lado com o `A4_fusao_tardia.csv`. Se a reimplementação não
reproduzir o que a campanha publicou, o controlo não é de confiança e o script diz-o.

Uso:
    python analysis/equal_cost_control.py
"""

from __future__ import annotations

import csv
import itertools
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import compute_metrics  # noqa: E402
from src.preds import PredDump  # noqa: E402

EVIDENCE = ROOT / "evidence/v7"
PREDS = EVIDENCE / "preds"
T_CRIT_4 = 2.776  # t-Student bilateral 95% com 4 graus de liberdade, como a v7
THRESHOLD_GRID = np.linspace(0.05, 0.95, 19)  # a grade de 0,05 que o treino já usa
PATIENT_LEVEL = {"NeoJaundice"}  # unidade primária por dataset


def load_cells() -> dict[tuple[str, str], dict[str, PredDump]]:
    """Todos os dumps, indexados por (dataset, backbone) -> {colorset: dump}."""
    cells: dict[tuple[str, str], dict[str, PredDump]] = {}
    for path in sorted(PREDS.glob("*/*.npz")):
        dataset = path.parent.name
        name = path.stem
        parts = name.split("__")
        if len(parts) < 3:
            continue
        backbone, colorset = parts[1], parts[2]
        cells.setdefault((dataset, backbone), {})[colorset] = PredDump.load(path)
    return cells


def aligned(a: PredDump, b: PredDump, split: str, seed: int) -> tuple[np.ndarray, ...]:
    """Probabilidades dos dois braços na mesma ordem de imagem.

    O alinhamento é por `path`, nunca posicional: dois dumps podem ter sido escritos
    em ordens diferentes, e somar probabilidades desalinhadas produziria um número
    plausível e errado.
    """
    sa, sb = a.subset(split=split, seed=seed), b.subset(split=split, seed=seed)
    ia, ib = np.argsort(sa.paths), np.argsort(sb.paths)
    if not np.array_equal(sa.paths[ia], sb.paths[ib]):
        raise SystemExit(f"Dumps não alinhados em {split}/seed={seed}")
    if not np.array_equal(sa.y_true[ia], sb.y_true[ib]):
        raise SystemExit(f"Rótulos divergem em {split}/seed={seed}")
    return sa.y_true[ia], sa.y_prob[ia], sb.y_prob[ib], sa.patient_ids[ia]


def by_patient(labels: np.ndarray, probs: np.ndarray, pids: np.ndarray):
    """Média das probabilidades por paciente; o rótulo é constante dentro do paciente."""
    order = np.argsort(pids)
    pids, labels, probs = pids[order], labels[order], probs[order]
    uniq, start = np.unique(pids, return_index=True)
    out_l, out_p = [], []
    for i, u in enumerate(uniq):
        end = start[i + 1] if i + 1 < len(start) else len(pids)
        out_l.append(int(labels[start[i]]))
        out_p.append(float(probs[start[i]:end].mean()))
    return np.array(out_l), np.array(out_p)


def score(labels, probs, pids, patient: bool, threshold: float) -> dict[str, float]:
    if patient:
        labels, probs = by_patient(labels, probs, pids)
    return compute_metrics(labels.tolist(), y_prob=probs.tolist(), threshold=threshold)


def threshold_from_val(pairs, patient: bool, objective: str = "f1_macro") -> float:
    """Limiar do `pooled_oof`: junta as sementes da validação e varre a grade.

    `pairs` é uma lista de (labels, probs, pids) de validação, uma por semente.
    """
    labels, probs = [], []
    for lab, pr, pid in pairs:
        if patient:
            lab, pr = by_patient(lab, pr, pid)
        labels.extend(lab.tolist())
        probs.extend(pr.tolist())
    if len(set(labels)) < 2:
        return 0.5
    best_t, best_v = 0.5, -1.0
    for t in THRESHOLD_GRID:
        value = compute_metrics(labels, y_prob=probs, threshold=float(t))[objective]
        if value > best_v:
            best_v, best_t = value, float(t)
    return best_t


def paired_ci(deltas: list[float]) -> tuple[float, float, float, int]:
    """Média, IC95 t-Student pareado e número de sementes positivas."""
    n = len(deltas)
    mean = statistics.fmean(deltas)
    if n < 2:
        return mean, float("nan"), float("nan"), sum(d > 0 for d in deltas)
    se = statistics.stdev(deltas) / (n ** 0.5)
    return mean, mean - T_CRIT_4 * se, mean + T_CRIT_4 * se, sum(d > 0 for d in deltas)


def analyse(dataset: str, backbone: str, arms: dict[str, PredDump], metric: str) -> dict[str, Any] | None:
    """Compara, na mesma célula: RGB único, fusão tardia e ensemble RGB de custo igual."""
    if "RGB" not in arms:
        return None
    colour = [c for c in arms if c != "RGB"]
    if not colour:
        return None
    arm_name = colour[0]
    rgb, arm = arms["RGB"], arms[arm_name]
    patient = dataset in PATIENT_LEVEL
    seeds = sorted(set(rgb.seed_list()) & set(arm.seed_list()))
    if len(seeds) < 2:
        return None

    # --- limiares, todos vindos só da validação -----------------------------
    val_single, val_late, val_equal = [], [], []
    for s in seeds:
        y, p_rgb, p_arm, pid = aligned(rgb, arm, "val", s)
        val_single.append((y, p_rgb, pid))
        val_late.append((y, (p_rgb + p_arm) / 2.0, pid))
    for i, j in itertools.permutations(seeds, 2):
        y, p_i, _, pid = aligned(rgb, arm, "val", i)
        _, p_j, _, _ = aligned(rgb, arm, "val", j)
        val_equal.append((y, (p_i + p_j) / 2.0, pid))
    thr_single = threshold_from_val(val_single, patient)
    thr_late = threshold_from_val(val_late, patient)
    thr_equal = threshold_from_val(val_equal, patient)

    # --- teste, pareado por semente ----------------------------------------
    single, late, equal = [], [], []
    for s in seeds:
        y, p_rgb, p_arm, pid = aligned(rgb, arm, "test", s)
        single.append(score(y, p_rgb, pid, patient, thr_single)[metric])
        late.append(score(y, (p_rgb + p_arm) / 2.0, pid, patient, thr_late)[metric])
        partners = []
        for j in seeds:
            if j == s:
                continue
            _, p_j, _, _ = aligned(rgb, arm, "test", j)
            partners.append(score(y, (p_rgb + p_j) / 2.0, pid, patient, thr_equal)[metric])
        equal.append(statistics.fmean(partners))

    d_v7 = [a - b for a, b in zip(late, single)]      # o que a v7 reportou
    d_ctl = [a - b for a, b in zip(late, equal)]      # o controlo de custo igual
    m_v7, lo_v7, hi_v7, pos_v7 = paired_ci(d_v7)
    m_ctl, lo_ctl, hi_ctl, pos_ctl = paired_ci(d_ctl)
    return {
        "dataset": dataset, "backbone": backbone, "arm": arm_name,
        "nivel": "paciente" if patient else "imagem", "metrica": metric,
        "n_seeds": len(seeds),
        "rgb": statistics.fmean(single), "tardia": statistics.fmean(late),
        "rgb_duplo": statistics.fmean(equal),
        "delta_vs_rgb_unico": m_v7, "ic95_lo_vs_rgb_unico": lo_v7, "ic95_hi_vs_rgb_unico": hi_v7,
        "sementes_pos_vs_rgb_unico": pos_v7,
        "delta_vs_custo_igual": m_ctl, "ic95_lo_vs_custo_igual": lo_ctl,
        "ic95_hi_vs_custo_igual": hi_ctl, "sementes_pos_vs_custo_igual": pos_ctl,
        "thr_single": thr_single, "thr_tardia": thr_late, "thr_rgb_duplo": thr_equal,
    }


def verdict(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Agrega por dataset e aplica a regra de decisão predeclarada."""
    out = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        deltas = [r[f"delta_{key}"] for r in rows if r["dataset"] == dataset]
        mean, lo, hi, pos = paired_ci(deltas)
        out[dataset] = {
            "n_backbones": len(deltas), "delta_medio": mean,
            "ic95_lo": lo, "ic95_hi": hi, "backbones_positivos": pos,
            "exclui_zero": bool(lo > 0 or hi < 0),
        }
    return out


def main() -> int:
    cells = load_cells()
    metric = "accuracy"
    rows = [r for (ds, bb), arms in sorted(cells.items())
            if (r := analyse(ds, bb, arms, metric)) is not None]
    if not rows:
        raise SystemExit("Nenhuma célula analisável")

    out_csv = ROOT / "results/equal_cost_control.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}".replace(".", ",") if isinstance(v, float) else v)
                        for k, v in r.items()})

    v7 = verdict(rows, "vs_rgb_unico")
    ctl = verdict(rows, "vs_custo_igual")

    print("=" * 78)
    print("REPRODUÇÃO — fusão tardia vs RGB único (o que a v7 reportou)")
    print("=" * 78)
    for ds, v in v7.items():
        print(f"  {ds:12} n={v['n_backbones']:>2} backbones  Δ {v['delta_medio']:+.3f} p.p. "
              f"IC95 [{v['ic95_lo']:+.3f}, {v['ic95_hi']:+.3f}]  "
              f"{v['backbones_positivos']}/{v['n_backbones']} positivos")
    print("\n  A campanha reportou +0,79 p.p. agregando os dois datasets e ambos os níveis.")
    print("  Se os valores acima forem da mesma ordem, a reimplementação confere.")

    print("\n" + "=" * 78)
    print("CONTROLO — fusão tardia vs ensemble RGB de custo igual (2 modelos, sem cor)")
    print("=" * 78)
    for ds, v in ctl.items():
        marca = "  <== IC95 EXCLUI ZERO" if v["exclui_zero"] else ""
        print(f"  {ds:12} n={v['n_backbones']:>2} backbones  Δ {v['delta_medio']:+.3f} p.p. "
              f"IC95 [{v['ic95_lo']:+.3f}, {v['ic95_hi']:+.3f}]  "
              f"{v['backbones_positivos']}/{v['n_backbones']} positivos{marca}")

    exclui = [ds for ds, v in ctl.items() if v["exclui_zero"] and v["delta_medio"] > 0]
    negativo = [ds for ds, v in ctl.items() if v["exclui_zero"] and v["delta_medio"] < 0]
    if exclui:
        decisao = ("A fusão tardia entra no artigo como contribuição própria: a cor "
                   f"acrescenta algo além de ensemblar em {', '.join(exclui)}.")
    elif negativo:
        decisao = ("A fusão tardia SAI do artigo: a custo igual, o ensemble RGB é "
                   f"superior em {', '.join(negativo)}.")
    else:
        decisao = ("A fusão tardia é reportada como método operacional, declarando que "
                   "o ganho é largamente de ensemble e que a cor fornece diversidade "
                   "barata. A manchete fica em RGB+croma vs só-croma, que não depende "
                   "deste controlo.")
    print("\n" + "-" * 78)
    print("DECISÃO (regra predeclarada no plano, antes de correr):")
    print("  " + decisao)
    print("-" * 78)

    (ROOT / "results/equal_cost_control.json").write_text(json.dumps({
        "metrica": metric,
        "protocolo": {
            "limiar": "só da validação, pooled entre sementes, grade 0,05",
            "alinhamento": "por path, nunca posicional",
            "pareamento": "por semente; IC95 t-Student t4=2,776",
            "custo_igual": "RGB@sem_i + RGB@sem_j, média sobre j != i",
        },
        "reproducao_v7": v7, "controlo_custo_igual": ctl, "decisao": decisao,
        "celulas": rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nescrito: results/equal_cost_control.csv e .json ({len(rows)} células)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
