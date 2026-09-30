#!/usr/bin/env python3
"""Escolhe o grupo de controlo: células que NÃO são máximos do seu backbone.

## A pergunta que este grupo responde

As 28 células remedidas a 2026-08-07 caem 0,60 p.p. em média face ao
`A1_factorial_accf1.csv`, com 22 das 28 negativas. Há duas explicações, e elas exigem
correcções muito diferentes:

* **Regressão à média.** Aquelas 28 foram escolhidas por serem o máximo de cada
  backbone (`coverage_report.py`, critério (c)), portanto tinham ruído favorável e
  perdem-no ao serem remedidas. Nesse caso o A1 está certo em geral e só os máximos
  estão inflados — que é o que `paper_tables.py` já assume ao preferir o dump.
* **Degradação sistemática.** Alguma diferença de ambiente faz qualquer retreino sair
  mais baixo. Nesse caso o A1 inteiro está inflado, e a decisão de manter a **T5** no
  A1 deixa de se sustentar, porque o painel completo herdaria o mesmo desvio.

Um grupo de células não-máximas separa as duas: sob regressão à média o seu Δ médio é
~0; sob degradação sistemática é ~−0,6, como o das máximas.

## Como são escolhidas

Amostra aleatória com semente fixa entre as células que satisfazem tudo isto:

1. estão no factorial A1, na condição canónica do seu dataset;
2. **não** têm dump ainda (senão não há nada a medir);
3. **não** são o máximo do seu backbone (é o que as torna controlo);
4. o braço RGB do mesmo backbone já tem dump, para o delta poder ser pareado por
   semente como na T4.

Estratificado por dataset, para os dois lados terem poder semelhante.

Saída: `results/control_cells.json`, que
`scripts/run_missing_dumps.py --cells` consome.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.dumps import canonical_dumps  # noqa: E402

EVIDENCE = ROOT / "evidence/v7"
OUT = ROOT / "results"
SEED = 42
PER_DATASET = 10


def num(value: str | None) -> float | None:
    text = (value or "").strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def main() -> int:
    with (EVIDENCE / "A1_factorial_accf1.csv").open(encoding="utf-8-sig", newline="") as fh:
        factorial = list(csv.DictReader(fh, delimiter=";"))
    dumps = canonical_dumps()
    rng = random.Random(SEED)

    chosen: list[dict[str, str]] = []
    for dataset in ("NJN", "NeoJaundice"):
        cells = [r for r in factorial if r["dataset"] == dataset
                 and num(r["accuracy_cor"]) is not None]

        # o máximo de cada backbone é exactamente o que já foi remedido: excluí-lo
        maxima = set()
        for backbone in {r["backbone"] for r in cells}:
            group = [r for r in cells if r["backbone"] == backbone]
            best = max(group, key=lambda r: num(r["accuracy_cor"]))
            maxima.add((backbone, best["colorset"]))

        pool = [r for r in cells
                if (r["backbone"], r["colorset"]) not in maxima
                and (dataset, r["backbone"], r["colorset"]) not in dumps
                and (dataset, r["backbone"], "RGB") in dumps]

        rng.shuffle(pool)
        picked = pool[:PER_DATASET]
        print(f"{dataset:12} elegíveis {len(pool):>3}  ->  escolhidas {len(picked)}")
        for r in picked:
            print(f"    {r['backbone']:20} {r['colorset']:22} A1={num(r['accuracy_cor']):.2f}")
            chosen.append({"dataset": dataset, "backbone": r["backbone"],
                           "colorset": r["colorset"]})

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "control_cells.json"
    path.write_text(json.dumps({
        "proposito": ("controlo de células não-máximas: separa regressão à média de "
                      "degradação sistemática no retreino"),
        "semente": SEED,
        "criterios": ["no factorial A1", "sem dump", "não é máximo do backbone",
                      "RGB do backbone tem dump (permite pareamento por semente)"],
        "prioritarias": chosen,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{len(chosen)} células escritas em {path}")
    print(f"estimativa: ~{len(chosen)*8/60:.1f} h a ~8 min por célula (mais, com a GPU partilhada)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
