#!/usr/bin/env python3
"""Que células do factorial têm dump de predições, e quais faltam.

Um dump é o que permite re-pontuar sem treinar: agregação por paciente, limiar
alternativo, ensembles, controlo de custo igual. A campanha v7 produziu 550
marcadores mas guardou dumps só de 160 células — o RGB e o braço seleccionado de
cada par (dataset, backbone). Tudo o resto existe como métrica agregada e não como
predição, o que fecha a porta a qualquer análise de re-scoring nessas células.

Este relatório diz exactamente o que falta, e prioriza: as células que o artigo
cita nas tabelas T2–T4 são as que precisam de dump, porque são as que um revisor
pode querer ver re-pontuadas.

Saída: `results/coverage.json` e a lista priorizada que
`scripts/run_missing_dumps.sh` consome.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.dumps import canonical_dumps, misplaced  # noqa: E402

EVIDENCE = ROOT / "evidence/v7"
PREDS = EVIDENCE / "preds"
OUT = ROOT / "results"


def read(name: str, base: Path = EVIDENCE) -> list[dict[str, str]]:
    with (base / name).open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=";"))


def num(value: str | None) -> float | None:
    text = (value or "").strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def have_dumps() -> set[tuple[str, str, str]]:
    """(dataset, backbone, colorset) com dump canónico em disco.

    Delegado a `analysis.dumps`, que resolve duas armadilhas que este relatório já
    teve: o dataset lido da pasta em vez do conteúdo (a cópia de `v7_extra` chegou a
    pôr o mesmo dump nas duas pastas, e cada célula contava a dobrar), e os dumps de
    outras arquitecturas de fusão — `stemcnn`, `ccat`, `notta`, `mt` — a passarem por
    cobertura da célula `adapter_v2` que o factorial A1 regista.
    """
    return set(canonical_dumps())


def main() -> int:
    factorial = read("A1_factorial_accf1.csv")
    dumps = have_dumps()

    # o universo: cada célula de cor, mais o baseline RGB de cada par
    cells = {(r["dataset"], r["backbone"], r["colorset"]) for r in factorial}
    pairs = {(d, b) for d, b, _ in cells}
    universe = cells | {(d, b, "RGB") for d, b in pairs}
    missing = sorted(universe - dumps)

    # Prioridade: só o que um revisor pode querer ver re-pontuado.
    #
    # Não é "toda a célula que o artigo cita": a T4 lista 116 células com IC95
    # excluindo zero, e 111 delas são só-croma negativas. Ninguém pede para
    # re-pontuar "LAB sozinho é mau" — isso já está fechado pela métrica agregada.
    # O que precisa de dump é o que sustenta uma afirmação positiva ou uma
    # recomendação de sistema.
    cited: set[tuple[str, str, str]] = set()

    # (a) os sistemas que a T2 recomenda
    if (OUT / "T2_melhores_sistemas.csv").is_file():
        for row in read("T2_melhores_sistemas.csv", OUT):
            cited.add((row["dataset"], row["backbone"], row["colorset"]))

    # (b) as células positivas da T4 — as que sustentam heterogeneidade favorável
    if (OUT / "T4_heterogeneidade.csv").is_file():
        for row in read("T4_heterogeneidade.csv", OUT):
            if row["sinal"] == "pos":
                cited.add((row["dataset"], row["backbone"], row["colorset"]))

    # (c) a melhor configuração de cada backbone, que é o que a T3 conta
    for dataset in {r["dataset"] for r in factorial}:
        for backbone in {r["backbone"] for r in factorial if r["dataset"] == dataset}:
            group = [r for r in factorial
                     if r["dataset"] == dataset and r["backbone"] == backbone
                     and num(r["accuracy_cor"]) is not None]
            if group:
                best = max(group, key=lambda r: num(r["accuracy_cor"]))
                cited.add((dataset, backbone, best["colorset"]))

    priority = sorted(c for c in cited if c not in dumps)

    print(f"universo do factorial          {len(universe):>4} células")
    print(f"com dump                       {len(dumps):>4} ({100*len(dumps)/len(universe):.0f}%)")
    print(f"sem dump                       {len(missing):>4}")
    print(f"sem dump E citadas nas tabelas {len(priority):>4}  <- alvo prioritário\n")

    # A cópia de v7_extra já correu mal uma vez; se voltar a correr, isto grita.
    bad = misplaced()
    if bad:
        print(f"AVISO: {len(bad)} dumps na pasta do dataset errado — corrija antes de usar:")
        for path, folder, real in bad[:5]:
            print(f"  {path.name[:66]}  em {folder}/ mas é {real}")
        print()

    print("alvo prioritário, por dataset:")
    for dataset in ("NJN", "NeoJaundice"):
        sub = [c for c in priority if c[0] == dataset]
        print(f"  {dataset:12} {len(sub):>3} células")
        for c in sub[:8]:
            print(f"      {c[1]:20} {c[2]}")
        if len(sub) > 8:
            print(f"      ... e mais {len(sub)-8}")

    # estimativa de custo: a v7 mediu ~5 min por célula (5 sementes) nos backbones
    # pequenos e ~12 min nos grandes; usamos 8 min como média conservadora
    minutes = len(priority) * 8
    print(f"\nestimativa: {len(priority)} células x ~8 min = ~{minutes/60:.1f} h")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coverage.json").write_text(json.dumps({
        "universo": len(universe), "com_dump": len(dumps), "sem_dump": len(missing),
        "prioritarias": [{"dataset": d, "backbone": b, "colorset": c} for d, b, c in priority],
        "todas_em_falta": [{"dataset": d, "backbone": b, "colorset": c} for d, b, c in missing],
        "estimativa_horas": round(minutes / 60, 1),
        "nota": ("prioritário = citado nas tabelas T2/T4 do artigo e sem dump. "
                 "São as células que um revisor pode querer ver re-pontuadas."),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("escrito: results/coverage.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
