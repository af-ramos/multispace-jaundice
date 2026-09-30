"""Exporta a partição congelada como artefacto verificável, sem precisar das imagens.

O `make_splits.py` gera a partição a partir dos ficheiros de imagem, e por isso só corre
onde os datasets existem. Este módulo faz o contrário: **reconstrói** a partição a partir
dos artefactos que o próprio repositório já versiona, que é melhor proveniência — o que
sai daqui é a partição que as 2.250 corridas realmente viram, não uma que um script
promete voltar a gerar.

Fontes, ambas versionadas:

* `evidence/v7/preds/*/f1_*.npz` — cada dump traz `paths`, `patient_ids`, `splits` e
  `y_true` para as partições de validação e teste, iguais em todas as 88 células e nas
  cinco sementes. É de lá que sai a atribuição val/test e o rótulo.
* `splits/njn_phash_d5.json` — o mapa de pseudo-paciente
  do NJN, 760 imagens em 755 grupos sob a regra pHash <= 5.

O que cada dataset permite, e a razão:

* **NJN: partição completa.** O mapa pHash enumera as 760 imagens, os dumps dizem quais
  as 228 de validação e teste, e o treino é o resto. Sai um CSV de 760 linhas.
* **NeoJaundice: validação e teste completos, 672 linhas.** Os identificadores de
  paciente são esparsos (0012 a 1351), portanto a lista dos 745 não se deriva sem o
  dataset. Isso **não** limita a verificação: o treino é, por definição, o complemento,
  logo publicar a pertença de validação e teste determina a partição inteira. Quem tiver
  o dataset público confirma o agrupamento com um `groupby` de duas linhas.

Uso:  python src/export_splits.py
Saída: splits/njn_split.csv, splits/neojaundice_split.csv, splits/split_meta.json
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "splits")
PHASH = os.path.join(ROOT, "splits", "njn_phash_d5.json")

# Tabela 2 do manuscrito. A exportação falha se não bater, que é o ponto do artefacto.
EXPECTED = {
    "NJN": {"train": (532, 529), "val": (76, 75), "test": (152, 151), "total": (760, 755)},
    "NeoJaundice": {"train": (1563, 521), "val": (225, 75), "test": (447, 149),
                    "total": (2235, 745)},
}


def key_of(path: str) -> str:
    """Chave estavel: os dois ultimos componentes do caminho.

    Os dumps nao concordam no prefixo --- uns guardam o caminho absoluto da maquina que
    treinou, outros o relativo --- mas concordam em `classe/ficheiro.jpg` (NJN) e
    `images/ficheiro.jpg` (NeoJaundice), que ja identifica a imagem sem ambiguidade.
    """
    parts = str(path).replace("\\", "/").split("/")
    return "/".join(parts[-2:])


def read_dump(dataset: str) -> dict[str, tuple[str, str, int]]:
    """`caminho -> (split, paciente, rótulo)`, lido de um dump canónico qualquer.

    Todas as células partilham a mesma partição congelada, por isso um dump chega; a
    concordância entre dumps é verificada em `check_consistency`.
    """
    files = sorted(glob.glob(os.path.join(ROOT, "evidence/v7/preds", dataset, "f1_*.npz")))
    if not files:
        raise FileNotFoundError(f"sem dumps canónicos para {dataset}")
    d = np.load(files[0], allow_pickle=True)
    out: dict[str, tuple[str, str, int]] = {}
    for p, s, pid, y in zip(d["paths"], d["splits"], d["patient_ids"], d["y_true"]):
        out[key_of(p)] = (str(s), str(pid), int(y))
    return out


def check_consistency(dataset: str, ref: dict, n: int = 8) -> int:
    """Confirma que outros dumps atribuem exactamente as mesmas imagens às mesmas partições."""
    files = sorted(glob.glob(os.path.join(ROOT, "evidence/v7/preds", dataset, "f1_*.npz")))[:n]
    for f in files:
        d = np.load(f, allow_pickle=True)
        for p, s, pid in zip(d["paths"], d["splits"], d["patient_ids"]):
            got = ref.get(key_of(p))
            if got is None or got[0] != str(s) or got[1] != str(pid):
                raise AssertionError(f"{os.path.basename(f)} discorda em {p}")
    return len(files)


def njn_rows() -> list[dict]:
    with open(PHASH, encoding="utf-8") as fh:
        pmap = json.load(fh)["map"]
    dump = read_dump("NJN")
    rows = []
    for path, group in sorted(pmap.items()):
        split, patient, label = dump.get(key_of(path), ("train", group, -1))
        if label == -1:  # treino: o rótulo lê-se do directório, como na campanha
            label = 1 if "/jaundice/" in path else 0
        rows.append({"image": os.path.basename(path), "pseudo_patient": group,
                     "label": label, "split": split})
    return rows


def neojaundice_rows() -> list[dict]:
    dump = read_dump("NeoJaundice")
    return [{"image": os.path.basename(p), "patient_id": v[1], "label": v[2], "split": v[0]}
            for p, v in sorted(dump.items())]


def summarise(rows: list[dict], patient_key: str) -> dict:
    per = defaultdict(set)
    counts = Counter()
    pos = Counter()
    for r in rows:
        counts[r["split"]] += 1
        per[r["split"]].add(r[patient_key])
        pos[r["split"]] += int(r["label"] == 1)
    return {s: {"images": counts[s], "patients": len(per[s]),
                "prevalence_pct": round(100 * pos[s] / counts[s], 2)}
            for s in sorted(counts)}


def disjoint(rows: list[dict], patient_key: str) -> dict:
    per = defaultdict(set)
    for r in rows:
        per[r["split"]].add(r[patient_key])
    out = {}
    keys = sorted(per)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            out[f"{a}|{b}"] = len(per[a] & per[b])
    return out


def write(name: str, rows: list[dict]) -> str:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main() -> None:
    meta = {"split_seed": 42, "grouping": {"NJN": "pseudo-patient, pHash <= 5",
                                           "NeoJaundice": "patient_id from the dataset"},
            "datasets": {}}
    for dataset, rows, key, note in (
        ("NJN", njn_rows(), "pseudo_patient", "complete: all 760 images"),
        ("NeoJaundice", neojaundice_rows(), "patient_id",
         "validation and test only; train is the complement over the 745 public patients"),
    ):
        n = check_consistency(dataset, read_dump(dataset))
        summary, cross = summarise(rows, key), disjoint(rows, key)
        assert not any(cross.values()), f"{dataset}: partições partilham pacientes: {cross}"
        for split, (img, pat) in EXPECTED[dataset].items():
            if split == "total" or split not in summary:
                continue
            got = summary[split]
            assert (got["images"], got["patients"]) == (img, pat), \
                f"{dataset}/{split}: {got} != Tabela 2 ({img} imagens / {pat} pacientes)"
        digest = write(f"{dataset.lower()}_split.csv", rows)
        meta["datasets"][dataset] = {"rows": len(rows), "coverage": note,
                                     "sha256": digest, "splits": summary,
                                     "patient_overlap": cross,
                                     "dumps_cross_checked": n}
        print(f"{dataset}: {len(rows)} linhas | {note}")
        for s, v in summary.items():
            print(f"   {s:6s} {v['images']:5d} imagens  {v['patients']:4d} pacientes  "
                  f"prev {v['prevalence_pct']:5.2f}%")
        print(f"   sobreposição de pacientes entre partições: {cross}")
    with open(os.path.join(OUT, "split_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"\nescrito em {OUT}/")


if __name__ == "__main__":
    main()
