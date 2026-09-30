#!/usr/bin/env python3
"""Fonte canónica do Register B para as análises do factorial.

O Register B é o registo agregado da campanha ``f1_*``: uma célula por
``(dataset, backbone, colour set)``, com um único limiar escolhido nas predições de
validação agregadas das cinco sementes. Esse limiar é aplicado separadamente ao teste
de cada semente; a métrica guardada no campaign record é a média das cinco métricas.

Figure 4, Table 5 (``T11_por_backbone_medio``) e Table 6
(``T5_rgb_indispensavel``) devem consumir exclusivamente este módulo. Prediction
dumps não entram aqui: os dumps disponíveis cobrem apenas parte do factorial, alguns
são retreinos, e o leitor histórico de dumps repontua a threshold 0.5 em vez de usar o
operating point do Register B.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "evidence/v7/aggregate.csv"

DATASETS = ("NJN", "NeoJaundice")
CNN = (
    "convnextv2_tiny", "densenet121", "efficientnet_b0", "efficientnet_b4",
    "inception_v3", "mobilenetv3_large", "resnet18", "resnet50",
)
TRANSFORMERS = (
    "deit_base", "deit_small", "deit_tiny", "dinov3_vits16",
    "vit_b_16", "vit_b_32", "vit_l_16",
)
BACKBONES = CNN + TRANSFORMERS

# A ordem é também a ordem das colunas da Figure 4. Os pares estão alinhados por
# índice, de modo que (RGB+S) - S seja inequívoco.
RGB_PRESERVING = (
    "RGB+HSV",
    "RGB+LAB",
    "RGB+YCrCb",
    "RGB+LAB+HSV",
    "RGB+LAB+YCrCb",
    "RGB+YCrCb+HSV",
    "RGB+LAB+YCrCb+HSV",
)
CHROMA_ONLY = (
    "HSV",
    "LAB",
    "YCrCb",
    "LAB+HSV",
    "LAB+YCrCb",
    "YCrCb+HSV",
    "LAB+YCrCb+HSV",
)
RGB_PAIRINGS = tuple(zip(RGB_PRESERVING, CHROMA_ONLY))
COLOUR_SETS = ("RGB",) + RGB_PRESERVING + CHROMA_ONLY

Key = tuple[str, str, str]


def number(value: str | None) -> float:
    """Lê um campo numérico do CSV pt-BR e falha se estiver ausente."""
    text = (value or "").strip().replace(",", ".")
    if not text:
        raise ValueError("campo numérico vazio no Register B")
    return float(text)


@dataclass(frozen=True)
class Cell:
    dataset: str
    backbone: str
    colorset: str
    accuracy: float
    f1_macro: float
    auc: float
    threshold: float
    tag: str
    std_seed_auc: float = float("nan")


def _is_register_b(row: dict[str, str]) -> bool:
    """Filtro literal da condição canónica da fase 1."""
    common = (
        row["fusion"] == "adapter_v2"
        and row["backbone_mode"] == "lora"
        and row["film_mode"] == "channel"
        and row["multitask"] == "False"
        and row["split_mode"] == "group"
        and row["folds"] == "1"
        and row["tag"].startswith("f1_")
    )
    dataset_condition = (
        row["dataset"] == "NJN"
        and row["wb"] == "on"
        and row["njn_mode"] == "full_image"
    ) or (
        row["dataset"] == "NeoJaundice"
        and row["wb"] == "off"
    )
    return common and dataset_condition


def load_cells(path: Path = CAMPAIGN) -> dict[Key, Cell]:
    """Carrega e valida as 450 células do campaign record final auditado."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle, delimiter=";")
                if _is_register_b(row)]

    cells: dict[Key, Cell] = {}
    for row in rows:
        key = (row["dataset"], row["backbone"], row["colorset_id"])
        if key in cells:
            raise ValueError(f"célula Register B duplicada: {key}")
        cells[key] = Cell(
            dataset=key[0], backbone=key[1], colorset=key[2],
            accuracy=number(row["acc_mean"]),
            f1_macro=number(row["f1_macro_mean"]),
            auc=number(row["auc_mean"]),
            threshold=number(row["threshold"]),
            tag=row["tag"],
            std_seed_auc=number(row["std_seed_auc"]),
        )

    expected = {(dataset, backbone, colorset)
                for dataset in DATASETS
                for backbone in BACKBONES
                for colorset in COLOUR_SETS}
    missing, extra = expected - set(cells), set(cells) - expected
    if missing or extra:
        raise ValueError(
            f"Register B incompleto: {len(cells)}/450 células; "
            f"missing={sorted(missing)}; extra={sorted(extra)}"
        )

    allowed_thresholds = {step / 20 for step in range(1, 20)}
    invalid = {cell.threshold for cell in cells.values()
               if cell.threshold not in allowed_thresholds}
    if invalid:
        raise ValueError(f"thresholds fora da grade 0.05--0.95: {sorted(invalid)}")
    return cells


def deltas_against_rgb(cells: dict[Key, Cell] | None = None,
                       metric: str = "accuracy") -> dict[Key, float]:
    """Retorna ``metric(colour set) - metric(RGB)`` para as 420 células não-RGB."""
    cells = cells or load_cells()
    out: dict[Key, float] = {}
    for dataset in DATASETS:
        for backbone in BACKBONES:
            baseline = getattr(cells[(dataset, backbone, "RGB")], metric)
            for colorset in RGB_PRESERVING + CHROMA_ONLY:
                out[(dataset, backbone, colorset)] = (
                    getattr(cells[(dataset, backbone, colorset)], metric) - baseline
                )
    return out


def adding_rgb_deltas(cells: dict[Key, Cell] | None = None,
                      metric: str = "accuracy") -> dict[Key, float]:
    """Retorna ``metric(RGB+S) - metric(S)`` para os 210 contrastes pareados."""
    cells = cells or load_cells()
    out: dict[Key, float] = {}
    for dataset in DATASETS:
        for backbone in BACKBONES:
            for with_rgb, base in RGB_PAIRINGS:
                out[(dataset, backbone, with_rgb)] = (
                    getattr(cells[(dataset, backbone, with_rgb)], metric)
                    - getattr(cells[(dataset, backbone, base)], metric)
                )
    return out
