"""
aggregate.py
============

Varre os marcadores de conclusao (``results/<dataset>/<modelo>/*.json`` — apenas
as pastas de backbones; as pastas historicas v1/, v2/, ... versionadas a mao pelo
usuario sao ignoradas) e consolida tudo em ``results/results_<dataset>.csv``.

O schema parte do dos arquivos de baseline (``baseline/Results_<dataset>.csv``):

    Dataset, Model, HO, DA, Accuracy, Precision, Recall, F1_Score

e acrescenta as colunas proprias do estudo (espaco de cor, fusao, seed), as
metricas extras (Specificity, Balanced_Accuracy, ROC_AUC, MCC, matriz de
confusao) e o **reporte dual de limiar**:

* ``Threshold`` / metricas principais — limiar que maximiza a metrica-objetivo
  no validacao (operacao clinica, F1_macro);
* ``Threshold_acc`` / ``Accuracy_acc_thr`` / ``Comparable_F1`` — limiar que
  maximiza a ACCURACY no validacao (operacao comparavel ao baseline).

NOTA METODOLOGICA — comparacao com o baseline: os CSVs do baseline calculam
Precision/Recall/F1 tratando a classe MAJORITARIA (healthy) como positiva (a
matriz de confusao salva nos artefatos soma TP+FN = nº de amostras healthy).
Portanto o ``F1_Score`` do baseline corresponde ao nosso ``F1_healthy`` — a
coluna ``Comparable_F1`` (F1_healthy sob o limiar de accuracy) e a comparacao
justa, e as colunas ``Baseline_Acc``/``Baseline_F1`` (melhor configuracao do
modelo no baseline) sao anexadas por Model para leitura lado a lado.

Com execucoes multi-seed, um segundo CSV ``results_<dataset>_<versao>_seeds.csv``
resume cada configuracao (Model x ColorSpace x Fusion) com mean±std e best
(o "best" e o numero comparavel ao protocolo do baseline, que nao fixava seeds).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path
from typing import List

import pandas as pd

# Funciona tanto importado como modulo (``from src.aggregate``) quanto rodado
# como script solto (``python src/aggregate.py``).
if __package__:
    from .config import BACKBONES, COLOR_SPACES, display_name
else:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from src.config import BACKBONES, COLOR_SPACES, display_name

# Ordem das colunas: primeiro o schema do baseline, depois as extensoes do estudo.
COLUMNS = [
    "Dataset", "Model", "ColorSpace", "Fusion", "Num_Classes", "NJN_Mode", "Seed", "HO", "DA",
    "Accuracy", "Precision", "Recall", "F1_Score",
    "F1_macro", "F1_jaundice", "F1_healthy",
    "Threshold_acc", "Accuracy_acc_thr", "Comparable_F1",
    "Accuracy_patient", "F1_macro_patient", "ROC_AUC_patient",
    "Threshold_patient", "n_patients_test",
    "Baseline_Acc", "Baseline_F1",
    "Specificity", "Balanced_Accuracy", "ROC_AUC", "MCC", "Threshold",
    "TN", "FP", "FN", "TP", "n_channels",
    "best_lr", "best_optimizer", "best_unfreeze", "adapter_init",
    "train_n", "val_n", "test_n", "elapsed_min",
]


def _colorset_sort_key(colorset_id: str) -> str:
    """Chave de ordenacao ESCALAR (string) para um subconjunto de cor."""
    spaces = colorset_id.split("+")
    order = {s: i for i, s in enumerate(COLOR_SPACES)}
    seq = "".join(f"{order.get(s, 9):02d}" for s in spaces)
    return f"{len(spaces)}_{seq}"


def load_baseline(dataset: str, baseline_dir: str = "baseline") -> pd.DataFrame | None:
    """Melhor linha do baseline por modelo (max Accuracy entre HOxDA)."""
    path = Path(baseline_dir) / f"Results_{dataset}.csv"
    if not path.exists():
        return None
    base = pd.read_csv(path)
    best = (base.sort_values("Accuracy", ascending=False)
                .groupby("Model", as_index=False).first())
    return best[["Model", "Accuracy", "F1_Score"]].rename(
        columns={"Accuracy": "Baseline_Acc", "F1_Score": "Baseline_F1"})


def _marker_files(results_dir: str, dataset: str, version: str) -> List[Path]:
    """Marcadores da rodada corrente.

    Sem ``version`` (default), varre APENAS as pastas de backbones na raiz de
    ``results/<dataset>/`` — isso exclui as pastas historicas (v1/, v2/, ...)
    que o usuario versiona manualmente ao lado delas.
    """
    base = Path(results_dir) / dataset
    if version:
        base = base / version
        return sorted(base.rglob("*.json")) if base.is_dir() else []
    files: List[Path] = []
    for backbone in BACKBONES:
        d = base / backbone
        if d.is_dir():
            files.extend(sorted(d.glob("*.json")))
    return files


def load_records(results_dir: str, dataset: str, version: str = "") -> List[dict]:
    rows: List[dict] = []
    for f in _marker_files(results_dir, dataset, version):
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        if rec.get("status") != "completed":
            continue
        tm = rec.get("test_metrics", {})
        tm_acc = rec.get("test_metrics_acc") or {}
        tm_pat = rec.get("test_metrics_patient") or {}
        hp = rec.get("best_hparams", {})
        rows.append({
            "Dataset": rec.get("dataset"),
            "Model": display_name(rec.get("backbone", "")),
            "ColorSpace": rec.get("colorset_id"),
            "Fusion": rec.get("fusion"),
            "Num_Classes": rec.get("num_classes"),
            "NJN_Mode": rec.get("njn_mode"),
            "Seed": rec.get("seed"),
            "HO": bool(rec.get("ho")),     # sempre True (Optuna executado)
            "DA": bool(rec.get("da", False)),
            "Accuracy": round(tm.get("accuracy", float("nan")), 2),
            "Precision": round(tm.get("precision", float("nan")), 3),
            "Recall": round(tm.get("recall", float("nan")), 3),
            "F1_Score": round(tm.get("f1", float("nan")), 3),
            "F1_macro": round(tm.get("f1_macro", float("nan")), 3),
            "F1_jaundice": round(tm.get("f1_jaundice", float("nan")), 3),
            "F1_healthy": round(tm.get("f1_healthy", float("nan")), 3),
            "Threshold_acc": rec.get("threshold_acc"),
            "Accuracy_acc_thr": round(tm_acc.get("accuracy", float("nan")), 2),
            "Comparable_F1": round(tm_acc.get("f1_healthy", float("nan")), 3),
            "Accuracy_patient": round(tm_pat.get("accuracy", float("nan")), 2),
            "F1_macro_patient": round(tm_pat.get("f1_macro", float("nan")), 3),
            "ROC_AUC_patient": round(tm_pat.get("roc_auc", float("nan")), 3),
            "Threshold_patient": rec.get("threshold_patient"),
            "n_patients_test": rec.get("n_patients_test") or tm_pat.get("n_patients"),
            "Specificity": round(tm.get("specificity", float("nan")), 3),
            "Balanced_Accuracy": round(tm.get("balanced_accuracy", float("nan")), 2),
            "ROC_AUC": round(tm.get("roc_auc", float("nan")), 3),
            "MCC": round(tm.get("mcc", float("nan")), 3),
            "Threshold": rec.get("threshold"),
            "TN": tm.get("tn"), "FP": tm.get("fp"), "FN": tm.get("fn"), "TP": tm.get("tp"),
            "n_channels": rec.get("n_input_channels"),
            "best_lr": hp.get("lr"), "best_optimizer": hp.get("optimizer"),
            "best_unfreeze": hp.get("n_unfreeze"),
            "adapter_init": rec.get("adapter_init"),
            "train_n": rec.get("sizes", {}).get("train"),
            "val_n": rec.get("sizes", {}).get("val"),
            "test_n": rec.get("sizes", {}).get("test"),
            "elapsed_min": rec.get("elapsed_min"),
        })
    return rows


def summarize_seeds(df: pd.DataFrame) -> pd.DataFrame | None:
    """Resumo multi-seed por configuracao: mean±std (numero defensavel) + best
    (comparavel ao protocolo do baseline, que reportava a melhor execucao)."""
    grouped = df.groupby(["Dataset", "Model", "ColorSpace", "Fusion", "Num_Classes", "NJN_Mode"],
                         dropna=False)
    if grouped["Seed"].nunique().max() <= 1:
        return None
    out = grouped.agg(
        n_seeds=("Seed", "nunique"),
        Accuracy_mean=("Accuracy", "mean"),
        Accuracy_std=("Accuracy", "std"),
        Accuracy_best=("Accuracy", "max"),
        Accuracy_acc_thr_mean=("Accuracy_acc_thr", "mean"),
        Accuracy_acc_thr_best=("Accuracy_acc_thr", "max"),
        F1_macro_mean=("F1_macro", "mean"),
        Comparable_F1_mean=("Comparable_F1", "mean"),
        Accuracy_patient_mean=("Accuracy_patient", "mean"),
        Accuracy_patient_best=("Accuracy_patient", "max"),
        F1_macro_patient_mean=("F1_macro_patient", "mean"),
        ROC_AUC_patient_mean=("ROC_AUC_patient", "mean"),
    ).reset_index()
    for c in out.columns:
        if out[c].dtype == "float64":
            out[c] = out[c].round(3)
    return out


def aggregate(results_dir: str, dataset: str, version: str = "",
              baseline_dir: str = "baseline", out: str | None = None) -> Path | None:
    rows = load_records(results_dir, dataset, version)
    if not rows:
        where = f"{dataset}/{version}" if version else dataset
        print(f"[aggregate] Nenhum resultado concluido para {where} em {results_dir}.")
        return None
    df = pd.DataFrame(rows)

    base = load_baseline(dataset, baseline_dir)
    if base is not None:
        df = df.merge(base, on="Model", how="left")
    else:
        df["Baseline_Acc"] = float("nan")
        df["Baseline_F1"] = float("nan")

    df = df.reindex(columns=COLUMNS)
    df["_sort"] = df["ColorSpace"].map(_colorset_sort_key)
    df = (df.sort_values(by=["Model", "_sort", "Fusion", "Seed"])
            .drop(columns="_sort").reset_index(drop=True))

    suffix = f"_{version}" if version else ""
    # --out permite gravar direto numa subpasta (ex.: results/NeoJaundice/), como o
    # report.py; default = raiz do results_dir (results_<dataset>.csv).
    out_csv = Path(out) if out else Path(results_dir) / f"results_{dataset}{suffix}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, sep=';', decimal=',')
    print(f"[aggregate] {len(df)} linhas -> {out_csv}")

    seeds = summarize_seeds(df)
    if seeds is not None:
        out_seeds = out_csv.with_name(f"results_{dataset}{suffix}_seeds.csv")
        seeds.to_csv(out_seeds, index=False, sep=';', decimal=',')
        print(f"[aggregate] resumo multi-seed ({len(seeds)} configs) -> {out_seeds}")
    return out_csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="local/runs")
    ap.add_argument("--dataset", required=True, choices=["NJN", "NeoJaundice"])
    ap.add_argument("--version", default="",
                    help="Subpasta opcional (vazio = raiz do dataset, ignora v1/v2/...).")
    ap.add_argument("--baseline-dir", default="baseline")
    ap.add_argument("--out", default=None,
                    help="Caminho do CSV de saida (default: <results-dir>/results_<dataset>.csv).")
    args = ap.parse_args()
    aggregate(args.results_dir, args.dataset, args.version, args.baseline_dir, out=args.out)
