"""
metrics.py
==========

Calculo das metricas de avaliacao. Mantemos as 4 metricas da baseline
(Accuracy, Precision, Recall, F1-Score) e acrescentamos metricas relevantes para
um cenario clinico/desbalanceado:

* **Specificity** (taxa de verdadeiros negativos) — importante para nao alarmar
  bebes saudaveis.
* **Balanced Accuracy** — media de sensibilidade e especificidade; robusta ao
  desbalanceamento do NJN.
* **ROC-AUC** — qualidade do ranqueamento independente do limiar.
* **MCC** (Matthews Correlation Coefficient) — resumo balanceado da matriz de
  confusao.
* **Matriz de confusao** (TN, FP, FN, TP).

Convencao de classe positiva = ``jaundice`` (rotulo 1). As metricas precision/
recall/f1 sao reportadas para a icterícia (minoria, clinicamente relevante).

Fase 2 — reporte comparavel ao baseline:
* ``f1_macro``    — media do F1 das duas classes; e a metrica otimizada no HPO
  (robusta a desbalanceamento, evita modelos degenerados "tudo-healthy").
* ``f1_jaundice`` — F1 da minoria (= ``f1``, nossa convenção clinica).
* ``f1_healthy``  — F1 da classe majoritaria; **mesma convenção do baseline**
  (cujo TP/FN era da classe de 121 amostras), permitindo comparacao justa.

``compute_metrics`` aceita opcionalmente um ``threshold`` sobre ``y_prob``: quando
fornecido, as predicoes sao ``y_prob >= threshold`` (usado pelo ajuste de limiar
no validacao). Sem ele, usa-se o ``y_pred`` ja calculado pelo argmax.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

POSITIVE_LABEL = 1  # jaundice


def compute_metrics(y_true: Sequence[int], y_pred: Optional[Sequence[int]] = None,
                    y_prob: Sequence[float] | None = None,
                    threshold: Optional[float] = None) -> Dict[str, float]:
    """Retorna um dicionario com todas as metricas (percentuais em [0,100] para
    accuracy/balanced-acc; demais em [0,1]).

    Se ``threshold`` for dado, as predicoes sao derivadas de ``y_prob >= threshold``
    (ignora ``y_pred``). Caso contrario usa ``y_pred`` (argmax)."""
    y_true = np.asarray(y_true)
    if threshold is not None:
        if y_prob is None:
            raise ValueError("threshold exige y_prob.")
        y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred)

    # Matriz de confusao binaria fixando os rotulos {0,1}.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # Balanced accuracy = media de sensibilidade e especificidade, derivada da
    # matriz {0,1} ja fixada acima. Calcular aqui (em vez de balanced_accuracy_score)
    # evita a confusion_matrix interna SEM labels do sklearn, que emite UserWarning
    # quando um subconjunto tem uma unica classe (ex.: splits minusculos do smoke).
    balanced_acc = 0.5 * (sensitivity + specificity)
    # F1 por classe: index 0 = healthy (baseline), index 1 = jaundice (nossa).
    f1_per = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)

    metrics: Dict[str, float] = {
        "accuracy": 100.0 * accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "specificity": float(specificity),
        "f1": float(f1_per[1]),
        "f1_jaundice": float(f1_per[1]),
        "f1_healthy": float(f1_per[0]),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": 100.0 * balanced_acc,
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, np.asarray(y_prob)))
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# Agregacao por paciente (NeoJaundice: NNNN-p.jpg -> bebe NNNN)
# ---------------------------------------------------------------------------
# A unidade clinica e o BEBE, mas a avaliacao base e por imagem. O NeoJaundice tem
# ~2,8 imagens por bebe (partes do corpo NNNN-1/-2/-3); agregar a probabilidade por
# paciente cancela o ruido de pose/iluminacao e casa a metrica com a tarefa clinica.
# Datasets sem ID de paciente recuperavel (NJN: "jaundice (N)") caem no caso
# degenerado 1 imagem = 1 "paciente" -> metrica-paciente == metrica-imagem (inocuo).

def patient_id_from_filename(filename: str) -> str:
    """ID do bebe a partir do nome do arquivo (``0005-1.jpg`` -> ``0005``).

    Convencao NeoJaundice: prefixo antes do primeiro ``-``. Sem ``-`` (ex.: NJN),
    o stem inteiro vira o id, tornando cada imagem o seu proprio "paciente"."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return stem.split("-")[0]


def _pids_for(paths: Sequence[str], patient_ids: Optional[Sequence[str]]) -> List[str]:
    """IDs de paciente explicitos (ex.: pseudo-IDs pHash da NJN) ou do nome do arquivo."""
    if patient_ids is not None:
        return [str(x) for x in patient_ids]
    return [patient_id_from_filename(p) for p in paths]


def aggregate_by_patient(paths: Sequence[str], y_true: Sequence[int],
                         y_prob: Sequence[float], agg: str = "mean",
                         patient_ids: Optional[Sequence[str]] = None
                         ) -> Tuple[List[str], List[int], List[float]]:
    """Agrega probabilidades por paciente. Devolve (patient_ids, labels, probs).

    ``agg`` = ``mean`` (default; cancela ruido) ou ``max`` (qualquer imagem ictérica
    eleva o bebe). ``patient_ids`` permite IDs explicitos (NJN: pseudo-IDs pHash);
    sem eles, deriva do nome do arquivo (NeoJaundice ``NNNN-p.jpg`` -> ``NNNN``)."""
    pids_in = _pids_for(paths, patient_ids)
    groups: "OrderedDict[str, Dict[str, list]]" = OrderedDict()
    for pid, yt, yp in zip(pids_in, y_true, y_prob):
        g = groups.setdefault(pid, {"labels": [], "probs": []})
        g["labels"].append(int(yt))
        g["probs"].append(float(yp))

    pids: List[str] = []
    labels: List[int] = []
    probs: List[float] = []
    for pid, g in groups.items():
        labs, ps = g["labels"], g["probs"]
        pids.append(pid)
        labels.append(int(round(sum(labs) / len(labs))))  # maioria (== valor comum)
        probs.append(max(ps) if agg == "max" else sum(ps) / len(ps))
    return pids, labels, probs


def compute_patient_metrics(paths: Sequence[str], y_true: Sequence[int],
                            y_prob: Sequence[float], threshold: Optional[float] = None,
                            agg: str = "mean",
                            patient_ids: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Agrega por paciente e calcula as mesmas metricas de ``compute_metrics``,
    acrescentando ``n_patients``."""
    _, p_lab, p_prob = aggregate_by_patient(paths, y_true, y_prob, agg=agg,
                                            patient_ids=patient_ids)
    m = compute_metrics(p_lab, y_prob=p_prob, threshold=threshold)
    m["n_patients"] = len(p_lab)
    return m


def denormalize_tsb(y_norm, mean: float, std: float):
    """Des-normaliza predicoes/targets de TSB com as stats do TREINO do fold.

    Simetria obrigatoria (ver plano/audit): as stats usadas para normalizar o alvo na
    entrada TEM de ser as mesmas usadas aqui na saida, senao o R²/MAE saem errados."""
    return np.asarray(y_norm, dtype=np.float64) * float(std) + float(mean)


def ita_degrees(L, b):
    """Individual Typology Angle (graus): ``atan2(L*-50, b*) · 180/π``.

    Medida dermatológica de tom de pele (Fitzpatrick); usada para estratificar o
    Δ(cor−RGB) por ITA°. Aceita escalares ou arrays (broadcast). L* e b* são as
    coordenadas CIELAB físicas (L*∈[0,100], b* eixo azul↔amarelo)."""
    L = np.asarray(L, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ita = np.arctan2(L - 50.0, b) * (180.0 / np.pi)
    return float(ita) if ita.ndim == 0 else ita


def compute_regression_metrics(y_true_mgdl, y_pred_mgdl) -> Dict[str, float]:
    """MAE e R² da regressao de TSB (em mg/dL, ja des-normalizado)."""
    yt = np.asarray(y_true_mgdl, dtype=np.float64)
    yp = np.asarray(y_pred_mgdl, dtype=np.float64)
    if yt.size == 0:
        return {"tsb_mae": float("nan"), "tsb_r2": float("nan"), "tsb_rmse": float("nan")}
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"tsb_mae": mae, "tsb_rmse": rmse, "tsb_r2": r2}


def format_metrics(m: Dict[str, float]) -> str:
    base = (f"Acc={m['accuracy']:.2f}% MacroF1={m['f1_macro']:.3f} "
            f"BalAcc={m['balanced_accuracy']:.2f}%")
    if "f1_jaundice" in m:  # binario
        return (f"{base} P={m['precision']:.3f} R={m['recall']:.3f} Spec={m['specificity']:.3f} "
                f"F1j={m['f1_jaundice']:.3f} F1h={m['f1_healthy']:.3f} "
                f"AUC={m['roc_auc']:.3f} MCC={m['mcc']:.3f} "
                f"[TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}]")
    pc = ",".join(f"{x:.2f}" for x in m.get("f1_per_class", []))
    return (f"{base} AUC_ovr={m.get('roc_auc', float('nan')):.3f} MCC={m['mcc']:.3f} "
            f"F1/cls=[{pc}]")


# ---------------------------------------------------------------------------
# Metricas MULTICLASSE (NeoJaundice 3 faixas de TSB) — decisao por argmax
# ---------------------------------------------------------------------------
def compute_metrics_multiclass(y_true: Sequence[int], y_score, num_classes: int) -> Dict[str, float]:
    """Metricas para K>2 classes. ``y_score`` = matriz [N,K] (softmax); argmax decide.

    Reporta accuracy, Macro-F1 (prioritaria), F1/precisao/recall por classe,
    balanced-accuracy, MCC multiclasse, AUC One-vs-Rest (macro) e matriz KxK."""
    y_true = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=np.float64)
    labels = list(range(num_classes))
    y_pred = scores.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    f1_per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    prec_per = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec_per = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    metrics: Dict[str, float] = {
        "accuracy": 100.0 * accuracy_score(y_true, y_pred),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        # balanced accuracy = media das recalls por classe (rec_per ja usa labels=labels),
        # evitando a confusion_matrix interna SEM labels do balanced_accuracy_score.
        "balanced_accuracy": 100.0 * float(rec_per.mean()),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "f1_per_class": [float(x) for x in f1_per],
        "precision_per_class": [float(x) for x in prec_per],
        "recall_per_class": [float(x) for x in rec_per],
        "cm": cm.tolist(),
    }
    try:
        if len(np.unique(y_true)) == num_classes:
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores, multi_class="ovr",
                                                     average="macro", labels=labels))
        else:
            metrics["roc_auc"] = float("nan")
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def aggregate_by_patient_multiclass(paths: Sequence[str], y_true: Sequence[int], y_score,
                                    patient_ids: Optional[Sequence[str]] = None
                                    ) -> Tuple[List[str], List[int], "np.ndarray"]:
    """Agrega por paciente a MEDIA do vetor de probabilidades [N,K]."""
    scores = np.asarray(y_score, dtype=np.float64)
    pids_in = _pids_for(paths, patient_ids)
    groups: "OrderedDict[str, Dict[str, list]]" = OrderedDict()
    for pid, yt, sc in zip(pids_in, y_true, scores):
        g = groups.setdefault(pid, {"labels": [], "scores": []})
        g["labels"].append(int(yt))
        g["scores"].append(np.asarray(sc, dtype=np.float64))
    pids, labels, agg = [], [], []
    for pid, g in groups.items():
        pids.append(pid)
        labels.append(int(round(sum(g["labels"]) / len(g["labels"]))))
        agg.append(np.mean(np.stack(g["scores"], axis=0), axis=0))
    return pids, labels, np.stack(agg, axis=0)


def compute_patient_metrics_multiclass(paths: Sequence[str], y_true: Sequence[int], y_score,
                                       num_classes: int,
                                       patient_ids: Optional[Sequence[str]] = None) -> Dict[str, float]:
    _, p_lab, p_score = aggregate_by_patient_multiclass(paths, y_true, y_score,
                                                        patient_ids=patient_ids)
    m = compute_metrics_multiclass(p_lab, p_score, num_classes)
    m["n_patients"] = len(p_lab)
    return m
