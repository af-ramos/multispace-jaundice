"""
preds.py — persistência das predições por imagem e re-scoring offline.

MOTIVO (emenda 2026-07-25). Até aqui o harness não persistia NADA além das métricas
agregadas: os pesos viviam só em RAM (``cv.py`` guarda ``best_state`` com
``copy.deepcopy``, nunca ``torch.save``) e o marcador só levava os totais da matriz de
confusão por seed. Consequência: qualquer pergunta nova sobre as MESMAS predições —
agregar por paciente, mover o limiar, juntar os seeds em ensemble — exigia re-treinar
tudo. Este módulo grava as predições uma vez (~26 KB por execução na NeoJaundice, ~9 KB
na NJN) e transforma essas perguntas em RE-SCORING LOCAL, sem GPU e sem re-treino.

O que ISTO NÃO FAZ: não muda treino, não muda seleção de modelo, não muda o limiar do
marcador. É estritamente aditivo — as métricas-imagem já existentes continuam idênticas
bit a bit (o teste ``test_rescore_reproduz_metricas_imagem`` trava isso).

Formato: ``.npz`` comprimido (já coberto pelo .gitignore — predição é dado derivado,
não vai para o Git). Uma linha por (split, seed, fold, imagem):

    split ∈ {val, test} · seed · fold · path · patient_id · y_true · y_prob

``patient_id`` vem do ``bundle.path_to_pid`` (ID real na NeoJaundice, pseudo-ID pHash na
NJN) — nunca é re-derivado do nome do arquivo aqui, para não divergir do agrupamento que
o split congelado usou.

ATENÇÃO AO PROTOCOLO: o limiar de qualquer métrica-paciente sai de
:func:`patient_threshold_from_val`, que só olha ``split == "val"``. Escolher limiar no
test é o vazamento da regra #2 e já foi medido neste projeto (~+0,009 AUC).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import aggregate_by_patient, compute_metrics

# Colunas do dump, em ordem canônica (uma linha = uma predição de uma imagem).
ROW_FIELDS = ("split", "seed", "fold", "path", "patient_id", "y_true", "y_prob")


@dataclass
class PredDump:
    """Predições por imagem de UMA unidade de experimento (1 marcador)."""

    splits: np.ndarray       # <U4   'val' | 'test'
    seeds: np.ndarray        # int64
    folds: np.ndarray        # int64
    paths: np.ndarray        # <U…
    patient_ids: np.ndarray  # <U…
    y_true: np.ndarray       # int64
    y_prob: np.ndarray       # float64

    # ------------------------------------------------------------------ #
    @property
    def n_rows(self) -> int:
        return int(self.splits.shape[0])

    @classmethod
    def from_rows(cls, rows: Sequence[Tuple]) -> "PredDump":
        if not rows:
            return cls(*(np.array([], dtype=d) for d in
                         ("<U4", "int64", "int64", "<U1", "<U1", "int64", "float64")))
        cols = list(zip(*rows))
        return cls(
            splits=np.asarray(cols[0], dtype=str),
            seeds=np.asarray(cols[1], dtype=np.int64),
            folds=np.asarray(cols[2], dtype=np.int64),
            paths=np.asarray(cols[3], dtype=str),
            patient_ids=np.asarray(cols[4], dtype=str),
            y_true=np.asarray(cols[5], dtype=np.int64),
            y_prob=np.asarray(cols[6], dtype=np.float64),
        )

    def copy(self) -> "PredDump":
        return PredDump(*(a.copy() for a in (self.splits, self.seeds, self.folds, self.paths,
                                             self.patient_ids, self.y_true, self.y_prob)))

    # ------------------------------------------------------------------ #
    def save(self, path) -> None:
        """Escrita ATÔMICA (tmp -> rename), como os marcadores."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        np.savez_compressed(
            tmp, splits=self.splits, seeds=self.seeds, folds=self.folds, paths=self.paths,
            patient_ids=self.patient_ids, y_true=self.y_true, y_prob=self.y_prob)
        # np.savez acrescenta '.npz' se o nome não terminar nele
        cand = tmp if tmp.is_file() else tmp.with_suffix(tmp.suffix + ".npz")
        cand.replace(path)

    @classmethod
    def load(cls, path) -> "PredDump":
        with np.load(Path(path), allow_pickle=False) as z:
            return cls(splits=z["splits"], seeds=z["seeds"], folds=z["folds"],
                       paths=z["paths"], patient_ids=z["patient_ids"],
                       y_true=z["y_true"], y_prob=z["y_prob"])

    # ------------------------------------------------------------------ #
    def subset(self, split: Optional[str] = None, seed: Optional[int] = None) -> "PredDump":
        m = np.ones(self.n_rows, dtype=bool)
        if split is not None:
            m &= (self.splits == split)
        if seed is not None:
            m &= (self.seeds == int(seed))
        return PredDump(self.splits[m], self.seeds[m], self.folds[m], self.paths[m],
                        self.patient_ids[m], self.y_true[m], self.y_prob[m])

    def seed_list(self) -> List[int]:
        return [int(s) for s in np.unique(self.seeds)]


# --------------------------------------------------------------------------- #
# Agregação / limiar
# --------------------------------------------------------------------------- #
def _agg_patient(d: PredDump, agg: str = "mean") -> Tuple[List[int], List[float], int]:
    """(labels, probs, n_pacientes) agregando por ``patient_id``."""
    if d.n_rows == 0:
        return [], [], 0
    _, lab, prob = aggregate_by_patient(list(d.paths), list(d.y_true), list(d.y_prob),
                                        agg=agg, patient_ids=list(d.patient_ids))
    return lab, prob, len(lab)


def patient_threshold_from_val(d: PredDump, objective: str = "f1_macro",
                               agg: str = "mean") -> float:
    """Limiar-paciente escolhido SÓ nas predições de validação (regra #2).

    Agrega por paciente DENTRO de cada seed (cada seed é um modelo distinto) e junta os
    seeds para varrer o limiar — o análogo, a nível de paciente, do ``pooled_oof``."""
    from .engine import _best_threshold

    lab: List[int] = []
    prob: List[float] = []
    val = d.subset(split="val")
    for s in val.seed_list():
        l_s, p_s, _ = _agg_patient(val.subset(seed=s), agg=agg)
        lab.extend(l_s); prob.extend(p_s)
    if not lab or len(set(lab)) < 2:
        return 0.5
    thr, _ = _best_threshold(lab, prob, objective)
    return float(thr)


def patient_metrics_from_dump(d: PredDump, threshold: float, split: str = "test",
                              agg: str = "mean") -> Dict[int, Dict[str, float]]:
    """Métricas por paciente, por seed. ``{}`` se o dump estiver vazio."""
    sub = d.subset(split=split)
    out: Dict[int, Dict[str, float]] = {}
    for s in sub.seed_list():
        lab, prob, npat = _agg_patient(sub.subset(seed=s), agg=agg)
        if not lab:
            continue
        m = compute_metrics(lab, y_prob=prob, threshold=threshold)
        m["n_patients"] = npat
        out[int(s)] = m
    return out


# --------------------------------------------------------------------------- #
# Re-scoring genérico (o que torna as alavancas gratuitas)
# --------------------------------------------------------------------------- #
def rescore_from_dump(d: PredDump, split: str = "test", threshold: float = 0.5,
                      by_patient: bool = False, agg: str = "mean",
                      ensemble_seeds: bool = False) -> Dict:
    """Recalcula métricas a partir das predições salvas — sem GPU, sem re-treino.

    ``by_patient``  agrega por paciente antes de pontuar.
    ``ensemble_seeds`` média das probabilidades entre seeds da MESMA imagem/paciente
    (unidade = 1 imagem, não 1 imagem×seed) — o ensemble que hoje se perde ao mediar
    métricas em vez de predições.
    """
    sub = d.subset(split=split)
    out: Dict = {"per_seed": {}}
    for s in sub.seed_list():
        ss = sub.subset(seed=s)
        if by_patient:
            lab, prob, npat = _agg_patient(ss, agg=agg)
            m = compute_metrics(lab, y_prob=prob, threshold=threshold)
            m["n_patients"] = npat
        else:
            lab, prob = list(ss.y_true), list(ss.y_prob)
            m = compute_metrics(lab, y_prob=prob, threshold=threshold)
        m["n"] = len(lab)
        out["per_seed"][int(s)] = m

    if ensemble_seeds and sub.n_rows:
        key = sub.patient_ids if by_patient else sub.paths
        order: List[str] = []
        acc: Dict[str, List] = {}
        for k, yt, yp in zip(key, sub.y_true, sub.y_prob):
            if k not in acc:
                acc[k] = [int(yt), []]
                order.append(k)
            acc[k][1].append(float(yp))
        lab = [acc[k][0] for k in order]
        prob = [float(np.mean(acc[k][1])) for k in order]
        m = compute_metrics(lab, y_prob=prob, threshold=threshold)
        m["n"] = len(lab)
        out["ensemble"] = m
    return out


def preds_path_for(marker_path) -> Path:
    """Caminho do dump ao lado do marcador: ``<dir>/preds/<stem>.npz``."""
    marker_path = Path(marker_path)
    return marker_path.parent / "preds" / (marker_path.stem + ".npz")
