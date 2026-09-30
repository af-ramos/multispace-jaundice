"""attn.py — persistência dos mapas de atenção do CCAT (insumo da Fig 5).

MOTIVO (emenda 2026-07-26). `CCATFusion.attn_maps` é populado a cada forward
(`models.py`) e **descartado no forward seguinte**: nada fora do módulo o lia. Como o
v7 nunca chama ``torch.save`` (os pesos vivem só em RAM, ver `preds.py`), o mapa não é
recuperável depois — ou se grava durante o run, ou se re-treina. Este módulo grava.

O que ISTO NÃO FAZ: não muda treino, não muda seleção de modelo, não muda métrica
nenhuma. É estritamente aditivo — a coleta roda DEPOIS do treino do fold, em
``model.eval()`` e sob ``no_grad``.

Formato: ``.npz`` comprimido em ``<dir_do_marcador>/attn/<run_id>.npz`` (gitignored,
como as predições — mapa de atenção é dado derivado). Uma linha por imagem guardada:

    path · seed · fold · y_true · y_prob · bucket ∈ {TP,TN,FP,FN} · maps[K,H,W]

**Amostragem, e por que ela é honesta:** guardar o test inteiro × 5 seeds seria centenas
de MB de dado redundante. Guarda-se uma cota fixa por *bucket* de acerto/erro
(``n_per_bucket``), pegando as PRIMEIRAS ocorrências na ordem determinística do loader —
sem escolher por confiança, sem escolher "as bonitas". Os FP/FN entram por cota própria
justamente para que a figura não possa mostrar só acertos. O ``y_prob`` vai junto para
que a legenda da figura seja auditável.

⚠️ O bucket é calculado com limiar 0,5 no momento da coleta (o limiar do marcador só
existe depois de agregar todos os folds/seeds). É rótulo de AMOSTRAGEM, não de métrica:
qualquer número reportado sai do marcador/`preds.py`, nunca daqui.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

#: Cota padrão por bucket, por (seed, fold). 4×4 = até 16 imagens por fold.
N_PER_BUCKET = 4

BUCKETS = ("TP", "TN", "FP", "FN")


def bucket_of(y_true: int, y_prob: float, threshold: float = 0.5) -> str:
    """Rótulo de amostragem: acerto/erro por classe ao limiar dado."""
    pred = int(y_prob >= threshold)
    if y_true == 1:
        return "TP" if pred == 1 else "FN"
    return "FP" if pred == 1 else "TN"


@dataclass
class AttnDump:
    """Mapas de atenção por espaço crômico de UMA unidade de experimento (1 marcador)."""

    paths: np.ndarray        # <U…
    seeds: np.ndarray        # int64
    folds: np.ndarray        # int64
    y_true: np.ndarray       # int64
    y_prob: np.ndarray       # float64
    buckets: np.ndarray      # <U2
    maps: np.ndarray         # float16 [N, K, H, W]  (softmax sobre os K espaços)
    space_names: np.ndarray  # <U…    [K]

    @property
    def n_rows(self) -> int:
        return int(self.paths.shape[0])

    @property
    def n_spaces(self) -> int:
        return int(self.space_names.shape[0])

    # ------------------------------------------------------------------ #
    @classmethod
    def from_records(cls, recs: Sequence[dict], space_names: Sequence[str]) -> "AttnDump":
        """Constrói a partir de dicts ``{path, seed, fold, y_true, y_prob, bucket, map}``."""
        if not recs:
            return cls(np.array([], dtype="<U1"), np.array([], dtype=np.int64),
                       np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                       np.array([], dtype=np.float64), np.array([], dtype="<U2"),
                       np.zeros((0, 0, 0, 0), dtype=np.float16),
                       np.asarray(list(space_names), dtype=str))
        return cls(
            paths=np.asarray([str(r["path"]) for r in recs], dtype=str),
            seeds=np.asarray([int(r["seed"]) for r in recs], dtype=np.int64),
            folds=np.asarray([int(r["fold"]) for r in recs], dtype=np.int64),
            y_true=np.asarray([int(r["y_true"]) for r in recs], dtype=np.int64),
            y_prob=np.asarray([float(r["y_prob"]) for r in recs], dtype=np.float64),
            buckets=np.asarray([str(r["bucket"]) for r in recs], dtype=str),
            maps=np.stack([np.asarray(r["map"], dtype=np.float16) for r in recs]),
            space_names=np.asarray(list(space_names), dtype=str),
        )

    def save(self, path) -> None:
        """Escrita ATÔMICA (tmp -> rename), como os marcadores e o dump de predições."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        np.savez_compressed(
            tmp, paths=self.paths, seeds=self.seeds, folds=self.folds, y_true=self.y_true,
            y_prob=self.y_prob, buckets=self.buckets, maps=self.maps,
            space_names=self.space_names)
        cand = tmp if tmp.is_file() else tmp.with_suffix(tmp.suffix + ".npz")
        cand.replace(path)

    @classmethod
    def load(cls, path) -> "AttnDump":
        with np.load(Path(path), allow_pickle=False) as z:
            return cls(paths=z["paths"], seeds=z["seeds"], folds=z["folds"],
                       y_true=z["y_true"], y_prob=z["y_prob"], buckets=z["buckets"],
                       maps=z["maps"], space_names=z["space_names"])

    # ------------------------------------------------------------------ #
    def subset(self, bucket: Optional[str] = None, seed: Optional[int] = None) -> "AttnDump":
        m = np.ones(self.n_rows, dtype=bool)
        if bucket is not None:
            m &= (self.buckets == bucket)
        if seed is not None:
            m &= (self.seeds == int(seed))
        return AttnDump(self.paths[m], self.seeds[m], self.folds[m], self.y_true[m],
                        self.y_prob[m], self.buckets[m], self.maps[m], self.space_names)

    def mean_map_by_space(self, bucket: Optional[str] = None) -> np.ndarray:
        """Mapa médio [K, H, W] (opcionalmente restrito a um bucket)."""
        d = self.subset(bucket=bucket) if bucket else self
        if d.n_rows == 0:
            return np.zeros((self.n_spaces, 0, 0), dtype=np.float32)
        return d.maps.astype(np.float32).mean(axis=0)

    def space_share(self) -> dict:
        """Fração média de atenção por espaço (média espacial do softmax).

        Como o softmax do CCAT é sobre os K espaços em cada pixel, as frações somam ~1.
        Espaço com fração ≫ 1/K = o modelo olha mais para aquele espaço.
        """
        if self.n_rows == 0:
            return {}
        m = self.maps.astype(np.float32).mean(axis=(0, 2, 3))
        return {str(nm): float(v) for nm, v in zip(self.space_names, m)}


def attn_path_for(marker_path) -> Path:
    """Caminho do dump ao lado do marcador: ``<dir>/attn/<stem>.npz``."""
    marker_path = Path(marker_path)
    return marker_path.parent / "attn" / (marker_path.stem + ".npz")


def select_records(paths: Sequence[str], y_true: Sequence[int], y_prob: Sequence[float],
                   maps: Sequence[np.ndarray], seed: int, fold: int,
                   n_per_bucket: int = N_PER_BUCKET,
                   threshold: float = 0.5) -> List[dict]:
    """Cota fixa por bucket, na ordem determinística do loader (sem cherry-picking)."""
    out: List[dict] = []
    contagem = {b: 0 for b in BUCKETS}
    for pa, yt, yp, mp in zip(paths, y_true, y_prob, maps):
        b = bucket_of(int(yt), float(yp), threshold)
        if contagem[b] >= n_per_bucket:
            continue
        contagem[b] += 1
        out.append({"path": str(pa), "seed": int(seed), "fold": int(fold),
                    "y_true": int(yt), "y_prob": float(yp), "bucket": b, "map": mp})
    return out
