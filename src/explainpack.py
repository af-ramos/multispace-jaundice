"""explainpack.py — persistência dos pesos do modelo e do gate POR IMAGEM.

MOTIVO (emenda 2026-08-18). A campanha v7 gravou duas coisas e não uma terceira:

* ``preds.py`` grava a probabilidade por imagem — o que permitiu repontuar tudo offline;
* ``cv._collect_interpretability`` grava ``alpha_space``, o peso do gate por espaço,
  **médio sobre o test loader**;
* mas os **pesos do modelo** nunca foram gravados (``cv.py`` guarda ``best_state`` num
  ``deepcopy`` em RAM e nunca chama ``torch.save``). Sem eles, qualquer atribuição
  espacial — Grad-CAM num CNN, attention rollout num ViT — exige re-treinar a célula.

Este módulo grava as duas coisas que faltam: o ``state_dict`` do modelo restaurado, e o
vector do gate **por imagem** em vez da sua média. O α médio já dizia PARA ONDE o gate
se inclina; o α por imagem diz se essa inclinação é uniforme ou se o modelo trata
imagens diferentes de maneira diferente, que a média não distingue de todo.

O que ISTO NÃO FAZ: não muda treino, não muda selecção de modelo, não muda métrica
nenhuma. É estritamente aditivo e corre DEPOIS do treino do fold, em ``model.eval()`` e
sob ``no_grad``, exactamente como ``attn.py``. Fica atrás de ``cfg.save_explain``, que é
falso por omissão, portanto uma execução sem a flag é bit a bit a de sempre.

Formato, ao lado do marcador (gitignored, como as predições — é dado derivado):

    <results>/<ds>/<bb>/explain/<célula>__s<seed>_f<fold>.pt   state_dict em CPU/float32
    <results>/<ds>/<bb>/explain/<célula>__gate.npz             path·seed·fold·y_true·w[K]

CUSTO. O ``.pt`` é o tamanho do backbone (de ~9 MB no MobileNetV3 a ~1,2 GB nos cinco
seeds do ViT-L/16), portanto isto é para um punhado de células escolhidas à mão, nunca
para o factorial.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch


def explain_dir_for(cfg) -> Path:
    """``<results>/<dataset>/<backbone>/explain/``, no molde de `attn.attn_path_for`."""
    return cfg.result_dir() / "explain"


def cell_stem(cfg) -> str:
    """O identificador da CÉLULA, sem o sufixo de seed.

    ``cfg.unit_id`` termina em ``__s<seed>`` e muda a cada repetição; o que queremos
    nomear aqui é a célula, que é a mesma para as cinco. É o mesmo ``base`` que o
    ``train.py`` calcula para montar o ``run_id``.
    """
    return cfg.unit_id.rsplit("__s", 1)[0]


def state_path_for(cfg, seed: int, fold: int) -> Path:
    return explain_dir_for(cfg) / f"{cell_stem(cfg)}__s{seed}_f{fold}.pt"


def gate_path_for(cfg) -> Path:
    return explain_dir_for(cfg) / f"{cell_stem(cfg)}__gate.npz"


@torch.no_grad()
def save_state(cfg, seed: int, fold: int, model) -> Optional[Path]:
    """Grava o ``state_dict`` do modelo já restaurado ao melhor checkpoint do fold.

    Em CPU e ``float32``, para o ficheiro não depender do dispositivo nem do dtype de
    treino. Guarda também o mínimo que permite reconstruir o modelo sem adivinhar:
    backbone, colorset, fusão e resolução.
    """
    dest = state_path_for(cfg, seed, fold)
    dest.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().to("cpu", torch.float32) if v.dtype.is_floating_point
             else v.detach().cpu()
             for k, v in model.state_dict().items()}
    torch.save({
        "state_dict": state,
        "backbone": cfg.backbone,
        "dataset": cfg.dataset,
        "colorspaces": list(cfg.colorspaces),
        "colorset_id": cfg.colorset_id,
        "fusion": cfg.fusion,
        "num_classes": cfg.num_classes,
        "input_size": cfg.spec.input_size,
        "backbone_mode": cfg.backbone_mode,
        "seed": int(seed),
        "fold": int(fold),
    }, dest)
    return dest


@torch.no_grad()
def collect_gate(model, loader, device, amp_ctx, channels_last,
                 paths: Sequence[str], y_true: Sequence[int],
                 seed: int, fold: int) -> List[dict]:
    """Um vector do gate por imagem do loader, alinhado com ``paths``.

    Robusto por contrato, como o resto da interpretabilidade: qualquer falha devolve
    ``[]``. E se o número de vectores não bater com o número de caminhos, devolve ``[]``
    em vez de arriscar atribuir o gate de uma imagem a outra --- a mesma regra que
    ``attn._collect_attn_maps`` aplica.
    """
    gate = getattr(getattr(model, "fusion", None), "gate", None)
    if gate is None:
        return []
    try:
        model.eval()
        chunks: List[np.ndarray] = []
        for batch in loader:
            imgs = batch[0].to(device, non_blocking=True)
            if channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)
            with amp_ctx:
                model(imgs)
            w = getattr(gate, "last_weights", None)
            if w is None:
                return []
            chunks.append(w.detach().float().cpu().numpy())
        if not chunks:
            return []
        weights = np.concatenate(chunks, axis=0)
        if len(weights) != len(paths) or len(weights) != len(y_true):
            return []
        return [{"path": str(p), "seed": int(seed), "fold": int(fold),
                 "y_true": int(y), "w": weights[i].astype(np.float32)}
                for i, (p, y) in enumerate(zip(paths, y_true))]
    except Exception:
        return []


def save_gate(cfg, records: List[dict], space_names: Sequence[str]) -> Optional[Path]:
    """Junta os registos de todos os folds×seeds num ``.npz`` comprimido."""
    if not records:
        return None
    dest = gate_path_for(cfg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        paths=np.array([r["path"] for r in records], dtype="U256"),
        seeds=np.array([r["seed"] for r in records], dtype=np.int64),
        folds=np.array([r["fold"] for r in records], dtype=np.int64),
        y_true=np.array([r["y_true"] for r in records], dtype=np.int64),
        w=np.stack([r["w"] for r in records]).astype(np.float32),
        space_names=np.array(list(space_names), dtype="U8"),
    )
    return dest
