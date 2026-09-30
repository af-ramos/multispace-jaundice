"""
cv.py (NOVO — v6)
=================

**Harness de validação cruzada 5-fold × 5-seed com ``pooled_oof``** — a peça central
do protocolo honesto do v6 (substitui o split único 70/10/20 da v5).

Protocolo (ver plano §4 e §7):

* Para cada ``seed`` e cada ``fold`` de um ``StratifiedGroupKFold(k=5)`` (por
  ``patient_id`` na NeoJaundice; por pseudo-paciente pHash na NJN), treina UM modelo.
* **val e test são disjuntos DENTRO do fold**: o fold retido é o **test**; do restante
  carve-se um **val** (:func:`splits.kfold_indices`). O limiar do ``pooled_oof`` é
  escolhido **só nas predições de validação** (juntadas entre folds/seeds) e aplicado ao
  **test** — nunca se escolhe limiar olhando o test.
* Métrica primária = **AUC** (independe de limiar). **Unidade da variância do GATE**:
  1 AUC por seed (predições de test agrupadas dos 5 folds daquele seed) → ``std`` sobre
  as ``n_seeds`` AUCs a nível de seed (NÃO por-fit).
* Estabilidade: **SWA/EMA** opcionais; backbone congelado em ``eval()`` (BN fixa).
* Multitarefa (só NeoJaundice): cabeça de regressão TSB com **uncertainty weighting
  (Kendall)**; predições des-normalizadas com as stats do treino do fold antes de MAE/R².

Saída: um marcador JSON agregado por config (média ± IC95%, std entre seeds, limiar,
métricas por seed e de regressão), gravado por :mod:`train`.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import attn as attnmod
from . import splits
from .config import ExperimentConfig
from .data import build_dataloaders_from_split
from .engine import (HParams, _aligned_paths, _amp_ctx, _best_threshold, _collect,
                     _cosine_warmup_scheduler, _fp16_scaler, _make_optimizer,
                     _shutdown_loaders, build_param_groups)
from .metrics import (compute_metrics, compute_metrics_multiclass,
                      compute_regression_metrics, denormalize_tsb)
from .models import build_model
from . import explainpack as explainmod
from .preds import PredDump, patient_metrics_from_dump, patient_threshold_from_val


# --------------------------------------------------------------------------- #
# EMA dos pesos (shadow) — estabilidade entre seeds
# --------------------------------------------------------------------------- #
class _EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def state_dict(self, model: nn.Module) -> dict:
        out = copy.deepcopy(model.state_dict())
        for k in self.shadow:
            out[k] = self.shadow[k].clone()
        return out


def _seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _kendall_loss(ce: torch.Tensor, reg: torch.Tensor,
                  log_var_cls: torch.Tensor, log_var_reg: torch.Tensor) -> torch.Tensor:
    """L = exp(-s1)·CE + exp(-s2)·Huber + s1 + s2 (Kendall & Gal, uncertainty weighting)."""
    return (torch.exp(-log_var_cls) * ce + torch.exp(-log_var_reg) * reg
            + log_var_cls + log_var_reg)


# --------------------------------------------------------------------------- #
# Treino de UM fold
# --------------------------------------------------------------------------- #
def _train_one_fold(model, bundle, device, cfg: ExperimentConfig, hp: HParams,
                    epochs: int, patience: int, multitask: bool,
                    swa: bool, ema: bool, verbose: bool = False):
    channels_last = (device.type == "cuda" and cfg.spec.family == "cnn")
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    huber = nn.HuberLoss()
    optim = _make_optimizer(hp.optimizer,
                            build_param_groups(model, hp.lr, cfg.backbone_lr_mult), hp.lr)
    sched = _cosine_warmup_scheduler(optim, epochs)
    scaler = _fp16_scaler(cfg.amp, cfg.amp_dtype, device)
    use_scaler = scaler is not None and scaler.is_enabled()
    ema_obj = _EMA(model, cfg.ema_decay) if ema else None
    swa_states: List[dict] = []
    swa_start = int(0.75 * epochs)

    best_metric, best_state = -1.0, copy.deepcopy(model.state_dict())
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        optim.zero_grad(set_to_none=True)
        for batch in bundle.train_loader:
            if multitask:
                imgs, labels, tsb = batch
                tsb = tsb.to(device, non_blocking=True).float()
            else:
                imgs, labels = batch
                tsb = None
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)
            with _amp_ctx(device, cfg.amp, cfg.amp_dtype):
                logits = model(imgs)
                ce = ce_loss(logits, labels)
                if multitask and getattr(model, "last_tsb", None) is not None:
                    reg = huber(model.last_tsb.float(), tsb)
                    loss = _kendall_loss(ce, reg, model.log_var_cls, model.log_var_reg)
                else:
                    loss = ce
            (scaler.scale(loss) if use_scaler else loss).backward()
            if use_scaler:
                scaler.step(optim); scaler.update()
            else:
                optim.step()
            optim.zero_grad(set_to_none=True)
        sched.step()
        if ema_obj is not None:
            ema_obj.update(model)
        # Seleção pelo objetivo de CLASSIFICAÇÃO no val (early stopping).
        vm = _eval_cls(model, bundle.val_loader, device, cfg, channels_last)
        monitor = vm.get(cfg.objective, 0.0)
        if verbose:
            print(f"    ep {epoch+1:2d}/{epochs} val_{cfg.objective}={monitor:.3f}")
        if monitor > best_metric:
            best_metric = monitor
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if swa and epoch >= swa_start:
            swa_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if no_improve >= patience:
            break

    # Escolha do estado final: SWA (média) > EMA (shadow) > melhor por early stopping.
    if swa and len(swa_states) >= 2:
        avg = {k: torch.stack([s[k].float() for s in swa_states]).mean(0)
               for k in swa_states[0]}
        model.load_state_dict(avg)
        _update_bn(model, bundle.train_loader, device, cfg, channels_last, multitask)
    elif ema_obj is not None:
        model.load_state_dict(ema_obj.state_dict(model))
    else:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _update_bn(model, loader, device, cfg, channels_last, multitask, max_batches: int = 50):
    """Recalcula as running stats de BN após a média SWA (só módulos treináveis;
    o backbone congelado permanece em eval)."""
    was_training = model.training
    model.train()  # ChromaFiLMNet.train() mantém o backbone em eval (BN fixa)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) and m.momentum is not None:
            m.reset_running_stats()
    n = 0
    for batch in loader:
        imgs = batch[0].to(device, non_blocking=True)
        if channels_last:
            imgs = imgs.to(memory_format=torch.channels_last)
        with _amp_ctx(device, cfg.amp, cfg.amp_dtype):
            model(imgs)
        n += 1
        if n >= max_batches:
            break
    if not was_training:
        model.eval()


@torch.no_grad()
def _eval_cls(model, loader, device, cfg, channels_last) -> Dict[str, float]:
    ys, probs, _ = _collect(model, loader, device, None, cfg.tta, amp=cfg.amp,
                            channels_last=channels_last, num_classes=cfg.num_classes,
                            amp_dtype=cfg.amp_dtype)
    if cfg.num_classes == 2:
        return compute_metrics(ys, y_prob=probs, threshold=0.5)
    return compute_metrics_multiclass(ys, probs, cfg.num_classes)


@torch.no_grad()
def _collect_preds(model, loader, device, cfg, channels_last):
    """Devolve (y_true, probs, tsb_pred_norm) — probs binário=p(1), multiclasse=vetor."""
    ys, probs, _ = _collect(model, loader, device, None, cfg.tta, amp=cfg.amp,
                            channels_last=channels_last, num_classes=cfg.num_classes,
                            amp_dtype=cfg.amp_dtype)
    tsb_pred: List[float] = []
    if cfg.multitask:
        model.eval()
        for batch in loader:
            imgs = batch[0].to(device, non_blocking=True)
            if channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)
            with _amp_ctx(device, cfg.amp, cfg.amp_dtype):
                model(imgs)
            if getattr(model, "last_tsb", None) is not None:
                tsb_pred.extend(model.last_tsb.detach().float().cpu().tolist())
    return ys, probs, tsb_pred


#: Fusões que expõem um vetor de pesos por espaço, e QUAIS espaços esse vetor indexa.
#: ⚠️ NÃO são os mesmos espaços: o α_space do ChromaFiLM percorre só os espaços CRÔMICOS
#: (o RGB é a âncora, fora do vetor), enquanto o ``_SpaceGate`` do adapter_v2 tem um peso
#: por GRUPO da pilha — RGB incluído. Rotular o gate com os nomes crômicos deslocaria a
#: tabela em uma posição (o peso do RGB sairia rotulado como LAB), em silêncio.
_ALPHA_FUSIONS = {"chromafilm": "chroma", "chromafilm_route": "chroma", "adapter_v2": "todos"}


def _alpha_names(cfg, escopo: str) -> List[str]:
    """Nomes dos espaços que o vetor de α indexa, na ordem da pilha."""
    if escopo == "todos":
        return [str(c) for c in cfg.colorspaces]
    return [str(c) for c in cfg.colorspaces if str(c).upper() != "RGB"]


@torch.no_grad()
def _collect_interpretability(model, loader, device, cfg, channels_last):
    """α por espaço e perfil ‖Δγ,Δβ‖ por estágio sobre ``loader``.

    Cobre duas fusões com semânticas diferentes (ver ``_ALPHA_FUSIONS``):

    * ``chromafilm`` / ``chromafilm_route`` — ``last_alpha``, softmax sobre os espaços
      CRÔMICOS, mais o perfil FiLM por estágio.
    * ``adapter_v2`` — ``fusion.gate.last_weights`` do :class:`~models._SpaceGate`
      (GAP → softmax × K), um peso por espaço da pilha, **RGB incluído**. Init neutra
      dá 1,0 em todos; α≈1 uniforme depois do treino = **gate inerte**, que é o
      diagnóstico registrado nos PRINCÍPIOS DE PROJETO do CLAUDE.md (a BN a jusante
      reabsorve um ganho escalar por espaço). Inerte é RESULTADO, não bug.

    Devolve ``(alpha, alpha_names, film, n_sites)``. **Robusto por contrato:** qualquer
    falha (ou fusão sem α) devolve ``(None, None, None, 0)`` — a interpretabilidade
    NUNCA quebra um run de treino.
    """
    escopo = _ALPHA_FUSIONS.get(getattr(model, "fusion_type", None))
    if escopo is None:
        return None, None, None, 0
    try:
        model.eval()
        alpha_sum = None
        film_sum: Dict[int, List[float]] = {}
        n = 0
        for batch in loader:
            imgs = batch[0].to(device, non_blocking=True)
            if channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)
            bs = int(imgs.size(0))
            with _amp_ctx(device, cfg.amp, cfg.amp_dtype):
                model(imgs)
            a = getattr(model, "last_alpha", None)
            if a is None:  # fallback: α_space vive no submódulo do gate
                gate = getattr(getattr(model, "fusion", None), "space_gate", None)
                a = getattr(gate, "last_alpha", None) if gate is not None else None
            if a is None:  # adapter_v2: pesos por espaço do _SpaceGate
                gate = getattr(getattr(model, "fusion", None), "gate", None)
                a = getattr(gate, "last_weights", None) if gate is not None else None
            if a is not None:
                av = a.detach().float().mean(dim=0).cpu().numpy()   # [K]
                alpha_sum = av * bs if alpha_sum is None else alpha_sum + av * bs
            for i, (dg, db) in getattr(model, "last_film_stats", {}).items():
                acc = film_sum.setdefault(int(i), [0.0, 0.0])
                acc[0] += float(dg) * bs
                acc[1] += float(db) * bs
            n += bs
        if n == 0:
            return None, None, None, 0
        alpha = (alpha_sum / n).tolist() if alpha_sum is not None else None
        names = _alpha_names(cfg, escopo)[:len(alpha)] if alpha is not None else None
        n_sites = int(getattr(model, "n_sites", len(film_sum)) or len(film_sum))
        film = {i: (v[0] / n, v[1] / n) for i, v in film_sum.items()}
        return alpha, names, film, n_sites
    except Exception as e:  # nunca quebrar o run por causa da interpretabilidade
        print(f"    [interp] aviso: coleta de α_space/FiLM falhou ({e}); seguindo sem ela.")
        return None, None, None, 0


@torch.no_grad()
def _collect_attn_maps(model, loader, device, cfg, channels_last, seed: int, fold: int,
                       n_per_bucket: int = attnmod.N_PER_BUCKET):
    """Mapas de atenção do CCAT para uma cota de imagens do loader (insumo da Fig 5).

    Só ``fusion == 'ccat'``: ``CCATFusion.attn_maps`` é [B, K_chroma, H, W], softmax
    espacial POR PIXEL sobre os espaços crômicos (por isso o gate do CCAT não é inerte —
    a BN não reabsorve uma distribuição por-pixel).

    Amostra por bucket TP/TN/FP/FN na ordem determinística do loader (`attn.select_records`)
    e alinha os paths com o mesmo helper das predições — se a ordem não casar, devolve
    ``[]`` em vez de arriscar atribuir um mapa à imagem errada. Robusto por contrato:
    qualquer falha devolve ``([], None)``.
    """
    if getattr(model, "fusion_type", None) != "ccat" or cfg.num_classes != 2:
        return [], None
    try:
        model.eval()
        ys: List[int] = []
        ps: List[float] = []
        mps: List[np.ndarray] = []
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            if channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)
            with _amp_ctx(device, cfg.amp, cfg.amp_dtype):
                logits = model(imgs)
            p = torch.softmax(logits.float(), dim=1)[:, 1]
            amaps = getattr(getattr(model, "fusion", None), "attn_maps", None)
            if amaps is None:
                return [], None
            ys.extend(int(v) for v in labels.cpu().tolist())
            ps.extend(float(v) for v in p.cpu().tolist())
            mps.extend(amaps.detach().float().cpu().numpy().astype(np.float16))
        paths = _aligned_paths(loader, ys)
        if paths is None:
            print(f"    [attn] aviso: ordem não confere (seed {seed}, fold {fold}) — "
                  "dump de atenção desta partição pulado.")
            return [], None
        recs = attnmod.select_records(paths, ys, ps, mps, seed=seed, fold=fold,
                                      n_per_bucket=n_per_bucket)
        names = [str(c) for c in cfg.colorspaces if str(c).upper() != "RGB"]
        return recs, names
    except Exception as e:
        print(f"    [attn] aviso: coleta de mapas de atenção falhou ({e}); seguindo sem ela.")
        return [], None


@torch.no_grad()
def _collect_loso(model, loader, device, cfg, channels_last):
    """ΔAUC por espaco crômico (leave-one-space-out): AUC(todos) − AUC(sem espaco k).

    Ablação rigorosa que cruza o α_space aprendido — só ChromaRouteFiLMNet (fatorado por
    espaco). Positivo = o espaco k AJUDA. Robusto por contrato: qualquer falha -> None."""
    if getattr(model, "fusion_type", None) != "chromafilm_route":
        return None
    if not hasattr(model, "set_loso_drop") or cfg.num_classes != 2:
        return None
    K = int(getattr(model, "n_chroma_spaces", 0) or 0)
    if K <= 0:
        return None
    try:
        from sklearn.metrics import roc_auc_score
        model.eval()

        def _auc(drop):
            model.set_loso_drop(drop)
            ys, probs, _ = _collect(model, loader, device, None, cfg.tta, amp=cfg.amp,
                                    channels_last=channels_last, num_classes=cfg.num_classes,
                                    amp_dtype=cfg.amp_dtype)
            return float(roc_auc_score(ys, probs))

        auc_full = _auc(None)
        deltas = [auc_full - _auc(k) for k in range(K)]
        model.set_loso_drop(None)
        return deltas
    except Exception as e:
        print(f"    [loso] aviso: ablação LOSO falhou ({e}); seguindo sem ela.")
        try:
            model.set_loso_drop(None)
        except Exception:
            pass
        return None


# --------------------------------------------------------------------------- #
# Resultado agregado
# --------------------------------------------------------------------------- #
@dataclass
class CVResult:
    threshold: float
    seed_aucs: List[float]
    std_seed_auc: float
    test_metrics_mean: Dict[str, float]
    test_metrics_ci95: Dict[str, float]
    per_seed_metrics: Dict[int, Dict[str, float]]
    regression: Optional[Dict[str, float]] = None
    n_seeds: int = 0
    n_folds: int = 0
    # AUC de VALIDAÇÃO (pooled_oof) — critério de seleção leakage-free da Fase 2.
    # ``oof_auc`` = AUC nas predições de val juntadas entre folds/seeds (escalar, estável);
    # ``oof_seed_aucs`` = 1 AUC de val por seed (p/ desempate/estabilidade). None se multiclasse.
    oof_auc: Optional[float] = None
    oof_seed_aucs: Optional[List[float]] = None
    # --- Agregação por PACIENTE (emenda 2026-07-25) ------------------------------
    # A unidade clínica é o bebê, não a imagem (NeoJaundice ~2,8 imgs/bebê). Estritamente
    # aditivo: não altera treino, seleção nem o limiar das métricas-imagem acima.
    # ``threshold_patient`` sai SÓ do val agregado por paciente (regra #2).
    # Na NJN cada imagem é seu próprio pseudo-paciente -> estas métricas coincidem com as
    # de imagem (degenerado e inócuo).
    threshold_patient: Optional[float] = None
    test_metrics_patient_mean: Optional[Dict[str, float]] = None
    test_metrics_patient_ci95: Optional[Dict[str, float]] = None
    per_seed_patient_metrics: Optional[Dict[int, Dict[str, float]]] = None
    seed_aucs_patient: Optional[List[float]] = None
    std_seed_auc_patient: Optional[float] = None
    # Dump de predições por imagem (val+test). NÃO vai para o marcador JSON: é gravado
    # em .npz ao lado dele por train.py, e é o insumo do re-scoring offline (preds.py).
    pred_dump: Optional["PredDump"] = field(default=None, repr=False)
    # Interpretabilidade: α médio por espaço + perfil ‖Δγ_l‖,‖Δβ_l‖ por estágio
    # (profundidade relativa 0–1, só ChromaFiLM). Em ``adapter_v2`` o α vem do _SpaceGate
    # e indexa TODOS os espaços (RGB incluído); em ChromaFiLM, só os crômicos — por isso
    # ``alpha_space_names`` é obrigatório para ler o vetor, e ``alpha_scope`` registra qual
    # das duas semânticas está gravada.
    alpha_space: Optional[List[float]] = None
    alpha_space_names: Optional[List[str]] = None
    alpha_scope: Optional[str] = None
    film_profile: Optional[List[List[float]]] = None
    # Mapas de atenção do CCAT (Fig 5). Como o dump de predições, NÃO vai para o marcador:
    # é gravado em .npz ao lado dele por train.py.
    attn_dump: Optional["attnmod.AttnDump"] = field(default=None, repr=False)
    # ΔAUC por espaco crômico (leave-one-space-out) — só chromafilm_route. Cruza o α_space.
    loso_delta_auc: Optional[List[float]] = None
    loso_names: Optional[List[str]] = None


def _safe_auc(ys: List[int], probs: List[float]) -> float:
    """ROC-AUC robusta a classe única/entrada vazia (devolve NaN nesses casos)."""
    if not ys or len(set(ys)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(ys, probs))
    except Exception:
        return float("nan")


def _mean_ci(values: List[float]) -> Tuple[float, float]:
    a = np.asarray([v for v in values if not (v is None or math.isnan(v))], dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    mean = float(a.mean())
    if a.size < 2:
        return mean, 0.0
    from scipy import stats as sstats
    sem = sstats.sem(a)
    h = float(sem * sstats.t.ppf(0.975, a.size - 1)) if sem > 0 else 0.0
    return mean, h


def run_cv(cfg: ExperimentConfig, hp: HParams, device, seeds: List[int],
           epochs: int, patience: int, verbose: bool = False) -> CVResult:
    """Agrega com pooled_oof (limiar só no val) sobre len(seeds) repetições.

    * ``split_source='kfold'`` (default): 5-fold × len(seeds) — o split varia com o seed.
    * ``split_source='frozen'``: split canônico ÚNICO 70/10/20 (dataset/<DS>/splits/),
      IDÊNTICO para todos os seeds; os seeds variam só a estocasticidade de treino
      (init/augment/ordem). ``std_seed_auc`` continua sendo o gate de estabilidade.
    """
    k = cfg.cv_folds
    mode = cfg.split_mode
    multitask = cfg.multitask
    frozen = cfg.split_source == "frozen"
    # No modo congelado carrega samples + split UMA vez (o mesmo p/ todos os seeds).
    fz_samples = fz_classes = fz_split = None
    if frozen:
        fz_samples, fz_classes = splits.load_samples(cfg)
        fz_split = splits.load_frozen_split(cfg, fz_samples)
    n_folds_report = 1 if frozen else k

    pooled_val: Dict[str, list] = {"y": [], "p": []}      # p/ limiar (binário)
    per_seed: Dict[int, Dict[str, list]] = {}
    reg_true: List[float] = []
    reg_pred: List[float] = []
    # acumuladores de interpretabilidade (ChromaFiLM): 1 vetor por fold×seed
    alpha_accum: List[List[float]] = []
    alpha_names_seen: Optional[List[str]] = None
    film_accum: Dict[int, List[Tuple[float, float]]] = {}
    loso_accum: List[List[float]] = []
    n_sites_seen = 0
    chroma_names = [c for c in cfg.colorspaces if str(c).upper() != "RGB"]
    # Mapas de atenção do CCAT: registros de todos os folds×seeds (cota por bucket).
    attn_recs: List[dict] = []
    attn_names: Optional[List[str]] = None
    # Pesos do modelo e gate por imagem (explainpack): só com cfg.save_explain.
    explain_recs: List[dict] = []
    explain_paths: List[Path] = []
    # Linhas do dump de predições: (split, seed, fold, path, patient_id, y_true, y_prob).
    # Só binário — a agregação por paciente de multiclasse tem API própria e não é usada
    # em nenhuma célula da campanha atual.
    pred_rows: List[Tuple] = []

    for seed in seeds:
        _seed_everything(seed)
        cfg.seed = seed
        if frozen:
            samples, classes, folds = fz_samples, fz_classes, [fz_split]
        else:
            samples, folds, classes = splits.make_kfold(cfg, seed, k=k, mode=mode)
        per_seed[seed] = {"y": [], "p": [], "yv": [], "pv": []}
        for fi in range(len(folds)):
            cfg.fold = fi
            tr, va, te = folds[fi]
            bundle = build_dataloaders_from_split(
                cfg, samples, tr, va, te, classes, augment=hp.augment,
                da_strength=hp.da_strength, batch_size=hp.batch_size,
                limit_per_split=cfg.limit_per_split, multitask=multitask)
            channels_last = (device.type == "cuda" and cfg.spec.family == "cnn")
            try:
                model = build_model(
                    cfg.spec, bundle.n_channels, fusion=cfg.fusion, activation=hp.activation,
                    n_unfreeze=hp.n_unfreeze, colorspaces=cfg.colorspaces,
                    adapter_init=cfg.adapter_init, hue_circular=cfg.hue_circular,
                    num_classes=cfg.num_classes, backbone_mode=cfg.backbone_mode,
                    film_mode=cfg.film_mode, lora_rank=cfg.lora_rank,
                    multitask=multitask).to(device)
                model = _train_one_fold(model, bundle, device, cfg, hp, epochs, patience,
                                        multitask, cfg.swa, cfg.ema, verbose=verbose)
                yv, pv, _ = _collect_preds(model, bundle.val_loader, device, cfg, channels_last)
                yt, pt, tsb_pred = _collect_preds(model, bundle.test_loader, device, cfg, channels_last)
                tsb_mean, tsb_std = bundle.tsb_mean, bundle.tsb_std  # captura antes do del
                # Paths na ORDEM das predições (sem shuffle/sampler em val/test). O helper
                # devolve None se a ordem não casar — nesse caso NÃO gravamos nada, em vez
                # de arriscar atribuir uma prob à imagem errada.
                if cfg.num_classes == 2:
                    pid_of = bundle.path_to_pid
                    for split_name, ys_, ps_, loader in (("val", yv, pv, bundle.val_loader),
                                                         ("test", yt, pt, bundle.test_loader)):
                        paths_ = _aligned_paths(loader, ys_)
                        if paths_ is None:
                            print(f"    [preds] aviso: ordem não confere em {split_name} "
                                  f"(seed {seed}, fold {fi}) — dump desta partição pulado.")
                            continue
                        pred_rows.extend(
                            (split_name, int(seed), int(fi), str(pa), str(pid_of.get(str(pa), pa)),
                             int(yy), float(pp))
                            for pa, yy, pp in zip(paths_, ys_, ps_))
                a_vec, a_names, film_prof, n_sites = _collect_interpretability(
                    model, bundle.test_loader, device, cfg, channels_last)
                if a_vec is not None:
                    alpha_accum.append(a_vec)
                    alpha_names_seen = a_names
                if getattr(cfg, "save_explain", False):
                    sp = explainmod.save_state(cfg, seed, fi, model)
                    if sp is not None:
                        explain_paths.append(sp)
                    _gp = _aligned_paths(bundle.test_loader, yt)
                    if _gp is not None:
                        explain_recs.extend(explainmod.collect_gate(
                            model, bundle.test_loader, device,
                            _amp_ctx(device, cfg.amp, cfg.amp_dtype), channels_last,
                            _gp, yt, seed=seed, fold=fi))
                ar, an = _collect_attn_maps(model, bundle.test_loader, device, cfg,
                                            channels_last, seed=seed, fold=fi)
                if ar:
                    attn_recs.extend(ar)
                    attn_names = an
                if film_prof:
                    n_sites_seen = max(n_sites_seen, n_sites)
                    for i, (dg, db) in film_prof.items():
                        film_accum.setdefault(i, []).append((dg, db))
                loso_vec = _collect_loso(model, bundle.test_loader, device, cfg, channels_last)
                if loso_vec is not None:
                    loso_accum.append(loso_vec)
            finally:
                _shutdown_loaders(bundle)
                del bundle
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if cfg.num_classes == 2:
                pooled_val["y"].extend(yv); pooled_val["p"].extend(pv)
                per_seed[seed]["yv"].extend(yv); per_seed[seed]["pv"].extend(pv)
            per_seed[seed]["y"].append(yt); per_seed[seed]["p"].append(pt)
            if multitask and tsb_pred:
                # des-normaliza com as stats do TREINO do fold (simetria; sem vazamento)
                reg_pred.extend(denormalize_tsb(tsb_pred, tsb_mean, tsb_std).tolist())
                reg_true.extend([samples[i].tsb for i in te][:len(tsb_pred)])
            if verbose:
                print(f"  seed {seed} fold {fi}: n_test={len(yt)}")

    # --- pooled_oof: um limiar escolhido SÓ no val (binário) ---
    if cfg.num_classes == 2:
        thr, _ = _best_threshold(pooled_val["y"], pooled_val["p"], cfg.objective)
    else:
        thr = 0.5  # multiclasse decide por argmax

    # --- AUC de VALIDAÇÃO (OOF): critério de seleção de colorset da Fase 2 (leakage-free) ---
    # NUNCA usa o test. Selecionar o braço por este número não olha o teste (guardrail #2).
    if cfg.num_classes == 2:
        oof_auc = _safe_auc(pooled_val["y"], pooled_val["p"])
        oof_seed_aucs = [_safe_auc(per_seed[s]["yv"], per_seed[s]["pv"]) for s in seeds]
    else:
        oof_auc, oof_seed_aucs = None, None

    # --- 1 AUC por seed (test agrupado dos folds) + métricas ao limiar ---
    seed_aucs: List[float] = []
    per_seed_metrics: Dict[int, Dict[str, float]] = {}
    metric_lists: Dict[str, List[float]] = {}
    for seed in seeds:
        y = [v for fold in per_seed[seed]["y"] for v in fold]
        if cfg.num_classes == 2:
            p = [v for fold in per_seed[seed]["p"] for v in fold]
            m = compute_metrics(y, y_prob=p, threshold=thr)
        else:
            p = [v for fold in per_seed[seed]["p"] for v in fold]
            m = compute_metrics_multiclass(y, p, cfg.num_classes)
        per_seed_metrics[seed] = m
        seed_aucs.append(m.get("roc_auc", float("nan")))
        for key, val in m.items():
            if isinstance(val, (int, float)):
                metric_lists.setdefault(key, []).append(float(val))

    std_seed_auc = float(np.nanstd(np.asarray(seed_aucs, dtype=np.float64))) if seed_aucs else float("nan")
    mean_metrics = {k2: _mean_ci(v)[0] for k2, v in metric_lists.items()}
    ci_metrics = {k2: _mean_ci(v)[1] for k2, v in metric_lists.items()}

    # --- agregação por PACIENTE (re-scoring das MESMAS predições; não re-treina nada) ---
    pred_dump = PredDump.from_rows(pred_rows)
    thr_patient = None
    pat_mean = pat_ci = pat_per_seed = None
    seed_aucs_pat = std_seed_auc_pat = None
    if pred_dump.n_rows and cfg.num_classes == 2:
        # limiar-paciente SÓ do val (mesmo objetivo do limiar-imagem) — regra #2
        thr_patient = patient_threshold_from_val(pred_dump, objective=cfg.objective)
        pat_per_seed = patient_metrics_from_dump(pred_dump, threshold=thr_patient, split="test")
        if pat_per_seed:
            pat_lists: Dict[str, List[float]] = {}
            for s in seeds:
                for key, val in (pat_per_seed.get(s) or {}).items():
                    if isinstance(val, (int, float)):
                        pat_lists.setdefault(key, []).append(float(val))
            pat_mean = {k2: _mean_ci(v)[0] for k2, v in pat_lists.items()}
            pat_ci = {k2: _mean_ci(v)[1] for k2, v in pat_lists.items()}
            seed_aucs_pat = [(pat_per_seed.get(s) or {}).get("roc_auc", float("nan"))
                             for s in seeds]
            std_seed_auc_pat = float(np.nanstd(np.asarray(seed_aucs_pat, dtype=np.float64)))

    regression = None
    if multitask and reg_true:
        regression = compute_regression_metrics(reg_true, reg_pred)

    # --- interpretabilidade: média entre folds×seeds ---
    alpha_space = None
    alpha_space_names = None
    alpha_scope = None
    if explain_recs or explain_paths:
        _names = alpha_names_seen or [str(c) for c in cfg.colorspaces]
        _gp = explainmod.save_gate(cfg, explain_recs, _names)
        print(f"    [explain] {len(explain_paths)} state_dicts"
              + (f" | gate por imagem: {len(explain_recs)} linhas -> {_gp.name}"
                 if _gp is not None else " | gate por imagem: nada gravado"))

    if alpha_accum:
        alpha_space = np.asarray(alpha_accum, dtype=np.float64).mean(axis=0).tolist()
        # Os nomes vêm do coletor (que sabe se o vetor inclui o RGB) — nunca são
        # re-derivados aqui, para não desalinhar o gate do adapter_v2.
        alpha_space_names = (alpha_names_seen or chroma_names)[:len(alpha_space)]
        alpha_scope = _ALPHA_FUSIONS.get(cfg.fusion)
    film_profile = None
    if film_accum:
        ns = n_sites_seen or (max(film_accum) + 1)
        film_profile = []
        for i in sorted(film_accum):
            dgs = [d for d, _ in film_accum[i]]
            dbs = [b for _, b in film_accum[i]]
            rel = round(i / (ns - 1), 4) if ns > 1 else 0.0
            film_profile.append([rel, float(np.mean(dgs)), float(np.mean(dbs))])
    loso_delta_auc = None
    loso_names = None
    if loso_accum:
        loso_delta_auc = np.asarray(loso_accum, dtype=np.float64).mean(axis=0).tolist()
        loso_names = chroma_names[:len(loso_delta_auc)]

    return CVResult(
        threshold=float(thr), seed_aucs=seed_aucs, std_seed_auc=std_seed_auc,
        test_metrics_mean=mean_metrics, test_metrics_ci95=ci_metrics,
        per_seed_metrics=per_seed_metrics, regression=regression,
        n_seeds=len(seeds), n_folds=n_folds_report,
        oof_auc=oof_auc, oof_seed_aucs=oof_seed_aucs,
        threshold_patient=thr_patient, test_metrics_patient_mean=pat_mean,
        test_metrics_patient_ci95=pat_ci, per_seed_patient_metrics=pat_per_seed,
        seed_aucs_patient=seed_aucs_pat, std_seed_auc_patient=std_seed_auc_pat,
        pred_dump=pred_dump,
        alpha_space=alpha_space, alpha_space_names=alpha_space_names,
        alpha_scope=alpha_scope, film_profile=film_profile,
        attn_dump=(attnmod.AttnDump.from_records(attn_recs, attn_names or [])
                   if attn_recs else None),
        loso_delta_auc=loso_delta_auc, loso_names=loso_names,
    )
