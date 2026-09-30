"""
engine.py
=========

Motor unico de treino/validacao/teste — serve CNNs e ViTs sem alteracao.

Responsabilidades:

* ``evaluate`` — roda o modelo num loader e devolve metricas + perda, alem das
  probabilidades por amostra. Suporta **TTA** (media das probabilidades da imagem
  e do seu flip horizontal) e um **limiar de decisao** configuravel.

* ``train_loop`` — treino com **early stopping pela metrica-objetivo** (default
  ``f1_macro``, robusta a desbalanceamento) e **scheduler cosseno + warmup**.
  Suporta **AMP bf16** e **channels_last** (CNNs) para acelerar em GPUs Ada/Ampere.
  Guarda o melhor estado; aceita ``on_epoch`` para o pruning do Optuna.

* ``select_thresholds`` — escolhe, **no validacao** (uma unica inferencia, sem
  olhar o teste), DOIS limiares sobre ``p(jaundice)``: o que maximiza a
  metrica-objetivo (operacao clinica) e o que maximiza a accuracy (operacao
  comparavel ao baseline, que decide por argmax).

* ``fit`` — orquestra UMA configuracao de ponta a ponta, com **fallback de OOM**
  (reduz batch + acumula gradiente). Balanceamento de classes vem do
  WeightedRandomSampler (em :mod:`data`), logo a loss e ``CrossEntropy`` simples.
"""

from __future__ import annotations

import copy
import gc
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import ExperimentConfig
from .data import build_dataloaders
from .metrics import (aggregate_by_patient, compute_metrics, compute_patient_metrics,
                      compute_metrics_multiclass, compute_patient_metrics_multiclass)
from .models import build_model

OOM_BATCH_FLOOR = 2  # menor batch antes de desistir


@dataclass
class HParams:
    lr: float = 1e-4
    optimizer: str = "adam"          # adam | adamax
    activation: str = "none"         # head linear (logits puros)
    n_unfreeze: int = 0              # 0 | 10 | 50 | 100 | -1 (backbone inteiro)
    fusion: str = "adapter"          # adapter | adapter_v2 | inflate
    augment: bool = False
    da_strength: float = 1.0
    batch_size: int = 32

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _amp_ctx(device, enabled: bool, dtype: str = "bf16"):
    """Autocast (bf16 padrao ou fp16); no-op em CPU.

    bf16 dispensa GradScaler (mesmo range do fp32). fp16 exige GradScaler (ver
    ``train_loop``). bf16 cai para fp16 se a GPU nao suportar bf16."""
    if not (enabled and device.type == "cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False)
    want = torch.float16 if dtype == "fp16" else torch.bfloat16
    if want is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        want = torch.float16
    return torch.autocast(device_type="cuda", dtype=want, enabled=True)


def _fp16_scaler(cfg_amp: bool, amp_dtype: str, device):
    """GradScaler ativo apenas para fp16 em CUDA (bf16/CPU -> no-op)."""
    enabled = bool(cfg_amp and amp_dtype == "fp16" and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)  # API nova
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)     # fallback torch antigo


def _make_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamax":
        return torch.optim.Adamax(params, lr=lr)
    raise ValueError(f"Otimizador desconhecido: {name}")


def build_param_groups(model, lr: float, backbone_lr_mult: float = 0.1) -> list:
    """Differential LR (portado da v4): backbone com LR reduzido, fusao+cabeca com LR
    cheio. Um LR unico destruiria os pesos pre-treinados do ViT antes do adapter
    aprender a misturar as cores. So inclui params com ``requires_grad`` (respeita
    ``set_finetune_depth``). Devolve 1 grupo se o backbone estiver congelado.

    Na fusao "inflate" a 1a convolucao recem-inicializada vive DENTRO do backbone;
    ela e tratada como parte da fusao (LR cheio), nao do backbone pre-treinado, pois
    parte do zero e precisa aprender rapido."""
    # ChromaFiLMNet define seus proprios grupos (film/heads/lora em LR cheio;
    # backbone congelado fica fora do otimizador) — delega ao metodo do modelo.
    if hasattr(model, "build_param_groups"):
        return model.build_param_groups(lr, backbone_lr_mult)
    inflated = getattr(model, "_inflated_conv", None)
    inflated_ids = {id(p) for p in inflated.parameters()} if inflated is not None else set()
    bb = [p for p in model.backbone.parameters()
          if p.requires_grad and id(p) not in inflated_ids]
    head_fusion = ([p for p in model.fusion.parameters() if p.requires_grad] +
                   [p for p in model.head.parameters() if p.requires_grad] +
                   ([p for p in inflated.parameters() if p.requires_grad] if inflated is not None else []))
    groups = [{"params": head_fusion, "lr": lr}]
    if bb:
        groups.append({"params": bb, "lr": lr * backbone_lr_mult})
    return groups


def _cosine_warmup_scheduler(optimizer, epochs: int, warmup_frac: float = 0.1):
    """LR: warmup linear (~10% das epocas) seguido de decaimento cosseno ate ~0."""
    warmup = max(1, int(round(warmup_frac * epochs)))

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def _collect(model: nn.Module, loader, device, criterion: Optional[nn.Module], tta: bool,
             amp: bool = False, channels_last: bool = False, num_classes: int = 2,
             amp_dtype: str = "bf16") -> Tuple[List[int], List, float]:
    """Roda o modelo no loader e devolve (y_true, probs, perda_media).

    Binario (num_classes==2): ``probs`` = lista de p(classe 1) (compat com a v1).
    Multiclasse: ``probs`` = lista de vetores [K] (softmax completo)."""
    model.eval()
    ys: List[int] = []
    probs: List = []
    total_loss, n_batches = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if channels_last:
            imgs = imgs.to(memory_format=torch.channels_last)
        with _amp_ctx(device, amp, amp_dtype):
            logits = model(imgs)
            if criterion is not None:
                total_loss += criterion(logits, labels).item(); n_batches += 1
            p = torch.softmax(logits.float(), dim=1)
            if tta:  # media com o flip horizontal
                p = 0.5 * (p + torch.softmax(model(torch.flip(imgs, dims=[3])).float(), dim=1))
            p = p[:, 1] if num_classes == 2 else p
        ys.extend(labels.cpu().tolist())
        probs.extend(p.cpu().tolist())
    return ys, probs, total_loss / max(n_batches, 1)


def _aligned_paths(loader, ys: List[int]) -> Optional[List[str]]:
    """Paths das amostras na ORDEM de ``ys`` (== ordem do loader sem shuffle/sampler).

    Devolve ``None`` se a ordem nao casar (seguranca: evita atribuir prob a imagem
    errada). Val/test usam ``shuffle=False`` sem sampler, entao
    ``loader.dataset.samples`` esta alinhado com a inferencia."""
    samples = getattr(loader.dataset, "samples", None)
    if samples is None:
        return None
    if [int(l) for _, l in samples] != [int(y) for y in ys]:
        return None
    return [p for p, _ in samples]


def evaluate(model, loader, device, criterion: Optional[nn.Module] = None,
             threshold: float = 0.5, tta: bool = False,
             amp: bool = False, channels_last: bool = False, num_classes: int = 2,
             amp_dtype: str = "bf16") -> Tuple[Dict[str, float], float]:
    ys, probs, avg_loss = _collect(model, loader, device, criterion, tta,
                                   amp=amp, channels_last=channels_last,
                                   num_classes=num_classes, amp_dtype=amp_dtype)
    if num_classes == 2:
        metrics = compute_metrics(ys, y_prob=probs, threshold=threshold)
    else:
        metrics = compute_metrics_multiclass(ys, probs, num_classes)
    return metrics, avg_loss


def _best_threshold(ys, probs, objective: str) -> Tuple[float, Dict[str, float]]:
    best_t, best_v, best_m = 0.5, -1.0, {}
    for t in np.linspace(0.05, 0.95, 19):
        m = compute_metrics(ys, y_prob=probs, threshold=float(t))
        if m[objective] > best_v:
            best_v, best_t, best_m = m[objective], float(t), m
    return best_t, best_m


def select_threshold(model, loader, device, objective: str = "f1_macro", tta: bool = False
                     ) -> Tuple[float, Dict[str, float]]:
    """Escolhe no VAL o limiar que maximiza a metrica-objetivo. Sem vazamento."""
    ys, probs, _ = _collect(model, loader, device, None, tta)
    return _best_threshold(ys, probs, objective)


def select_thresholds(model, loader, device, objective: str = "f1_macro", tta: bool = False,
                      amp: bool = False, channels_last: bool = False
                      ) -> Tuple[float, Dict[str, float], float, Dict[str, float]]:
    """Dual threshold no VAL (uma unica passada de inferencia, sem vazamento):

    * limiar que maximiza a metrica-objetivo (default f1_macro — operacao clinica);
    * limiar que maximiza a ACCURACY (operacao comparavel ao baseline, que usa argmax).

    Devolve (thr_obj, val_metrics_obj, thr_acc, val_metrics_acc)."""
    ys, probs, _ = _collect(model, loader, device, None, tta,
                            amp=amp, channels_last=channels_last)
    t_obj, m_obj = _best_threshold(ys, probs, objective)
    t_acc, m_acc = _best_threshold(ys, probs, "accuracy")
    return t_obj, m_obj, t_acc, m_acc


def train_loop(model, train_loader, val_loader, device, criterion, optimizer,
               epochs: int, patience: int, accum_steps: int = 1, objective: str = "f1_macro",
               scheduler=None, on_epoch: Optional[Callable[[int, float], None]] = None,
               verbose: bool = False, amp: bool = False, channels_last: bool = False,
               num_classes: int = 2, amp_dtype: str = "bf16", scaler=None
               ) -> Tuple[Dict[str, float], dict, list]:
    """Treina com early stopping pela metrica-objetivo; devolve
    (best_val_metrics, best_state_cpu, history).

    ``scaler`` (GradScaler) ativo so para fp16; bf16/CPU usam backward/step direto."""
    best_metric = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_val_metrics: Dict[str, float] = {}
    history = []
    epochs_no_improve = 0
    use_scaler = scaler is not None and scaler.is_enabled()

    def _step():
        if use_scaler:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        did_step = False
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)
            with _amp_ctx(device, amp, amp_dtype):
                loss = criterion(model(imgs), labels) / accum_steps
            (scaler.scale(loss) if use_scaler else loss).backward()
            did_step = True
            if (i + 1) % accum_steps == 0:
                _step(); did_step = False
        if did_step:  # flush residual do ultimo micro-batch incompleto
            _step()
        if scheduler is not None:
            scheduler.step()

        val_metrics, val_loss = evaluate(model, val_loader, device, criterion,
                                         amp=amp, channels_last=channels_last,
                                         num_classes=num_classes, amp_dtype=amp_dtype)
        monitor = val_metrics[objective]
        history.append({"epoch": epoch + 1, "val_loss": val_loss,
                        **{k: v for k, v in val_metrics.items() if not isinstance(v, list)}})
        if verbose:
            print(f"  epoch {epoch+1:3d}/{epochs} | val_loss {val_loss:.4f} "
                  f"| val_{objective} {monitor:.3f} | val_acc {val_metrics['accuracy']:.2f}%")

        if on_epoch is not None:
            on_epoch(epoch, monitor)  # pode levantar optuna.TrialPruned

        if monitor > best_metric:
            best_metric = monitor
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  early stopping na epoca {epoch+1} (sem melhora ha {patience}).")
                break

    best_state = {k: v.detach().cpu() for k, v in best_state.items()}
    return best_val_metrics, best_state, history


@dataclass
class FitResult:
    val_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]]
    best_state: dict
    history: list
    threshold: float
    effective_batch: int
    n_channels: int
    sizes: Tuple[int, int, int]
    classes: Tuple[str, ...]
    # Dual threshold: operacao orientada a accuracy (comparavel ao baseline).
    threshold_acc: float = 0.5
    val_metrics_acc: Optional[Dict[str, float]] = None
    test_metrics_acc: Optional[Dict[str, float]] = None
    # Avaliacao por paciente (unidade clinica): limiar escolhido no val agregado por
    # paciente e aplicado ao test agregado. None se o dataset nao tem id de paciente.
    threshold_patient: float = 0.5
    val_metrics_patient: Optional[Dict[str, float]] = None
    test_metrics_patient: Optional[Dict[str, float]] = None


def fit(cfg: ExperimentConfig, hp: HParams, device, epochs: int, patience: int,
        return_test: bool = False, on_epoch=None, verbose: bool = False) -> FitResult:
    """Treina UMA configuracao com fallback de OOM (reduz batch + acumula)."""
    batch = hp.batch_size
    accum = 1
    while True:
        try:
            return _fit_once(cfg, hp, device, epochs, patience, batch, accum,
                             return_test, on_epoch, verbose)
        except torch.cuda.OutOfMemoryError:
            _free_cuda()
            if batch // 2 < OOM_BATCH_FLOOR:
                raise RuntimeError(
                    f"OOM mesmo com batch={batch} (piso={OOM_BATCH_FLOOR}). "
                    f"Use --gpu com mais memoria ou um backbone menor."
                )
            accum *= 2
            batch //= 2
            print(f"[engine] OOM -> reduzindo batch para {batch} (accum={accum}, "
                  f"batch efetivo={batch*accum}) e refazendo o treino.")


def _shutdown_loaders(bundle) -> None:
    """Desliga explicitamente os workers persistentes dos DataLoaders do bundle.

    Critico no HPO: trials PODADOS (MedianPruner) ou que falham abandonam o bundle;
    sem desligar os workers 'spawn'/persistentes, os pipes/descritores vazam e ao
    longo de dezenas de trials estouram em ``OSError: [Errno 24] Too many open
    files``. Idempotente e tolerante a erros."""
    for name in ("train_loader", "val_loader", "test_loader"):
        loader = getattr(bundle, name, None)
        it = getattr(loader, "_iterator", None) if loader is not None else None
        if it is not None:
            try:
                it._shutdown_workers()
            except Exception:
                pass
            try:
                loader._iterator = None
            except Exception:
                pass


def _fit_once(cfg, hp, device, epochs, patience, batch, accum, return_test, on_epoch, verbose):
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # input de tamanho fixo
    bundle = build_dataloaders(cfg, augment=hp.augment, da_strength=hp.da_strength,
                               batch_size=batch, limit_per_split=cfg.limit_per_split)
    # try/finally garante o desligamento dos workers em QUALQUER saida (sucesso,
    # OOM, TrialPruned), evitando o vazamento de descritores entre trials do HPO.
    model = optimizer = None
    try:
        K = cfg.num_classes
        # multitask=False no HPO: o objetivo do Optuna e a metrica de CLASSIFICACAO;
        # a regressao TSB entra so no cv.py (retrain final), com o alvo no loader.
        model = build_model(cfg.spec, bundle.n_channels, fusion=hp.fusion,
                            activation=hp.activation, n_unfreeze=hp.n_unfreeze,
                            colorspaces=cfg.colorspaces, adapter_init=cfg.adapter_init,
                            hue_circular=cfg.hue_circular, num_classes=K,
                            backbone_mode=cfg.backbone_mode, film_mode=cfg.film_mode,
                            lora_rank=cfg.lora_rank, multitask=False).to(device)
        # channels_last beneficia CNNs com AMP; ViTs nao ganham.
        channels_last = (device.type == "cuda" and cfg.spec.family == "cnn")
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        amp = cfg.amp
        amp_dtype = cfg.amp_dtype
        # Balanceamento vem do sampler (data.py) -> CrossEntropy simples (sem pesos).
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        # Differential LR: backbone com LR reduzido, fusao+cabeca com LR cheio.
        optimizer = _make_optimizer(
            hp.optimizer, build_param_groups(model, hp.lr, cfg.backbone_lr_mult), hp.lr)
        scheduler = _cosine_warmup_scheduler(optimizer, epochs)
        scaler = _fp16_scaler(amp, amp_dtype, device)

        val_metrics, best_state, history = train_loop(
            model, bundle.train_loader, bundle.val_loader, device, criterion, optimizer,
            epochs=epochs, patience=patience, accum_steps=accum, objective=cfg.objective,
            scheduler=scheduler, on_epoch=on_epoch, verbose=verbose,
            amp=amp, channels_last=channels_last, num_classes=K,
            amp_dtype=amp_dtype, scaler=scaler,
        )

        # Recarrega o melhor estado para ajuste de limiar e teste.
        model.load_state_dict(best_state)
        threshold, threshold_acc, threshold_patient = 0.5, 0.5, 0.5
        val_metrics_acc = val_metrics_patient = None
        test_metrics = test_metrics_acc = test_metrics_patient = None

        pid_of = bundle.path_to_pid  # path -> patient_id (real ou pseudo pHash NJN)

        def pids(paths):
            return [pid_of.get(p, p) for p in paths] if paths is not None else None

        if K > 2:
            # MULTICLASSE (NeoJaundice3): decisao por argmax, sem limiar. Agregacao por
            # paciente = media do vetor de probabilidades -> argmax.
            ys_v, probs_v, _ = _collect(model, bundle.val_loader, device, None, cfg.tta,
                                        amp=amp, channels_last=channels_last, num_classes=K,
                                        amp_dtype=amp_dtype)
            val_metrics = compute_metrics_multiclass(ys_v, probs_v, K)
            val_paths = _aligned_paths(bundle.val_loader, ys_v)
            if val_paths is not None:
                val_metrics_patient = compute_patient_metrics_multiclass(
                    val_paths, ys_v, probs_v, K, patient_ids=pids(val_paths))
            if return_test:
                ys_t, probs_t, _ = _collect(model, bundle.test_loader, device, None, cfg.tta,
                                            amp=amp, channels_last=channels_last, num_classes=K,
                                            amp_dtype=amp_dtype)
                test_metrics = compute_metrics_multiclass(ys_t, probs_t, K)
                test_paths = _aligned_paths(bundle.test_loader, ys_t)
                if test_paths is not None:
                    test_metrics_patient = compute_patient_metrics_multiclass(
                        test_paths, ys_t, probs_t, K, patient_ids=pids(test_paths))
        else:
            # BINARIO (v1 original, inalterado salvo agregacao por pseudo-paciente).
            if cfg.tune_threshold:
                ys_v, probs_v, _ = _collect(model, bundle.val_loader, device, None, cfg.tta,
                                            amp=amp, channels_last=channels_last, amp_dtype=amp_dtype)
                threshold, val_metrics = _best_threshold(ys_v, probs_v, cfg.objective)
                threshold_acc, val_metrics_acc = _best_threshold(ys_v, probs_v, "accuracy")
                val_paths = _aligned_paths(bundle.val_loader, ys_v)
                if val_paths is not None:
                    _, ylab_v, yprob_v = aggregate_by_patient(val_paths, ys_v, probs_v,
                                                              patient_ids=pids(val_paths))
                    threshold_patient, val_metrics_patient = _best_threshold(ylab_v, yprob_v, cfg.objective)
                    val_metrics_patient["n_patients"] = len(ylab_v)
            if return_test:
                ys_t, probs_t, _ = _collect(model, bundle.test_loader, device, None, cfg.tta,
                                            amp=amp, channels_last=channels_last, amp_dtype=amp_dtype)
                test_metrics = compute_metrics(ys_t, y_prob=probs_t, threshold=threshold)
                test_metrics_acc = compute_metrics(ys_t, y_prob=probs_t, threshold=threshold_acc)
                test_paths = _aligned_paths(bundle.test_loader, ys_t)
                if test_paths is not None:
                    test_metrics_patient = compute_patient_metrics(
                        test_paths, ys_t, probs_t, threshold=threshold_patient,
                        patient_ids=pids(test_paths))

        result = FitResult(
            val_metrics=val_metrics, test_metrics=test_metrics, best_state=best_state,
            history=history, threshold=threshold, effective_batch=batch * accum,
            n_channels=bundle.n_channels, sizes=bundle.sizes, classes=bundle.classes,
            threshold_acc=threshold_acc, val_metrics_acc=val_metrics_acc,
            test_metrics_acc=test_metrics_acc,
            threshold_patient=threshold_patient, val_metrics_patient=val_metrics_patient,
            test_metrics_patient=test_metrics_patient,
        )
        return result
    finally:
        _shutdown_loaders(bundle)
        del model, optimizer, bundle
        _free_cuda()


def _free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
