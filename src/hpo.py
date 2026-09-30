"""
hpo.py
======

Otimizacao de hiperparametros (HO) com **Optuna**, executada *por dentro* de cada
unidade de experimento (um subconjunto de cor por vez).

Decisao metodologica
--------------------
O **espaco de cor NAO e** um hiperparametro do Optuna — ele e a variavel
independente do estudo (queremos a tabela comparativa por espaco). O Optuna
otimiza tudo o que esta *em volta*:

* ``lr``           — taxa de aprendizado (loguniform 1e-5..1e-1), como na baseline;
* ``optimizer``    — adam | adamax;
* ``activation``   — fixo "none" (head linear, logits puros para CrossEntropy);
* ``n_unfreeze``   — profundidade de fine-tuning 0/10/50/100/-1, onde -1 =
  backbone INTEIRO (fine-tuning total, condicao usada pelas CNNs da baseline);
* ``augment``      — **Data Augmentation ON/OFF** (sempre otimizado, a pedido) + intensidade;
* ``fusion``       — opcionalmente adapter | inflate (quando ``fusion_as_hparam``).

Busca eficiente: ``TPESampler`` + ``MedianPruner`` (poda trials ruins cedo). Cada
trial treina por ``search_epochs`` com early stopping; o melhor e re-treinado por
``final_epochs`` em :mod:`run_experiment`. Trials que estouram memoria sao podados
(nao derrubam o estudo).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import optuna
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.samplers import TPESampler

from . import engine
from .config import ExperimentConfig
from .engine import HParams

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Datasets aceitam apelidos minúsculos na CLI (o config usa "NeoJaundice"/"NJN").
_DATASET_ALIAS = {"neojaundice": "NeoJaundice", "njn": "NJN",
                  "NeoJaundice": "NeoJaundice", "NJN": "NJN"}


def suggest_hparams(trial: optuna.Trial, cfg: ExperimentConfig) -> HParams:
    augment = trial.suggest_categorical("augment", [False, True])
    hp = HParams(
        lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True),
        optimizer=trial.suggest_categorical("optimizer", ["adam", "adamax"]),
        # Head LINEAR fixo. Ativacoes (tanh/sigmoid/relu) sobre os logits
        # antes do CrossEntropy distorcem o gradiente e causavam colapsos.
        activation="none",
        # -1 = fine-tuning total (backbone inteiro), como nas CNNs da baseline.
        n_unfreeze=trial.suggest_categorical("n_unfreeze", [0, 10, 50, 100, -1]),
        augment=augment,
        da_strength=trial.suggest_float("da_strength", 0.3, 1.0) if augment else 0.0,
        batch_size=cfg.batch_size,
        fusion=(trial.suggest_categorical("fusion", ["adapter", "inflate"])
                if cfg.fusion_as_hparam else cfg.fusion),
    )
    return hp


def _objective(trial: optuna.Trial, cfg: ExperimentConfig, device) -> float:
    hp = suggest_hparams(trial, cfg)

    def on_epoch(epoch: int, monitor: float):
        trial.report(monitor, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    try:
        result = engine.fit(cfg, hp, device, epochs=cfg.search_epochs,
                            patience=cfg.patience, return_test=False, on_epoch=on_epoch)
    except RuntimeError as e:
        # OOM irrecuperavel neste trial -> poda em vez de quebrar o estudo.
        if "OOM" in str(e) or "out of memory" in str(e).lower():
            raise optuna.TrialPruned()
        raise
    # Otimiza a metrica-objetivo configurada (default f1_macro).
    return result.val_metrics.get(cfg.objective, 0.0)


def _build_pruner(name: str, search_epochs: int):
    """Fábrica de pruner. ``hyperband`` (§8, ASHA/multi-fidelity) é o do contrato;
    ``median`` fica como fallback barato para smokes."""
    if name == "hyperband":
        # min_resource=1 época; max = orçamento de busca; fator 3 (ASHA clássico).
        return HyperbandPruner(min_resource=1, max_resource=max(1, search_epochs),
                               reduction_factor=3)
    return MedianPruner(n_startup_trials=3, n_warmup_steps=5)


def run_hpo(cfg: ExperimentConfig, device, pruner: str = "median",
            storage: Optional[str] = None,
            study_name: Optional[str] = None) -> Tuple[HParams, optuna.Study]:
    """Roda o estudo Optuna e devolve (melhores_hparams, study).

    ``pruner`` = ``hyperband`` (contrato) | ``median`` (default, retrocompatível com o
    smoke). ``storage`` = URL RDB (ex.: ``sqlite:///local/runs/hpo.db``) para retomada/2 GPUs no
    mesmo estudo; ``load_if_exists`` permite anexar trials a um estudo já criado."""
    # A campanha v7 correu com hpo_sampler_seed=None, logo seed=cfg.seed=42 nos 30
    # estudos: os 10 trials de arranque do TPE são idênticos em todos eles, e 20 dos
    # 30 vencedores saíram desse bloco partilhado. Ver docs/PENDENCIAS.md item 15.
    sampler_seed = cfg.hpo_sampler_seed if cfg.hpo_sampler_seed is not None else cfg.seed
    sampler = TPESampler(seed=sampler_seed)
    pruner_obj = _build_pruner(pruner, cfg.search_epochs)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner_obj,
                                storage=storage, study_name=study_name,
                                load_if_exists=storage is not None)
    study.optimize(lambda t: _objective(t, cfg, device), n_trials=cfg.hpo_trials,
                   gc_after_trial=True, show_progress_bar=False)

    # Reconstroi HParams a partir do melhor trial; se todos falharam, usa default.
    try:
        best = study.best_trial
        params = best.params
        augment = params.get("augment", False)
        best_hp = HParams(
            lr=params["lr"], optimizer=params["optimizer"], activation="none",
            n_unfreeze=params["n_unfreeze"], augment=augment,
            da_strength=params.get("da_strength", 0.0), batch_size=cfg.batch_size,
            fusion=params.get("fusion", cfg.fusion),
        )
    except ValueError:
        print("[hpo] Nenhum trial valido — usando hiperparametros padrao.")
        best_hp = HParams(batch_size=cfg.batch_size, fusion=cfg.fusion)
    return best_hp, study


# --------------------------------------------------------------------------- #
# CLI desacoplada (IMPLEMENTATION.md §8 + EXPERIMENT_MATRIX.md)
# 1× por (backbone, dataset) no baseline RGB → dump results/best_params/<bb>_<ds>.json.
# --------------------------------------------------------------------------- #
def _best_params_path(best_params_dir: str, backbone: str, dataset: str) -> Path:
    """<best_params_dir>/<backbone>_<dataset>.json (default: results/best_params/)."""
    return Path(best_params_dir) / f"{backbone}_{dataset}.json"


def dump_best_params(path: Path, best_hp: HParams, study, *, backbone: str,
                     dataset: str, colorset: str, objective: str, hpo_fold: int,
                     search_epochs: int, code_version: str = "") -> None:
    """Grava o JSON de best_params (escrita atômica tmp→rename).

    ``best_params`` é o dict de HParams congelado, reutilizado por TODA a matriz
    (proibido re-otimizar por colorset). Metadados do estudo entram para auditoria."""
    try:
        best_value = float(study.best_value)
    except Exception:
        best_value = None
    record = {
        "backbone": backbone, "dataset": dataset, "colorset": colorset,
        "objective": objective, "hpo_fold": hpo_fold, "search_epochs": search_epochs,
        "n_trials": len(study.trials), "best_value": best_value,
        "best_params": best_hp.to_dict(),
        "code_version": code_version,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2))
    os.replace(tmp, path)


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description="v7 — HPO desacoplado (Optuna + HyperbandPruner) 1×/(backbone,dataset) "
                    "no baseline RGB; congela results/best_params/<backbone>_<dataset>.json.")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--colorset", default="RGB",
                    help="Baseline RGB (controle). NUNCA re-otimizar por colorset.")
    ap.add_argument("--hpo-trials", type=int, default=20)
    ap.add_argument("--sampler-seed", type=int, default=None,
                    help="Semente do TPESampler. Omitido => cfg.seed (comportamento da "
                         "campanha, que partilhou o bloco de arranque nos 30 estudos).")
    ap.add_argument("--hpo-fold", type=int, default=0,
                    help="Fold de baixa fidelidade (split único 70/10/20 semeado).")
    ap.add_argument("--pruner", default="hyperband", choices=["hyperband", "median"])
    ap.add_argument("--storage", default=None,
                    help="URL RDB compartilhada (ex.: sqlite:///local/runs/hpo.db) p/ retomada.")
    ap.add_argument("--study-name", default=None,
                    help="Default: <backbone>_<dataset> (determinístico p/ retomar).")
    ap.add_argument("--gpu", type=int, default=0)
    # Regime da busca (low-fidelity §8): frozen/LoRA, épocas reduzidas.
    ap.add_argument("--fusion", default="adapter_v2",
                    choices=["adapter", "adapter_v2", "inflate", "stemcnn", "ccat",
                             "chromafilm", "chromafilm_route", "ds_lfn", "naive_concat"])
    ap.add_argument("--backbone-mode", default="lora", choices=["frozen", "lora", "full"])
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--num-classes", type=int, default=2, choices=[2, 3])
    ap.add_argument("--objective", default="roc_auc",
                    help="Métrica-alvo do Optuna (primária = AUC; ou f1_macro).")
    ap.add_argument("--search-epochs", type=int, default=None,
                    help="Épocas por trial (default: ExperimentConfig.search_epochs).")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    # NeoJaundice / NJN pré-proc (baseline: sem calibração — H1/H2 é regime não calibrado).
    ap.add_argument("--wb", default="off", choices=["on", "off"])
    ap.add_argument("--njn-mode", default="full_image", choices=["skin_roi", "full_image"])
    ap.add_argument("--frozen-split", action="store_true",
                    help="busca hparams no split canônico congelado (dataset/<DS>/splits/), "
                         "coerente com o treino --frozen-split. Gere antes com make_splits.")
    # recursos / caminhos
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--data-root", default="dataset")
    ap.add_argument("--results-dir", default="local/runs")
    ap.add_argument("--cache-dir", default="local/cache/roi")
    ap.add_argument("--best-params-dir", default="local/runs/best_params")
    ap.add_argument("--limit-per-split", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="Refaz mesmo que results/best_params/<bb>_<ds>.json já exista.")
    return ap.parse_args(argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = _parse_args(argv)
    dataset = _DATASET_ALIAS.get(args.dataset, args.dataset)

    # Idempotência ANTES de qualquer import pesado/GPU: se o best_params já existe, pula.
    bp_path = _best_params_path(args.best_params_dir, args.backbone, dataset)
    if bp_path.exists() and not args.force:
        print(f"[skip] best_params já existe em {bp_path}. Use --force para refazer.")
        return 0

    # Seleção de GPU antes de qualquer contexto CUDA (init é lazy → honrado no fit).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    import torch
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass

    from .config import parse_colorset

    wb_method = "graypatch" if args.wb == "on" else "off"
    cfg = ExperimentConfig(
        backbone=args.backbone, dataset=dataset,
        colorspaces=parse_colorset(args.colorset), num_classes=args.num_classes,
        fusion=args.fusion, gpu=args.gpu, num_workers=args.num_workers,
        batch_size=args.batch_size, wb_method=wb_method, njn_mode=args.njn_mode,
        amp=not args.no_amp, amp_dtype=args.amp_dtype, use_cache=not args.no_cache,
        data_root=args.data_root, results_dir=args.results_dir, cache_dir=args.cache_dir,
        objective=args.objective, backbone_mode=args.backbone_mode, lora_rank=args.lora_rank,
        hpo_trials=args.hpo_trials, fold=args.hpo_fold, limit_per_split=args.limit_per_split,
        hpo_sampler_seed=args.sampler_seed,
        split_source=("frozen" if args.frozen_split else "kfold"),
    )
    if args.search_epochs is not None:
        cfg.search_epochs = args.search_epochs
    if args.patience is not None:
        cfg.patience = args.patience

    study_name = args.study_name or f"{args.backbone}_{dataset}"
    print("=" * 78)
    print(f"HPO: backbone={args.backbone} dataset={dataset} colorset={cfg.colorset_id} "
          f"fusion={cfg.fusion} mode={cfg.backbone_mode}")
    print(f"  sampler_seed={cfg.hpo_sampler_seed if cfg.hpo_sampler_seed is not None else cfg.seed}"
          f"{'' if cfg.hpo_sampler_seed is None else '  (NÃO é o da campanha)'}")
    print(f"  trials={cfg.hpo_trials} fold={cfg.fold} search_epochs={cfg.search_epochs} "
          f"pruner={args.pruner} objective={cfg.objective}")
    print(f"  storage={args.storage or '(in-memory)'} study={study_name}")
    print("=" * 78)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[aviso] CUDA indisponível — CPU (só smoke).")

    t0 = time.time()
    best_hp, study = run_hpo(cfg, device, pruner=args.pruner, storage=args.storage,
                             study_name=study_name)
    elapsed = (time.time() - t0) / 60

    dump_best_params(bp_path, best_hp, study, backbone=args.backbone, dataset=dataset,
                     colorset=cfg.colorset_id, objective=cfg.objective,
                     hpo_fold=cfg.fold, search_epochs=cfg.search_epochs,
                     code_version=cfg.version)
    try:
        best_value = f"{study.best_value:.4f}"
    except Exception:
        best_value = "n/a"
    print(f"[ok] best_{cfg.objective}={best_value} | {len(study.trials)} trials | "
          f"{elapsed:.1f} min")
    print(f"[ok] best_params salvo em {bp_path}")
    print(f"     {json.dumps(best_hp.to_dict())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
