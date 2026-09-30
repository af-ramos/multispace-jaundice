"""
train.py (NOVO — v6)
====================

CLI de UMA configuração sob **CV 5-fold × N-seed** (chama :mod:`cv`) e grava um
**marcador JSON agregado** (média ± IC95%, std entre seeds p/ o GATE, limiar
``pooled_oof``, métricas por seed, regressão TSB opcional).

Os hiperparâmetros vêm da linha de comando (congelados do HPO da Fase 1a); este
script NÃO roda Optuna por config (isso seria 1 estudo por combo). Uso:

    python -m src.train --dataset neojaundice --backbone efficientnet_b4 \\
        --fusion chromafilm --backbone-mode frozen --film-mode channel \\
        --colors RGB+YCrCb+HSV --wb on --seeds "42 123 456 789 1010" \\
        --folds 5 --swa --gpu 0 --tag f1b_neo_efficientnet_b4

Seleção de GPU via ``--gpu`` ANTES de importar torch; retomada por marcador.
"""

from __future__ import annotations

import argparse
import os
import sys

# aliases amigáveis de dataset (o config usa "NeoJaundice"/"NJN")
_DATASET_ALIAS = {"neojaundice": "NeoJaundice", "njn": "NJN",
                  "NeoJaundice": "NeoJaundice", "NJN": "NJN"}


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="v7 — treino CV (5-fold × N-seed) de 1 config.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--colors", "--colorspaces", dest="colors", default="RGB")
    ap.add_argument("--fusion", default="chromafilm",
                    choices=["adapter", "adapter_v2", "inflate", "stemcnn", "ccat",
                             "chromafilm", "chromafilm_route", "ds_lfn", "naive_concat"])
    ap.add_argument("--num-classes", type=int, default=2, choices=[2, 3])
    # ChromaFiLM / estabilidade
    ap.add_argument("--backbone-mode", default="frozen", choices=["frozen", "lora", "full"])
    ap.add_argument("--film-mode", default="channel", choices=["channel", "spatial"])
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--multitask", action="store_true")
    ap.add_argument("--swa", action="store_true")
    ap.add_argument("--ema", action="store_true")
    # CV
    ap.add_argument("--seeds", default="42 123 456 789 1010")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--split-mode", default="group", choices=["group", "random"])
    ap.add_argument("--frozen-split", action="store_true",
                    help="usa o split canônico ÚNICO 70/10/20 congelado em "
                         "dataset/<DS>/splits/ (mesmo split p/ todos os seeds; gere com make_splits).")
    ap.add_argument("--threshold", dest="threshold_mode", default="pooled_oof",
                    choices=["pooled_oof", "youden", "per_fold"])
    # NeoJaundice pré-proc
    ap.add_argument("--wb", default="on", choices=["on", "off"],
                    help="on=graypatch (calibrado) | off=sem white-balance.")
    ap.add_argument("--roi-lo", type=float, default=0.30)
    ap.add_argument("--roi-hi", type=float, default=0.70)
    # NJN
    ap.add_argument("--njn-mode", default="skin_roi", choices=["skin_roi", "full_image"])
    ap.add_argument("--phash-dist", type=int, default=5)
    # hparams (congelados do HPO / defaults)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optimizer", default="adam", choices=["adam", "adamax"])
    ap.add_argument("--n-unfreeze", type=int, default=0)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--da-strength", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--save-explain", action="store_true",
                    help="grava state_dict por seed/fold e o gate por imagem "
                         "(explainpack). Aditivo: nao altera treino nem metricas.")
    # recursos / caminhos
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--mp-context", default="forkserver", choices=["spawn", "fork", "forkserver"],
                    help="start method dos workers do DataLoader. PADRÃO forkserver: o servidor sobe "
                         "por exec de um interpretador LIMPO (sem CUDA/threads do pai) e os workers "
                         "forkam dele -> sem a corrupção nativa do 'fork' (segfault) e sem o re-import "
                         "concorrente do 'spawn' (SyntaxError). 'fork' é UNSAFE com CUDA já inicializada.")
    ap.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--data-root", default="dataset")
    ap.add_argument("--results-dir", default="local/runs")
    ap.add_argument("--cache-dir", default="local/cache/roi")
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit-per-split", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--refresh-missing-oof", action="store_true",
                    help="Refaz apenas marcadores 'completed' que ainda não têm 'oof_auc' "
                         "(campo de seleção da Fase 2). Idempotente: pula os que já têm.")
    ap.add_argument("--refresh-missing-interp", action="store_true",
                    help="Refaz marcadores 'completed' sem interpretabilidade: sem "
                         "'alpha_space' (gate do adapter_v2 / α do ChromaFiLM) ou, em "
                         "células CCAT, sem o dump de mapas de atenção. Idempotente. "
                         "Como o v7 não persiste pesos, α e mapas só existem se coletados "
                         "DURANTE o run — daí o re-treino. O re-run é determinístico: "
                         "acrescenta campos sem mudar nenhuma métrica já publicada.")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


def _falta_interp(prev: dict, cfg, args) -> bool:
    """O marcador 'completed' está sem a interpretabilidade que ESTA fusão sabe produzir?

    ``adapter_v2``/ChromaFiLM → precisa de ``alpha_space``. ``ccat`` → precisa também do
    ``.npz`` de mapas de atenção EXISTINDO em disco (o campo no marcador não basta: os
    dumps são gitignored e podem não ter vindo no rsync). Fusões sem interpretabilidade
    (ex.: ``naive_concat``) nunca disparam refresh — senão o flag re-rodaria a campanha
    inteira sem produzir nada.
    """
    from pathlib import Path

    fusion = str(cfg.fusion)
    if fusion == "ccat":
        # O CCAT não tem α por espaço (a atenção é por-pixel, não um escalar): exigir
        # alpha_space aqui faria o flag re-rodar a célula para sempre.
        af = prev.get("attn_file")
        return not af or not (Path(args.results_dir) / af).is_file()
    if fusion in ("adapter_v2", "chromafilm", "chromafilm_route"):
        return prev.get("alpha_space") is None
    return False


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = _parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    import json
    import time
    from pathlib import Path

    import torch
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass

    # forkserver: o servidor (interpretador limpo) pré-importa 'data' UMA vez, então
    # cada worker nasce por fork barato — sem re-import concorrente (o que quebrava o
    # 'spawn' em SyntaxError) e sem herdar o CUDA/threads do pai (o que fazia o 'fork'
    # segfaultar). Precisa ser configurado ANTES de o 1º DataLoader subir o servidor.
    if args.mp_context == "forkserver":
        try:
            import multiprocessing as _mp
            _mp.set_forkserver_preload(["src.data"])
        except Exception:
            pass

    from .config import ExperimentConfig, parse_colorset
    from .engine import HParams
    from . import attn as attnmod
    from . import cv as cvmod
    from . import preds as predsmod

    dataset = _DATASET_ALIAS.get(args.dataset, args.dataset)
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    wb_method = "graypatch" if args.wb == "on" else "off"

    cfg = ExperimentConfig(
        backbone=args.backbone, dataset=dataset, colorspaces=parse_colorset(args.colors),
        num_classes=args.num_classes, fusion=args.fusion, gpu=args.gpu, seed=seeds[0],
        num_workers=args.num_workers, mp_context=args.mp_context, batch_size=args.batch_size,
        wb_method=wb_method, roi=(args.roi_lo, args.roi_hi),
        njn_mode=args.njn_mode, phash_dist=args.phash_dist,
        amp=not args.no_amp, amp_dtype=args.amp_dtype, use_cache=not args.no_cache,
        backbone_lr_mult=args.backbone_lr_mult, data_root=args.data_root,
        results_dir=args.results_dir, cache_dir=args.cache_dir,
        objective="f1_macro", tta=not args.no_tta, label_smoothing=args.label_smoothing,
        limit_per_split=args.limit_per_split, force=args.force,
        backbone_mode=args.backbone_mode, film_mode=args.film_mode, lora_rank=args.lora_rank,
        multitask=args.multitask, cv_folds=args.folds, split_mode=args.split_mode,
        split_source=("frozen" if args.frozen_split else "kfold"),
        threshold_mode=args.threshold_mode, swa=args.swa, ema=args.ema,
        save_explain=args.save_explain,
    )

    base = cfg.unit_id.rsplit("__s", 1)[0]
    # split congelado (fixo p/ todos os seeds) tem sufixo próprio -> marcador não colide com o 5x5 CV.
    run_id = (f"{base}__fixedsplit_x{len(seeds)}" if cfg.split_source == "frozen"
              else f"{base}__cv{args.folds}x{len(seeds)}")
    out_dir = Path(args.results_dir) / dataset / args.backbone
    marker = out_dir / f"{(args.tag + '__' if args.tag else '')}{run_id}.json"

    print("=" * 78)
    _split_desc = "frozen(70/10/20 fixo)" if cfg.split_source == "frozen" else f"{args.folds}-fold"
    print(f"CV UNIT: {run_id}  |  seeds={seeds}  split={args.split_mode}/{_split_desc}")
    print(f"  dataset={dataset} colors={cfg.colorset_id} fusion={cfg.fusion} "
          f"mode={cfg.backbone_mode} film={cfg.film_mode} wb={args.wb} mt={cfg.multitask}")
    print("=" * 78)

    if marker.exists() and not args.force:
        try:
            _prev = json.loads(marker.read_text())
            if _prev.get("status") == "completed":
                # Refresh seletivo: só refaz se faltar o oof_auc (seleção da Fase 2) ou a
                # interpretabilidade (α do gate / mapas de atenção do CCAT).
                if args.refresh_missing_oof and _prev.get("oof_auc") is None:
                    print(f"[refresh-oof] {marker} sem oof_auc — refazendo p/ logar validação.")
                elif args.refresh_missing_interp and _falta_interp(_prev, cfg, args):
                    print(f"[refresh-interp] {marker} sem α/mapas de atenção — refazendo "
                          "p/ coletar (determinístico: as métricas não mudam).")
                else:
                    print(f"[skip] já concluído em {marker}. Use --force para refazer.")
                    return 0
        except Exception:
            pass

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[aviso] CUDA indisponível — CPU (só smoke).")

    hp = HParams(lr=args.lr, optimizer=args.optimizer, activation="none",
                 n_unfreeze=args.n_unfreeze, fusion=cfg.fusion, augment=args.augment,
                 da_strength=args.da_strength if args.augment else 0.0,
                 batch_size=args.batch_size)

    t0 = time.time()
    res = cvmod.run_cv(cfg, hp, device, seeds, epochs=args.epochs,
                       patience=args.patience, verbose=args.verbose)
    elapsed = time.time() - t0

    m, ci = res.test_metrics_mean, res.test_metrics_ci95
    auc = m.get("roc_auc", float("nan"))
    print(f"[cv] AUC={auc:.4f}±{ci.get('roc_auc', 0):.4f} | "
          f"Acc={m.get('accuracy', float('nan')):.2f}% | "
          f"MacroF1={m.get('f1_macro', float('nan')):.4f} | "
          f"thr={res.threshold:.2f} | std_seed_AUC={res.std_seed_auc:.4f}")
    if res.regression:
        print(f"[cv] TSB regressão: MAE={res.regression['tsb_mae']:.2f} mg/dL  "
              f"R²={res.regression['tsb_r2']:.3f}")
    if res.test_metrics_patient_mean:
        pm = res.test_metrics_patient_mean
        print(f"[cv] PACIENTE: AUC={pm.get('roc_auc', float('nan')):.4f} | "
              f"Acc={pm.get('accuracy', float('nan')):.2f}% | "
              f"MacroF1={pm.get('f1_macro', float('nan')):.4f} | "
              f"thr={res.threshold_patient:.2f} | n={pm.get('n_patients', float('nan')):.0f} "
              f"(Δacc vs imagem: {pm.get('accuracy', 0) - m.get('accuracy', 0):+.2f} p.p.)")
    gate_ok = (not (auc != auc)) and res.std_seed_auc < 0.01
    print(f"[gate] std_seed_AUC<0.01 ? {'OK' if gate_ok else 'FALHOU'} "
          f"({res.std_seed_auc:.4f}) | tempo {elapsed/60:.1f} min")

    record = {
        "status": "completed", "run_id": run_id, "tag": args.tag,
        "dataset": dataset, "backbone": args.backbone, "colorspaces": list(cfg.colorspaces),
        "colorset_id": cfg.colorset_id, "fusion": cfg.fusion,
        "backbone_mode": cfg.backbone_mode, "film_mode": cfg.film_mode,
        "multitask": cfg.multitask, "wb": args.wb, "wb_method": wb_method,
        "njn_mode": (None if cfg.is_neojaundice else cfg.njn_mode),
        "split_mode": cfg.split_mode, "split_source": cfg.split_source,
        "threshold_mode": cfg.threshold_mode,
        "seeds": seeds, "folds": res.n_folds, "swa": cfg.swa, "ema": cfg.ema,
        "hparams": hp.to_dict(), "threshold": round(res.threshold, 4),
        "std_seed_auc": res.std_seed_auc, "seed_aucs": res.seed_aucs,
        "oof_auc": res.oof_auc, "oof_seed_aucs": res.oof_seed_aucs,
        "gate_stable": bool(gate_ok),
        "test_metrics_mean": m, "test_metrics_ci95": ci,
        "per_seed_metrics": {str(s): v for s, v in res.per_seed_metrics.items()},
        # Agregação por PACIENTE (unidade clínica). Aditivo: as métricas-imagem acima
        # seguem idênticas. thr escolhido só no val agregado por paciente (regra #2).
        "threshold_patient": (round(res.threshold_patient, 4)
                              if res.threshold_patient is not None else None),
        "test_metrics_patient_mean": res.test_metrics_patient_mean,
        "test_metrics_patient_ci95": res.test_metrics_patient_ci95,
        "per_seed_patient_metrics": ({str(s): v for s, v in res.per_seed_patient_metrics.items()}
                                     if res.per_seed_patient_metrics else None),
        "seed_aucs_patient": res.seed_aucs_patient,
        "std_seed_auc_patient": res.std_seed_auc_patient,
        "preds_file": (str(predsmod.preds_path_for(marker).relative_to(Path(args.results_dir)))
                       if res.pred_dump is not None and res.pred_dump.n_rows else None),
        "attn_file": (str(attnmod.attn_path_for(marker).relative_to(Path(args.results_dir)))
                      if res.attn_dump is not None and res.attn_dump.n_rows else None),
        "regression": res.regression,
        "alpha_space": res.alpha_space, "alpha_space_names": res.alpha_space_names,
        "alpha_scope": res.alpha_scope,
        "film_profile": res.film_profile,
        "loso_delta_auc": res.loso_delta_auc, "loso_names": res.loso_names,
        "elapsed_min": round(elapsed / 60, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    # Dump de predições ANTES do marcador: o marcador é o sinal de "unidade concluída"
    # (a retomada o usa para pular), então ele deve ser o ÚLTIMO a aparecer. Assim nunca
    # existe marcador 'completed' apontando para um .npz que não foi escrito.
    if res.pred_dump is not None and res.pred_dump.n_rows:
        pfile = predsmod.preds_path_for(marker)
        res.pred_dump.save(pfile)
        print(f"[ok] predições salvas em {pfile} ({res.pred_dump.n_rows} linhas)")
    if res.attn_dump is not None and res.attn_dump.n_rows:
        afile = attnmod.attn_path_for(marker)
        res.attn_dump.save(afile)
        print(f"[ok] mapas de atenção salvos em {afile} "
              f"({res.attn_dump.n_rows} imagens × {res.attn_dump.n_spaces} espaços)")
    tmp = marker.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2))
    os.replace(tmp, marker)
    print(f"[ok] marcador salvo em {marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
