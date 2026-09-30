#!/usr/bin/env python3
"""Atribuição espacial dos doze modelos retreinados: Grad-CAM (CNN) e rollout (ViT).

## O que isto responde

O artigo mostra QUE o efeito dos espaços cromáticos depende do par (backbone, coleção).
Não mostra ONDE. Esta análise pergunta se o braço multi-espaço olha para sítios
diferentes do braço RGB, ou se olha para os mesmos sítios e apenas decide de outra
maneira. São hipóteses distintas e a figura tinha de as poder separar.

## O piso de ruído, que é a parte que não se pode saltar

Uma correlação de 0,7 entre o mapa RGB e o mapa multi-espaço não diz nada sozinha,
porque duas sementes do MESMO braço também não dão 1,0 --- o treino é estocástico e o
mapa herda essa variância. Portanto mede-se sempre o par:

* ``r_between``     --- correlação entre uma semente do braço RGB e uma do multi-espaço,
  média dos 25 pares cruzados;
* ``r_within_rgb``  --- correlação entre duas sementes do MESMO braço RGB, média dos 10
  pares. Os dois lados são pares de sementes individuais, ao mesmo nível de agregação ---
  comparar a média das cinco sementes contra pares soltos mediria a supressão de
  variância pela média, e não a cromância.

Se ``r_between ≈ r_within``, a cromância não moveu a atenção para além do que o ruído
de semente já move, e qualquer diferença de acurácia veio de outro sítio que não a
localização. Se ``r_between < r_within``, moveu. É a única leitura defensável.

## Proveniência

Os pesos vêm de `local/experiments/explain/`, gravados por `src/explainpack.py` no retreino de
`scripts/run_explain_cells.py`. **Não são os pesos que produziram a Tabela 4** --- a
campanha não os gravou. A configuração é a mesma (mesmos hiperparâmetros, mesma partição
congelada, mesmas cinco sementes); o treino é que é estocástico.

## Anonimização

Só o NeoJaundice entra em figura com fotografia: o recorte do protocolo [0,30, 0,70]
é pele, sem rosto (verificado por inspecção). O NJN corre para a estatística, e nunca
para uma imagem --- `--njn-mode full_image` faz o modelo ver o rosto inteiro, e a
deteção automática de face não foi reprodutível nestas imagens.

Uso:
    python analysis/make_fig_attribution.py --shift          # CSV, todas as 6 células
    python analysis/make_fig_attribution.py --shift --only NeoJaundice/vit_l_16
    python analysis/make_fig_attribution.py --fig            # figura (usa o CSV + mapas)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPLAIN = ROOT / "local/experiments/explain"
OUT = ROOT / "results"
FIGDIR = ROOT / "local/generated/figures"
FIGS = ROOT / "paper/figs"
MAPS = OUT / "attr_maps"

#: Os seis pares retreinados. O braço RGB é sempre o comparador.
CELLS = [
    ("NJN",         "mobilenetv3_large", "RGB+LAB+HSV"),
    ("NeoJaundice", "vit_l_16",          "RGB+LAB+YCrCb+HSV"),
    ("NJN",         "densenet121",       "RGB+YCrCb+HSV"),
    ("NJN",         "dinov3_vits16",     "RGB+YCrCb+HSV"),
    ("NeoJaundice", "deit_base",         "RGB+LAB+YCrCb+HSV"),
    ("NeoJaundice", "deit_small",        "RGB+LAB+YCrCb+HSV"),
]

SEEDS = [42, 123, 456, 7, 99]
BLUE, RED, INK, MUTED, EDGE = "#2a78d6", "#e34948", "#0b0b0b", "#52514e", "#d8d7d3"


# ------------------------------------------------------------------ config
def cfg_and_hp(dataset: str, backbone: str, colorset: str):
    """Reconstrói exactamente a configuração do retreino, pelo mesmo parser."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rec", ROOT / "scripts/run_explain_cells.py")
    rec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rec)

    argv = rec.build_command(dataset, backbone, colorset, gpu=0)[3:]  # tira python -m src.train
    argv = [a for a in argv if a != "--save-explain"]

    from src.train import _parse_args, _DATASET_ALIAS
    from src.config import ExperimentConfig, parse_colorset
    from src.engine import HParams

    a = _parse_args(argv)
    ds = _DATASET_ALIAS.get(a.dataset, a.dataset)
    cfg = ExperimentConfig(
        backbone=a.backbone, dataset=ds, colorspaces=parse_colorset(a.colors),
        num_classes=a.num_classes, fusion=a.fusion, gpu=a.gpu, seed=SEEDS[0],
        num_workers=a.num_workers, mp_context=a.mp_context, batch_size=a.batch_size,
        wb_method=("graypatch" if a.wb == "on" else "off"), roi=(a.roi_lo, a.roi_hi),
        njn_mode=a.njn_mode, phash_dist=a.phash_dist,
        amp=not a.no_amp, amp_dtype=a.amp_dtype, use_cache=not a.no_cache,
        backbone_lr_mult=a.backbone_lr_mult, data_root=a.data_root,
        results_dir=a.results_dir, cache_dir=a.cache_dir,
        objective="f1_macro", tta=not a.no_tta, label_smoothing=a.label_smoothing,
        limit_per_split=a.limit_per_split, force=a.force,
        backbone_mode=a.backbone_mode, film_mode=a.film_mode, lora_rank=a.lora_rank,
        multitask=a.multitask, cv_folds=a.folds, split_mode=a.split_mode,
        split_source=("frozen" if a.frozen_split else "kfold"),
        threshold_mode=a.threshold_mode, swa=a.swa, ema=a.ema,
    )
    hp = HParams(lr=a.lr, optimizer=a.optimizer, activation="none",
                 n_unfreeze=a.n_unfreeze, fusion=cfg.fusion, augment=False,
                 da_strength=0.0, batch_size=a.batch_size)
    return cfg, hp


def test_loader(cfg, hp):
    """O MESMO loader de teste do retreino: partição congelada, sem augment."""
    from src import splits
    from src.data import build_dataloaders_from_split
    samples, classes = splits.load_samples(cfg)
    tr, va, te = splits.load_frozen_split(cfg, samples)
    bundle = build_dataloaders_from_split(cfg, samples, tr, va, te, classes,
                                          augment=False, da_strength=0.0,
                                          batch_size=hp.batch_size)
    paths = [str(samples[i].path) for i in te]
    return bundle, paths


def load_model(pt: Path, n_channels: int, cfg, hp, device):
    import torch
    from src.models import build_model
    blob = torch.load(pt, map_location="cpu", weights_only=False)
    model = build_model(cfg.spec, n_channels, fusion=cfg.fusion,
                        activation=hp.activation, n_unfreeze=hp.n_unfreeze,
                        colorspaces=cfg.colorspaces, adapter_init=cfg.adapter_init,
                        hue_circular=cfg.hue_circular, num_classes=cfg.num_classes,
                        backbone_mode=cfg.backbone_mode, film_mode=cfg.film_mode,
                        lora_rank=cfg.lora_rank, multitask=False)
    missing, unexpected = model.load_state_dict(blob["state_dict"], strict=False)
    if missing or unexpected:
        raise SystemExit(f"state_dict não bate em {pt.name}: "
                         f"{len(missing)} em falta, {len(unexpected)} a mais")
    return model.to(device).eval()


# -------------------------------------------------------------- atribuição
def _cnn_target(model):
    """A última pilha convolucional. densenet121 e mobilenetv3_large expõem `features`."""
    feats = getattr(model.backbone, "features", None)
    if feats is None:
        raise SystemExit(f"sem .features em {type(model.backbone).__name__}")
    return feats


def gradcam(model, x, device):
    """Grad-CAM na saída da última pilha conv, para a classe 'icterícia' (índice 1)."""
    import torch
    store = {}

    def fwd(_m, _i, o):
        store["a"] = o
        o.retain_grad()

    h = _cnn_target(model).register_forward_hook(fwd)
    try:
        x = x.to(device).requires_grad_(False)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[:, 1].sum().backward()
        a, g = store["a"], store["a"].grad
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * a).sum(1))                     # [B,h,w]
        return cam.detach().float().cpu().numpy()
    finally:
        h.remove()
        model.zero_grad(set_to_none=True)


def _attn_hooks(model):
    """Captura a matriz de atenção por bloco, em timm e em torchvision.

    timm usa SDPA fundido (`fused_attn`), que não materializa a matriz --- desliga-se e
    engancha-se o `attn_drop`, por onde ela passa. O torchvision chama a
    `MultiheadAttention` com `need_weights=False`; reescreve-se o kwarg no pre-hook.
    """
    grabs, handles = [], []
    blocks = getattr(model.backbone, "blocks", None)
    if blocks is not None:                                    # timm
        for blk in blocks:
            blk.attn.fused_attn = False

            def fwd(_m, inp, _o, _g=grabs):
                _g.append(inp[0].detach())                    # [B,heads,L,L]
            handles.append(blk.attn.attn_drop.register_forward_hook(fwd))
        return grabs, handles, len(blocks)

    layers = model.backbone.encoder.layers                    # torchvision
    for blk in layers:
        def pre(_m, args, kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = True
            return args, kwargs

        def fwd(_m, _i, out, _g=grabs):
            _g.append(out[1].detach())                        # [B,L,L]
        handles.append(blk.self_attention.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(blk.self_attention.register_forward_hook(fwd))
    return grabs, handles, len(layers)


def rollout(model, x, device, n_prefix: int):
    """Attention rollout de Abnar & Zuidema: A_hat = ½A + ½I, normalizada e composta."""
    import torch
    grabs, handles, n_blocks = _attn_hooks(model)
    try:
        with torch.no_grad():
            model(x.to(device))
        if len(grabs) != n_blocks:
            raise SystemExit(f"rollout: {len(grabs)} matrizes para {n_blocks} blocos")
        roll = None
        for a in grabs:
            if a.dim() == 4:
                a = a.mean(1)                                 # média sobre as cabeças
            a = a.float()
            eye = torch.eye(a.size(-1), device=a.device).expand_as(a)
            a = 0.5 * a + 0.5 * eye
            a = a / a.sum(-1, keepdim=True)
            roll = a if roll is None else torch.bmm(a, roll)
        cls = roll[:, 0, n_prefix:]                           # linha do CLS -> patches
        g = int(round(cls.size(1) ** 0.5))
        return cls.reshape(cls.size(0), g, g).cpu().numpy()
    finally:
        for h in handles:
            h.remove()
        grabs.clear()


def maps_for(cfg, hp, colorset_dir: Path, colorset: str, device):
    """Um mapa por (semente, imagem). Devolve [S, N, h, w] e os caminhos alinhados."""
    import torch
    is_vit = cfg.spec.family == "vit"
    bundle, paths = test_loader(cfg, hp)
    n_prefix = 0
    stack = []
    for seed in SEEDS:
        pt = colorset_dir / f"{_stem(cfg)}__s{seed}_f0.pt"
        if not pt.is_file():
            raise SystemExit(f"em falta: {pt}")
        model = load_model(pt, bundle.n_channels, cfg, hp, device)
        if is_vit and not n_prefix:
            n_prefix = int(getattr(model.backbone, "num_prefix_tokens", 1) or 1)
        per_seed = []
        for batch in bundle.test_loader:
            xb = batch[0]
            m = (rollout(model, xb, device, n_prefix) if is_vit
                 else gradcam(model, xb.to(device), device))
            per_seed.append(m.astype(np.float32))
        stack.append(np.concatenate(per_seed, 0))
        del model
        torch.cuda.empty_cache()
    return np.stack(stack), paths


def _stem(cfg) -> str:
    return cfg.unit_id.rsplit("__s", 1)[0]


# ------------------------------------------------------------------ shift
def _norm(m: np.ndarray) -> np.ndarray:
    """Normaliza cada mapa para [0,1] --- a correlação é de forma, não de escala."""
    flat = m.reshape(m.shape[0], -1).astype(np.float64)
    lo = flat.min(1, keepdims=True)
    hi = flat.max(1, keepdims=True)
    return (flat - lo) / np.maximum(hi - lo, 1e-12)


def _pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.divide((a * b).sum(1), den, out=np.full(len(a), np.nan), where=den > 0)


def shift(dataset: str, backbone: str, colorset: str, device) -> dict:
    """r_between (RGB vs multi) contra r_within (RGB vs RGB, outra semente)."""
    cfg_m, hp_m = cfg_and_hp(dataset, backbone, colorset)
    cfg_r, hp_r = cfg_and_hp(dataset, backbone, "RGB")
    d = EXPLAIN / dataset / backbone / "explain"

    print(f"  [{dataset}/{backbone}] {colorset} ...", flush=True)
    mm, paths_m = maps_for(cfg_m, hp_m, d, colorset, device)
    print(f"  [{dataset}/{backbone}] RGB ...", flush=True)
    mr, paths_r = maps_for(cfg_r, hp_r, d, "RGB", device)
    if paths_m != paths_r:
        raise SystemExit("ordem do loader difere entre braços — abortado")

    MAPS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MAPS / f"{dataset}__{backbone}.npz",
                        multi=mm.astype(np.float16), rgb=mr.astype(np.float16),
                        paths=np.array(paths_m, dtype="U256"),
                        colorset=colorset)

    # Os dois lados TÊM de ser medidos ao mesmo nível de agregação. Comparar o mapa
    # médio das cinco sementes de um braço com o mapa médio do outro, e chamar piso de
    # ruído à correlação entre DUAS sementes soltas, mede a média contra o ruído bruto e
    # dá um "efeito" que é só a supressão de variância pela média. Aqui os dois lados são
    # pares de sementes individuais: 25 pares cruzados contra 10 pares dentro do RGB.
    nm = [_norm(mm[i]) for i in range(len(SEEDS))]
    nr = [_norm(mr[i]) for i in range(len(SEEDS))]

    between = [_pearson_rows(a, b) for a in nm for b in nr]
    within_r = [_pearson_rows(nr[i], nr[j])
                for i in range(len(SEEDS)) for j in range(i + 1, len(SEEDS))]
    within_m = [_pearson_rows(nm[i], nm[j])
                for i in range(len(SEEDS)) for j in range(i + 1, len(SEEDS))]

    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # Um mapa constante (Grad-CAM todo zero após o ReLU) tem variância nula e a
        # correlação é indefinida. Conta-se em n_degenerate em vez de se fingir um valor.
        warnings.simplefilter("ignore", RuntimeWarning)
        rb = np.nanmean(np.stack(between), 0)
        wr = np.nanmean(np.stack(within_r), 0)
        wm = np.nanmean(np.stack(within_m), 0)

    ok = np.isfinite(rb) & np.isfinite(wr)
    n_bad = int((~ok).sum())

    return {
        "dataset": dataset, "backbone": backbone, "colorset": colorset,
        "family": cfg_m.spec.family, "n_images": len(paths_m),
        "grid": f"{mm.shape[-2]}x{mm.shape[-1]}", "n_degenerate": n_bad,
        "r_between": float(np.nanmedian(rb)),
        "r_within_rgb": float(np.nanmedian(wr)),
        "r_within_multi": float(np.nanmedian(wm)),
        "gap": float(np.nanmedian(wr[ok]) - np.nanmedian(rb[ok])),
        "frac_below": float(np.mean(rb[ok] < wr[ok])) if ok.any() else float("nan"),
    }


def run_shift(only: str | None, device_str: str) -> int:
    import torch
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    rows = []
    for dataset, backbone, colorset in CELLS:
        if only and only not in f"{dataset}/{backbone}":
            continue
        rows.append(shift(dataset, backbone, colorset, device))

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "G2_attribution_shift.csv"
    cols = ["dataset", "backbone", "colorset", "family", "n_images", "grid",
            "n_degenerate", "r_between", "r_within_rgb", "r_within_multi",
            "gap", "frac_below"]
    import csv
    write_header = not dest.exists() or only is None
    mode = "w" if write_header else "a"
    with dest.open(mode, encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter=";")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in cols})
    print(f"\n{dest.relative_to(ROOT)}")
    for r in rows:
        print(f"  {r['dataset']:12s} {r['backbone']:18s} "
              f"r_between={r['r_between']:.3f} "
              f"r_within(RGB)={r['r_within_rgb']:.3f} "
              f"r_within(multi)={r['r_within_multi']:.3f} "
              f"gap={r['gap']:+.3f} abaixo do piso: {100*r['frac_below']:.0f}%")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shift", action="store_true", help="calcula r_between vs r_within")
    ap.add_argument("--fig", action="store_true", help="desenha a figura (precisa dos mapas)")
    ap.add_argument("--only", default=None, help="filtra por 'dataset/backbone'")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    if args.shift:
        return run_shift(args.only, args.device)
    if args.fig:
        # Por caminho, e não por pacote: `analysis/` não é um pacote e transformá-lo num
        # mudaria a resolução de imports dos outros módulos que lá vivem.
        import importlib.util
        sp = importlib.util.spec_from_file_location(
            "_attr_fig", Path(__file__).with_name("_attr_fig.py"))
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
        return mod.build()
    ap.print_help()
    return 1


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    raise SystemExit(main())
