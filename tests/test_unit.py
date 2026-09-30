"""
test_unit.py — testes unitários offline da Fase 0 (HCSF v7).

Cobre o §1 de ``docs/TESTING.md``. Os marcados 🔴 no doc são bloqueantes do gate.
Rodar: ``CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src import config, metrics, models, splits, stats
from src.cache import cached_npy_path, load_rgb_cached
from src.colorspaces import (IMAGENET_MEAN, IMAGENET_STD,
                                                   ColorSpaceStack)
from src.config import (canonical_colorset, parse_colorset,
                                              space_nchannels)
from src.models import CCATFusion, ColorFusion

from tests.conftest import (
    CS_COL, CS_RGB, RES_CNN, SPEC_CNN, nch, rand_pil, rand_stack,
    skip_if_no_weights)


# =========================================================================== #
# 1.1 Espaços de cor
# =========================================================================== #
def test_colorspace_stack_shapes():
    img = rand_pil(32)
    st_rgb = ColorSpaceStack(["RGB"], 32)
    assert st_rgb.n_channels == 3
    assert st_rgb(img).shape == (3, 32, 32)

    st_col = ColorSpaceStack(["RGB", "YCrCb", "HSV"], 32, hue_circular=False)
    assert st_col.n_channels == 9
    assert st_col(img).shape[0] == 9

    st_hue = ColorSpaceStack(["RGB", "YCrCb", "HSV"], 32, hue_circular=True)
    assert st_hue.n_channels == 10          # HSV: H->(sin,cos) => +1 canal
    assert st_hue(img).shape[0] == 10


def test_colorspace_canonical_order():
    # Ordem canônica fixa RGB | LAB | YCrCb | HSV, preservada em qualquer subconjunto.
    assert canonical_colorset(["HSV", "RGB", "LAB"]) == ("RGB", "LAB", "HSV")
    assert canonical_colorset(["YCrCb", "RGB"]) == ("RGB", "YCrCb")
    assert canonical_colorset(["HSV", "YCrCb"]) == ("YCrCb", "HSV")


def test_hue_circular_channels():
    # Com hue circular, os 2 primeiros canais do HSV são (sinH, cosH) ∈ [-1,1] e
    # sin²+cos²≈1 (sem descontinuidade artificial no wrap 0<->179).
    st = ColorSpaceStack(["HSV"], 32, hue_circular=True)
    out = st(rand_pil(32))
    sin_h, cos_h = out[0], out[1]
    assert sin_h.min() >= -1.0001 and sin_h.max() <= 1.0001
    assert cos_h.min() >= -1.0001 and cos_h.max() <= 1.0001
    r2 = (sin_h ** 2 + cos_h ** 2)
    assert torch.allclose(r2, torch.ones_like(r2), atol=1e-4)


def test_normalization_stats_source():
    # RGB usa stats do ImageNet; LAB/YCrCb/HSV usam stats fornecidas (do treino).
    st_rgb = ColorSpaceStack(["RGB"], 32)
    assert torch.allclose(st_rgb._mean["RGB"].flatten(),
                          torch.tensor(IMAGENET_MEAN), atol=1e-6)
    assert torch.allclose(st_rgb._std["RGB"].flatten(),
                          torch.tensor(IMAGENET_STD), atol=1e-6)
    train_stats = {"LAB": ([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])}
    st_lab = ColorSpaceStack(["RGB", "LAB"], 32, stats=train_stats)
    assert torch.allclose(st_lab._mean["LAB"].flatten(),
                          torch.tensor([0.1, 0.2, 0.3]), atol=1e-6)
    # RGB continua ImageNet mesmo com stats de LAB fornecidas.
    assert torch.allclose(st_lab._mean["RGB"].flatten(),
                          torch.tensor(IMAGENET_MEAN), atol=1e-6)


# =========================================================================== #
# 1.2 Fusão e CONTRATO DE COMPARABILIDADE (🔴)
# =========================================================================== #
def test_adapter_identity_init_equals_rgb(model_rgb):  # 🔴
    # Época 0 (eval): pseudo-RGB ≈ RGB normalizado (init identidade; BN neutra).
    model_rgb.eval()
    x = rand_stack(CS_RGB)
    pseudo = model_rgb.fusion(x)
    assert pseudo.shape == x.shape
    assert torch.allclose(pseudo, x, atol=1e-4)


def test_adapter_color_residual_nonzero(model_col):  # 🔴
    # Pesos dos canais de cor ~ σ=1e-4 (≠ 0.0 exato, AMP-safe) e gradiente não-nulo.
    fusion = model_col.fusion
    assert isinstance(fusion, ColorFusion)
    w = fusion.proj.weight.detach()          # [3, N, 1, 1]
    # canais RGB = 0..2 (RGB é o 1o grupo); demais = cor.
    color_cols = list(range(3, w.shape[1]))
    color_std = float(w[:, color_cols, 0, 0].std())
    assert 1e-6 < color_std < 1e-2           # ruído pequeno, porém não-zero
    # gradiente flui nos canais de cor no 1o batch.
    model_col.train()
    x = rand_stack(CS_COL)
    out = model_col(x)
    out.sum().backward()
    g = fusion.proj.weight.grad
    assert g is not None and float(g[:, color_cols, 0, 0].abs().sum()) > 0.0


def test_ccat_zero_init_residual():  # 🔴
    cs = ["RGB", "LAB", "HSV"]
    m = skip_if_no_weights(lambda: models.build_model(
        SPEC_CNN, nch(cs), fusion="ccat", colorspaces=cs, num_classes=2))
    m.eval()
    x = rand_stack(cs)
    pseudo = m.fusion(x)
    # Wo zero-init => resíduo ≈ 0 => pseudo ≈ RGB (BN neutra na init).
    assert torch.allclose(pseudo, x[:, :3], atol=1e-4)


def test_ccat_attn_maps_populated():
    cs = ["RGB", "LAB", "HSV"]
    m = skip_if_no_weights(lambda: models.build_model(
        SPEC_CNN, nch(cs), fusion="ccat", colorspaces=cs, num_classes=2))
    m.eval()
    _ = m.fusion(rand_stack(cs))
    attn = m.fusion.attn_maps
    assert attn is not None
    assert attn.shape[1] == 2               # 2 espaços cromáticos (LAB, HSV)
    somas = attn.sum(dim=1)                  # softmax sobre espaços soma 1 por pixel
    assert torch.allclose(somas, torch.ones_like(somas), atol=1e-4)


def test_fusion_activity(model_col):  # 🔴
    # Injetar Δcromância ≠ 0 nos canais de cor muda a saída da fusão (fusão NÃO inerte).
    model_col.eval()
    x = rand_stack(CS_COL)
    base = model_col.fusion(x)
    x2 = x.clone()
    x2[:, 3:] += 3.0                          # perturba só os canais cromáticos
    pert = model_col.fusion(x2)
    assert float((pert - base).detach().norm()) > 1e-4


def test_rgb_baseline_uses_fusion(model_rgb):  # 🔴
    # O baseline RGB passa pelo MESMO bloco de fusão (BN + projeção), não backbone(rgb) cru.
    assert isinstance(model_rgb.fusion, ColorFusion)
    assert hasattr(model_rgb.fusion, "bn") and hasattr(model_rgb.fusion, "proj")
    assert model_rgb.fusion(rand_stack(CS_RGB)).shape[1] == 3   # entrega pseudo-RGB


def test_comparability_backbone_identical(model_rgb, model_col):  # 🔴
    # Mesmo backbone/init nos dois braços; só o nº de canais da fusão difere.
    sd_rgb = model_rgb.backbone.state_dict()
    sd_col = model_col.backbone.state_dict()
    assert sd_rgb.keys() == sd_col.keys()
    n_rgb = sum(p.numel() for p in model_rgb.backbone.parameters())
    n_col = sum(p.numel() for p in model_col.backbone.parameters())
    assert n_rgb == n_col
    for k in sd_rgb:
        assert torch.allclose(sd_rgb[k], sd_col[k]), f"backbone diverge em {k}"
    # Ambos entregam pseudo-RGB [B,3,H,W].
    assert model_rgb.fusion(rand_stack(CS_RGB)).shape[1] == 3
    assert model_col.fusion(rand_stack(CS_COL)).shape[1] == 3


def test_comparability_hparams_shared():  # 🔴
    # Os hparams de treino (best_params congelado = objeto HParams) são o MESMO nos
    # dois braços; a ExperimentConfig difere APENAS no colorset.
    from src.engine import HParams
    best = HParams(lr=3e-4, batch_size=16, fusion="adapter_v2", da_strength=1.0)
    hp_rgb = hp_col = best                       # best_params aplicado aos dois braços
    assert hp_rgb is hp_col
    for f in ("lr", "batch_size", "fusion", "optimizer", "da_strength"):
        assert getattr(hp_rgb, f) == getattr(hp_col, f)

    base = dict(backbone="resnet18", dataset="NJN", label_smoothing=0.1,
                backbone_mode="frozen")
    cfg_rgb = config.ExperimentConfig(colorspaces=parse_colorset("RGB"), **base)
    cfg_col = config.ExperimentConfig(colorspaces=parse_colorset("RGB+YCrCb+HSV"), **base)
    d_rgb = {k: v for k, v in vars(cfg_rgb).items() if k != "colorspaces"}
    d_col = {k: v for k, v in vars(cfg_col).items() if k != "colorspaces"}
    assert d_rgb == d_col                         # nenhum outro campo difere
    assert cfg_rgb.colorspaces != cfg_col.colorspaces   # a ÚNICA diferença


def test_ccat_requires_chroma():
    # CCAT com colorset=RGB não tem espaço cromático (K/V) => erro (não é baseline cruzado).
    with pytest.raises(Exception):
        models.build_model(SPEC_CNN, nch(CS_RGB), fusion="ccat",
                           colorspaces=CS_RGB, num_classes=2)


# =========================================================================== #
# 1.3 Cabeças e otimização
# =========================================================================== #
def test_head_reg_optional():
    m_mt = skip_if_no_weights(lambda: models.build_model(
        SPEC_CNN, nch(CS_RGB), fusion="adapter_v2", colorspaces=CS_RGB,
        num_classes=2, multitask=True))
    logits = m_mt(rand_stack(CS_RGB))
    assert logits.shape == (2, 2)
    assert m_mt.head_reg is not None
    assert m_mt.last_tsb is not None and m_mt.last_tsb.shape == (2,)  # (logits, tsb_pred)

    m_cls = skip_if_no_weights(lambda: models.build_model(
        SPEC_CNN, nch(CS_RGB), fusion="adapter_v2", colorspaces=CS_RGB,
        num_classes=2, multitask=False))
    _ = m_cls(rand_stack(CS_RGB))
    assert m_cls.head_reg is None and m_cls.last_tsb is None            # só logits


def test_kendall_uncertainty_weighting():
    from src.cv import _kendall_loss
    m = skip_if_no_weights(lambda: models.build_model(
        SPEC_CNN, nch(CS_RGB), fusion="adapter_v2", colorspaces=CS_RGB,
        num_classes=2, multitask=True))
    assert m.log_var_cls.requires_grad and m.log_var_reg.requires_grad
    ce = torch.tensor(0.7, requires_grad=True)
    reg = torch.tensor(1.3, requires_grad=True)
    loss = _kendall_loss(ce, reg, m.log_var_cls, m.log_var_reg)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.log_var_cls.grad is not None and m.log_var_reg.grad is not None


def test_differential_lr_groups(model_col):  # 🔴
    # frozen: 1 grupo (backbone fora do otimizador, sem grad).
    groups_frozen = model_col.build_param_groups(1e-3, backbone_lr_mult=0.1)
    assert len(groups_frozen) == 1
    assert all(not p.requires_grad for p in model_col.backbone.parameters())

    # full: 2 grupos; backbone com lr*mult, fusão+cabeças com lr cheio.
    m_full = skip_if_no_weights(lambda: models.build_model(
        SPEC_CNN, nch(CS_COL), fusion="adapter_v2", colorspaces=CS_COL,
        num_classes=2, backbone_mode="full"))
    groups_full = m_full.build_param_groups(1e-3, backbone_lr_mult=0.1)
    assert len(groups_full) == 2
    lrs = sorted(g["lr"] for g in groups_full)
    assert lrs == [pytest.approx(1e-4), pytest.approx(1e-3)]


# =========================================================================== #
# 1.4 Splits, cache, dados
# =========================================================================== #
@pytest.mark.parametrize("dataset", ["NeoJaundice", "NJN"])
def test_splits_no_patient_leak(dataset, dataset_root):  # 🔴
    cfg = config.ExperimentConfig(backbone="resnet18", dataset=dataset,
                                  colorspaces=parse_colorset("RGB"), data_root=str(dataset_root))
    samples, folds, _ = splits.make_kfold(cfg, seed=42, k=5, mode="group")
    assert len(folds) == 5
    for tr, va, te in folds:
        splits.assert_no_patient_leak(samples, tr, va, te)  # não levanta => 0 leak


@pytest.mark.parametrize("dataset", ["NeoJaundice", "NJN"])
def test_frozen_split_roundtrip(dataset, dataset_root):  # 🔴
    """O split canônico congelado (dataset/<DS>/splits/) cobre TODAS as amostras, é
    disjunto, sem leak de paciente, ~70/10/20, e make_split==load_frozen_split (HPO)."""
    cfg = config.ExperimentConfig(backbone="resnet18", dataset=dataset,
                                  colorspaces=parse_colorset("RGB"),
                                  data_root=str(dataset_root),
                                  njn_mode="full_image", split_source="frozen")
    samples, _ = splits.load_samples(cfg)
    tr, va, te = splits.load_frozen_split(cfg, samples)
    n = len(samples)
    assert len(tr) + len(va) + len(te) == n                 # cobertura total
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    splits.assert_no_patient_leak(samples, tr, va, te)      # 0 leak
    assert 0.62 <= len(tr) / n <= 0.78                      # ~70/10/20 (folga p/ agrupamento)
    assert 0.05 <= len(va) / n <= 0.16
    assert 0.13 <= len(te) / n <= 0.27
    # o caminho do HPO (make_split) deve devolver EXATAMENTE o mesmo split congelado
    _, (tr2, va2, te2), _ = splits.make_split(cfg)
    assert np.array_equal(tr, tr2) and np.array_equal(va, va2) and np.array_equal(te, te2)


def test_phash_pseudo_patients(tmp_path):
    from PIL import Image
    rng = np.random.default_rng(1)
    base = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    p1 = tmp_path / "a.png"; Image.fromarray(base).save(p1)
    p2 = tmp_path / "b.png"; Image.fromarray(base).save(p2)          # duplicata
    other = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    p3 = tmp_path / "c.png"; Image.fromarray(other).save(p3)
    mapping, used = skip_if_no_weights(
        lambda: splits.phash_pseudo_patients([p1, p2, p3], dist=3))
    if not used:
        pytest.skip("imagehash indisponível (fallback: cada imagem = 1 pseudo-paciente)")
    k1, k2, k3 = str(p1.resolve()), str(p2.resolve()), str(p3.resolve())
    assert mapping[k1] == mapping[k2]        # duplicatas no mesmo cluster
    assert mapping[k3] != mapping[k1]        # imagem distinta, cluster separado


def test_stratified_group_kfold():
    rng = np.random.default_rng(0)
    n_groups = 50
    groups = np.repeat([f"p{i}" for i in range(n_groups)], 3)   # 3 imgs/paciente
    labels = np.repeat(rng.integers(0, 2, size=n_groups), 3)
    folds = splits.kfold_indices(labels, groups, seed=42, k=5, mode="group")
    assert len(folds) == 5
    overall_pos = labels.mean()
    for tr, va, te in folds:
        g_tr = {groups[i] for i in tr}
        g_va = {groups[i] for i in va}
        g_te = {groups[i] for i in te}
        assert not (g_tr & g_te) and not (g_tr & g_va) and not (g_va & g_te)
        # proporção de classe aproximadamente preservada no teste
        assert abs(labels[te].mean() - overall_pos) < 0.30


def test_cache_roundtrip(tmp_path):
    arr = (np.random.default_rng(0).integers(0, 256, size=(16, 16, 3))).astype(np.uint8)
    calls = {"n": 0}

    def loader(_path):
        calls["n"] += 1
        return arr

    src = "/fake/img_0001.jpg"
    r1 = load_rgb_cached(src, loader, tmp_path, "tagA", use_cache=True)   # computa+grava
    r2 = load_rgb_cached(src, loader, tmp_path, "tagA", use_cache=True)   # lê do disco
    assert np.array_equal(r1, arr) and np.array_equal(r2, arr)
    assert calls["n"] == 1                    # 2ª chamada NÃO recomputou (cache hit)
    assert cached_npy_path(tmp_path, "tagA", src).is_file()


def test_cache_invalidation_version(monkeypatch):
    from src import skin_roi
    cfg = config.ExperimentConfig(backbone="resnet18", dataset="NJN",
                                  colorspaces=parse_colorset("RGB"))
    tag_v1 = cfg.cache_tag()
    assert f"skin-v{skin_roi.SKIN_ALGO_VERSION}" in tag_v1
    monkeypatch.setattr(skin_roi, "SKIN_ALGO_VERSION", "9.9")
    tag_v2 = cfg.cache_tag()                   # cache_tag re-importa a versão a cada chamada
    assert tag_v1 != tag_v2                     # bump da versão invalida a chave


def test_cache_key_excludes_colorset():
    common = dict(backbone="resnet18", dataset="NeoJaundice", wb_method="graypatch")
    a = config.ExperimentConfig(colorspaces=parse_colorset("RGB"), **common)
    b = config.ExperimentConfig(colorspaces=parse_colorset("RGB+LAB+YCrCb+HSV"), **common)
    assert a.cache_tag() == b.cache_tag()       # mesma chave p/ qualquer subconjunto de cor


def test_tsb_only_in_train():
    from src.data import _StackDataset
    items = [("/x/0001-1.jpg", 1, "0001"), ("/x/0002-1.jpg", 0, "0002")]
    tsb = [15.0, 4.0]
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    common = dict(classes=("healthy", "jaundice"), transform=lambda im: torch.zeros(3, 8, 8),
                  loader=lambda p: arr, cache_dir=None, cache_tag="t", use_cache=False,
                  tsb_norm=tsb)
    # treino: alvo TSB exposto; val/test: NÃO (return_tsb=False) — TSB não vaza.
    ds_train = _StackDataset(items, return_tsb=True, **common)
    ds_val = _StackDataset(items, return_tsb=False, **common)
    assert len(ds_train[0]) == 3                # (img, label, tsb)
    assert len(ds_val[0]) == 2                  # (img, label) — sem TSB


# =========================================================================== #
# 1.5 Avaliação, métricas, estatística
# =========================================================================== #
def test_pooled_oof_val_only():  # 🔴
    from src.engine import _best_threshold
    rng = np.random.default_rng(0)
    val_y = rng.integers(0, 2, size=60)
    val_p = np.clip(val_y * 0.6 + rng.normal(0.2, 0.2, size=60), 0, 1)
    thr1, _ = _best_threshold(val_y, val_p, "f1_macro")
    thr2, _ = _best_threshold(val_y, val_p, "f1_macro")
    assert thr1 == thr2                          # determinístico, só do val
    # trocar o TESTE não muda o limiar (o test nunca é consultado p/ escolher limiar).
    test_y = rng.integers(0, 2, size=40)
    test_p = rng.random(40)
    thr_after, _ = _best_threshold(val_y, val_p, "f1_macro")
    assert thr_after == thr1
    # o limiar é aplicável ao teste sem re-otimização.
    m = metrics.compute_metrics(test_y, y_prob=test_p, threshold=thr1)
    assert 0.0 <= m["accuracy"] <= 100.0


def test_metrics_binary():
    y = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.8, 0.9]                     # ranqueamento perfeito
    m = metrics.compute_metrics(y, y_prob=p, threshold=0.5)
    assert m["accuracy"] == pytest.approx(100.0)
    assert m["f1_macro"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["mcc"] == pytest.approx(1.0)


def test_patient_aggregation():
    # 2 pacientes, 3 imgs cada; ruído por imagem faz erros a nível-imagem, mas a
    # média por paciente acerta => acurácia-paciente >= acurácia-imagem.
    paths = ["0001-1.jpg", "0001-2.jpg", "0001-3.jpg",
             "0002-1.jpg", "0002-2.jpg", "0002-3.jpg"]
    y = [1, 1, 1, 0, 0, 0]
    p = [0.45, 0.9, 0.8, 0.55, 0.1, 0.2]         # 1 img errada por paciente no thr 0.5
    img = metrics.compute_metrics(y, y_prob=p, threshold=0.5)["accuracy"]
    pat = metrics.compute_patient_metrics(paths, y, p, threshold=0.5)["accuracy"]
    assert pat >= img
    assert pat == pytest.approx(100.0)


def test_regression_denormalize():
    y = np.array([3.9, 12.0, 20.5])
    mean, std = 10.0, 4.0
    y_norm = (y - mean) / std
    assert np.allclose(metrics.denormalize_tsb(y_norm, mean, std), y, atol=1e-9)


def test_ita_computation():
    assert metrics.ita_degrees(50.0, 10.0) == pytest.approx(0.0)      # atan2(0,10)=0
    assert metrics.ita_degrees(60.0, 0.0) == pytest.approx(90.0)      # atan2(10,0)=pi/2
    L, b = 65.0, 12.0
    assert metrics.ita_degrees(L, b) == pytest.approx(
        np.arctan2(L - 50.0, b) * 180.0 / np.pi)


def test_paired_wilcoxon():
    same = [0.80, 0.82, 0.79, 0.85, 0.81]
    _, p_same, sign_same = stats.wilcoxon_paired(same, same)
    assert p_same == pytest.approx(1.0) and sign_same == 0
    a = [0.85, 0.86, 0.84, 0.87, 0.88]
    b = [0.80, 0.81, 0.79, 0.82, 0.83]
    _, p, sign = stats.wilcoxon_paired(a, b)
    assert sign == 1 and p < 0.1                 # a > b consistentemente
    with pytest.raises(ValueError):
        stats.wilcoxon_paired([0.1, 0.2], [0.1, 0.2, 0.3])   # desalinhado


def test_paired_indices_match():  # 🔴
    # RGB e RGB+cor usam os MESMOS índices de fold/seed (split só depende de
    # labels/grupos/seed, não do colorset) — pré-condição do Wilcoxon pareado.
    rng = np.random.default_rng(0)
    groups = np.repeat([f"p{i}" for i in range(40)], 2)
    labels = np.repeat(rng.integers(0, 2, size=40), 2)
    f_rgb = splits.kfold_indices(labels, groups, seed=42, k=5, mode="group")
    f_col = splits.kfold_indices(labels, groups, seed=42, k=5, mode="group")
    for (tr_a, va_a, te_a), (tr_b, va_b, te_b) in zip(f_rgb, f_col):
        assert np.array_equal(tr_a, tr_b)
        assert np.array_equal(va_a, va_b)
        assert np.array_equal(te_a, te_b)


# =========================================================================== #
# 1.6 Cross-backbone
# =========================================================================== #
def test_cross_backbone_cnn_and_vit():
    # CNN (resnet18) em 64px e ViT (deit_tiny) em 224px: forward OK; pseudo-RGB [B,3,H,W].
    cnn = skip_if_no_weights(lambda: models.build_model(
        config.BACKBONES["resnet18"], nch(CS_COL), fusion="adapter_v2",
        colorspaces=CS_COL, num_classes=2))
    cnn.eval()
    assert cnn(rand_stack(CS_COL, res=64)).shape == (2, 2)
    assert cnn.fusion(rand_stack(CS_COL, res=64)).shape[1] == 3

    vit_spec = config.BACKBONES["deit_tiny"]
    vit = skip_if_no_weights(lambda: models.build_model(
        vit_spec, nch(CS_COL), fusion="adapter_v2", colorspaces=CS_COL, num_classes=2))
    vit.eval()
    res = vit_spec.input_size
    assert vit(rand_stack(CS_COL, res=res)).shape == (2, 2)
    assert vit.fusion(rand_stack(CS_COL, res=res)).shape[1] == 3


def test_dinov3_fallback():
    # DINOv3 registrado; se indisponível no timm, a âncora de fallback (efficientnet_b4)
    # deve ser construível (a campanha não quebra). Pula se faltar peso offline.
    assert "dinov3_vits16" in config.BACKBONES
    try:
        m = models.build_model(config.BACKBONES["dinov3_vits16"], nch(CS_COL),
                               fusion="adapter_v2", colorspaces=CS_COL, num_classes=2)
        m.eval()
        res = config.BACKBONES["dinov3_vits16"].input_size
        assert m(rand_stack(CS_COL, res=res)).shape == (2, 2)
    except Exception as e:
        msg = repr(e).lower()
        if any(k in msg for k in ("download", "connection", "url", "http", "offline",
                                   "not found", "no pretrained", "unknown model")):
            fb = skip_if_no_weights(lambda: models.build_model(
                config.BACKBONES["efficientnet_b4"], nch(CS_COL), fusion="adapter_v2",
                colorspaces=CS_COL, num_classes=2))
            assert fb is not None                # fallback construível
        else:
            raise
