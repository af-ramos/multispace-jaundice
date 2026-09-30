"""
test_smoke.py — smoke tests ponta-a-ponta minúsculos (§2 de docs/TESTING.md).

Objetivo: validar o PIPELINE (wiring forward/backward/param-groups/otimizador,
fusão ativa, atenção CCAT, audit), NÃO a métrica. Rodam em CPU.

Os smokes que exigem o DATASET + laço de treino real (cv_harness, hpo_lowfidelity,
multitask_neo, skin_roi_sam, resume_idempotent) são pesados para a máquina local
(ver política do projeto: treino pesado só no cluster). Ficam **gated** por
``RUN_SLOW_SMOKES=1`` e trazem o comando real no corpo; por padrão são pulados,
mantendo o gate da Fase 0 offline e rápido.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from src import models
from tests.conftest import (CS_COL, CS_RGB, SPEC_CNN,
                                                      nch, rand_stack,
                                                      skip_if_no_weights)

SLOW = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_SMOKES"),
    reason="smoke pesado (dataset+treino): rodar no cluster com RUN_SLOW_SMOKES=1")


def _one_train_step(model, x, y):
    """Um passo de otimização usando os param-groups do modelo (differential LR)."""
    opt = torch.optim.AdamW(model.build_param_groups(1e-3))
    model.train()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.detach())


def _fusion_nparams(model) -> int:
    return sum(p.numel() for p in model.fusion.parameters())


# --------------------------------------------------------------------------- #
# Smokes leves (offline, sempre executados)
# --------------------------------------------------------------------------- #
def test_smoke_rgb_vs_color_same_hparams(tmp_path):  # 🔴
    """Contrato §0 fim-a-fim: os dois braços com o MESMO best_params; backbone com
    nº de params idêntico; fusão difere (3->3 vs 9->3); 2 marcadores escritos."""
    from src.engine import HParams
    best = HParams(lr=1e-3, batch_size=8, fusion="adapter_v2")     # best_params congelado

    torch.manual_seed(0)
    m_rgb = skip_if_no_weights(lambda: models.build_model(
        SPEC_CNN, nch(CS_RGB), fusion=best.fusion, colorspaces=CS_RGB, num_classes=2))
    torch.manual_seed(0)
    m_col = skip_if_no_weights(lambda: models.build_model(
        SPEC_CNN, nch(CS_COL), fusion=best.fusion, colorspaces=CS_COL, num_classes=2))

    n_bb_rgb = sum(p.numel() for p in m_rgb.backbone.parameters())
    n_bb_col = sum(p.numel() for p in m_col.backbone.parameters())
    assert n_bb_rgb == n_bb_col                                    # backbone idêntico
    assert _fusion_nparams(m_rgb) != _fusion_nparams(m_col)        # fusão 3->3 vs 9->3

    y = torch.tensor([0, 1])
    markers = {}
    for tag, m, cs in [("RGB", m_rgb, CS_RGB), ("RGB+YCrCb+HSV", m_col, CS_COL)]:
        loss = _one_train_step(m, rand_stack(cs, batch=2), y)
        assert loss == loss                                        # finito (não NaN)
        mk = tmp_path / f"marker_{tag.replace('+', '_')}.json"
        mk.write_text(json.dumps({"colorset": tag, "best_params": vars(best)}))
        markers[tag] = json.loads(mk.read_text())
    assert len(list(tmp_path.glob("marker_*.json"))) == 2          # 2 marcadores
    assert markers["RGB"]["best_params"] == markers["RGB+YCrCb+HSV"]["best_params"]


def test_smoke_explain_activity():  # 🔴
    """TESTE DE ATIVIDADE DA FUSÃO ponta-a-ponta: injetar cromância muda os logits."""
    m = skip_if_no_weights(lambda: models.build_model(
        SPEC_CNN, nch(CS_COL), fusion="adapter_v2", colorspaces=CS_COL, num_classes=2))
    m.eval()
    x = rand_stack(CS_COL)
    with torch.no_grad():
        base = m(x)
        x2 = x.clone(); x2[:, 3:] += 3.0
        pert = m(x2)
    assert float((pert - base).norm()) > 1e-6, "fusão inerte -> depurar antes de interpretar nulos"


def test_smoke_audit(tmp_path, dataset_root, monkeypatch):  # 🔴
    """Auditoria offline: leak=0 nos 5 folds (NeoJaundice + NJN); relatório gerado."""
    from src import audit
    original_cfg = audit._cfg
    monkeypatch.setattr(audit, "_cfg", lambda *a, **kw:
                        original_cfg(*a, data_root=str(dataset_root), **kw))
    out = tmp_path / "audit_report.md"
    ok = audit.run_audit(["neojaundice", "njn"], out)
    assert ok, "audit sujo (leak>0 ou contagem divergente) — ver relatório"
    assert out.is_file() and "leak=0" in out.read_text()


def test_smoke_ccat_attention():
    """CCAT: attn_maps populado e resíduo ≈ 0 na época 0 (Wo zero-init)."""
    cs = ["RGB", "LAB", "HSV"]
    m = skip_if_no_weights(lambda: models.build_model(
        SPEC_CNN, nch(cs), fusion="ccat", colorspaces=cs, num_classes=2))
    m.eval()
    x = rand_stack(cs)
    pseudo = m.fusion(x)
    assert m.fusion.attn_maps is not None
    assert torch.allclose(pseudo, x[:, :3], atol=1e-4)             # resíduo ≈ 0 na init


# --------------------------------------------------------------------------- #
# Smokes pesados (dataset + treino real, minúsculo). Gated por RUN_SLOW_SMOKES.
# CPU-safe: num_workers=0 (o fork+OpenCV derruba workers em algumas máquinas locais;
# no cluster/GPU o default num_workers=8 é usado). limit_per_split=8, 1 época.
# --------------------------------------------------------------------------- #
def _unit_args(res, cache, dataset="njn", extra=None):
    a = ["--dataset", dataset, "--backbone", "resnet18", "--colors", "RGB",
         "--fusion", "adapter_v2", "--folds", "2", "--seeds", "42",
         "--epochs", "1", "--patience", "1", "--batch-size", "4",
         "--limit-per-split", "8", "--num-workers", "0",
         "--results-dir", str(res), "--cache-dir", str(cache)]
    return a + (extra or [])


def _find_marker(res_dir):
    ms = list(res_dir.rglob("*.json"))
    assert ms, "nenhum marcador JSON escrito"
    return json.loads(ms[0].read_text()), ms[0]


@SLOW
def test_smoke_cv_harness(tmp_path):
    """5-fold×seed (aqui 2×1) com pooled_oof; marcador com std_seed_auc/gate_stable."""
    from src import train
    res, cache = tmp_path / "res", tmp_path / "cache"
    assert train.main(_unit_args(res, cache)) == 0
    rec, _ = _find_marker(res)
    assert rec["status"] == "completed"
    for k in ("std_seed_auc", "gate_stable", "threshold", "per_seed_metrics",
              "test_metrics_mean", "folds", "seeds"):
        assert k in rec, f"marcador sem campo {k}"


@SLOW
def test_smoke_hpo_lowfidelity():
    """HPO low-fidelity: 2 trials, low-fidelity; best_params dumpado; sem tocar o test."""
    import torch as _t
    from src import config, hpo
    cfg = config.ExperimentConfig(backbone="resnet18", dataset="NJN", colorspaces=("RGB",),
                                  fusion="adapter_v2", hpo_trials=2, search_epochs=1,
                                  num_workers=0, limit_per_split=8, cv_folds=2, batch_size=4)
    best_hp, study = hpo.run_hpo(cfg, _t.device("cpu"))
    assert len(study.trials) == 2
    bp = best_hp.to_dict()
    assert "lr" in bp and bp["fusion"] == "adapter_v2"


@SLOW
def test_smoke_multitask_neo(tmp_path):
    """NeoJaundice multitask: saídas cls+reg; MAE/R² no marcador; loss Kendall finita."""
    from src import train
    res, cache = tmp_path / "res", tmp_path / "cache"
    assert train.main(_unit_args(res, cache, dataset="neojaundice",
                                 extra=["--multitask"])) == 0
    rec, _ = _find_marker(res)
    assert rec["multitask"] is True
    assert rec["regression"] is not None
    assert "tsb_mae" in rec["regression"] and "tsb_r2" in rec["regression"]


@SLOW
def test_smoke_skin_roi(tmp_path):
    """NJN: pipeline de ROI roda em ambos os modos (skin_roi heurístico e full_image);
    ablação de ROI que a NJN exige (skin_roi heurístico piora -> comparar com full_image)."""
    from src import train
    for mode in ("skin_roi", "full_image"):
        res, cache = tmp_path / mode, tmp_path / f"c_{mode}"
        assert train.main(_unit_args(res, cache, extra=["--njn-mode", mode])) == 0
        rec, _ = _find_marker(res)
        assert rec["status"] == "completed"


@SLOW
def test_smoke_resume_idempotent(tmp_path):
    """Rodar 2× a mesma unidade: a 2ª PULA (marcador existe); nada é sobrescrito."""
    from src import train
    res, cache = tmp_path / "res", tmp_path / "cache"
    args = _unit_args(res, cache)
    assert train.main(args) == 0
    _, marker = _find_marker(res)
    mtime1 = marker.stat().st_mtime_ns
    assert train.main(args) == 0                  # 2ª execução: deve PULAR
    assert marker.stat().st_mtime_ns == mtime1    # marcador NÃO reescrito


@SLOW
def test_smoke_preds_dump_e_metricas_paciente(tmp_path):
    """Emenda 2026-07-25: o dump de predições sai alinhado com os loaders REAIS e o
    re-scoring reproduz as métricas-imagem do marcador.

    Usa NeoJaundice, o único dataset com patient_id real (NNNN-p.jpg -> NNNN); a NJN cai
    no caso degenerado 1 imagem = 1 pseudo-paciente e não exercitaria a agregação."""
    import numpy as np
    from src import train
    from src.metrics import compute_metrics
    from src.preds import PredDump, rescore_from_dump

    res, cache = tmp_path / "res", tmp_path / "cache"
    assert train.main(_unit_args(res, cache, dataset="neojaundice")) == 0
    rec, marker = _find_marker(res)

    # 1. o marcador aponta para um dump que EXISTE
    assert rec["preds_file"], "marcador sem preds_file"
    pfile = res / rec["preds_file"]
    assert pfile.is_file(), f"dump ausente: {pfile}"

    d = PredDump.load(pfile)
    assert d.n_rows > 0
    assert set(np.unique(d.splits)) <= {"val", "test"}
    assert set(d.seed_list()) == set(rec["seeds"])

    # 2. RE-SCORING reproduz as métricas-imagem do marcador (prova de alinhamento:
    #    se path e prob estivessem trocados, a AUC não bateria).
    got = rescore_from_dump(d, split="test", threshold=rec["threshold"], by_patient=False)
    for s, esperado in rec["per_seed_metrics"].items():
        obtido = got["per_seed"][int(s)]
        for k in ("accuracy", "f1_macro", "roc_auc"):
            assert obtido[k] == pytest.approx(esperado[k], abs=1e-9), \
                f"re-scoring divergiu em {k} (seed {s}): {obtido[k]} != {esperado[k]}"

    # 3. métricas-paciente no marcador, com MENOS unidades que imagens
    assert rec["test_metrics_patient_mean"] is not None
    assert rec["threshold_patient"] is not None
    n_pat = rec["test_metrics_patient_mean"]["n_patients"]
    n_img = len(d.subset(split="test", seed=rec["seeds"][0]).y_true)
    assert 0 < n_pat <= n_img

    # 4. o limiar-paciente NÃO veio do test: recomputá-lo só do val dá o mesmo número
    from src.preds import patient_threshold_from_val
    assert patient_threshold_from_val(d) == pytest.approx(rec["threshold_patient"], abs=1e-9)


@SLOW
def test_smoke_gate_alpha_dump(tmp_path):
    """Emenda 2026-07-26: o gate por espaço do adapter_v2 chega ao marcador.

    Fecha metade da pendência de interpretabilidade da Fase 4: `_SpaceGate.last_weights`
    existia mas nada o lia, e os 496 marcadores da campanha saíram com alpha_space=None.
    O teste trava também o ROTULAMENTO: o vetor do gate inclui o RGB, então
    `alpha_space_names` tem de ter um nome por espaço da pilha — se herdasse os nomes
    crômicos do ChromaFiLM, a tabela sairia deslocada em uma posição, em silêncio."""
    from src import train

    res, cache = tmp_path / "res", tmp_path / "cache"
    args = _unit_args(res, cache, extra=["--colors", "RGB+LAB"])
    args[args.index("--colors") + 1] = "RGB+LAB"          # sobrepõe o RGB do default
    assert train.main(args) == 0
    rec, _ = _find_marker(res)

    assert rec["alpha_space"] is not None, "adapter_v2 sem alpha_space no marcador"
    assert rec["alpha_scope"] == "todos"
    assert rec["alpha_space_names"] == ["RGB", "LAB"], \
        f"nomes desalinhados: {rec['alpha_space_names']}"
    assert len(rec["alpha_space"]) == 2
    # Init neutra do _SpaceGate = 1,0 por espaço; após 1 época ainda deve ficar perto
    # disso (o diagnóstico de inércia é justamente esse). Só travamos a ORDEM DE GRANDEZA.
    assert all(0.0 < v < len(rec["alpha_space"]) for v in rec["alpha_space"])


@SLOW
def test_smoke_attn_dump_ccat(tmp_path):
    """Emenda 2026-07-26: os mapas de atenção do CCAT são persistidos (insumo da Fig 5).

    `CCATFusion.attn_maps` era sobrescrito a cada forward e perdido; como o v7 não salva
    pesos, sem este dump a Fig 5 exigiria re-treino toda vez."""
    import numpy as np
    from src import train
    from src.attn import AttnDump

    res, cache = tmp_path / "res", tmp_path / "cache"
    args = _unit_args(res, cache, extra=["--fusion", "ccat"])
    args[args.index("--colors") + 1] = "RGB+LAB"           # CCAT exige RGB + >=1 croma
    args[args.index("--fusion") + 1] = "ccat"
    assert train.main(args) == 0
    rec, _ = _find_marker(res)

    assert rec["attn_file"], "marcador CCAT sem attn_file"
    afile = res / rec["attn_file"]
    assert afile.is_file(), f"dump de atenção ausente: {afile}"

    d = AttnDump.load(afile)
    assert d.n_rows > 0
    assert list(d.space_names) == ["LAB"]                  # só o crômico entra
    assert d.maps.ndim == 4 and d.maps.shape[1] == 1
    assert set(np.unique(d.buckets)) <= {"TP", "TN", "FP", "FN"}
    # softmax sobre os espaços: com 1 espaço crômico o mapa é constante 1
    assert float(d.maps.astype(np.float32).mean()) == pytest.approx(1.0, abs=1e-2)
    # cota por bucket respeitada (default 4 por (seed, fold))
    for b in set(np.unique(d.buckets)):
        for s in set(d.seeds.tolist()):
            for f in set(d.folds.tolist()):
                n = int(((d.buckets == b) & (d.seeds == s) & (d.folds == f)).sum())
                assert n <= 4, f"cota estourada em {b}/{s}/{f}: {n}"
