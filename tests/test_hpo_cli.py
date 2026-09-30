"""
test_hpo_cli.py — CLI do HPO desacoplado (IMPLEMENTATION.md §8 + EXPERIMENT_MATRIX.md).

Contrato coberto:
  python -m src.hpo --backbone <BB> --dataset <DS> --colorset RGB \\
     --hpo-trials 20 --hpo-fold 0 --pruner hyperband --storage sqlite:///hpo.db --gpu 0
  → best_params/<backbone>_<dataset>.json  (idempotente: pula se já existe)

Os testes LEVES (default, offline, sem dataset/GPU) exercitam o parser, a fábrica de
pruner (HyperbandPruner), o dump do best_params e a IDEMPOTÊNCIA (pula sem tocar
torch/dataset). O teste PESADO ponta-a-ponta (roda Optuna real sobre o dataset) é
gated por ``RUN_SLOW_SMOKES=1``, como os demais smokes de treino (política do projeto:
treino pesado só no cluster).
"""

from __future__ import annotations

import json
import os

import pytest

from src import hpo

SLOW = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_SMOKES"),
    reason="smoke pesado (dataset+Optuna): rodar no cluster com RUN_SLOW_SMOKES=1")


# --------------------------------------------------------------------------- #
# Testes leves (offline, sem dataset/GPU) — sempre executados
# --------------------------------------------------------------------------- #
def test_parse_args_contrato():
    """O parser expõe EXATAMENTE os flags do contrato documentado."""
    ns = hpo._parse_args([
        "--backbone", "resnet18", "--dataset", "NJN", "--colorset", "RGB",
        "--hpo-trials", "20", "--hpo-fold", "0", "--pruner", "hyperband",
        "--storage", "sqlite:///hpo.db", "--gpu", "0",
    ])
    assert ns.backbone == "resnet18"
    assert ns.dataset == "NJN"
    assert ns.colorset == "RGB"
    assert ns.hpo_trials == 20
    assert ns.hpo_fold == 0
    assert ns.pruner == "hyperband"
    assert ns.storage == "sqlite:///hpo.db"
    assert ns.gpu == 0


def test_parse_args_defaults():
    """Defaults sensatos: colorset=RGB (controle), pruner=hyperband, 20 trials, fold 0."""
    ns = hpo._parse_args(["--backbone", "deit_tiny", "--dataset", "NeoJaundice"])
    assert ns.colorset == "RGB"
    assert ns.pruner == "hyperband"
    assert ns.hpo_trials == 20
    assert ns.hpo_fold == 0
    assert ns.storage is None


def test_build_pruner_hyperband():
    """--pruner hyperband → HyperbandPruner (multi-fidelity, §8); median → MedianPruner."""
    from optuna.pruners import HyperbandPruner, MedianPruner
    assert isinstance(hpo._build_pruner("hyperband", search_epochs=10), HyperbandPruner)
    assert isinstance(hpo._build_pruner("median", search_epochs=10), MedianPruner)


def test_best_params_path():
    """O caminho do dump é best_params/<backbone>_<dataset>.json (nome canônico)."""
    p = hpo._best_params_path("best_params", "resnet18", "NJN")
    assert p.name == "resnet18_NJN.json"
    p2 = hpo._best_params_path("best_params", "efficientnet_b0", "NeoJaundice")
    assert p2.name == "efficientnet_b0_NeoJaundice.json"


def test_dump_best_params_schema(tmp_path):
    """dump_best_params grava JSON atômico com best_params + metadados do estudo."""
    from src.engine import HParams

    class _StudyStub:
        best_value = 0.842
        trials = [1, 2, 3]  # len() = nº de trials

    best = HParams(lr=3e-4, n_unfreeze=10, augment=True, da_strength=0.5, fusion="adapter_v2")
    out = tmp_path / "resnet18_NJN.json"
    hpo.dump_best_params(out, best, _StudyStub(), backbone="resnet18", dataset="NJN",
                         colorset="RGB", objective="roc_auc", hpo_fold=0, search_epochs=20)

    rec = json.loads(out.read_text())
    assert rec["backbone"] == "resnet18"
    assert rec["dataset"] == "NJN"
    assert rec["objective"] == "roc_auc"
    assert rec["best_params"]["lr"] == 3e-4
    assert rec["best_params"]["fusion"] == "adapter_v2"
    assert rec["best_value"] == 0.842
    assert rec["n_trials"] == 3
    assert not (tmp_path / "resnet18_NJN.json.tmp").exists()  # escrita atômica limpa


def test_main_idempotente_pula(tmp_path, capsys):
    """Se best_params/<bb>_<ds>.json já existe, main() PULA (return 0) sem tocar
    torch/dataset e SEM reescrever o arquivo."""
    bp_dir = tmp_path / "best_params"
    bp_dir.mkdir()
    marker = bp_dir / "resnet18_NJN.json"
    sentinel = {"best_params": {"lr": 0.123}, "sentinel": True}
    marker.write_text(json.dumps(sentinel))
    mtime0 = marker.stat().st_mtime_ns

    rc = hpo.main(["--backbone", "resnet18", "--dataset", "NJN",
                   "--best-params-dir", str(bp_dir)])
    assert rc == 0
    assert marker.stat().st_mtime_ns == mtime0           # NÃO reescrito
    assert json.loads(marker.read_text()) == sentinel    # conteúdo intacto
    assert "skip" in capsys.readouterr().out.lower()


# --------------------------------------------------------------------------- #
# Teste pesado ponta-a-ponta (Optuna real sobre o dataset). Gated por RUN_SLOW_SMOKES.
# --------------------------------------------------------------------------- #
@SLOW
def test_hpo_cli_end_to_end(tmp_path):
    """CLI roda Optuna low-fidelity (2 trials) e dumpa best_params/<bb>_<ds>.json,
    sem NUNCA tocar o test (só val)."""
    bp_dir = tmp_path / "best_params"
    storage = f"sqlite:///{tmp_path/'hpo.db'}"
    rc = hpo.main([
        "--backbone", "resnet18", "--dataset", "NJN", "--colorset", "RGB",
        "--hpo-trials", "2", "--hpo-fold", "0", "--pruner", "hyperband",
        "--storage", storage, "--gpu", "0",
        "--search-epochs", "1", "--num-workers", "0", "--limit-per-split", "8",
        "--batch-size", "4", "--best-params-dir", str(bp_dir),
        "--results-dir", str(tmp_path / "res"), "--cache-dir", str(tmp_path / "cache"),
    ])
    assert rc == 0
    marker = bp_dir / "resnet18_NJN.json"
    assert marker.is_file()
    rec = json.loads(marker.read_text())
    assert "lr" in rec["best_params"]
    assert rec["backbone"] == "resnet18" and rec["dataset"] == "NJN"
