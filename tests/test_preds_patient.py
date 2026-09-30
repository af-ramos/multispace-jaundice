"""
test_preds_patient.py — persistência de predições + métricas por paciente no harness de CV.

Cobre a emenda de 2026-07-25: o ``cv.py`` passa a (a) gravar as predições de val/test
por imagem e (b) logar métricas agregadas por PACIENTE no marcador. Isso destrava, como
RE-SCORING LOCAL (sem GPU, sem re-treino), as alavancas de agregação por paciente,
limiar e ensemble de seeds — hoje impossíveis porque nada é persistido.

Contratos verificados aqui:
  1. o dump de predições ida-e-volta preserva valores e alinhamento path↔prob;
  2. a agregação por paciente NÃO é escolhida no test (limiar vem do val);
  3. re-scoring a partir do dump reproduz EXATAMENTE as métricas-imagem do marcador
     (se não reproduz, o dump está desalinhado e qualquer métrica-paciente é lixo);
  4. dataset sem ID de paciente recuperável (NJN) cai no caso degenerado
     1 imagem = 1 paciente -> métrica-paciente == métrica-imagem.

Rodar: ``CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import preds as predsmod
from src.metrics import compute_metrics
from src.preds import (PredDump, patient_metrics_from_dump,
                                             rescore_from_dump)


def _fake_dump(n_seeds: int = 2, n_pat: int = 6, per_pat: int = 3, seed: int = 0) -> PredDump:
    """Dump sintético: n_pat pacientes × per_pat imagens, rótulo constante por paciente."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_seeds):
        for p in range(n_pat):
            y = int(p % 2)
            for i in range(per_pat):
                # prob correlacionada com o rótulo, mas ruidosa por imagem
                prob = float(np.clip(rng.normal(0.35 + 0.3 * y, 0.18), 0.001, 0.999))
                for split in ("val", "test"):
                    rows.append((split, 42 + s, 0, f"ds/{p:04d}-{i}.jpg", f"{p:04d}", y, prob))
    return PredDump.from_rows(rows)


# --------------------------------------------------------------------------- #
# 1. round-trip do dump (alinhamento path <-> prob)
# --------------------------------------------------------------------------- #
def test_dump_roundtrip_preserva_alinhamento(tmp_path):
    d = _fake_dump()
    p = tmp_path / "preds.npz"
    d.save(p)
    back = PredDump.load(p)

    assert back.n_rows == d.n_rows
    assert list(back.paths) == list(d.paths)
    assert list(back.patient_ids) == list(d.patient_ids)
    np.testing.assert_allclose(back.y_prob, d.y_prob, rtol=0, atol=0)
    np.testing.assert_array_equal(back.y_true, d.y_true)
    np.testing.assert_array_equal(back.seeds, d.seeds)
    # o path continua casando com a MESMA prob depois do round-trip
    for path, prob in zip(back.paths, back.y_prob):
        j = list(d.paths).index(path)
        assert d.y_prob[j] == pytest.approx(prob) or path in [d.paths[k] for k in range(d.n_rows)
                                                              if d.y_prob[k] == pytest.approx(prob)]


def test_dump_vazio_nao_quebra(tmp_path):
    d = PredDump.from_rows([])
    p = tmp_path / "vazio.npz"
    d.save(p)
    assert PredDump.load(p).n_rows == 0
    assert patient_metrics_from_dump(PredDump.load(p), threshold=0.5) == {}


# --------------------------------------------------------------------------- #
# 2/3. re-scoring reproduz as métricas-imagem (prova de alinhamento)
# --------------------------------------------------------------------------- #
def test_rescore_reproduz_metricas_imagem():
    d = _fake_dump(n_seeds=3)
    thr = 0.5
    got = rescore_from_dump(d, split="test", threshold=thr, by_patient=False)
    for s in np.unique(d.seeds):
        m = d.subset(split="test", seed=int(s))
        esperado = compute_metrics(list(m.y_true), y_prob=list(m.y_prob), threshold=thr)
        for k in ("accuracy", "f1_macro", "roc_auc"):
            assert got["per_seed"][int(s)][k] == pytest.approx(esperado[k], abs=1e-12)


# --------------------------------------------------------------------------- #
# 2. limiar-paciente NUNCA vem do test
# --------------------------------------------------------------------------- #
def test_limiar_paciente_escolhido_so_no_val():
    d = _fake_dump(n_seeds=2)
    thr_val = predsmod.patient_threshold_from_val(d, objective="f1_macro")
    # embaralhar os rótulos do TEST não pode mudar o limiar (ele só olha o val)
    d2 = d.copy()
    mask = d2.splits == "test"
    d2.y_true[mask] = 1 - d2.y_true[mask]
    assert predsmod.patient_threshold_from_val(d2, objective="f1_macro") == pytest.approx(thr_val)


# --------------------------------------------------------------------------- #
# 4. agregação por paciente: reduz n e é degenerada sem ID
# --------------------------------------------------------------------------- #
def test_agregacao_por_paciente_reduz_n():
    d = _fake_dump(n_pat=6, per_pat=3, n_seeds=1)
    img = rescore_from_dump(d, split="test", threshold=0.5, by_patient=False)
    pat = rescore_from_dump(d, split="test", threshold=0.5, by_patient=True)
    assert pat["per_seed"][42]["n_patients"] == 6
    assert img["per_seed"][42]["n"] == 18


def test_sem_patient_id_metrica_paciente_igual_a_imagem():
    """NJN degenerado: 1 imagem = 1 paciente -> as duas métricas coincidem."""
    rows = [("test", 42, 0, f"njn/img{i}.jpg", f"njn/img{i}.jpg", i % 2, 0.2 + 0.1 * (i % 5))
            for i in range(12)]
    d = PredDump.from_rows(rows)
    img = rescore_from_dump(d, split="test", threshold=0.5, by_patient=False)
    pat = rescore_from_dump(d, split="test", threshold=0.5, by_patient=True)
    for k in ("accuracy", "f1_macro", "roc_auc"):
        assert img["per_seed"][42][k] == pytest.approx(pat["per_seed"][42][k])


# --------------------------------------------------------------------------- #
# 5. ensemble de seeds é re-scoring puro (não precisa de re-treino)
# --------------------------------------------------------------------------- #
def test_ensemble_de_seeds_disponivel_offline():
    d = _fake_dump(n_seeds=3)
    ens = rescore_from_dump(d, split="test", threshold=0.5, by_patient=False, ensemble_seeds=True)
    assert "ensemble" in ens and "per_seed" not in ens.get("ensemble", {})
    # o ensemble tem o mesmo nº de unidades de UM seed (média das probs entre seeds)
    assert ens["ensemble"]["n"] == d.subset(split="test", seed=42).n_rows
