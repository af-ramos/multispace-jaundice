"""Testes das primitivas estatísticas da consolidação final (`fase6_final_stats`).

Guardam exatamente a classe de bug que já ocorreu neste projeto: **t crítico errado**
(tabela hard-coded chaveada por df mas indexada por n ⇒ IC ~7% estreito demais) e
pareamento desalinhado entre seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.fase6_final_stats import (cliffs_delta, pair_markers,
                                                         paired, seed_metric)


def test_t_critico_n5_e_2776():  # 🔴
    """Com n=5 o IC95 usa t₄=2,776 — nunca t=2,571 (df=5) nem z=1,96."""
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [0.0] * 5
    r = paired(a, b)
    sd = float(np.std(np.array(a), ddof=1))
    half = (r["hi"] - r["lo"]) / 2
    assert r["delta"] == pytest.approx(3.0)
    assert half / (sd / np.sqrt(5)) == pytest.approx(2.776, abs=1e-3)


def test_ic_exclui_zero_coerente_com_limites():
    assert paired([0.1] * 5, [0.0] * 5)["exclui_zero"] == "sim"     # variância 0, Δ>0
    assert paired([1, -1, 1, -1, 0], [0] * 5)["exclui_zero"] == "nao"


def test_wilcoxon_piso_n5():  # 🔴
    """Com n=5 seeds o menor p bilateral do Wilcoxon é 0,0625 — p<0,05 é INATINGÍVEL.

    É a razão de o critério de decisão ser o IC95, não o p (HYPOTHESES.md).
    """
    r = paired([0.9, 0.8, 0.7, 0.6, 0.5], [0.1, 0.2, 0.3, 0.4, 0.45])
    assert r["p_wilcoxon"] == pytest.approx(0.0625)
    assert r["p_wilcoxon"] > 0.05


def test_wilcoxon_diferencas_nulas_nao_levanta():
    """Contrato do projeto: entradas idênticas → p=1,0 (scipy levantaria ValueError)."""
    r = paired([0.5] * 5, [0.5] * 5)
    assert r["p_wilcoxon"] == 1.0 and r["delta"] == 0.0


def test_cliffs_delta_extremos_e_empate():
    assert cliffs_delta([3, 4, 5], [0, 1, 2]) == 1.0
    assert cliffs_delta([0, 1, 2], [3, 4, 5]) == -1.0
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0


def _marker(vals: dict) -> dict:
    return {"per_seed_metrics": {s: {"roc_auc": v} for s, v in vals.items()}}


def test_pair_markers_alinha_por_seed():  # 🔴
    """O pareamento é POR SEED — a ordem em que os seeds aparecem no JSON é irrelevante."""
    ma = _marker({"42": 0.90, "7": 0.80, "123": 0.85})
    mb = _marker({"7": 0.70, "123": 0.75, "42": 0.80})
    r = pair_markers(ma, mb)
    assert r["n"] == 3
    assert r["delta"] == pytest.approx(0.10)          # 3 pares de +0,10 cada


def test_pair_markers_usa_so_seeds_em_comum():
    ma = _marker({"42": 0.9, "7": 0.8, "999": 0.1})   # 999 não existe no outro braço
    mb = _marker({"42": 0.8, "7": 0.7})
    r = pair_markers(ma, mb)
    assert r["n"] == 2 and r["delta"] == pytest.approx(0.10)


def test_pair_markers_insuficiente_devolve_none():
    assert pair_markers(_marker({"42": 0.9}), _marker({"42": 0.8})) is None


def test_seed_metric_nivel_paciente():
    m = {"per_seed_metrics": {"42": {"roc_auc": 0.8}},
         "per_seed_patient_metrics": {"42": {"roc_auc": 0.9}}}
    assert seed_metric(m, nivel="imagem") == {"42": 0.8}
    assert seed_metric(m, nivel="paciente") == {"42": 0.9}
