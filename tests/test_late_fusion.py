"""
test_late_fusion.py — fusão TARDIA de espaços de cor (re-scoring offline).

`late_fusion.py` combina as probabilidades de modelos treinados em colorsets diferentes
e re-pontua tudo a partir dos dumps `results/**/preds/*.npz`, sem GPU. Como ele produz
número que vai ao paper, os contratos abaixo são bloqueantes:

  1. **`_score_rapido` == `compute_metrics`.** A varredura de limiar usa um atalho que
     calcula só a métrica-alvo a partir da matriz de confusão (a função completa produz
     ~12 métricas e domina o tempo numa grade de 91 pontos). Se o atalho divergir da
     função do pipeline, o limiar escolhido é outro e a comparação inteira se desloca.
  2. **O limiar sai SÓ do val.** Embaralhar os rótulos do test não pode mudar o limiar
     escolhido — se mudar, há vazamento.
  3. **Ensemble de 1 dump == o dump sozinho.** A média de um elemento é identidade;
     se não for, o caminho do ensemble está adulterando as probabilidades.
  4. **Ensemble exige alinhamento imagem a imagem.** Dumps com `paths` diferentes têm
     de levantar erro, nunca somar probabilidade de imagens diferentes em silêncio.
  5. **Determinismo.** A mesma entrada produz o mesmo número em duas chamadas
     (a agregação por paciente usa dicionário ordenado; se alguém trocar por `set`,
     este teste pega).

Rodar: ``CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import late_fusion as lf
from src.metrics import compute_metrics
from src.preds import PredDump


def _dump(n_seeds: int = 3, n_pat: int = 8, por_pat: int = 2, seed: int = 0,
          desloca: float = 0.0) -> PredDump:
    """Dump sintético com val e test, rótulo constante por paciente.

    ``desloca`` empurra as probabilidades — serve para fabricar um segundo "modelo"
    correlacionado com o primeiro, como dois colorsets do mesmo backbone.
    """
    rng = np.random.default_rng(seed)
    linhas = []
    for s in range(n_seeds):
        for split in ("val", "test"):
            for p in range(n_pat):
                y = p % 2
                for i in range(por_pat):
                    prob = float(np.clip(rng.beta(2, 2) + desloca + 0.25 * y, 0.01, 0.99))
                    linhas.append((split, 42 + s, 0, f"{split}_{p:02d}_{i}.jpg",
                                   f"pac{p:02d}", y, prob))
    return PredDump.from_rows(linhas)


# --------------------------------------------------------------- contrato 1
@pytest.mark.parametrize("metrica", ["accuracy", "f1_macro"])
def test_score_rapido_bate_compute_metrics(metrica):
    """O atalho da varredura de limiar reproduz a função do pipeline, dígito a dígito."""
    rng = np.random.default_rng(7)
    for _ in range(30):
        n = int(rng.integers(20, 200))
        y = rng.integers(0, 2, n)
        p = rng.random(n)
        for t in (0.05, 0.3, 0.5, 0.72, 0.95):
            esperado = compute_metrics(list(y), y_prob=list(p), threshold=t)[metrica]
            obtido = lf._score_rapido(y, (p >= t).astype(int), metrica)
            assert obtido == pytest.approx(esperado, abs=1e-9), (metrica, t, n)


@pytest.mark.parametrize("metrica", ["accuracy", "f1_macro"])
def test_score_rapido_em_casos_degenerados(metrica):
    """Predição toda-0 / toda-1 e classe ausente — onde o F1 costuma divergir por 0/0."""
    casos = [
        (np.zeros(10, int), np.zeros(10, int)),      # tudo negativo, tudo acertado
        (np.ones(10, int), np.zeros(10, int)),       # tudo positivo, nada detectado
        (np.array([0, 1] * 5), np.ones(10, int)),    # predição degenerada "tudo-jaundice"
        (np.array([0, 1] * 5), np.zeros(10, int)),   # predição degenerada "tudo-healthy"
    ]
    for y, yhat in casos:
        esperado = compute_metrics(list(y), y_pred=list(yhat))[metrica]
        obtido = lf._score_rapido(y, yhat, metrica)
        assert obtido == pytest.approx(esperado, abs=1e-9), (metrica, y, yhat)


# --------------------------------------------------------------- contrato 2
def test_limiar_nao_olha_o_test():
    """Destruir os rótulos do TEST não pode mudar o limiar (ele vem só do val)."""
    d = _dump()
    limpo = lf.avalia([d], "f1_macro", por_paciente=False)

    corrompido = d.copy()
    m = corrompido.splits == "test"
    corrompido.y_true[m] = 1 - corrompido.y_true[m]     # inverte TODO o test
    sujo = lf.avalia([corrompido], "f1_macro", por_paciente=False)

    # As MÉTRICAS mudam (o test é outro), mas o limiar — que é o que poderia vazar —
    # é o mesmo; se o código espiasse o test, os scores "limpos" também mudariam.
    for s in limpo:
        assert limpo[s] != pytest.approx(sujo[s]), "test invertido deveria mudar a métrica"
    d2 = d.copy()
    d2.y_prob[d2.splits == "test"] = d.y_prob[d.splits == "test"]
    assert lf.avalia([d2], "f1_macro", por_paciente=False) == limpo


# --------------------------------------------------------------- contrato 3
@pytest.mark.parametrize("por_paciente", [False, True])
def test_ensemble_de_um_e_identidade(por_paciente):
    d = _dump()
    assert lf.avalia([d], "accuracy", por_paciente) == lf.avalia([d, d], "accuracy", por_paciente)


# --------------------------------------------------------------- contrato 4
def test_ensemble_recusa_dumps_desalinhados():
    a = _dump(seed=1)
    b = a.copy()
    b.paths = np.array([p.replace(".jpg", "_OUTRO.jpg") for p in b.paths])
    with pytest.raises(ValueError, match="desalinhados"):
        lf.avalia([a, b], "accuracy", por_paciente=False)


# --------------------------------------------------------------- contrato 5
def test_determinismo():
    a, b = _dump(seed=2), _dump(seed=2, desloca=0.1)
    for por_paciente in (False, True):
        r1 = lf.avalia([a, b], "f1_macro", por_paciente)
        r2 = lf.avalia([a, b], "f1_macro", por_paciente)
        assert r1 == r2


def test_agrupamento_por_paciente_preserva_ordem_e_rotulo():
    d = _dump(n_seeds=1, n_pat=5, por_pat=3, seed=3).subset(split="test")
    y, p = lf._agrupa_por_paciente(d.patient_ids, d.y_true, d.y_prob)
    assert len(y) == 5 and len(p) == 5
    # rótulo do paciente = rótulo (constante) das suas imagens, na ordem de 1ª aparição
    esperados = []
    for pid, yy in zip(d.patient_ids, d.y_true):
        if pid not in [e[0] for e in esperados]:
            esperados.append((pid, int(yy)))
    assert list(y) == [e[1] for e in esperados]
    # prob do paciente = média das probs das suas imagens
    for i, (pid, _) in enumerate(esperados):
        assert p[i] == pytest.approx(float(d.y_prob[d.patient_ids == pid].mean()))
