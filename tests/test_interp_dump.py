"""Testes da coleta de interpretabilidade (α do gate + mapas de atenção do CCAT).

Fecham a metade de INTERPRETABILIDADE da Fase 4, que ficou pendente por falta de
coleta: `CCATFusion.attn_maps` e `_SpaceGate.last_weights` existiam, mas nada os lia.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src import attn as attnmod
from src import config, models
from src.cv import _ALPHA_FUSIONS, _alpha_names
from tests.conftest import nch, rand_stack, skip_if_no_weights

CS_COL = ["RGB", "LAB", "YCrCb"]


class _Cfg:
    """Stub mínimo com o que `_alpha_names` lê."""

    def __init__(self, colorspaces):
        self.colorspaces = colorspaces


# --------------------------------------------------------------- α por espaço
def test_alpha_names_gate_inclui_rgb():  # 🔴
    """O gate do adapter_v2 tem um peso por GRUPO — RGB incluído.

    Rotular esse vetor com os nomes crômicos (que excluem o RGB) deslocaria a tabela
    em uma posição, em silêncio: o peso do RGB sairia como LAB.
    """
    cfg = _Cfg(CS_COL)
    assert _alpha_names(cfg, "todos") == ["RGB", "LAB", "YCrCb"]
    assert _alpha_names(cfg, "chroma") == ["LAB", "YCrCb"]
    assert _ALPHA_FUSIONS["adapter_v2"] == "todos"
    assert _ALPHA_FUSIONS["chromafilm"] == "chroma"


def test_space_gate_expoe_pesos_por_espaco():  # 🔴
    """`_SpaceGate.last_weights` é [B, K] e nasce ==1 (init neutra = adapter simples)."""
    gate = models._SpaceGate(n_channels=nch(CS_COL), n_groups=len(CS_COL))
    gate.eval()
    x = torch.randn(2, nch(CS_COL), 8, 8)
    w = gate(x)
    assert w.shape == (2, len(CS_COL))
    assert gate.last_weights is not None
    assert gate.last_weights.shape == (2, len(CS_COL))
    assert torch.allclose(w, torch.ones_like(w), atol=1e-5)   # logits zero -> softmax*K == 1


def test_fusion_adapter_v2_tem_gate_acessivel():  # 🔴
    """O coletor lê `model.fusion.gate.last_weights` — o caminho precisa existir."""
    fusion = models.ColorFusion([3, 3, 3], init="identity", gated=True, rgb_group=0)
    fusion.eval()
    fusion(torch.randn(2, 9, 16, 16))
    assert fusion.gate is not None and fusion.gate.last_weights is not None
    assert fusion.gate.last_weights.shape[1] == 3        # 3 espaços, RGB incluído


# --------------------------------------------------- mapas de atenção do CCAT
def test_ccat_attn_maps_forma_e_normalizacao():  # 🔴
    """attn_maps é [B, K_chroma, H, W] e soma ~1 por pixel (softmax sobre espaços)."""
    fusion = models.CCATFusion([3, 3, 3], rgb_group=0, dim=8, n_heads=2)
    fusion.eval()
    fusion(torch.randn(2, 9, 16, 16))
    am = fusion.attn_maps
    assert am is not None and am.shape == (2, 2, 16, 16)   # 2 espaços crômicos
    soma = am.float().sum(dim=1)
    assert torch.allclose(soma, torch.ones_like(soma), atol=1e-3)


def test_attn_dump_roundtrip(tmp_path):
    recs = [{"path": f"img{i}.jpg", "seed": 42, "fold": 0, "y_true": i % 2,
             "y_prob": 0.1 * i, "bucket": "TP", "map": np.random.rand(2, 8, 8)}
            for i in range(3)]
    d = attnmod.AttnDump.from_records(recs, ["LAB", "YCrCb"])
    p = tmp_path / "a.npz"
    d.save(p)
    r = attnmod.AttnDump.load(p)
    assert r.n_rows == 3 and r.n_spaces == 2
    assert list(r.space_names) == ["LAB", "YCrCb"]
    assert r.maps.shape == (3, 2, 8, 8)
    np.testing.assert_allclose(r.maps, d.maps)


def test_attn_dump_vazio_nao_quebra(tmp_path):
    d = attnmod.AttnDump.from_records([], ["LAB"])
    assert d.n_rows == 0 and d.space_share() == {}


def test_bucket_of_classifica_acerto_e_erro():
    assert attnmod.bucket_of(1, 0.9) == "TP"
    assert attnmod.bucket_of(1, 0.1) == "FN"
    assert attnmod.bucket_of(0, 0.9) == "FP"
    assert attnmod.bucket_of(0, 0.1) == "TN"


def test_select_records_cota_por_bucket_sem_cherry_picking():  # 🔴
    """A amostragem é por COTA na ordem do loader — não escolhe as mais confiantes.

    Se ordenasse por confiança, a Fig 5 mostraria só os casos fáceis; a cota fixa
    garante que os erros (FP/FN) apareçam sempre que existirem.
    """
    n = 20
    paths = [f"i{i}.jpg" for i in range(n)]
    y = [1] * 10 + [0] * 10
    p = [0.9 - 0.05 * i for i in range(10)] + [0.1 + 0.05 * i for i in range(10)]
    maps = [np.zeros((1, 4, 4)) for _ in range(n)]
    recs = attnmod.select_records(paths, y, p, maps, seed=1, fold=0, n_per_bucket=2)
    por_bucket = {}
    for r in recs:
        por_bucket.setdefault(r["bucket"], []).append(r["path"])
    assert all(len(v) <= 2 for v in por_bucket.values())
    assert set(por_bucket) == {"TP", "FN", "TN", "FP"}      # erros incluídos
    assert por_bucket["TP"] == ["i0.jpg", "i1.jpg"]         # primeiros da ordem, não os "melhores"


def test_attn_path_fica_ao_lado_do_marcador(tmp_path):
    m = tmp_path / "NJN" / "deit_tiny" / "f4_x__deit_tiny__RGB+HSV__ccat.json"
    assert attnmod.attn_path_for(m).name == "f4_x__deit_tiny__RGB+HSV__ccat.npz"
    assert attnmod.attn_path_for(m).parent.name == "attn"


def test_space_share_soma_um():
    """Como o softmax do CCAT é sobre os espaços, as frações somam ~1."""
    maps = np.random.rand(4, 3, 8, 8)
    maps /= maps.sum(axis=1, keepdims=True)
    recs = [{"path": f"i{i}.jpg", "seed": 1, "fold": 0, "y_true": 1, "y_prob": 0.7,
             "bucket": "TP", "map": maps[i]} for i in range(4)]
    d = attnmod.AttnDump.from_records(recs, ["LAB", "YCrCb", "HSV"])
    assert sum(d.space_share().values()) == pytest.approx(1.0, abs=1e-2)


# ------------------------------------------------------------ gate de refresh
def test_refresh_interp_so_dispara_onde_produz_algo():  # 🔴
    """`--refresh-missing-interp` não pode re-rodar células que não geram interpretabilidade."""
    from src.train import _falta_interp

    class _A:
        results_dir = "/nao/existe"

    class _C:
        def __init__(self, f):
            self.fusion = f

    completo = {"alpha_space": [1.0, 1.0], "attn_file": None}
    vazio = {"alpha_space": None, "attn_file": None}
    assert _falta_interp(vazio, _C("adapter_v2"), _A()) is True
    assert _falta_interp(completo, _C("adapter_v2"), _A()) is False
    assert _falta_interp(vazio, _C("naive_concat"), _A()) is False   # nunca re-roda
    assert _falta_interp(vazio, _C("ccat"), _A()) is True            # sem npz -> refaz


def test_refresh_interp_ccat_exige_npz_em_disco(tmp_path):  # 🔴
    """O campo no marcador não basta: o .npz é gitignored e pode não ter vindo no rsync."""
    from src.train import _falta_interp

    class _A:
        results_dir = str(tmp_path)

    class _C:
        fusion = "ccat"

    prev = {"alpha_space": None, "attn_file": "NJN/deit_tiny/attn/x.npz"}
    assert _falta_interp(prev, _C(), _A()) is True                   # arquivo ausente
    p = tmp_path / "NJN" / "deit_tiny" / "attn"
    p.mkdir(parents=True)
    (p / "x.npz").write_bytes(b"0")
    assert _falta_interp(prev, _C(), _A()) is False


# ------------------------------------------------- integração leve (sem GPU)
def test_ccat_model_expoe_attn_apos_forward():
    spec = config.BACKBONES["deit_tiny"]
    m = skip_if_no_weights(lambda: models.build_model(
        spec, nch(CS_COL), fusion="ccat", colorspaces=CS_COL, num_classes=2))
    m.eval()
    with torch.no_grad():
        m(rand_stack(CS_COL, res=spec.input_size))
    assert m.fusion_type == "ccat"
    assert m.fusion.attn_maps is not None
    assert m.fusion.attn_maps.shape[1] == 2      # LAB, YCrCb


# ------------------------------------------------------- tabela e figura
def test_gate_rows_detecta_inercia():  # 🔴
    """Gate ~1,0 em todos os espaços = INERTE (a BN reabsorve um escalar por espaço)."""
    from src.explain import GATE_INERTE_SPREAD, gate_rows

    inerte = {"dataset": "NJN", "backbone": "deit_tiny", "colorset_id": "RGB+LAB",
              "fusion": "adapter_v2", "njn_mode": "full_image", "alpha_scope": "todos",
              "alpha_space": [1.001, 0.999], "alpha_space_names": ["RGB", "LAB"]}
    ativo = dict(inerte, alpha_space=[1.4, 0.6])
    r_in, r_at = gate_rows([inerte])[0], gate_rows([ativo])[0]
    assert r_in["veredito"] == "INERTE" and r_in["spread"] < GATE_INERTE_SPREAD
    assert r_at["veredito"] == "ativo"
    assert r_in["pesos"] == {"RGB": 1.001, "LAB": 0.999}


def test_gate_rows_ignora_marcador_sem_alpha():
    from src.explain import gate_rows
    assert gate_rows([{"dataset": "NJN", "alpha_space": None}]) == []


def test_fig5_escala_simetrica_em_torno_do_empate():  # 🔴
    """A escala da Fig 5 é simétrica em torno de 1/K e compartilhada entre painéis.

    Se cada painel normalizasse pelo próprio min/max, a mesma cor significaria desvios
    diferentes em painéis diferentes — a figura mentiria por construção.
    """
    from src.make_fig5 import _escala

    maps = np.full((4, 2, 8, 8), 0.5, dtype=np.float16)
    maps[0, 0] = 0.8
    maps[0, 1] = 0.2
    recs = [{"path": f"i{i}.jpg", "seed": 1, "fold": 0, "y_true": 1, "y_prob": 0.7,
             "bucket": "TP", "map": maps[i]} for i in range(4)]
    d = attnmod.AttnDump.from_records(recs, ["LAB", "YCrCb"])
    half = _escala(d)
    assert half > 0
    assert half <= 0.3 + 1e-3            # |0,8 − 0,5| é o desvio máximo (tol. float16)


def test_fig5_escala_nunca_degenera():
    """Mapa constante (K=1, softmax trivial) não pode gerar vmin==vmax."""
    from src.make_fig5 import _escala

    recs = [{"path": "i.jpg", "seed": 1, "fold": 0, "y_true": 1, "y_prob": 0.7,
             "bucket": "TP", "map": np.ones((1, 4, 4))}]
    assert _escala(attnmod.AttnDump.from_records(recs, ["LAB"])) > 0


def test_fig5_cmap_tem_cinza_neutro_no_meio():  # 🔴
    """Rampa divergente: polos opostos e MEIO NEUTRO (nunca uma cor no midpoint)."""
    from src.make_fig5 import _cmap_divergente

    cm = _cmap_divergente()
    r, g, b, _ = cm(0.5)
    assert abs(r - g) < 0.05 and abs(g - b) < 0.05      # cinza: canais ~iguais
    lo, hi = cm(0.0), cm(1.0)
    assert lo[2] > lo[0] and hi[0] > hi[2]              # frio vs quente


def test_fig5_casos_incluem_erros():  # 🔴
    """A figura não pode mostrar só acertos — FP/FN entram por cota própria."""
    from src.make_fig5 import _casos

    recs = []
    for b, y, p in [("TP", 1, 0.9), ("TN", 0, 0.1), ("FP", 0, 0.8), ("FN", 1, 0.2)]:
        for j in range(3):
            recs.append({"path": f"{b}{j}.jpg", "seed": 42 + j, "fold": 0, "y_true": y,
                         "y_prob": p, "bucket": b, "map": np.zeros((1, 4, 4))})
    d = attnmod.AttnDump.from_records(recs, ["LAB"])
    idx = _casos(d, por_bucket=1)
    assert {str(d.buckets[i]) for i in idx} == {"TP", "TN", "FP", "FN"}
