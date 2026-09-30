"""
models.py
=========

A arquitetura **Hybrid Color-Space Fusion (HCSF)** e o registry dos 10 backbones.

Fluxo do modelo (``HCSFModel``):

    entrada [N, H, W]  (N = 3 x nº de espacos de cor)
        │
        ├─ FUSAO  ──────────────────────────────────────────────
        │   • "adapter" (metodo principal): BatchNorm(N) + Conv1x1(N→3).
        │       Projeta a pilha multi-espaco num "pseudo-RGB" de 3 canais,
        │       preservando 100% dos pesos pre-treinados do backbone.
        │       Init configuravel: "identity" (pseudo-RGB inicial == canais RGB,
        │       demais espacos como residuo zero) ou "balanced" (todos os
        │       espacos contribuem igualmente desde o inicio).
        │   • "adapter_v2": adapter + gating SE por espaco de cor — pesos
        │       dinamicos por imagem (interpretaveis), init neutra (gate==1).
        │   • "inflate" (ablacao): remove a fusao e substitui a 1a convolucao
        │       do backbone por uma versao de N canais, inicializada a partir
        │       da media dos pesos RGB pre-treinados (mantem a escala).
        │
        ├─ BACKBONE pre-treinado (ResNet/DenseNet/Inception/EfficientNet/ViT/DeiT)
        │   com profundidade de fine-tuning configuravel (0/10/50/100 ultimas
        │   camadas descongeladas, ou -1 = backbone inteiro, como na baseline CNN).
        │
        └─ CABECA  Linear(feat → 2)  (+ ativacao opcional)  → logits de 2 classes

O backbone e sempre exposto como extrator de features (classificador trocado por
Identity / ``num_classes=0`` no timm), de modo que um unico motor de treino sirva
CNNs e ViTs.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm

from .config import BackboneSpec, timm_name_of


# ---------------------------------------------------------------------------
# Carregamento dos backbones (torchvision e timm)
# ---------------------------------------------------------------------------
def _load_backbone(spec: BackboneSpec) -> Tuple[nn.Module, int, Callable[[], nn.Conv2d], Callable[[nn.Conv2d], None]]:
    """Devolve (backbone_extrator, feat_dim, get_first_conv, set_first_conv).

    ``backbone_extrator(x) -> [B, feat_dim]`` (sem cabeca de classificacao).
    Os getters/setters da 1a convolucao permitem a fusao "inflate".
    """
    b = spec.builder

    if spec.source == "torchvision":
        if b == "resnet18":
            m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            feat = m.fc.in_features; m.fc = nn.Identity()
            get = lambda: m.conv1; set_ = lambda c: setattr(m, "conv1", c)
        elif b == "resnet50":
            m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
            feat = m.fc.in_features; m.fc = nn.Identity()
            get = lambda: m.conv1; set_ = lambda c: setattr(m, "conv1", c)
        elif b == "densenet121":
            m = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1)
            feat = m.classifier.in_features; m.classifier = nn.Identity()
            get = lambda: m.features.conv0; set_ = lambda c: setattr(m.features, "conv0", c)
        elif b == "inception_v3":
            m = tvm.inception_v3(weights=tvm.Inception_V3_Weights.IMAGENET1K_V1)
            m.aux_logits = False; m.AuxLogits = None  # desliga ramo auxiliar
            m.transform_input = False                 # nao re-normalizar (entrada e pseudo-RGB)
            feat = m.fc.in_features; m.fc = nn.Identity()
            get = lambda: m.Conv2d_1a_3x3.conv; set_ = lambda c: setattr(m.Conv2d_1a_3x3, "conv", c)
        elif b == "efficientnet_b0":
            m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
            feat = m.classifier[1].in_features; m.classifier = nn.Identity()
            get = lambda: m.features[0][0]; set_ = lambda c: m.features[0].__setitem__(0, c)
        elif b == "efficientnet_b4":
            m = tvm.efficientnet_b4(weights=tvm.EfficientNet_B4_Weights.IMAGENET1K_V1)
            feat = m.classifier[1].in_features; m.classifier = nn.Identity()
            get = lambda: m.features[0][0]; set_ = lambda c: m.features[0].__setitem__(0, c)
        elif b == "mobilenetv3_large":
            m = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            feat = m.classifier[0].in_features; m.classifier = nn.Identity()
            get = lambda: m.features[0][0]; set_ = lambda c: m.features[0].__setitem__(0, c)
        elif b == "vit_b_16":
            m = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
            feat = m.heads.head.in_features; m.heads = nn.Identity()
            get = lambda: m.conv_proj; set_ = lambda c: setattr(m, "conv_proj", c)
        elif b == "vit_b_32":
            m = tvm.vit_b_32(weights=tvm.ViT_B_32_Weights.IMAGENET1K_V1)
            feat = m.heads.head.in_features; m.heads = nn.Identity()
            get = lambda: m.conv_proj; set_ = lambda c: setattr(m, "conv_proj", c)
        elif b == "vit_l_16":
            m = tvm.vit_l_16(weights=tvm.ViT_L_16_Weights.IMAGENET1K_V1)
            feat = m.heads.head.in_features; m.heads = nn.Identity()
            get = lambda: m.conv_proj; set_ = lambda c: setattr(m, "conv_proj", c)
        else:
            raise ValueError(f"Backbone torchvision desconhecido: {b}")
        return m, feat, get, set_

    elif spec.source == "timm":
        import timm
        m = timm.create_model(b, pretrained=True, num_classes=0)  # extrator de features
        feat = m.num_features
        # A 1a "conv" varia entre familias: ViT/Swin expoem patch_embed.proj;
        # ConvNeXtV2/EffNetV2 nao (tem 'stem'). So a fusao "inflate" precisa dela,
        # entao caimos para no-op nesses casos (inflate indisponivel, demais fusoes ok).
        if hasattr(m, "patch_embed") and hasattr(m.patch_embed, "proj"):
            get = lambda: m.patch_embed.proj
            set_ = lambda c: setattr(m.patch_embed, "proj", c)
        else:
            get = lambda: None
            set_ = lambda c: None
        return m, feat, get, set_

    raise ValueError(f"Fonte de backbone desconhecida: {spec.source}")


# ---------------------------------------------------------------------------
# Bloco de fusao "adapter" / "adapter_v2"
# ---------------------------------------------------------------------------
class _SpaceGate(nn.Module):
    """Gating SE por espaco de cor (adapter_v2).

    GAP sobre os N canais -> MLP -> softmax sobre os K espacos, escalado por K
    para que a inicializacao (logits zero) produza gate == 1 em todos os espacos
    (comportamento identico ao adapter simples no inicio do treino). Os pesos por
    espaco ficam em ``last_weights`` para inspecao/interpretabilidade.
    """

    def __init__(self, n_channels: int, n_groups: int):
        super().__init__()
        hidden = max(8, n_channels // 2)
        self.fc = nn.Sequential(
            nn.Linear(n_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_groups),
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)
        self.n_groups = n_groups
        self.last_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3))                       # [B, N]
        w = torch.softmax(self.fc(pooled), dim=1) * self.n_groups  # [B, K], ==1 na init
        self.last_weights = w.detach()
        return w


class ColorFusion(nn.Module):
    """BatchNorm(N) + [gating opcional por espaco] + Conv1x1(N -> 3) -> pseudo-RGB.

    Inicializacao da projecao (``init``):

    * ``"identity"`` — se o RGB esta na pilha, a submatriz 3x3 da posicao dos
      canais RGB inicia como identidade e o restante como zero: na epoca 0 a
      projecao ENCAMINHA os canais RGB sem os alterar e os demais espacos entram
      como residuo aprendido (zerar pesos NAO zera gradientes — eles aprendem
      desde o 1o batch). Sem RGB na pilha, cai para "balanced".

      CUIDADO com o que isto NAO significa (auditoria de 2026-08-10): a ``bn``
      abaixo vem ANTES da ``proj`` e, em ``train()``, padroniza por estatisticas
      de BATCH — logo o backbone NAO recebe o RGB normalizado do pre-treino
      ImageNet. Medido num minibatch real do NJN: max|fused - rgb_norm| = 1,585
      (e o braco RGB puro, sem cromancia nenhuma, da o mesmo valor — a causa e a
      BN, nao a projecao). O que a init identity garante e a PARIDADE ENTRE
      BRACOS: a BN e por-canal, entao os canais RGB sao padronizados igualmente
      com ou sem cromancia, e max|fused_multi - fused_baseline| = 2,5e-3, devido
      so ao ruido abaixo. Ver ``docs/FIGURES.md`` §"Fig 3" e paper.tex §3.4.
    * ``"balanced"`` — cada espaco inicia contribuindo igualmente (identidade
      3x3 escalada por 1/K por grupo), sem privilegiar nenhum espaco.

    ``gated=True`` (fusao "adapter_v2") adiciona o :class:`_SpaceGate` entre a
    BatchNorm e a projecao, com init neutra (gate==1).
    """

    def __init__(self, channel_groups: Sequence[int], init: str = "identity",
                 gated: bool = False, rgb_group: Optional[int] = None):
        super().__init__()
        self.groups: Tuple[int, ...] = tuple(channel_groups)
        n_channels = sum(self.groups)
        k = len(self.groups)
        self.bn = nn.BatchNorm2d(n_channels)
        self.gate = _SpaceGate(n_channels, k) if gated else None
        self.proj = nn.Conv2d(n_channels, 3, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj.bias)
        self._init_proj(init, rgb_group)

    def _init_proj(self, init: str, rgb_group: Optional[int]) -> None:
        if init not in ("identity", "balanced", "naive"):
            raise ValueError(f"adapter_init invalido: {init!r} (use identity|balanced|naive)")
        if init == "naive":
            # Projecao "naive_concat": mantem a init padrao do Conv1x1 (kaiming) e
            # so zera o bias — o backbone aprende a mistura N->3 do zero, sem herdar
            # a identidade RGB (controle honesto do ganho da fusao estruturada).
            return
        k = len(self.groups)
        with torch.no_grad():
            w = self.proj.weight  # [3, N, 1, 1]
            w.zero_()
            if init == "identity" and rgb_group is not None:
                # Pseudo-RGB inicial == canais RGB; demais espacos como residuo ~zero.
                off = sum(self.groups[:rgb_group])
                rgb_cols = set(range(off, off + 3))
                # Ruido gaussiano minusculo (sigma=1e-4) nos canais NAO-RGB, em vez de
                # zero exato. NAO e por underflow de gradiente sob bf16/fp16 (justificativa
                # antiga, incorreta): dL/dW de uma conv NAO depende de W, entao as colunas
                # cromaticas recebem gradiente identico com zero exato — medido bit-a-bit
                # igual — e bf16 tem o MESMO expoente do fp32 (nada sub-flui; os grads sao
                # acumulados em fp32 sob autocast). O que o zero exato bloqueia e o
                # gradiente que ATRAVESSA essas colunas para tras: dL/dgamma da BN nos
                # canais cromaticos fica 0,0 no passo 0 (recupera no passo 1). O ruido
                # quebra essa simetria, ao custo de 2,5e-3 no tensor da epoca 0.
                n_in = w.shape[1]
                noise_cols = [c for c in range(n_in) if c not in rgb_cols]
                if noise_cols:
                    w[:, noise_cols, 0, 0] = torch.randn(3, len(noise_cols)) * 1e-4
                for c in range(3):
                    w[c, off + c, 0, 0] = 1.0
                return
            # "balanced" (ou identity sem RGB): cada espaco contribui igualmente.
            off = 0
            for g in self.groups:
                if g == 3:
                    for c in range(3):
                        w[c, off + c, 0, 0] = 1.0 / k
                else:
                    # Grupo nao-3x3 (ex.: HSV com hue circular = 4 canais):
                    # media do grupo distribuida igualmente nos 3 canais de saida.
                    w[:, off:off + g, 0, 0] = 1.0 / (k * g)
                off += g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        if self.gate is not None:
            w = self.gate(x)  # [B, K]
            parts = torch.split(x, list(self.groups), dim=1)
            x = torch.cat([p * w[:, j].view(-1, 1, 1, 1)
                           for j, p in enumerate(parts)], dim=1)
        return self.proj(x)


class CCATFusion(nn.Module):
    """Cross-Channel/Color Attention (CCAT): fusao por atencao cruzada por-pixel.

    Substitui a Conv1x1 do early-fusion. O RGB e o stream-ANCORA (Query); cada espaco
    de cromaticidade (LAB/HSV/YCrCb) fornece Key/Value. Para cada posicao espacial p:

        a_c(p)   = softmax_c( <Wq(rgb)(p), Wk_c(c)(p)> / sqrt(d_h) )   # softmax SOBRE os espacos
        fused(p) = sum_c a_c(p) * Wv_c(c)(p)
        pseudo_rgb = rgb + Wo(fused)                                   # Wo zero-init -> epoca 0 == RGB puro

    Porque NAO cai no "gate inerte" (v2/v3/v4): a fusao e uma DISTRIBUICAO softmax
    espacial por-pixel sobre os espacos (nao um escalar global), logo a BatchNorm a
    jusante nao a reabsorve; e o residuo zero-init preserva o backbone pre-treinado.
    Saida = 3 canais (pseudo-RGB) -> qualquer backbone (CNN ou ViT) fica intacto.

    Os mapas de atencao por espaco ficam em ``self.attn_maps`` ([B, C_chroma, H, W])
    para explicabilidade (provar foco na cromaticidade nos pacientes ictericos).
    """

    def __init__(self, channel_groups: Sequence[int], rgb_group: Optional[int],
                 dim: int = 32, n_heads: int = 4):
        super().__init__()
        if rgb_group is None:
            raise ValueError("CCAT exige RGB na pilha de espacos (stream-ancora Query).")
        self.groups: Tuple[int, ...] = tuple(channel_groups)
        self.rgb_group = rgb_group
        self.chroma_groups = [i for i in range(len(self.groups)) if i != rgb_group]
        if not self.chroma_groups:
            raise ValueError("CCAT exige >=1 espaco cromatico alem do RGB.")
        n_channels = sum(self.groups)
        self.bn = nn.BatchNorm2d(n_channels)
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        self.q = nn.Conv2d(self.groups[rgb_group], dim, kernel_size=1)
        self.k = nn.ModuleList([nn.Conv2d(self.groups[i], dim, 1) for i in self.chroma_groups])
        self.v = nn.ModuleList([nn.Conv2d(self.groups[i], dim, 1) for i in self.chroma_groups])
        self.o = nn.Conv2d(dim, 3, kernel_size=1)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)  # residuo ~0 na init -> entrada do backbone == RGB
        self.attn_maps: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        parts = torch.split(x, list(self.groups), dim=1)
        rgb = parts[self.rgb_group]                              # [B, 3, H, W]
        B, _, H, W = rgb.shape
        h = self.n_heads
        q = self.q(rgb)
        d = q.shape[1]
        d_h = d // h
        q = q.view(B, h, d_h, H * W)                             # [B, h, d_h, HW]
        vs: list = []
        scores: list = []
        for ci, g in enumerate(self.chroma_groups):
            c = parts[g]
            k = self.k[ci](c).view(B, h, d_h, H * W)
            v = self.v[ci](c).view(B, h, d_h, H * W)
            scores.append((q * k).sum(dim=2, keepdim=True) * self.scale)  # [B, h, 1, HW]
            vs.append(v)
        attn = torch.softmax(torch.cat(scores, dim=2), dim=2)    # [B, h, C, HW] softmax/espacos
        fused = q.new_zeros(B, h, d_h, H * W)
        for ci in range(len(vs)):                                # acumula sem materializar [B,h,C,d_h,HW]
            fused = fused + attn[:, :, ci:ci + 1, :] * vs[ci]
        self.attn_maps = attn.mean(dim=1).detach().reshape(B, len(vs), H, W)
        return rgb + self.o(fused.reshape(B, d, H, W))           # pseudo-RGB enriquecido


class ColorFusionStem(nn.Module):
    """Stem CNN de fusao de cor (fusao "stemcnn").

    Inspirado nos baselines 'Hybrid Early CNN' (baseline/codes/V1ClassificacaoArtigo),
    que fundem os N canais multi-espaco com um pequeno CNN antes do backbone — mas
    CORRIGIDO em relacao a eles: a projecao final NAO leva ReLU/BatchNorm, entao a
    pseudo-RGB entregue ao backbone pre-treinado mantem o sinal (valores negativos)
    e fica compativel com a distribuicao ImageNet do pre-treino (a versao deles
    aplicava ReLU terminal, descasando a entrada do backbone).

    Estrutura: BN(N) -> Conv3x3(N->hidden)+BN+ReLU -> Conv3x3(hidden->3). As convs
    3x3 (padding=1, preservam a resolucao) dao contexto espacial + nao-linearidade
    interna que o adapter 1x1 nao possui. Saida = 3 canais -> backbone intacto.
    O stem treina do zero (sempre treinavel; ``set_finetune_depth`` so congela o
    backbone).
    """

    def __init__(self, n_channels: int, hidden: int = 16):
        super().__init__()
        self.bn_in = nn.BatchNorm2d(n_channels)
        self.block = nn.Sequential(
            nn.Conv2d(n_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        # Projecao final SEM ativacao/BN: pseudo-RGB com sinal (~ ImageNet).
        self.proj = nn.Conv2d(hidden, 3, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.block(self.bn_in(x)))


class DSLFNFusion(nn.Module):
    """DS-LFN (Dual-Stream Late-Fusion Net) — porte simplificado da v4, como CONTROLE.

    Cada espaco recebe uma projecao Conv1x1(3->3); um gate escalar por espaco
    (aprendido) pondera a soma, produzindo pseudo-RGB = rgb + sum_s g_s * proj_s(chroma_s).
    O gate parte de zero (residuo ~0 na init -> backbone recebe RGB puro). NB: a v4
    mostrou que este gate escalar tende a ficar INERTE (a BN a jusante reabsorve a
    escala) — por isso o ChromaFiLM (modulacao por-canal/por-pixel multi-estagio) e a
    aposta, e o DS-LFN entra na ablacao como baseline de fusao tardia. Ver plano.
    """

    def __init__(self, channel_groups: Sequence[int], rgb_group: Optional[int]):
        super().__init__()
        if rgb_group is None:
            raise ValueError("ds_lfn exige RGB na pilha (stream-ancora).")
        self.groups: Tuple[int, ...] = tuple(channel_groups)
        self.rgb_group = rgb_group
        self.chroma_groups = [i for i in range(len(self.groups)) if i != rgb_group]
        n_channels = sum(self.groups)
        self.bn = nn.BatchNorm2d(n_channels)
        self.proj = nn.ModuleList([nn.Conv2d(self.groups[i], 3, 1) for i in self.chroma_groups])
        # Gate por espaco: parametro escalar, init 0 -> sigmoid(0)=0.5 mas escalado por
        # zero-init do proj bias? Usamos gate multiplicativo com init 0 (neutro).
        self.gate = nn.Parameter(torch.zeros(len(self.chroma_groups)))
        for p in self.proj:
            nn.init.zeros_(p.weight); nn.init.zeros_(p.bias)  # residuo 0 na init
        self.last_gate: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        parts = torch.split(x, list(self.groups), dim=1)
        rgb = parts[self.rgb_group]
        g = torch.tanh(self.gate)  # em [-1,1], 0 na init
        self.last_gate = g.detach()
        out = rgb
        for j, ci in enumerate(self.chroma_groups):
            out = out + g[j] * self.proj[j](parts[ci])
        return out


def _inflate_conv(old: nn.Conv2d, n_channels: int) -> nn.Conv2d:
    """Substitui um Conv2d de 3 canais de entrada por um de ``n_channels``.

    Inicializa repetindo a media (sobre os canais RGB) dos pesos pre-treinados,
    reescalada por 3/n para preservar a magnitude da ativacao de saida.
    """
    new = nn.Conv2d(n_channels, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, dilation=old.dilation,
                    groups=old.groups, bias=old.bias is not None)
    with torch.no_grad():
        mean_w = old.weight.mean(dim=1, keepdim=True)         # [out,1,kh,kw]
        new.weight.copy_(mean_w.repeat(1, n_channels, 1, 1) * (3.0 / n_channels))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    return new


_ACTIVATIONS = {
    "none": nn.Identity,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def channel_groups_for(colorspaces: Optional[Sequence[str]], n_channels: int,
                       hue_circular: bool = False) -> Tuple[Tuple[int, ...], Optional[int]]:
    """Devolve (tamanhos dos grupos de canais por espaco, indice do grupo RGB).

    Sem ``colorspaces`` (uso legado/testes), assume grupos uniformes de 3 e
    considera RGB apenas quando a pilha e um unico espaco de 3 canais.
    """
    if colorspaces:
        from .config import space_nchannels
        groups = tuple(space_nchannels(sp, hue_circular) for sp in colorspaces)
        rgb_group = colorspaces.index("RGB") if "RGB" in colorspaces else None
        if sum(groups) != n_channels:
            raise ValueError(f"n_channels={n_channels} incompativel com "
                             f"{colorspaces} (esperado {sum(groups)}).")
        return groups, rgb_group
    if n_channels % 3 != 0:
        raise ValueError(f"n_channels={n_channels} sem colorspaces: precisa ser multiplo de 3.")
    groups = tuple([3] * (n_channels // 3))
    return groups, (0 if n_channels == 3 else None)


# ---------------------------------------------------------------------------
# Modelo HCSF
# ---------------------------------------------------------------------------
class HCSFModel(nn.Module):
    def __init__(self, spec: BackboneSpec, n_channels: int, fusion: str = "adapter",
                 activation: str = "none", num_classes: int = 2,
                 colorspaces: Optional[Sequence[str]] = None,
                 adapter_init: str = "identity", hue_circular: bool = False,
                 multitask: bool = False, backbone_mode: str = "frozen",
                 lora_rank: int = 8):
        super().__init__()
        backbone, feat_dim, get_fc, set_fc = _load_backbone(spec)
        self.spec = spec
        self.fusion_type = fusion
        self.n_channels = n_channels
        self.backbone_mode = backbone_mode
        self.multitask = bool(multitask)
        self._inflated_conv: Optional[nn.Conv2d] = None
        self._lora_modules: List["LoRALinear"] = []

        if fusion in ("adapter", "adapter_v2"):
            groups, rgb_group = channel_groups_for(colorspaces, n_channels, hue_circular)
            self.fusion = ColorFusion(groups, init=adapter_init,
                                      gated=(fusion == "adapter_v2"), rgb_group=rgb_group)
        elif fusion == "naive_concat":
            # Controle "naive": Conv1x1(N->3) com init padrao (sem identity/balanced) —
            # o backbone aprende a projecao do zero, sem preservar o pre-treino RGB.
            groups, rgb_group = channel_groups_for(colorspaces, n_channels, hue_circular)
            self.fusion = ColorFusion(groups, init="naive", gated=False, rgb_group=rgb_group)
        elif fusion == "ds_lfn":
            groups, rgb_group = channel_groups_for(colorspaces, n_channels, hue_circular)
            self.fusion = DSLFNFusion(groups, rgb_group)
        elif fusion == "ccat":
            groups, rgb_group = channel_groups_for(colorspaces, n_channels, hue_circular)
            self.fusion = CCATFusion(groups, rgb_group)
        elif fusion == "stemcnn":
            self.fusion = ColorFusionStem(n_channels)
        elif fusion == "inflate":
            self.fusion = nn.Identity()
            if n_channels != 3:
                new_conv = _inflate_conv(get_fc(), n_channels)
                set_fc(new_conv)
                self._inflated_conv = new_conv
            else:
                self._inflated_conv = get_fc()  # ja e 3-canais; mantem treinavel
        else:
            raise ValueError(f"Fusao invalida: {fusion}")

        self.backbone = backbone
        self.feat_dim = feat_dim
        # Cabeça de classificação (logits crus; CrossEntropy) + cabeça de regressão TSB
        # opcional (só multitask/NeoJaundice). Segue o contrato §2.3: pesos de
        # incerteza (Kendall) log_var_cls/log_var_reg aprendíveis quando multitask.
        self.head_cls = nn.Linear(feat_dim, num_classes)
        self.head_reg = nn.Linear(feat_dim, 1) if self.multitask else None
        if self.multitask:
            self.log_var_cls = nn.Parameter(torch.zeros(()))
            self.log_var_reg = nn.Parameter(torch.zeros(()))
        self.last_tsb: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stack [B,N,H,W] -> fusão -> pseudo-RGB [B,3,H,W] -> backbone -> logits.

        Retorna sempre os ``logits`` (compatível com o engine/cv, que consomem
        ``model(imgs)`` como tensor). Em multitask a predição de TSB fica em
        ``self.last_tsb`` [B] (convenção idêntica ao ChromaFiLMNet)."""
        x = self.fusion(x)
        feats = self.backbone(x)
        logits = self.head_cls(feats)
        self.last_tsb = (self.head_reg(feats).squeeze(-1)
                         if (self.multitask and self.head_reg is not None) else None)
        return logits

    # -- controle de fine-tuning --
    def set_finetune_depth(self, n_unfreeze: int) -> None:
        """Congela o backbone e descongela as ``n_unfreeze`` ultimas camadas.

        ``n_unfreeze == -1`` descongela o backbone INTEIRO (fine-tuning total,
        como na baseline CNN). Fusao e cabeca sao SEMPRE treinaveis; na fusao
        "inflate" a 1a convolucao (recem-inicializada) tambem e mantida treinavel.
        """
        params = list(self.backbone.parameters())
        if n_unfreeze == -1:
            for p in params:
                p.requires_grad = True
        else:
            for p in params:
                p.requires_grad = False
            if n_unfreeze > 0:
                for p in params[-n_unfreeze:]:
                    p.requires_grad = True
        for p in self.fusion.parameters():
            p.requires_grad = True
        for p in self._head_params():
            p.requires_grad = True
        if self._inflated_conv is not None:
            for p in self._inflated_conv.parameters():
                p.requires_grad = True

    def _head_params(self) -> List[torch.nn.Parameter]:
        head = list(self.head_cls.parameters())
        if self.head_reg is not None:
            head += list(self.head_reg.parameters())
        if self.multitask:
            head += [self.log_var_cls, self.log_var_reg]
        return head

    def train(self, mode: bool = True):
        """No modo frozen/lora o backbone fica em ``eval()`` (BN com running stats
        FIXAS) — condição do contrato §0 (mesma init/estatística nos dois braços)."""
        super().train(mode)
        if self.backbone_mode != "full":
            self.backbone.eval()
        return self

    def build_param_groups(self, lr: float, backbone_lr_mult: float = 0.1) -> list:
        """Differential LR (contrato §2.4): **2 grupos** — (1) fusão + cabeças (+ conv
        inflate + LoRA) em LR cheio; (2) backbone pré-treinado em ``lr × mult``. Se o
        backbone estiver congelado (frozen/lora), ele fica FORA do otimizador."""
        head_fusion = ([p for p in self.fusion.parameters() if p.requires_grad] +
                       [p for p in self._head_params() if p.requires_grad])
        lora = [p for m in self._lora_modules for p in (m.A, m.B)]
        if self._inflated_conv is not None:
            head_fusion += [p for p in self._inflated_conv.parameters() if p.requires_grad]
        groups = [{"params": head_fusion + lora, "lr": lr}]
        inflated_ids = ({id(p) for p in self._inflated_conv.parameters()}
                        if self._inflated_conv is not None else set())
        bb = [p for p in self.backbone.parameters()
              if p.requires_grad and id(p) not in inflated_ids]
        if bb:
            groups.append({"params": bb, "lr": lr * backbone_lr_mult})
        return groups

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# ChromaFiLM — WRAPPER de backbone (NAO fusao pseudo-RGB). Modula features
# intermediarias em varios estagios via forward hooks: F_l <- (1+dg_l)*F_l + db_l.
# Ver plano §3 (correcao de contrato) e a suite de testes (multi-site + zero-init).
# ===========================================================================
class LoRALinear(nn.Module):
    """nn.Linear com adapter LoRA (base congelado; so A,B treinam; B=0 -> neutro)."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # B=0 -> saida == base na init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * F.linear(F.linear(x, self.A), self.B)


class _AlphaSpaceGate(nn.Module):
    """Peso por espaco cromatico (interpretabilidade "qual espaco importa").

    softmax sobre os espacos (distribuicao em ``last_alpha``), escalado por K para
    init neutra (~1 por espaco, como o _SpaceGate do adapter_v2)."""

    def __init__(self, chroma_group_sizes: Sequence[int]):
        super().__init__()
        self.sizes = list(chroma_group_sizes)
        self.k = len(self.sizes)
        self.fc = nn.Linear(sum(self.sizes), self.k)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)  # logits 0 -> softmax uniforme (neutro)
        self.last_alpha: Optional[torch.Tensor] = None

    def forward(self, chroma_parts: List[torch.Tensor]) -> torch.Tensor:
        pooled = torch.cat([p.mean(dim=(2, 3)) for p in chroma_parts], dim=1)  # [B, sumC]
        soft = torch.softmax(self.fc(pooled), dim=1)                            # [B, K]
        self.last_alpha = soft.detach()
        w = soft * self.k                                                       # ~1 na init
        return torch.cat([p * w[:, j:j + 1, None, None]
                          for j, p in enumerate(chroma_parts)], dim=1)


class _ChromaEncoder(nn.Module):
    """Encoder leve (DSConv) dos canais cromaticos -> descritor Z [B, dim, H/4, W/4]."""

    def __init__(self, in_ch: int, dim: int = 64):
        super().__init__()

        def ds(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
                nn.Conv2d(cin, cout, 1, bias=False),
                nn.BatchNorm2d(cout), nn.GELU())

        h = max(8, dim // 2)
        self.net = nn.Sequential(ds(in_ch, h, 2), ds(h, dim, 2), ds(dim, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _FiLMGenerator(nn.Module):
    """Z -> (Delta_gamma, Delta_beta) para UM estagio.

    ``zero_init=True`` (default, ChromaFiLMNet): ultima camada zero-init => dg=db=0 na
    epoca 0 (backbone RGB puro). ``zero_init=False`` (ChromaRouteFiLMNet): init padrao —
    o zero-init do modelo passa a viver no escalar de magnitude ``s_i`` por estagio, o que
    evita o deadlock de gradiente da normalizacao de um vetor nulo (ver ChromaRouteFiLMNet).
    """

    def __init__(self, in_dim: int, out_ch: int, film_mode: str, zero_init: bool = True):
        super().__init__()
        self.film_mode = film_mode
        self.out_ch = out_ch
        if film_mode == "channel":
            self.mlp = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(),
                                     nn.Linear(in_dim, 2 * out_ch))
            if zero_init:
                nn.init.zeros_(self.mlp[-1].weight)
                nn.init.zeros_(self.mlp[-1].bias)   # dg=db=0 na epoca 0
        else:  # spatial (so CNN)
            self.conv = nn.Conv2d(in_dim, 2 * out_ch, kernel_size=1)
            if zero_init:
                nn.init.zeros_(self.conv.weight)
                nn.init.zeros_(self.conv.bias)

    def forward(self, Z: torch.Tensor, out: torch.Tensor, is_token: bool):
        if self.film_mode == "channel":
            z = Z.mean(dim=(2, 3))                  # [B, in_dim]
            dgb = self.mlp(z)                       # [B, 2*C]
            dg, db = dgb[:, :self.out_ch], dgb[:, self.out_ch:]
            return dg, db, None
        # spatial: casa a resolucao do estagio
        Hh, Ww = out.shape[-2], out.shape[-1]
        z = F.interpolate(Z, size=(Hh, Ww), mode="bilinear", align_corners=False)
        dgb = self.conv(z)                          # [B, 2*C, H, W]
        dg, db = dgb[:, :self.out_ch], dgb[:, self.out_ch:]
        return dg, db, dg.mean(dim=1).detach()      # gmap [B,H,W] p/ overlay


class ChromaFiLMNet(nn.Module):
    """Backbone (via timm) + modulacao FiLM por estagio condicionada na cromancia.

    Contrato: entrada = stack [B, N, H, W]; ``rgb`` -> backbone; cromancia -> α_space
    -> ChromaEncoder(Z) -> geradores FiLM por estagio. Hooks aplicam
    ``F_l <- (1+dg_l)*F_l + db_l``. Zero-init nos geradores => epoca 0 == backbone(rgb).
    Exposto p/ explicabilidade: ``last_alpha``, ``last_film_stats`` (‖dg_l‖,‖db_l‖ por
    estagio = perfil por profundidade), ``last_film_maps`` (spatial), ``last_tsb``.
    """

    _FUSION_TYPE = "chromafilm"

    def __init__(self, spec: BackboneSpec, n_channels: int,
                 colorspaces: Optional[Sequence[str]] = None, num_classes: int = 2,
                 film_mode: str = "channel", backbone_mode: str = "frozen",
                 lora_rank: int = 8, multitask: bool = False,
                 hue_circular: bool = False, encoder_dim: int = 64, n_sites: int = 4):
        super().__init__()
        import timm
        self.spec = spec
        self.fusion_type = self._FUSION_TYPE
        self.film_mode = film_mode
        self.backbone_mode = backbone_mode
        self.multitask = bool(multitask)

        groups, rgb_group = channel_groups_for(colorspaces, n_channels, hue_circular)
        if rgb_group is None:
            raise ValueError("chromafilm exige RGB na pilha (backbone recebe RGB puro).")
        self.groups = tuple(groups)
        self.rgb_group = rgb_group
        self.chroma_groups = [i for i in range(len(groups)) if i != rgb_group]
        off, self._offsets = 0, []
        for g in groups:
            self._offsets.append((off, off + g)); off += g
        self.n_chroma_ch = sum(groups[i] for i in self.chroma_groups)

        name = timm_name_of(spec)
        try:
            self.backbone = timm.create_model(name, pretrained=True, num_classes=0)
        except Exception as e:
            raise RuntimeError(
                f"chromafilm: falha ao criar backbone timm '{name}' ({spec.name}): {e}. "
                f"Se for o dinov3, use efficientnet_b4 como fallback (ver plano §6).")
        self.feat_dim = int(self.backbone.num_features)
        self._sites = self._identify_sites(self.backbone, spec.family, n_sites)

        self.encoder_dim = encoder_dim
        self.fusion = self._build_fusion(groups, film_mode, encoder_dim)

        self.head_cls = nn.Linear(self.feat_dim, num_classes)
        self.head_reg = nn.Linear(self.feat_dim, 1) if self.multitask else None
        if self.multitask:
            self.log_var_cls = nn.Parameter(torch.zeros(()))
            self.log_var_reg = nn.Parameter(torch.zeros(()))

        # estado dos hooks (limpo a cada forward -> sem vazamento entre batches)
        self._Z: Optional[torch.Tensor] = None
        self.last_alpha: Optional[torch.Tensor] = None
        self.last_film_stats: Dict[int, Tuple[float, float]] = {}
        self.last_film_maps: Dict[int, torch.Tensor] = {}
        self.last_tsb: Optional[torch.Tensor] = None
        self._hooks = []
        self._lora_modules: List[LoRALinear] = []
        self._register_hooks()
        self._apply_backbone_mode(lora_rank)

    # -- construcao do modulo de fusao (sobrescrito por ChromaRouteFiLMNet) --
    def _build_fusion(self, groups, film_mode, encoder_dim) -> nn.Module:
        """α_space (gate na ENTRADA) + encoder cromatico compartilhado + FiLM por estagio."""
        fusion = nn.Module()
        if self.chroma_groups:
            fusion.space_gate = _AlphaSpaceGate([groups[i] for i in self.chroma_groups])
            fusion.encoder = _ChromaEncoder(self.n_chroma_ch, encoder_dim)
            fusion.film_gens = nn.ModuleList(
                [_FiLMGenerator(encoder_dim, C, film_mode) for (_n, _m, C, _t) in self._sites])
        return fusion

    # -- identificacao de estagios (CNN via feature_info; ViT via .blocks) --
    def _identify_sites(self, backbone, family, n_sites):
        named = dict(backbone.named_modules())
        fi = getattr(backbone, "feature_info", None)
        if fi is not None and family == "cnn":
            infos: list = []
            try:
                if isinstance(fi, (list, tuple)):
                    infos = list(fi)                         # timm sem features_only -> list de dicts
                elif hasattr(fi, "get_dicts"):
                    infos = fi.get_dicts()                   # FeatureInfo
                elif hasattr(fi, "module_name"):
                    infos = [{"module": fi.module_name(i), "num_chs": fi.channels(i)}
                             for i in range(len(fi))]
            except Exception:
                infos = []
            sites = [(inf["module"], named[inf["module"]], int(inf["num_chs"]), False)
                     for inf in infos
                     if isinstance(inf, dict) and inf.get("module") in named and inf.get("num_chs")]
            if len(sites) >= 2:
                return sites
        blocks = getattr(backbone, "blocks", None)
        if blocks is not None and len(blocks) >= 2:
            C = int(getattr(backbone, "embed_dim", None) or backbone.num_features)
            n = len(blocks)
            picks = sorted({min(n - 1, max(0, int(round((j + 1) * n / n_sites)) - 1))
                            for j in range(n_sites)})
            return [(f"blocks.{k}", blocks[k], C, True) for k in picks]
        raise RuntimeError(
            f"chromafilm: nao identifiquei estagios p/ {self.spec.name} "
            f"(sem feature_info CNN nem .blocks ViT).")

    def _make_hook(self, i):
        def hook(module, inp, out):
            if not self.chroma_groups or self._Z is None:
                return out
            is_token = self._sites[i][3]
            dg, db, gmap = self.fusion.film_gens[i](self._Z, out, is_token)
            if is_token:                       # out [B,T,C]
                out = out * (1 + dg.unsqueeze(1)) + db.unsqueeze(1)
            elif dg.dim() == 2:                # channel: out [B,C,H,W]
                out = out * (1 + dg[:, :, None, None]) + db[:, :, None, None]
            else:                              # spatial: dg,db [B,C,H,W]
                out = out * (1 + dg) + db
            self.last_film_stats[i] = (float(dg.detach().abs().mean()),
                                       float(db.detach().abs().mean()))
            if gmap is not None:
                self.last_film_maps[i] = gmap
            return out
        return hook

    def _register_hooks(self):
        if not self.chroma_groups:
            return
        for i, (_n, mod, _C, _t) in enumerate(self._sites):
            self._hooks.append(mod.register_forward_hook(self._make_hook(i)))

    def _apply_backbone_mode(self, lora_rank):
        if self.backbone_mode == "full":
            for p in self.backbone.parameters():
                p.requires_grad = True     # full fine-tuning (regime da v1 onde a cor ajudou)
            return
        for p in self.backbone.parameters():
            p.requires_grad = False
        if self.backbone_mode == "lora":
            self._inject_lora(lora_rank)

    def _inject_lora(self, rank):
        targets = [name for name, mod in self.backbone.named_modules()
                   if isinstance(mod, nn.Linear)
                   and any(k in name for k in ("attn", "mlp", "qkv", "fc", "proj"))]
        for name in targets:
            *parents, attr = name.split(".")
            parent = self.backbone
            for p in parents:
                parent = getattr(parent, p)
            base = getattr(parent, attr)
            if not isinstance(base, nn.Linear):
                continue
            lin = LoRALinear(base, rank=rank)
            setattr(parent, attr, lin)
            self._lora_modules.append(lin)
        if not self._lora_modules:
            print(f"[chromafilm] WARN: nenhum Linear alvo p/ LoRA em {self.spec.name} "
                  f"(CNN?) — backbone segue congelado (equivale a frozen).")

    @property
    def n_sites(self) -> int:
        return len(self._sites)

    def forward(self, x: torch.Tensor):
        parts = [x[:, a:b] for (a, b) in self._offsets]
        rgb = parts[self.rgb_group]
        self.last_film_stats = {}
        self.last_film_maps = {}
        if self.chroma_groups:
            chroma = [parts[i] for i in self.chroma_groups]
            chroma_w = self.fusion.space_gate(chroma)
            self.last_alpha = self.fusion.space_gate.last_alpha  # expõe α_space no nível do modelo
            self._Z = self.fusion.encoder(chroma_w)
        else:
            self.last_alpha = None
            self._Z = None
        feats = self.backbone(rgb)     # hooks modulam as saidas dos estagios
        self._Z = None                 # limpa o estado (sem vazamento entre batches)
        logits = self.head_cls(feats)
        self.last_tsb = (self.head_reg(feats).squeeze(-1)
                         if (self.multitask and self.head_reg is not None) else None)
        return logits

    # -- interface p/ o engine/cv --
    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_mode != "full":
            self.backbone.eval()   # frozen/lora: BN do backbone com running stats FIXAS (§7)
        return self

    def set_finetune_depth(self, n_unfreeze: int) -> None:
        """No-op: no ChromaFiLM o backbone e sempre congelado (frozen) ou LoRA."""
        return

    def build_param_groups(self, lr: float, backbone_lr_mult: float = 0.1) -> list:
        head = list(self.head_cls.parameters())
        if self.head_reg is not None:
            head += list(self.head_reg.parameters())
        if self.multitask:
            head += [self.log_var_cls, self.log_var_reg]
        fus = list(self.fusion.parameters())
        lora = [p for m in self._lora_modules for p in (m.A, m.B)]
        params = [p for p in (head + fus) if p.requires_grad]
        groups = [{"params": params, "lr": lr}]
        if lora:
            groups.append({"params": lora, "lr": lr})
        if self.backbone_mode == "full":
            bb = [p for p in self.backbone.parameters() if p.requires_grad]
            if bb:  # LR diferencial menor no backbone pre-treinado (full fine-tuning)
                groups.append({"params": bb, "lr": lr * backbone_lr_mult})
        return groups

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _sparsemax(logits: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Sparsemax (Martins & Astudillo, 2016): projecao no simplex que gera zeros exatos.

    Diferente do softmax, atribui probabilidade **zero** a espacos irrelevantes — logo o
    α_space fica legivel ("qual espaco NAO importa" vira 0). sparsemax(0) = uniforme (init
    neutro)."""
    z = logits
    z_sorted, _ = torch.sort(z, descending=True, dim=dim)
    rng = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim(); shape[dim] = -1
    rng = rng.view(shape)
    z_cumsum = z_sorted.cumsum(dim)
    support = (1 + rng * z_sorted) > z_cumsum
    k_z = support.to(z.dtype).sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (z_cumsum.gather(dim, (k_z.long() - 1)) - 1) / k_z
    return torch.clamp(z - tau, min=0)


class _RouteGate(nn.Module):
    """Roteamento IDENTIFICAVEL por espaco crômico: pooled Z_k -> logit_k -> sparsemax.

    Ao contrario do `_AlphaSpaceGate` (que multiplica a cromância na ENTRADA e e reabsorvido
    pela BN do encoder -> α inerte 50/50), aqui o α pondera as modulacoes FiLM JA
    normalizadas em magnitude, no ponto de aplicacao. Como a magnitude vive num escalar
    proprio por estagio, o α controla SO a divisao entre espacos -> identificavel."""

    def __init__(self, k: int, dim: int):
        super().__init__()
        self.k = k
        self.fc = nn.ModuleList([nn.Linear(dim, 1) for _ in range(k)])
        for lin in self.fc:
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)  # logits 0 -> α uniforme neutro
        self.last_alpha: Optional[torch.Tensor] = None

    def forward(self, Zs: List[torch.Tensor]) -> torch.Tensor:
        logits = torch.cat([self.fc[j](Zs[j].mean(dim=(2, 3))) for j in range(self.k)], dim=1)
        alpha = _sparsemax(logits, dim=1)          # [B, K]
        self.last_alpha = alpha.detach()
        return alpha


class ChromaRouteFiLMNet(ChromaFiLMNet):
    """Substituto do ChromaFiLMNet com **roteamento de cor identificavel** (v6 headline).

    Conserta a inercia do α_space (50/50 em v2/v3/v4/v6): (1) um encoder + gerador FiLM
    **por espaco** (Z_k, dg_{l,k}); (2) cada modulacao e **normalizada em magnitude** (L2
    unitario) antes de entrar; (3) um **escalar de magnitude ``stage_scale`` por estagio**
    (init 0 -> epoca 0 == backbone RGB) carrega o "quanto"; (4) o α (sparsemax) carrega SO o
    "qual espaco" -> identificavel e nao-reabsorvivel. Suporta **LOSO** (zera um espaco por
    vez em avaliacao) p/ ablação rigorosa cruzando o α aprendido. Mantem hooks/LoRA/multitask.
    """

    _FUSION_TYPE = "chromafilm_route"
    _loso_drop: Optional[int] = None

    def _build_fusion(self, groups, film_mode, encoder_dim) -> nn.Module:
        fusion = nn.Module()
        if self.chroma_groups:
            k = len(self.chroma_groups)
            fusion.encoders = nn.ModuleList(
                [_ChromaEncoder(groups[i], encoder_dim) for i in self.chroma_groups])
            fusion.route = _RouteGate(k, encoder_dim)
            fusion.film_gens = nn.ModuleList([
                nn.ModuleList([_FiLMGenerator(encoder_dim, C, film_mode, zero_init=False)
                               for _ in range(k)])
                for (_n, _m, C, _t) in self._sites])
            # magnitude por estagio (init 0 => modulacao nula na epoca 0 => backbone RGB puro).
            fusion.stage_scale = nn.Parameter(torch.zeros(len(self._sites)))
        return fusion

    def set_loso_drop(self, k: Optional[int]) -> None:
        """Zera o espaco crômico ``k`` na proxima avaliacao (None = usa todos)."""
        self._loso_drop = k

    @property
    def n_chroma_spaces(self) -> int:
        return len(self.chroma_groups)

    def _make_hook(self, i):
        eps = 1e-6

        def hook(module, inp, out):
            if not self.chroma_groups or self._Zs is None:
                return out
            is_token = self._sites[i][3]
            alpha = self._alpha                     # [B, K]
            drop = self._loso_drop
            dg_tot = db_tot = None
            for k in range(len(self.chroma_groups)):
                if drop is not None and k == drop:
                    continue
                dg_k, db_k, _ = self.fusion.film_gens[i][k](self._Zs[k], out, is_token)
                if dg_k.dim() == 2:                 # channel/token: [B, C]
                    dg_k = dg_k / (dg_k.norm(dim=1, keepdim=True) + eps)
                    db_k = db_k / (db_k.norm(dim=1, keepdim=True) + eps)
                    w = alpha[:, k:k + 1]
                else:                               # spatial: [B, C, H, W]
                    dg_k = dg_k / (dg_k.flatten(1).norm(dim=1)[:, None, None, None] + eps)
                    db_k = db_k / (db_k.flatten(1).norm(dim=1)[:, None, None, None] + eps)
                    w = alpha[:, k][:, None, None, None]
                dg_tot = w * dg_k if dg_tot is None else dg_tot + w * dg_k
                db_tot = w * db_k if db_tot is None else db_tot + w * db_k
            if dg_tot is None:                      # todos os espacos zerados (LOSO extremo)
                return out
            s = self.fusion.stage_scale[i]
            dg_tot, db_tot = s * dg_tot, s * db_tot
            if is_token:
                out = out * (1 + dg_tot.unsqueeze(1)) + db_tot.unsqueeze(1)
            elif dg_tot.dim() == 2:
                out = out * (1 + dg_tot[:, :, None, None]) + db_tot[:, :, None, None]
            else:
                out = out * (1 + dg_tot) + db_tot
            self.last_film_stats[i] = (float(dg_tot.detach().abs().mean()),
                                       float(db_tot.detach().abs().mean()))
            return out
        return hook

    def forward(self, x: torch.Tensor):
        parts = [x[:, a:b] for (a, b) in self._offsets]
        rgb = parts[self.rgb_group]
        self.last_film_stats = {}
        self.last_film_maps = {}
        if self.chroma_groups:
            chroma = [parts[i] for i in self.chroma_groups]
            self._Zs = [self.fusion.encoders[j](chroma[j]) for j in range(len(chroma))]
            self._alpha = self.fusion.route(self._Zs)        # [B, K] sparsemax
            self.last_alpha = self.fusion.route.last_alpha
            p = self._alpha.clamp_min(1e-9)
            self.last_route_entropy = float((-(p * p.log()).sum(dim=1)).mean().detach())
        else:
            self._Zs = self._alpha = self.last_alpha = None
        feats = self.backbone(rgb)                            # hooks modulam os estagios
        self._Zs = self._alpha = None                        # limpa (sem vazamento)
        logits = self.head_cls(feats)
        self.last_tsb = (self.head_reg(feats).squeeze(-1)
                         if (self.multitask and self.head_reg is not None) else None)
        return logits


def build_model(spec: BackboneSpec, n_channels: int, fusion: str = "adapter",
                activation: str = "none", n_unfreeze: int = 0,
                colorspaces: Optional[Sequence[str]] = None,
                adapter_init: str = "identity", hue_circular: bool = False,
                num_classes: int = 2, backbone_mode: str = "frozen",
                film_mode: str = "channel", lora_rank: int = 8,
                multitask: bool = False):
    """Constroi o modelo e aplica a profundidade de fine-tuning (-1 = total).

    ``fusion == "chromafilm"`` retorna um :class:`ChromaFiLMNet` (wrapper de backbone
    via hooks, NAO fusao pseudo-RGB); ``"chromafilm_route"`` -> :class:`ChromaRouteFiLMNet`
    (roteamento de cor identificavel). As demais fusoes usam o :class:`HCSFModel`.
    """
    if fusion in ("chromafilm", "chromafilm_route"):
        cls = ChromaRouteFiLMNet if fusion == "chromafilm_route" else ChromaFiLMNet
        return cls(spec, n_channels, colorspaces=colorspaces,
                   num_classes=num_classes, film_mode=film_mode,
                   backbone_mode=backbone_mode, lora_rank=lora_rank,
                   multitask=multitask, hue_circular=hue_circular)
    model = HCSFModel(spec, n_channels, fusion=fusion, activation=activation,
                      colorspaces=colorspaces, adapter_init=adapter_init,
                      hue_circular=hue_circular, num_classes=num_classes,
                      multitask=multitask, backbone_mode=backbone_mode,
                      lora_rank=lora_rank)
    # backbone_mode=full => descongela o backbone inteiro; frozen/lora usam n_unfreeze
    # (0 = congelado; o HPO pode buscar unfreeze parcial). LoRA no HCSFModel trata o
    # backbone como frozen (o wiring LoRA vive no ChromaFiLMNet).
    model.set_finetune_depth(-1 if backbone_mode == "full" else n_unfreeze)
    return model
