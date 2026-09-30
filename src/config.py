"""
config.py
=========

Ponto único de configuração do estudo **Hybrid Color-Space Fusion (HCSF) — versão
final/definitiva**.

Esta versão une a arquitetura vencedora da v1 (Single-Stream / Early Fusion via
adapter Conv 1x1, com subconjuntos de espaço de cor como variável independente) ao
pipeline clínico avançado: white-balance pela carta + crop ROI central (NeoJaundice),
extração de ROI de pele (NJN), split agrupado por paciente (StratifiedGroupKFold) e
cabeça dinâmica 2/3 classes via CLI.

Aqui ficam:

* O *power set* dos espaços de cor {RGB, LAB, YCrCb, HSV} (15 subconjuntos).
* O registry de backbones — **restrito (estritamente) aos campeões/baselines**:
  ``deit_tiny`` (campeão NJN), ``inception_v3`` (campeão NeoJaundice), ``resnet18`` e
  ``efficientnet_b0`` (baselines).
* A dataclass :class:`ExperimentConfig`, unidade atômica de trabalho.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Espaços de cor
# ---------------------------------------------------------------------------
# Ordem canônica: RGB primeiro (referência/baseline interna; init identity do adapter
# preserva os pesos pré-treinados), depois os espaços de luminância-cromância padrão e,
# por fim, o "banco fisiológico" v6 (features de cor auto-normalizadas, sem estatística
# de dataset — ver colorspaces.CUSTOM_SPACES):
#   • BSTAR     — b* do CIELAB (eixo azul↔amarelo; sinal direto de bilirrubina).       1 canal
#   • ITA       — Individual Typology Angle atan2(L*−50, b*)·180/π (tom de pele).       1 canal
#   • OPP       — cromaticidade oponente normalizada (rg, yb), robusta a iluminação.    2 canais
#   • LOGCHROMA — log(R/G), log(B/G): aprox. invariante ao iluminante (mira NJN).       2 canais
COLOR_SPACES: Tuple[str, ...] = (
    "RGB", "LAB", "YCrCb", "HSV", "BSTAR", "ITA", "OPP", "LOGCHROMA")
CHANNELS_PER_SPACE: int = 3

# Nº de canais por espaço (fonte única de verdade; consumido por config/colorspaces/models).
SPACE_CHANNELS: Dict[str, int] = {
    "RGB": 3, "LAB": 3, "YCrCb": 3, "HSV": 3,
    "BSTAR": 1, "ITA": 1, "OPP": 2, "LOGCHROMA": 2,
}


def space_nchannels(space: str, hue_circular: bool = False) -> int:
    """Nº de canais de um espaço (HSV circular = 4; demais via ``SPACE_CHANNELS``)."""
    if space == "HSV" and hue_circular:
        return 4
    return SPACE_CHANNELS.get(space, CHANNELS_PER_SPACE)


def canonical_colorset(spaces) -> Tuple[str, ...]:
    """Ordena um subconjunto de espaços na ordem canônica de ``COLOR_SPACES``."""
    order = {s: i for i, s in enumerate(COLOR_SPACES)}
    uniq = {s for s in spaces}
    unknown = uniq - set(COLOR_SPACES)
    if unknown:
        raise ValueError(f"Espaco(s) de cor desconhecido(s): {sorted(unknown)}. "
                         f"Validos: {COLOR_SPACES}")
    return tuple(sorted(uniq, key=lambda s: order[s]))


def colorset_id(spaces) -> str:
    """Identificador textual estável, ex.: ``RGB+LAB+YCrCb``."""
    return "+".join(canonical_colorset(spaces))


def parse_colorset(text: str) -> Tuple[str, ...]:
    """``"RGB+LAB"`` -> ``("RGB", "LAB")`` (ordem canônica)."""
    return canonical_colorset(text.split("+"))


def powerset_colorspaces() -> List[Tuple[str, ...]]:
    """Todos os 15 subconjuntos não vazios de ``COLOR_SPACES`` (tamanho crescente)."""
    subsets: List[Tuple[str, ...]] = []
    for k in range(1, len(COLOR_SPACES) + 1):
        for combo in combinations(COLOR_SPACES, k):
            subsets.append(combo)
    return subsets


INDIVIDUAL_COLORSPACES: List[Tuple[str, ...]] = [(s,) for s in COLOR_SPACES]


# ---------------------------------------------------------------------------
# Limiares clínicos (NeoJaundice)
# ---------------------------------------------------------------------------
# TSB (bilirrubina sérica total) >= 12.9 mg/dL -> jaundice (limiar de fototerapia).
TSB_BINARY_THRESHOLD: float = 12.9
# Bins de 3 classes (mg/dL): [0,5) / [5,10) / [10, inf).
TSB_3CLASS_BINS: Tuple[Tuple[float, float], ...] = (
    (0.0, 5.0), (5.0, 10.0), (10.0, float("inf")),
)
CLASS_NAMES_2: Tuple[str, ...] = ("healthy", "jaundice")
CLASS_NAMES_3: Tuple[str, ...] = ("00-05", "05-10", "10-inf")


# ---------------------------------------------------------------------------
# Registry de backbones (restrito aos campeões/baselines)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BackboneSpec:
    name: str
    family: str         # "cnn" ou "vit"
    source: str         # "torchvision" ou "timm"
    builder: str
    input_size: int
    feature_dim: int
    # Nome timm equivalente — usado SÓ pelo ChromaFiLMNet, que constrói o backbone
    # via timm (uniformiza feature_info/.blocks p/ os hooks de FiLM por estágio).
    # Vazio => usa ``builder`` (já é um nome timm quando source == "timm").
    timm_name: str = ""


BACKBONES: Dict[str, BackboneSpec] = {
    # CNNs
    "resnet18":        BackboneSpec("resnet18", "cnn", "torchvision", "resnet18", 256, 512, "resnet18"),
    "efficientnet_b0": BackboneSpec("efficientnet_b0", "cnn", "torchvision", "efficientnet_b0", 256, 1280, "efficientnet_b0"),
    "inception_v3":    BackboneSpec("inception_v3", "cnn", "torchvision", "inception_v3", 299, 2048, "inception_v3"),
    # ViT (timm)
    "deit_tiny":       BackboneSpec("deit_tiny", "vit", "timm", "deit_tiny_patch16_224", 224, 192, "deit_tiny_patch16_224"),
    # SOTA modernos (timm >= 0.9). family controla channels_last (cnn) e display.
    "convnextv2_tiny":  BackboneSpec("convnextv2_tiny", "cnn", "timm", "convnextv2_tiny", 224, 768, "convnextv2_tiny"),
    "swinv2_tiny":      BackboneSpec("swinv2_tiny", "vit", "timm", "swinv2_tiny_window16_256", 256, 768, "swinv2_tiny_window16_256"),
    "efficientnetv2_s": BackboneSpec("efficientnetv2_s", "cnn", "timm", "tf_efficientnetv2_s", 256, 1280, "tf_efficientnetv2_s"),
    # --- v6: âncoras da literatura + canônico CNN (efficientnet_b4) + ViT grande + DINOv3 ---
    "efficientnet_b4":     BackboneSpec("efficientnet_b4", "cnn", "torchvision", "efficientnet_b4", 256, 1792, "efficientnet_b4"),
    "resnet50":            BackboneSpec("resnet50", "cnn", "torchvision", "resnet50", 256, 2048, "resnet50"),
    "densenet121":         BackboneSpec("densenet121", "cnn", "torchvision", "densenet121", 256, 1024, "densenet121"),
    "mobilenetv3_large":   BackboneSpec("mobilenetv3_large", "cnn", "torchvision", "mobilenetv3_large", 256, 960, "mobilenetv3_large_100"),
    "vit_b_16":            BackboneSpec("vit_b_16", "vit", "torchvision", "vit_b_16", 224, 768, "vit_base_patch16_224"),
    # --- v7: comparação direta com o pesquisador externo (mesma partição canônica) ---
    # DeiT não-tiny (completa a escada de capacidade tiny→small→base p/ a tese "fraqueza
    # do backbone") + ViT-B/32 (patch grosso) e ViT-L/16 (grande). Loaders já em models.py.
    "vit_b_32":            BackboneSpec("vit_b_32", "vit", "torchvision", "vit_b_32", 224, 768, "vit_base_patch32_224"),
    "vit_l_16":            BackboneSpec("vit_l_16", "vit", "torchvision", "vit_l_16", 224, 1024, "vit_large_patch16_224"),
    "deit_small":          BackboneSpec("deit_small", "vit", "timm", "deit_small_patch16_224", 224, 384, "deit_small_patch16_224"),
    "deit_base":           BackboneSpec("deit_base", "vit", "timm", "deit_base_patch16_224", 224, 768, "deit_base_patch16_224"),
    # DINOv3 (caminho de maior risco; validar em 1 fold no PASS 2). timm >= 1.0 expõe
    # o DINOv3 ViT-S/16; se indisponível no ambiente, cai p/ efficientnet_b4 no run.
    "dinov3_vits16":       BackboneSpec("dinov3_vits16", "vit", "timm", "vit_small_patch16_dinov3.lvd1689m", 224, 384, "vit_small_patch16_dinov3.lvd1689m"),
}


def timm_name_of(spec: BackboneSpec) -> str:
    """Nome timm p/ o ChromaFiLMNet (uniformiza os hooks); fallback = builder."""
    return spec.timm_name or spec.builder


FAMILIES: Dict[str, List[str]] = {
    "cnn": [n for n, s in BACKBONES.items() if s.family == "cnn"],
    "vit": [n for n, s in BACKBONES.items() if s.family == "vit"],
}

DISPLAY_NAMES: Dict[str, str] = {
    "resnet18": "ResNet18",
    "efficientnet_b0": "EfficientNetB0",
    "inception_v3": "Inception",
    "deit_tiny": "DeiT-Tiny",
    "convnextv2_tiny": "ConvNeXtV2-T",
    "swinv2_tiny": "SwinV2-T",
    "efficientnetv2_s": "EffNetV2-S",
    "efficientnet_b4": "EfficientNetB4",
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "mobilenetv3_large": "MobileNetV3-L",
    "vit_b_16": "ViT-B/16",
    "vit_b_32": "ViT-B/32",
    "vit_l_16": "ViT-L/16",
    "deit_small": "DeiT-Small",
    "deit_base": "DeiT-Base",
    "dinov3_vits16": "DINOv3-ViT-S/16",
}


def display_name(backbone: str) -> str:
    return DISPLAY_NAMES.get(backbone, backbone)


def resolve_targets(target: str) -> List[str]:
    """Resolve ``--target`` (família ``cnn``/``vit``, ``all`` ou um modelo)."""
    target = target.strip().lower()
    if target in FAMILIES:
        return list(FAMILIES[target])
    if target == "all":
        return list(BACKBONES.keys())
    if target in BACKBONES:
        return [target]
    raise ValueError(
        f"Alvo desconhecido: {target!r}. Use uma familia {list(FAMILIES)}, "
        f"'all', ou um modelo {list(BACKBONES)}."
    )


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str          # "neojaundice_live" (CSV + WB + ROI) | "njn_flat" (pasta plana + skin-ROI)
    subdir: str
    supports_3class: bool


DATASETS: Dict[str, DatasetSpec] = {
    # NeoJaundice: imagens cruas 567x567 + CSV; WB pela carta + crop central (live).
    "NeoJaundice": DatasetSpec("NeoJaundice", "neojaundice_live", "NeoJaundice", supports_3class=True),
    # NJN: corpo na incubadora; pasta plana jaundice/normal (sem IDs); split por
    # pseudo-paciente (pHash) + skin-ROI opcional.
    "NJN":         DatasetSpec("NJN", "njn_flat", "NJN", supports_3class=False),
}


# ---------------------------------------------------------------------------
# Unidade atômica de experimento
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    backbone: str
    dataset: str
    colorspaces: Tuple[str, ...]          # subconjunto (ordem canônica)
    num_classes: int = 2                  # 2 (healthy/jaundice) | 3 (faixas TSB, só NeoJaundice)
    fusion: str = "adapter"               # adapter (principal) | adapter_v2 | inflate | stemcnn

    # Recursos / execução
    gpu: int = 0
    seed: int = 42
    num_workers: int = 8
    mp_context: str = "forkserver"        # start method dos workers: forkserver (seguro c/ CUDA + rápido) | spawn | fork

    # Semente do TPESampler. None => cfg.seed, que é o comportamento da campanha v7 e
    # a razão de os 30 estudos partilharem o bloco de arranque (docs/PENDENCIAS.md item
    # 15). Existe para se poder medir a alternativa, NÃO para a mudar por omissão.
    hpo_sampler_seed: Optional[int] = None

    # Orçamento Optuna / treino
    hpo_trials: int = 50
    search_epochs: int = 50
    final_epochs: int = 100
    patience: int = 7
    batch_size: int = 32

    # Pré-processamento NeoJaundice (live)
    wb_method: str = "graypatch"          # graypatch | grayworld | lab | gain | off
    roi: Tuple[float, float] = (0.30, 0.70)  # janela central fracionária (descarta cartão)

    # NJN (ROI de pele)
    njn_mode: str = "skin_roi"            # skin_roi | full_image
    skin_n_patches: int = 3               # nº de maiores contornos de pele
    phash_dist: int = 5                   # distância pHash p/ unir pseudo-pacientes

    # Performance
    amp: bool = True
    amp_dtype: str = "bf16"               # bf16 (default, sem GradScaler) | fp16 (+GradScaler)
    use_cache: bool = True                # cache de disco de imagens pré-processadas
    backbone_lr_mult: float = 0.1         # LR do backbone = lr * mult (differential LR)

    # Caminhos
    data_root: str = "dataset"
    results_dir: str = "local/runs"
    cache_dir: str = "local/cache/roi"
    version: str = ""

    # Comportamento
    force: bool = False
    fusion_as_hparam: bool = False
    objective: str = "f1_macro"
    tune_threshold: bool = True
    tta: bool = True
    label_smoothing: float = 0.0          # regularizacao da CrossEntropy (0.1 tipico)
    adapter_init: str = "identity"        # identity | balanced
    save_explain: bool = False            # grava state_dict + gate por imagem (explainpack)
    hue_circular: bool = False

    # Explicabilidade
    generate_heatmaps: bool = False
    heatmap_domain: str = "both"          # roi | original | both (NJN); NeoJaundice -> roi

    # Smoke test (trunca cada split; None = dataset completo)
    limit_per_split: Optional[int] = None

    # ---- v6: ChromaFiLM / CV / estabilidade ----
    # ATENCAO ao rotulo "lora": no caminho HCSFModel (adapter_v2, que e o da campanha
    # v7) NENHUM modulo LoRA e instanciado -- _lora_modules fica vazio e o wiring LoRA
    # so existe em ChromaFiLMNet, que tem zero execucoes na campanha. Com
    # backbone_mode=lora o regime efectivo e: backbone congelado excepto os ultimos
    # n_unfreeze tensores (LR = lr x backbone_lr_mult) e backbone em eval(). O rotulo e
    # historico e esta preservado nos registos; ver README, "Rotulos historicos".
    backbone_mode: str = "frozen"         # frozen | lora | full (full fine-tuning)
    film_mode: str = "channel"            # channel (headline) | spatial (só CNN)
    lora_rank: int = 8                    # rank do LoRA (só ChromaFiLMNet; não usado na v7)
    multitask: bool = False               # cabeça de regressão TSB (só NeoJaundice)
    cv_folds: int = 5                     # k do StratifiedGroupKFold
    fold: int = 0                         # fold ativo [0, cv_folds) — setado pelo cv.py
    split_mode: str = "group"             # group (honesto) | random (Track A vazamento)
    split_source: str = "kfold"           # kfold (5-fold gerado on-the-fly) | frozen (split canônico
                                          # 70/10/20 congelado em dataset/<DS>/splits/, ver make_splits.py)
    threshold_mode: str = "pooled_oof"    # pooled_oof | youden | per_fold
    swa: bool = False                     # Stochastic Weight Averaging
    ema: bool = False                     # Exponential Moving Average dos pesos
    ema_decay: float = 0.999

    def __post_init__(self):
        self.colorspaces = canonical_colorset(self.colorspaces)
        if self.backbone not in BACKBONES:
            raise ValueError(f"Backbone invalido: {self.backbone} (use {list(BACKBONES)}).")
        if self.dataset not in DATASETS:
            raise ValueError(f"Dataset invalido: {self.dataset} (use {list(DATASETS)}).")
        if self.num_classes not in (2, 3):
            raise ValueError(f"num_classes invalido: {self.num_classes} (use 2 ou 3).")
        if self.num_classes == 3 and not self.dataset_spec.supports_3class:
            raise ValueError(
                f"Dataset {self.dataset} nao suporta 3 classes (sem TSB/CSV). Use --num-classes 2.")
        if self.fusion not in ("adapter", "adapter_v2", "inflate", "stemcnn", "ccat",
                               "chromafilm", "chromafilm_route", "ds_lfn", "naive_concat"):
            raise ValueError(f"Fusion invalida: {self.fusion}")
        if self.adapter_init not in ("identity", "balanced"):
            raise ValueError(f"adapter_init invalido: {self.adapter_init}")
        if self.backbone_mode not in ("frozen", "lora", "full"):
            raise ValueError(f"backbone_mode invalido: {self.backbone_mode} (frozen|lora|full).")
        if self.film_mode not in ("channel", "spatial"):
            raise ValueError(f"film_mode invalido: {self.film_mode} (channel|spatial).")
        if self.split_mode not in ("group", "random"):
            raise ValueError(f"split_mode invalido: {self.split_mode} (group|random).")
        if self.split_source not in ("kfold", "frozen"):
            raise ValueError(f"split_source invalido: {self.split_source} (kfold|frozen).")
        if self.threshold_mode not in ("pooled_oof", "youden", "per_fold"):
            raise ValueError(f"threshold_mode invalido: {self.threshold_mode}.")
        if self.film_mode == "spatial" and self.spec.family == "vit":
            raise ValueError("film_mode=spatial só é suportado em CNN (ver plano §6); "
                             "use channel para backbones ViT.")
        if self.multitask and not self.is_neojaundice:
            raise ValueError("multitask (regressão TSB) só existe na NeoJaundice (tem CSV).")
        if self.cv_folds > 0 and not (0 <= self.fold < self.cv_folds):
            raise ValueError(f"fold {self.fold} fora de [0,{self.cv_folds}).")
        if self.wb_method not in ("graypatch", "grayworld", "lab", "gain", "off"):
            raise ValueError(f"wb_method invalido: {self.wb_method}.")
        if self.njn_mode not in ("skin_roi", "full_image"):
            raise ValueError(f"njn_mode invalido: {self.njn_mode} (use skin_roi|full_image).")
        if self.amp_dtype not in ("bf16", "fp16"):
            raise ValueError(f"amp_dtype invalido: {self.amp_dtype} (use bf16|fp16).")
        if self.heatmap_domain not in ("roi", "original", "both"):
            raise ValueError(f"heatmap_domain invalido: {self.heatmap_domain}.")
        self.roi = (float(self.roi[0]), float(self.roi[1]))
        if not (0.0 <= self.roi[0] < self.roi[1] <= 1.0):
            raise ValueError(f"roi invalida: {self.roi} (exige 0 <= lo < hi <= 1).")

    # ---- derivados ----
    @property
    def spec(self) -> BackboneSpec:
        return BACKBONES[self.backbone]

    @property
    def dataset_spec(self) -> DatasetSpec:
        return DATASETS[self.dataset]

    @property
    def is_neojaundice(self) -> bool:
        return self.dataset_spec.kind == "neojaundice_live"

    @property
    def n_input_channels(self) -> int:
        return sum(space_nchannels(sp, self.hue_circular) for sp in self.colorspaces)

    @property
    def colorset_id(self) -> str:
        return colorset_id(self.colorspaces)

    @property
    def class_names(self) -> Tuple[str, ...]:
        return CLASS_NAMES_3 if self.num_classes == 3 else CLASS_NAMES_2

    @property
    def is_binary(self) -> bool:
        return self.num_classes == 2

    @property
    def unit_id(self) -> str:
        """Identificador único: backbone__cores__fusao[__variantes]__c{K}[__njnmode]__s{seed}."""
        tags = ""
        if self.fusion.startswith("adapter") and self.adapter_init != "identity":
            tags += f"__{self.adapter_init}"
        if self.hue_circular:
            tags += "__hcirc"
        # Eixos v6 que mudam o modelo/protocolo — entram na chave p/ não colidir markers.
        if self.fusion in ("chromafilm", "chromafilm_route"):
            tags += f"__film-{self.film_mode}"
        if self.backbone_mode != "frozen":
            tags += f"__{self.backbone_mode}"
        if self.multitask:
            tags += "__mt"
        if self.wb_method != "graypatch" and self.is_neojaundice:
            tags += f"__wb-{self.wb_method}"
        if self.split_mode != "group":
            tags += f"__split-{self.split_mode}"
        njn = f"__{self.njn_mode}" if not self.is_neojaundice else ""
        return (f"{self.backbone}__{self.colorset_id}__{self.fusion}{tags}"
                f"__c{self.num_classes}{njn}__s{self.seed}")

    def result_dir(self) -> Path:
        base = Path(self.results_dir) / self.dataset
        if self.version:
            base = base / self.version
        return base / self.backbone

    def marker_path(self) -> Path:
        return self.result_dir() / f"{self.unit_id}.json"

    def weights_path(self) -> Path:
        return self.result_dir() / f"{self.unit_id}.pth"

    def cache_tag(self) -> str:
        """Tag da subpasta de cache: dataset/wb/roi/njn_mode/resize/algo_version.

        ``SKIN_ALGO_VERSION`` entra na chave para que ajustes nos limiares de skin
        invalidem o cache antigo automaticamente.
        """
        from .skin_roi import SKIN_ALGO_VERSION
        lo, hi = self.roi
        if self.is_neojaundice:
            return (f"{self.dataset}_wb-{self.wb_method}_roi{lo:.2f}-{hi:.2f}"
                    f"_r{self.spec.input_size}")
        return (f"{self.dataset}_{self.njn_mode}_n{self.skin_n_patches}"
                f"_r{self.spec.input_size}_skin-v{SKIN_ALGO_VERSION}")

    def stats_path(self) -> Path:
        """Cache de estatísticas de normalização (por dataset/wb/roi/njn/resize)."""
        return Path(self.results_dir) / "stats" / f"{self.cache_tag()}.json"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["colorspaces"] = list(self.colorspaces)
        d["roi"] = list(self.roi)
        d["unit_id"] = self.unit_id
        d["n_input_channels"] = self.n_input_channels
        d["class_names"] = list(self.class_names)
        return d
