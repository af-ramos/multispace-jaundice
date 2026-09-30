"""
make_fig5.py — Figura 5 do paper: mapas de atenção do CCAT por espaço de cor.

Lê os dumps ``results/<ds>/<bb>/attn/<run_id>.npz`` gravados por :mod:`attn` durante o
run (o v7 não persiste pesos — se não foi coletado no run, não existe) e monta:

  linha 1  a imagem original de cada caso amostrado, rotulada com o bucket (TP/TN/FP/FN)
           e a probabilidade predita. Os ERROS entram por cota própria: uma figura só
           com acertos seria propaganda, não evidência.
  linha 2+ um mapa por espaço crômico, sobreposto à imagem em ESCALA DE CINZA (o canal
           de cor fica exclusivamente para o dado). Como a atenção do CCAT é um softmax
           POR PIXEL sobre os espaços, os mapas de uma coluna somam 1 — a leitura correta
           é "onde este espaço GANHA dos outros", não "intensidade".

**Codificação de cor (método dataviz).** O trabalho aqui é POLARIDADE em torno de um
ponto neutro com significado — o empate 1/K, onde nenhum espaço ganha — então a rampa é
**divergente**: azul (o espaço perde) ↔ cinza neutro no empate ↔ vermelho (o espaço
ganha). Nunca uma rampa sequencial tipo magma/viridis, que não tem meio e faria o
empate parecer um valor qualquer. A escala é SIMÉTRICA em torno de 1/K e COMPARTILHADA
por todos os painéis (senão dois painéis com a mesma cor significariam coisas
diferentes). Para o P&B do paper impresso, os dois polos escurecem e o empate clareia:
o número da fração média vem impresso em cada painel, para que o sinal seja legível sem cor.

Também escreve ``results/fig5_atencao_por_espaco.csv`` com a fração média de atenção por
espaço e bucket — o número que a legenda cita, para a figura não ser a única fonte.

⚠️ Diferença de natureza em relação ao gate do ``adapter_v2``: aquele é um escalar por
espaço (e por isso pode ficar inerte — a BN o reabsorve); este é uma distribuição
espacial, que a BN não reabsorve. É a razão de o CCAT existir no estudo.

Local, sem GPU. Uso:
    python -m src.make_fig5
    python -m src.explain --attn-maps
Saída: paper/figs/fig5_atencao_ccat.{pdf,png} + results/fig5_atencao_por_espaco.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .attn import AttnDump
from .stats import load_markers

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2dd"
#: Par divergente da paleta de referência: polos que leem como opostos (quente/frio)
#: + cinza neutro no meio. blue↔red, midpoint #f0efec.
DIV_LO, DIV_MID, DIV_HI = "#2a78d6", "#f0efec", "#e34948"
#: Ordem de exibição dos casos: acertos primeiro, erros no fim (mas sempre presentes).
BUCKET_ORDER = ["TP", "FN", "TN", "FP"]
BUCKET_LABEL = {"TP": "ictérico correto", "FN": "ictérico PERDIDO",
                "TN": "saudável correto", "FP": "falso alarme"}
CSV_SEP = ";"


def _cmap_divergente():
    """blue → cinza neutro → red, com o cinza EXATAMENTE no meio da rampa."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "hcsf_div", [(0.0, DIV_LO), (0.5, DIV_MID), (1.0, DIV_HI)])


def _escala(d: AttnDump, q: float = 98.0) -> float:
    """Meia-amplitude simétrica em torno de 1/K, robusta a outliers (percentil q).

    Compartilhada por TODOS os painéis: sem isso, a mesma cor significaria desvios
    diferentes em painéis diferentes e a figura mentiria por construção.
    """
    K = max(d.n_spaces, 1)
    desvio = np.abs(d.maps.astype(np.float32) - 1.0 / K)
    v = float(np.percentile(desvio, q)) if desvio.size else 0.0
    return max(v, 1e-3)                       # nunca 0 (escala degenerada)


def _dec(x) -> str:
    return f"{float(x):.6g}".replace(".", ",")


def _load_image(path: str, size: int) -> Optional[np.ndarray]:
    """Imagem original em RGB [size, size, 3] float 0–1; None se ilegível.

    Sem pré-processamento (WB/ROI): o objetivo é o leitor reconhecer o bebê, não
    reproduzir a entrada do modelo — o mapa é que carrega a informação do modelo.
    """
    try:
        import cv2
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (size, size)).astype(np.float32) / 255.0
    except Exception:
        return None


def _casos(d: AttnDump, por_bucket: int = 1) -> List[int]:
    """Índices dos casos a exibir: uma cota por bucket, seed mais baixo (determinístico)."""
    idx: List[int] = []
    for b in BUCKET_ORDER:
        cand = [i for i in range(d.n_rows) if str(d.buckets[i]) == b]
        cand.sort(key=lambda i: (int(d.seeds[i]), int(d.folds[i]), str(d.paths[i])))
        idx.extend(cand[:por_bucket])
    return idx


def _dumps(results_dir: str, marker: Optional[str]) -> List[tuple]:
    """(marcador, AttnDump) de todas as células CCAT com dump em disco."""
    out = []
    for rec in load_markers(results_dir):
        af = rec.get("attn_file")
        if not af:
            continue
        if marker and not any(marker in str(rec.get(k, ""))
                              for k in ("_path", "tag", "run_id")):
            continue
        p = Path(results_dir) / af
        if not p.is_file():
            print(f"[fig5] aviso: {rec.get('tag')} aponta {af}, ausente em disco "
                  "(dump é gitignored — traga-o no rsync).")
            continue
        out.append((rec, AttnDump.load(p)))
    return out


def write_share_csv(pares: List[tuple], out: Path) -> int:
    """Fração média de atenção por espaço e bucket (o número citado na legenda)."""
    linhas = [CSV_SEP.join(["dataset", "backbone", "colorset_id", "bucket", "n_imagens",
                            "espaco", "fracao_atencao"])]
    n = 0
    for rec, d in pares:
        for b in [None] + BUCKET_ORDER:
            sub = d.subset(bucket=b) if b else d
            if sub.n_rows == 0:
                continue
            for nome, v in sub.space_share().items():
                linhas.append(CSV_SEP.join([
                    rec.get("dataset", ""), rec.get("backbone", ""),
                    rec.get("colorset_id", ""), b or "todos", str(sub.n_rows),
                    nome, _dec(v)]))
                n += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(linhas) + "\n", encoding="utf-8-sig")
    return n


def _painel(rec: dict, d: AttnDump, fig_path_stem: str, out_dir: Path) -> Optional[str]:
    idx = _casos(d)
    if not idx:
        return None
    K = d.n_spaces
    H = d.maps.shape[-1]
    tie = 1.0 / max(K, 1)
    half = _escala(d)
    cmap = _cmap_divergente()
    ncols, nrows = len(idx), K + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.35 * ncols, 2.5 * nrows),
                             squeeze=False)
    fig.patch.set_facecolor("white")
    im = None
    for c, i in enumerate(idx):
        img = _load_image(str(d.paths[i]), H)
        base = img if img is not None else np.ones((H, H, 3), dtype=np.float32) * 0.9
        # Luminância Rec.601: o fundo vira contexto anatômico, o canal de cor fica
        # inteiro para o dado (senão a cor da pele compete com a cor do mapa).
        cinza = base @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        ax = axes[0][c]
        ax.imshow(base)
        b = str(d.buckets[i])
        ax.set_title(f"{b} — {BUCKET_LABEL.get(b, b)}\np={float(d.y_prob[i]):.2f}"
                     + ("" if img is not None else "\n(imagem indisponível)"),
                     fontsize=8.5, color=INK, pad=6)
        for k in range(K):
            axk = axes[k + 1][c]
            axk.imshow(cinza, cmap="gray", vmin=0.0, vmax=1.0)
            m = d.maps[i, k].astype(np.float32)
            im = axk.imshow(m, cmap=cmap, alpha=0.70,
                            vmin=tie - half, vmax=tie + half)
            # O número torna o painel legível em P&B (os dois polos escurecem igual).
            axk.set_xlabel(f"média {float(m.mean()):.2f}", fontsize=7.5, color=MUTED,
                           labelpad=2)
            if c == 0:
                axk.set_ylabel(str(d.space_names[k]), fontsize=10, color=INK)
    for row in axes:
        for ax in row:
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    ds, bb = rec.get("dataset", ""), rec.get("backbone", "")
    cs = rec.get("colorset_id", "")
    share = d.space_share()
    resumo = " · ".join(f"{k}={v:.2f}" for k, v in share.items())
    fig.suptitle(f"Fig 5 — atenção cruzada do CCAT por espaço · {bb} · {ds} · {cs}",
                 fontsize=12.5, color=INK, y=1.0)
    fig.text(0.5, 0.975,
             f"Softmax POR PIXEL sobre os {K} espaços crômicos: os mapas de uma coluna "
             f"somam 1 em cada pixel, e o empate é 1/{K}={tie:.2f}. A escala divergente é "
             f"simétrica em torno do empate e COMPARTILHADA por todos os painéis. "
             f"Fração média de atenção: {resumo}. Casos por cota de bucket, erros incluídos.",
             ha="center", va="top", fontsize=8.5, color=MUTED, wrap=True)
    # A colorbar entra DEPOIS do tight_layout, em eixo próprio: pedi-la sobre os eixos
    # de dados faz o matplotlib roubar largura da última coluna e a escala acaba
    # desenhada por cima da 4ª imagem.
    fig.tight_layout(rect=[0, 0, 0.93, 0.955])
    if im is not None:
        cax = fig.add_axes([0.945, 0.10, 0.013, 0.52])
        cb = fig.colorbar(im, cax=cax, ticks=[tie - half, tie, tie + half])
        cb.ax.set_yticklabels([f"{tie - half:.2f}\nperde", f"{tie:.2f}\nempate",
                               f"{tie + half:.2f}\nganha"], fontsize=7.5, color=INK)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = None
    for ext in ("pdf", "png"):
        p = out_dir / f"{fig_path_stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[fig5] {p}")
        saved = str(p)
    plt.close(fig)
    return saved


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fig 5 — mapas de atenção do CCAT.")
    ap.add_argument("--results-dir", default="local/runs")
    ap.add_argument("--marker", default=None,
                    help="Restringe a um marcador (caminho, tag ou run_id — substring).")
    ap.add_argument("--out-dir", default="local/generated/figs",
                    help="Onde gravar a figura (default: paper/figs).")
    args = ap.parse_args(argv)

    pares = _dumps(args.results_dir, args.marker)
    if not pares:
        print("[fig5] nenhum dump de atenção encontrado.\n"
              "  Os mapas só existem se coletados DURANTE o run (o v7 não persiste pesos).\n"
              "  Colete com: ./colorspace_experiments_v7/run_fase4_interp_1gpu.sh\n"
              "  e traga results/**/attn/*.npz no rsync (são gitignored).")
        return 0

    n = write_share_csv(pares, Path(args.results_dir) / "fig5_atencao_por_espaco.csv")
    print(f"[fig5] frações de atenção: {n} linhas -> "
          f"{Path(args.results_dir)/'fig5_atencao_por_espaco.csv'}")
    for rec, d in pares:
        # O colorset_id ENTRA no nome do arquivo. Sem ele, as células K=1 (colorset
        # mínimo do CCAT, atenção identicamente 1,000) e K=3 colidiam no mesmo PDF e o
        # vencedor era a ordem do filesystem — na prática a NJN ficava com a figura
        # degenerada. Ver PENDENCIAS_V7.md §P1.1 e CONSOLIDADO_V7.md §Fig 5.
        stem = "fig5_atencao_ccat" if len(pares) == 1 else (
            f"fig5_atencao_ccat_{rec.get('dataset','')}_{rec.get('backbone','')}"
            f"_{(rec.get('colorset_id') or '').replace('+', '')}")
        _painel(rec, d, stem, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
