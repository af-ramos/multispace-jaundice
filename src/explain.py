"""
explain.py (NOVO — v6, Fase 3: interpretabilidade)
==================================================

Consolida os campos de interpretabilidade que ``train.py`` passou a gravar no
marcador JSON (via :func:`cv._collect_interpretability`):

* ``alpha_space`` / ``alpha_space_names`` — o α_space (softmax por espaço crômico):
  **"qual espaço de cor o modelo usa"** (LAB vs HSV vs YCrCb), por dataset/condição.
* ``film_profile`` — perfil ``[profundidade_relativa, ‖Δγ_l‖, ‖Δβ_l‖]`` por estágio:
  **quão forte** a cromância modula as features, **por profundidade (0–1)** — comparável
  entre backbones com nº de estágios diferente.
* ``loso_delta_auc`` / ``loso_names`` (só ``chromafilm_route``) — ΔAUC de deixar-um-espaço-de-
  fora: **ablação rigorosa** que cruza o α_space aprendido ("o espaço que o α prioriza é o que,
  removido, mais derruba a AUC?"). Fecha o buraco do gate inerte das versões anteriores.

Produz (CSV pt-BR ``;``/``,``, Excel-friendly) e, com ``--plots``, PNGs prontos p/ o paper:

    python -m src.explain --results-dir local/runs --plots

**Narrativa-alvo:** a modulação (‖Δγ,Δβ‖) deve ser MAIOR onde a cor NÃO foi calibrada
(NeoJaundice ``wb off`` > ``wb on``; NJN ``full_image`` > ``skin_roi``) — o mesmo padrão
assimétrico do ganho de classificação.

Emenda 2026-07-26 (fecha a pendência de interpretabilidade da Fase 4):

* ``--gate-alpha`` — tabela do **gate por espaço do ``adapter_v2``** (``_SpaceGate``),
  com o **teste de inércia**: se o gate for ~1,0 em todos os espaços (spread≈0), ele
  não está selecionando nada e a BN a jusante o reabsorve — o diagnóstico que os
  PRINCÍPIOS DE PROJETO do CLAUDE.md registram desde a v2. Inerte é RESULTADO
  reportável (a fusão que REAGE à cor é a por-pixel), não bug a esconder.
  ⚠️ O vetor do gate inclui o **RGB**; o α_space do ChromaFiLM, não. Por isso a tabela
  traz a coluna ``escopo`` e os nomes vêm sempre de ``alpha_space_names``.
* ``--attn-maps`` — delega para :mod:`make_fig5` (mapas de atenção do CCAT, Fig 5).
* ``--marker`` — restringe a um marcador (caminho ou substring de tag/run_id).

**Onde estão as outras análises** (NÃO são flags deste módulo — foram entregues como
scripts próprios, e duplicá-las aqui criaria duas fontes para o mesmo número):

    espaço-por-espaço (T3) → python -m src.fase6_final_stats
    estratificação por ITA° → python -m src.ita_strat
    teste de atividade da fusão → pytest -k "fusion_activity" + smoke explain_activity
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from .stats import load_markers

CSV_SEP = ";"


def _dec(x) -> str:
    """Float -> texto com vírgula decimal (pt-BR); None/str tratados."""
    if x is None:
        return ""
    if not isinstance(x, (int, float)):
        return str(x)
    return f"{float(x):.6g}".replace(".", ",")


def _condition(rec: dict) -> str:
    """Rótulo da condição de calibração (o eixo da tese)."""
    if rec.get("dataset") == "NeoJaundice":
        return f"wb-{rec.get('wb', '?')}"
    return str(rec.get("njn_mode") or "?")   # skin_roi | full_image


def _has_interp(rec: dict) -> bool:
    return bool(rec.get("alpha_space")) or bool(rec.get("film_profile")) \
        or bool(rec.get("loso_delta_auc"))


#: Abaixo deste spread (max−min dos pesos por espaço) o gate é considerado INERTE.
#: 0,05 é o mesmo limiar usado na auditoria do v4a (`audit_v4a`), onde o conserto do
#: gate falhou por σ(α) ≫ 0,05. Aqui o critério é o spread, mais legível a K pequeno.
GATE_INERTE_SPREAD = 0.05


def gate_rows(recs: List[dict]) -> List[dict]:
    """Uma linha por célula com gate por espaço + diagnóstico de inércia.

    ``spread`` = max−min dos pesos. Com a init neutra todos valem 1,0 (spread 0); se o
    treino não move isso, o gate não seleciona espaço nenhum — a BN a jusante reabsorve
    um ganho escalar por espaço, exatamente a armadilha registrada no CLAUDE.md.
    """
    out: List[dict] = []
    for r in sorted(recs, key=lambda x: (x.get("dataset", ""), x.get("backbone", ""),
                                         x.get("colorset_id", ""))):
        a = r.get("alpha_space")
        nomes = r.get("alpha_space_names") or []
        if not a or not nomes:
            continue
        vals = [float(v) for v in a]
        spread = max(vals) - min(vals)
        out.append({
            "dataset": r.get("dataset", ""), "backbone": r.get("backbone", ""),
            "colorset_id": r.get("colorset_id", ""), "fusion": r.get("fusion", ""),
            "condicao": _condition(r), "escopo": r.get("alpha_scope") or "",
            "pesos": {nm: v for nm, v in zip(nomes, vals)},
            "spread": spread,
            "veredito": ("INERTE" if spread < GATE_INERTE_SPREAD else "ativo"),
            "tag": r.get("tag", ""),
        })
    return out


def write_gate_csv(recs: List[dict], out: Path) -> int:
    """CSV do gate por espaço + spread + veredito de inércia."""
    rows = gate_rows(recs)
    spaces: List[str] = []
    for r in rows:
        for nm in r["pesos"]:
            if nm not in spaces:
                spaces.append(nm)
    header = (["dataset", "backbone", "colorset_id", "fusion", "condicao", "escopo"]
              + [f"gate_{s}" for s in spaces] + ["spread", "veredito", "tag"])
    lines = [CSV_SEP.join(header)]
    for r in rows:
        lines.append(CSV_SEP.join(
            [r["dataset"], r["backbone"], r["colorset_id"], r["fusion"], r["condicao"],
             r["escopo"]]
            + [_dec(r["pesos"].get(s)) for s in spaces]
            + [_dec(r["spread"]), r["veredito"], r["tag"]]))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return len(rows)


# --------------------------------------------------------------------------- #
# Tabelas
# --------------------------------------------------------------------------- #
def write_alpha_csv(recs: List[dict], out: Path) -> int:
    """Uma linha por config: α por espaço crômico (colunas dinâmicas por espaço)."""
    spaces: List[str] = []
    for r in recs:
        for nm in (r.get("alpha_space_names") or []):
            if nm not in spaces:
                spaces.append(nm)
    header = ["dataset", "backbone", "colorset_id", "condition", "tag"] + [f"alpha_{s}" for s in spaces]
    lines = [CSV_SEP.join(header)]
    n = 0
    for r in sorted(recs, key=lambda x: (x.get("dataset", ""), x.get("backbone", ""), x.get("colorset_id", ""))):
        a = r.get("alpha_space")
        if not a:
            continue
        amap = dict(zip(r.get("alpha_space_names") or [], a))
        row = [r.get("dataset", ""), r.get("backbone", ""), r.get("colorset_id", ""),
               _condition(r), r.get("tag", "")] + [_dec(amap.get(s)) for s in spaces]
        lines.append(CSV_SEP.join(row))
        n += 1
    out.write_text("\n".join(lines), encoding="utf-8-sig")
    return n


def write_film_csv(recs: List[dict], out: Path) -> int:
    """Uma linha por (config × estágio): perfil ‖Δγ‖,‖Δβ‖ por profundidade relativa."""
    header = ["dataset", "backbone", "colorset_id", "condition", "tag",
              "rel_depth", "dgamma_norm", "dbeta_norm"]
    lines = [CSV_SEP.join(header)]
    n = 0
    for r in sorted(recs, key=lambda x: (x.get("dataset", ""), x.get("backbone", ""), x.get("colorset_id", ""))):
        prof = r.get("film_profile")
        if not prof:
            continue
        for rel, dg, db in prof:
            lines.append(CSV_SEP.join([
                r.get("dataset", ""), r.get("backbone", ""), r.get("colorset_id", ""),
                _condition(r), r.get("tag", ""), _dec(rel), _dec(dg), _dec(db)]))
            n += 1
    out.write_text("\n".join(lines), encoding="utf-8-sig")
    return n


def write_loso_csv(recs: List[dict], out: Path) -> int:
    """Uma linha por (config × espaço): α_space aprendido vs ΔAUC do leave-one-space-out.

    Se o mecanismo é honesto, ``alpha`` e ``loso_delta_auc`` correlacionam (o espaço mais
    roteado é o que mais faz falta ao ser removido)."""
    header = ["dataset", "backbone", "colorset_id", "condition", "tag",
              "space", "alpha", "loso_delta_auc"]
    lines = [CSV_SEP.join(header)]
    n = 0
    for r in sorted(recs, key=lambda x: (x.get("dataset", ""), x.get("backbone", ""), x.get("colorset_id", ""))):
        loso = r.get("loso_delta_auc")
        if not loso:
            continue
        names = r.get("loso_names") or r.get("alpha_space_names") or []
        amap = dict(zip(r.get("alpha_space_names") or [], r.get("alpha_space") or []))
        for j, nm in enumerate(names):
            lines.append(CSV_SEP.join([
                r.get("dataset", ""), r.get("backbone", ""), r.get("colorset_id", ""),
                _condition(r), r.get("tag", ""), str(nm),
                _dec(amap.get(nm)), _dec(loso[j] if j < len(loso) else None)]))
            n += 1
    out.write_text("\n".join(lines), encoding="utf-8-sig")
    return n


# --------------------------------------------------------------------------- #
# Figuras (best-effort; requer matplotlib)
# --------------------------------------------------------------------------- #
def make_plots(recs: List[dict], out_dir: Path) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:  # pragma: no cover
        print(f"[explain] matplotlib indisponível ({e}); pulando figuras.")
        return []

    saved: List[str] = []

    # (1) α_space médio por espaço, agrupado por dataset × condição.
    alpha_recs = [r for r in recs if r.get("alpha_space")]
    if alpha_recs:
        agg: Dict[tuple, Dict[str, List[float]]] = {}
        for r in alpha_recs:
            key = (r.get("dataset", ""), _condition(r))
            d = agg.setdefault(key, {})
            for nm, v in zip(r.get("alpha_space_names") or [], r["alpha_space"]):
                d.setdefault(nm, []).append(float(v))
        keys = sorted(agg)
        spaces = sorted({s for d in agg.values() for s in d})
        x = np.arange(len(keys)); w = 0.8 / max(1, len(spaces))
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(keys)), 4))
        for j, s in enumerate(spaces):
            ax.bar(x + j * w, [float(np.mean(agg[k].get(s, [0]))) for k in keys], w, label=s)
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels([f"{d}\n{c}" for d, c in keys], fontsize=8)
        ax.set_ylabel("α_space (softmax médio)"); ax.axhline(1.0 / max(1, len(spaces)), ls=":", c="gray", lw=1)
        ax.set_title("α_space por dataset × condição de calibração"); ax.legend(fontsize=8)
        fig.tight_layout(); p = out_dir / "alpha_space.png"; fig.savefig(p, dpi=150); plt.close(fig)
        saved.append(str(p))

    # (2) Perfil FiLM por profundidade — overlay das condições (a assimetria da tese).
    film_recs = [r for r in recs if r.get("film_profile")]
    by_ds: Dict[str, List[dict]] = {}
    for r in film_recs:
        by_ds.setdefault(r.get("dataset", "?"), []).append(r)
    for ds, rs in sorted(by_ds.items()):
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in sorted(rs, key=_condition):
            prof = sorted(r["film_profile"], key=lambda t: t[0])
            xs = [t[0] for t in prof]; dg = [t[1] for t in prof]
            ax.plot(xs, dg, marker="o", ms=3, label=f"{r.get('backbone','')}·{r.get('colorset_id','')}·{_condition(r)}")
        ax.set_xlabel("profundidade relativa (0=raso, 1=fundo)")
        ax.set_ylabel("‖Δγ_l‖ (magnitude da modulação)")
        ax.set_title(f"Perfil FiLM por profundidade — {ds}"); ax.legend(fontsize=7)
        fig.tight_layout(); p = out_dir / f"film_profile_{ds}.png"; fig.savefig(p, dpi=150); plt.close(fig)
        saved.append(str(p))

    # (3) α_space aprendido vs ΔAUC do LOSO — a validação cruzada da atribuição por espaço.
    loso_recs = [r for r in recs if r.get("loso_delta_auc")]
    if loso_recs:
        fig, ax = plt.subplots(figsize=(5, 5))
        xs, ys, labels = [], [], []
        for r in loso_recs:
            amap = dict(zip(r.get("alpha_space_names") or [], r.get("alpha_space") or []))
            names = r.get("loso_names") or r.get("alpha_space_names") or []
            for j, nm in enumerate(names):
                a = amap.get(nm)
                if a is None or j >= len(r["loso_delta_auc"]):
                    continue
                xs.append(float(a)); ys.append(float(r["loso_delta_auc"][j])); labels.append(nm)
        for nm in sorted(set(labels)):
            pts = [(x, y) for x, y, l in zip(xs, ys, labels) if l == nm]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=28, label=nm)
        ax.axhline(0, ls=":", c="gray", lw=1)
        ax.set_xlabel("α_space (peso de roteamento aprendido)")
        ax.set_ylabel("ΔAUC leave-one-space-out (importância causal)")
        ax.set_title("Roteamento aprendido vs importância por ablação")
        ax.legend(fontsize=8, title="espaço")
        fig.tight_layout(); p = out_dir / "alpha_vs_loso.png"; fig.savefig(p, dpi=150); plt.close(fig)
        saved.append(str(p))
    return saved


def _match_marker(rec: dict, alvo: str) -> bool:
    """`--marker` casa por caminho do JSON, tag ou run_id (substring)."""
    return any(alvo in str(rec.get(k, "")) for k in ("_path", "tag", "run_id"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="v7 — interpretabilidade (α_space/gate + perfil FiLM + atenção CCAT).")
    ap.add_argument("--results-dir", default="local/runs")
    ap.add_argument("--out-dir", default=None, help="default: <results-dir>/explain")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--marker", default=None,
                    help="Restringe a um marcador (caminho, tag ou run_id — substring).")
    ap.add_argument("--gate-alpha", action="store_true",
                    help="Tabela do gate por espaço do adapter_v2 + teste de inércia "
                         "(spread < 0,05 = gate INERTE, a BN o reabsorve).")
    ap.add_argument("--attn-maps", action="store_true",
                    help="Gera a Fig 5 (mapas de atenção do CCAT) via make_fig5.")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir or (Path(args.results_dir) / "explain"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.attn_maps:
        from .make_fig5 import main as fig5_main
        rc = fig5_main(["--results-dir", args.results_dir]
                       + (["--marker", args.marker] if args.marker else []))
        if not (args.gate_alpha or args.plots):
            return rc

    todos = load_markers(args.results_dir)
    if args.marker:
        todos = [r for r in todos if _match_marker(r, args.marker)]

    if args.gate_alpha:
        gr = gate_rows(todos)
        if not gr:
            print("[explain] nenhum marcador com gate/α gravado.\n"
                  "  Os marcadores de adapter_v2 anteriores a 2026-07-26 têm alpha_space=None: "
                  "α só existe se coletado DURANTE o run (o v7 não persiste pesos).\n"
                  "  Colete com: ./colorspace_experiments_v7/run_fase4_interp_1gpu.sh")
        else:
            n = write_gate_csv(todos, out_dir / "gate_alpha.csv")
            inertes = sum(1 for r in gr if r["veredito"] == "INERTE")
            print(f"[explain] gate por espaço: {n} células -> {out_dir/'gate_alpha.csv'} | "
                  f"INERTES: {inertes}/{len(gr)} (spread < {GATE_INERTE_SPREAD})")
        if not args.plots:
            return 0

    recs = [r for r in todos if _has_interp(r)]
    if not recs:
        print("[explain] nenhum marcador com α_space/film_profile encontrado.\n"
              "  α/FiLM só existem se coletados DURANTE o run (o v7 não persiste pesos): "
              "colete com run_fase4_interp_1gpu.sh e repita.")
        return 0

    na = write_alpha_csv(recs, out_dir / "alpha_space.csv")
    nf = write_film_csv(recs, out_dir / "film_profile.csv")
    nl = write_loso_csv(recs, out_dir / "alpha_vs_loso.csv")
    print(f"[explain] {len(recs)} configs c/ interpretabilidade | "
          f"α_space: {na} linhas -> {out_dir/'alpha_space.csv'} | "
          f"FiLM: {nf} linhas -> {out_dir/'film_profile.csv'} | "
          f"LOSO: {nl} linhas -> {out_dir/'alpha_vs_loso.csv'}")
    if args.plots:
        for p in make_plots(recs, out_dir):
            print(f"[explain] figura -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
