"""accf1_stats.py — Reanálise da campanha sob ACURÁCIA e F1-MACRO.

Motivação (`docs/ANALISE_OPORTUNIDADES.md`): a consolidação da Fase 6 organiza tudo por
ROC-AUC, que é métrica de RANQUEAMENTO. A literatura de icterícia reporta acurácia e
F1 — métricas do PONTO DE OPERAÇÃO. As duas não coincidem: há células onde a cromância
quase não move a AUC e move a acurácia em vários pontos percentuais (ela melhora a
separação em torno do limiar, não a ordenação).

Roda 100% offline: lê só `results/**/*.json` (campo ``per_seed_metrics``, presente nos
450 marcadores `f1_*`). Sem GPU, sem tocar o cluster, sem re-treino.

Convenções mantidas da Fase 6:
* Comparações **pareadas por seed** (n=5, mesmo split congelado ⇒ mesmo test set).
* Δ + **IC95 t-Student** com t crítico via `scipy` (t₄=2,776) — nunca tabelado à mão.
* CSV pt-BR: separador ``;``, decimal ``,``, ``utf-8-sig``.

Diferença deliberada em relação à Fase 6: além do IC95 por célula, reporta-se a
**contagem de sinal** (quantos dos 7 colorsets com RGB vão na mesma direção). Com
n=5 seeds o IC95 por célula quase nunca fecha; a consistência entre colorsets é o
sinal que de fato distingue backbone-que-ganha de backbone-que-não-ganha.

Saídas:
  results/A1_factorial_accf1.csv        — as 420 comparações, Acc/F1/AUC
  results/A2_backbone_consistencia.csv  — por backbone: média + contagem de sinal
  results/A3_colorset_medio.csv         — por colorset: média sobre os 15 backbones
  (+ espelhos .md em paper/tables/)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats as sps

CSV_SEP = ";"
ALPHA = 0.05

#: Métricas reanalisadas. Acurácia e F1-macro são as PRIMÁRIAS aqui; a AUC entra como
#: coluna de contraste (é o que mostra que a divergência entre as duas leituras é real).
METRICAS = ("accuracy", "f1_macro", "roc_auc")

#: Os 4 espaços do factorial; um colorset é "com RGB" quando o RGB está na pilha.
#: Só esses interessam à tese do paper (RGB + cromância). Os chroma-only entram no
#: A1 para completude, mas ficam FORA das agregações A2/A3 — descartar o RGB é uma
#: intervenção diferente e a perda que ela causa afundaria a média.
def tem_rgb(colorset: str) -> bool:
    return "RGB" in colorset.split("+")


# --------------------------------------------------------------------------- utils
def _n(x, nd: int = 6) -> str:
    """Número no padrão pt-BR (decimal ','); vazio p/ None/NaN."""
    if x is None:
        return ""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f != f:
        return ""
    return f"{f:.{nd}g}".replace(".", ",")


def write_csv(path: Path, cols: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    linhas = [CSV_SEP.join(cols)]
    for r in rows:
        linhas.append(CSV_SEP.join(str(r.get(c, "")) for c in cols))
    path.write_text("\n".join(linhas) + "\n", encoding="utf-8-sig")
    print(f"[accf1] {len(rows)} linhas -> {path}")


def write_md(path: Path, titulo: str, cols: Sequence[str],
             rows: Sequence[Dict[str, object]], nota: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [f"# {titulo}", ""]
    if nota:
        out += [nota, ""]
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[accf1] {len(rows)} linhas -> {path}")


def pareado(a: Sequence[float], b: Sequence[float]) -> Dict[str, object]:
    """Δ = a − b pareado por seed, com IC95 t-Student.

    Pré-condição: ``a`` e ``b`` vêm dos MESMOS seeds, na mesma ordem.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    media = float(d.mean())
    if n < 2:
        return {"n": n, "media_a": float(a.mean()), "media_b": float(b.mean()),
                "delta": media, "lo": None, "hi": None, "sinal": ""}
    meia = float(sps.t.ppf(1 - ALPHA / 2, df=n - 1)) * float(d.std(ddof=1)) / np.sqrt(n)
    lo, hi = media - meia, media + meia
    return {"n": n, "media_a": float(a.mean()), "media_b": float(b.mean()),
            "delta": media, "lo": lo, "hi": hi,
            "sinal": "pos" if lo > 0 else ("neg" if hi < 0 else "nulo")}


def resumo_grupo(deltas: Sequence[float]) -> Dict[str, object]:
    """Média de um grupo de Δ + IC95 + contagem de sinal + teste de sinais.

    ⚠️ Os Δ de um mesmo backbone NÃO são independentes (compartilham baseline e seeds).
    O IC95 aqui é DESCRITIVO de dispersão; a afirmação forte é a contagem (ex.: 7/7).
    """
    v = np.asarray([x for x in deltas if x is not None], dtype=float)
    n = len(v)
    if n == 0:
        return {}
    media = float(v.mean())
    pos = int((v > 0).sum())
    neg = int((v < 0).sum())
    out = {"n": n, "delta_medio": media, "positivos": pos, "negativos": neg}
    if n >= 2:
        meia = float(sps.t.ppf(1 - ALPHA / 2, df=n - 1)) * float(v.std(ddof=1)) / np.sqrt(n)
        out["lo"], out["hi"] = media - meia, media + meia
        out["ic_exclui_zero"] = "sim" if (media - meia > 0 or media + meia < 0) else "nao"
    if pos + neg:
        out["p_sinais"] = float(sps.binomtest(pos, pos + neg, 0.5).pvalue)
    return out


# --------------------------------------------------------------------------- carga
def carregar_celulas(rdir: Path, prefixo: str = "f1_") -> Dict:
    """{(dataset, backbone, colorset): {metrica: array de 5 seeds}} dos marcadores."""
    celulas: Dict = {}
    for f in sorted(rdir.glob("*/*/*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if d.get("status") != "completed" or not str(d.get("tag", "")).startswith(prefixo):
            continue
        ps = d.get("per_seed_metrics") or {}
        if len(ps) < 2:
            continue
        seeds = sorted(ps, key=int)          # ordem canônica p/ o pareamento
        celulas[(d["dataset"], d["backbone"], d["colorset_id"])] = {
            m: np.array([ps[s][m] for s in seeds], dtype=float) for m in METRICAS
        }
    return celulas


# --------------------------------------------------------------------------- tabelas
def tabela_a1(celulas: Dict) -> List[Dict]:
    """Factorial completo: cada colorset ≠ RGB vs o RGB do MESMO backbone/dataset."""
    linhas: List[Dict] = []
    for (ds, bb, cs), cur in sorted(celulas.items()):
        if cs == "RGB":
            continue
        base = celulas.get((ds, bb, "RGB"))
        if base is None:
            continue
        r: Dict[str, object] = {"dataset": ds, "backbone": bb, "colorset": cs,
                                "contem_rgb": "sim" if tem_rgb(cs) else "nao"}
        for m in METRICAS:
            p = pareado(cur[m], base[m])
            r[f"{m}_rgb"] = _n(p["media_b"])
            r[f"{m}_cor"] = _n(p["media_a"])
            r[f"delta_{m}"] = _n(p["delta"])
            r[f"ic95_lo_{m}"] = _n(p["lo"])
            r[f"ic95_hi_{m}"] = _n(p["hi"])
            r[f"sinal_{m}"] = p["sinal"]
            r[f"_d_{m}"] = p["delta"]          # numérico, p/ as agregações (não vai ao CSV)
        linhas.append(r)
    return linhas


def tabela_a2(a1: List[Dict]) -> List[Dict]:
    """Por backbone: consistência do efeito sobre os colorsets COM RGB."""
    linhas: List[Dict] = []
    for ds in sorted({r["dataset"] for r in a1}):
        for bb in sorted({r["backbone"] for r in a1 if r["dataset"] == ds}):
            sub = [r for r in a1 if r["dataset"] == ds and r["backbone"] == bb
                   and r["contem_rgb"] == "sim"]
            if not sub:
                continue
            r: Dict[str, object] = {"dataset": ds, "backbone": bb, "n_colorsets": len(sub)}
            for m in METRICAS:
                g = resumo_grupo([x[f"_d_{m}"] for x in sub])
                r[f"delta_{m}_medio"] = _n(g.get("delta_medio"))
                r[f"ic95_lo_{m}"] = _n(g.get("lo"))
                r[f"ic95_hi_{m}"] = _n(g.get("hi"))
                r[f"positivos_{m}"] = f"{g.get('positivos', 0)}/{g.get('n', 0)}"
                r[f"_d_{m}"] = g.get("delta_medio")
            # Ordena a tabela pelo que interessa ao paper.
            r["_ord"] = g.get("delta_medio", 0.0)
            linhas.append(r)
    linhas.sort(key=lambda r: (r["dataset"], -float(r.get("_d_accuracy") or 0.0)))
    return linhas


def tabela_a3(a1: List[Dict]) -> List[Dict]:
    """Por colorset (só os com RGB): média sobre os 15 backbones."""
    linhas: List[Dict] = []
    for ds in sorted({r["dataset"] for r in a1}):
        for cs in sorted({r["colorset"] for r in a1 if r["contem_rgb"] == "sim"}):
            sub = [r for r in a1 if r["dataset"] == ds and r["colorset"] == cs]
            if not sub:
                continue
            r: Dict[str, object] = {"dataset": ds, "colorset": cs, "n_backbones": len(sub)}
            for m in METRICAS:
                g = resumo_grupo([x[f"_d_{m}"] for x in sub])
                r[f"delta_{m}_medio"] = _n(g.get("delta_medio"))
                r[f"ic95_lo_{m}"] = _n(g.get("lo"))
                r[f"ic95_hi_{m}"] = _n(g.get("hi"))
                r[f"positivos_{m}"] = f"{g.get('positivos', 0)}/{g.get('n', 0)}"
            linhas.append(r)
    return linhas


def painel_global(a1: List[Dict]) -> str:
    """Bloco de texto: o quadro médio do painel, por dataset e recorte."""
    out: List[str] = []
    for rotulo, filtro in (("factorial completo (14 colorsets)", lambda r: True),
                           ("só colorsets COM RGB (7)", lambda r: r["contem_rgb"] == "sim")):
        out.append(f"\n### {rotulo}\n")
        out.append("| dataset | métrica | Δ>0 | IC95 positivo | IC95 negativo | Δ médio |")
        out.append("|---|---|---|---|---|---|")
        for ds in sorted({r["dataset"] for r in a1}):
            sub = [r for r in a1 if r["dataset"] == ds and filtro(r)]
            for m in METRICAS:
                d = [r[f"_d_{m}"] for r in sub]
                pos = sum(1 for r in sub if r[f"sinal_{m}"] == "pos")
                neg = sum(1 for r in sub if r[f"sinal_{m}"] == "neg")
                out.append(f"| {ds} | {m} | {sum(1 for x in d if x > 0)}/{len(d)} | "
                           f"{pos} | {neg} | {_n(float(np.mean(d)), 4)} |")
    return "\n".join(out)


# --------------------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="local/runs")
    ap.add_argument("--tables-dir", default="local/generated/tables")
    ap.add_argument("--tag-prefix", default="f1_", help="prefixo dos marcadores (default: Fase 1)")
    args = ap.parse_args(argv)

    rdir, tdir = Path(args.results_dir), Path(args.tables_dir)
    celulas = carregar_celulas(rdir, args.tag_prefix)
    print(f"[accf1] {len(celulas)} células '{args.tag_prefix}*' com per_seed_metrics")

    a1 = tabela_a1(celulas)
    a2 = tabela_a2(a1)
    a3 = tabela_a3(a1)
    print(f"[accf1] {len(a1)} comparações pareadas")

    cols_a1 = ["dataset", "backbone", "colorset", "contem_rgb"]
    for m in METRICAS:
        cols_a1 += [f"{m}_rgb", f"{m}_cor", f"delta_{m}",
                    f"ic95_lo_{m}", f"ic95_hi_{m}", f"sinal_{m}"]
    cols_a2 = ["dataset", "backbone", "n_colorsets"]
    cols_a3 = ["dataset", "colorset", "n_backbones"]
    for m in METRICAS:
        suf = [f"delta_{m}_medio", f"ic95_lo_{m}", f"ic95_hi_{m}", f"positivos_{m}"]
        cols_a2 += suf
        cols_a3 += suf

    write_csv(rdir / "A1_factorial_accf1.csv", cols_a1, a1)
    write_csv(rdir / "A2_backbone_consistencia.csv", cols_a2, a2)
    write_csv(rdir / "A3_colorset_medio.csv", cols_a3, a3)

    write_md(tdir / "A2_backbone_consistencia.md",
             "A2 — Efeito da cromância por backbone (Acc/F1), colorsets com RGB",
             cols_a2, a2,
             "Δ médio sobre os 7 colorsets que contêm RGB, pareado por seed (n=5) contra o "
             "baseline RGB do MESMO backbone. `positivos_X` = quantos dos 7 colorsets vão na "
             "direção positiva — é a estatística de consistência, mais informativa que o IC95 "
             "(os 7 Δ não são independentes: compartilham baseline e seeds).")
    write_md(tdir / "A3_colorset_medio.md",
             "A3 — Efeito por colorset (Acc/F1), média sobre os 15 backbones",
             cols_a3, a3,
             "Cada linha agrega os 15 backbones para um colorset. Complementa a A2: mostra que "
             "nenhum colorset ganha em média — o que varia é o backbone, não o espaço de cor.")

    print("\n" + "=" * 78)
    print("PAINEL GLOBAL — quadro médio (o que a média sobre o painel esconde)")
    print("=" * 78)
    print(painel_global(a1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
