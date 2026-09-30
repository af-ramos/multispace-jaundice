"""late_fusion.py — Fusão TARDIA de espaços de cor + alavancas de re-scoring (offline).

A campanha v7 mediu apenas **fusão precoce**: empilhar os canais dos N espaços e
projetá-los de volta para 3 antes do backbone (`adapter_v2`, `ccat`, `stemcnn`). O
resultado é aproximadamente nulo (`A1_factorial_accf1.csv`, `T3_espaco_por_espaco.csv`).

Este módulo testa a alternativa que a campanha nunca testou: **treinar um modelo por
representação e combinar as PROBABILIDADES na decisão**. Isso não exige GPU nenhuma —
os dumps `results/**/preds/*.npz` já contêm val+test por imagem, por seed, para toda
célula rodada depois que `preds.py` entrou no pipeline.

Protocolo (mesmas regras de honestidade da campanha):
* **Limiar escolhido SÓ no val**, por seed, na mesma grade fina para todos os braços.
* Comparações **pareadas por seed** (mesmo split congelado ⇒ mesmo test set).
* O braço RGB passa exatamente pelo mesmo re-scoring que o ensemble — a única
  variável é quantos espaços entram na média.
* Nível **imagem** e nível **paciente** (agregação por `patient_id`, a unidade clínica).

Alavancas medidas em separado (`--alavancas`), todas leakage-free e de custo zero:
  1. limiar fino (grade de 0,01) **por seed**, em vez da grade de 0,05 pooled do treino;
  2. agregação por paciente;
  3. ensemble de seeds (média das probabilidades dos 5 seeds, em vez da média das
     5 métricas) — é o modelo que de fato se implanta.

Saídas:
  results/A4_fusao_tardia.csv     — RGB vs cada braço vs ensemble, por célula/nível/métrica
  results/A5_alavancas.csv        — incremento de cada alavanca sobre o baseline publicado
  (+ espelhos .md em paper/tables/)
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps

from .metrics import compute_metrics
from .preds import PredDump

CSV_SEP = ";"
ALPHA = 0.05

#: Métricas-alvo do paper. Cada uma tem o SEU limiar ótimo no val — otimizar acurácia
#: e reportar F1 (ou vice-versa) seria incoerente, então o limiar é escolhido por métrica.
METRICAS = ("accuracy", "f1_macro")

#: Grade fina de limiar. O treino usa `np.linspace(0.05, 0.95, 19)` (passo 0,05); aqui
#: refinamos para 0,01. Continua tudo no VAL — não é vazamento, é resolução.
GRADE_FINA = np.round(np.arange(0.05, 0.951, 0.01), 3)
GRADE_TREINO = np.linspace(0.05, 0.95, 19)


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
    print(f"[late] {len(rows)} linhas -> {path}")


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
    print(f"[late] {len(rows)} linhas -> {path}")


def pareado(a: Sequence[float], b: Sequence[float]) -> Dict[str, object]:
    """Δ = a − b pareado por seed, com IC95 t-Student (t crítico via scipy)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    media = float(d.mean())
    if n < 2:
        return {"delta": media, "lo": None, "hi": None, "sinal": ""}
    meia = float(sps.t.ppf(1 - ALPHA / 2, df=n - 1)) * float(d.std(ddof=1)) / np.sqrt(n)
    return {"delta": media, "lo": media - meia, "hi": media + meia,
            "sinal": "pos" if media - meia > 0 else ("neg" if media + meia < 0 else "nulo")}


# ----------------------------------------------------------------- pontuação
def _agrupa_por_paciente(pids: np.ndarray, y: np.ndarray,
                         p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Média das probabilidades por paciente. Preserva a ORDEM de 1ª aparição
    (determinismo: o mesmo dump sempre produz o mesmo vetor)."""
    prob: "OrderedDict[str, List[float]]" = OrderedDict()
    lab: Dict[str, int] = {}
    for k, yy, pp in zip(pids, y, p):
        prob.setdefault(k, []).append(float(pp))
        lab.setdefault(k, int(yy))
    chaves = list(prob)
    return (np.array([lab[k] for k in chaves]),
            np.array([float(np.mean(prob[k])) for k in chaves]))


def _score(y: np.ndarray, p: np.ndarray, thr: float, metrica: str) -> float:
    """Métrica no test — via `compute_metrics`, a MESMA função do pipeline de treino."""
    m = compute_metrics(list(y), y_prob=list(p), threshold=float(thr))
    return float(m[metrica])


def _score_rapido(y: np.ndarray, yhat: np.ndarray, metrica: str) -> float:
    """Só a métrica-alvo, direto da matriz de confusão — usado na VARREDURA de limiar.

    `compute_metrics` calcula ~12 métricas (inclusive AUC e MCC) a cada chamada; numa
    grade de 91 pontos × 5 seeds × 4 braços isso domina o tempo. Aqui só sai o número
    que o argmax precisa. Os valores REPORTADOS continuam vindo de `_score`
    (`compute_metrics`) — este atalho escolhe o limiar, não pontua o resultado.

    Verificado idêntico a `compute_metrics` por `tests/test_late_fusion.py`.
    """
    tp = float(np.sum((yhat == 1) & (y == 1)))
    tn = float(np.sum((yhat == 0) & (y == 0)))
    fp = float(np.sum((yhat == 1) & (y == 0)))
    fn = float(np.sum((yhat == 0) & (y == 1)))
    if metrica == "accuracy":
        return 100.0 * (tp + tn) / max(tp + tn + fp + fn, 1.0)
    if metrica == "f1_macro":
        # `sklearn.f1_score(average='macro')` faz a média sobre os rótulos PRESENTES em
        # y_true ∪ y_pred — não sobre as duas classes sempre. Numa partição degenerada
        # (só a classe 0 aparece e o modelo acerta tudo) o macro é 1,0, não 0,5. Nos
        # dados reais as duas classes sempre aparecem (prevalência 26–42%) e os dois
        # cálculos coincidem; a distinção existe para o atalho ser EQUIVALENTE, não
        # aproximadamente igual.
        f1s = []
        if tn + fp + fn:      # classe 0 aparece em y_true (tn+fp) ou em y_pred (tn+fn)
            f1s.append((2 * tn / (2 * tn + fn + fp)) if (2 * tn + fn + fp) else 0.0)
        if tp + fn + fp:      # classe 1 aparece em y_true (tp+fn) ou em y_pred (tp+fp)
            f1s.append((2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0)
        return float(np.mean(f1s)) if f1s else 0.0
    raise ValueError(f"métrica não suportada na varredura rápida: {metrica}")


def _melhor_limiar(y: np.ndarray, p: np.ndarray, metrica: str,
                   grade: np.ndarray = GRADE_FINA) -> float:
    """Argmax da métrica na grade — avaliado SEMPRE no val do próprio seed.

    Empate resolvido pelo MAIOR limiar (determinístico e idêntico em todos os braços,
    logo não enviesa a comparação).
    """
    pontos = [(_score_rapido(y, (p >= t).astype(int), metrica), float(t)) for t in grade]
    return max(pontos)[1]


def _checa_alinhamento(dumps: Sequence[PredDump]) -> None:
    """Pré-condição do ensemble: as linhas têm de casar imagem a imagem."""
    for outro in dumps[1:]:
        if not np.array_equal(dumps[0].paths, outro.paths):
            raise ValueError("dumps desalinhados: paths diferem (ensemble seria inválido)")


def avalia(dumps: Sequence[PredDump], metrica: str, por_paciente: bool,
           grade: np.ndarray = GRADE_FINA, pooled: bool = False) -> Dict[int, float]:
    """Média das probabilidades de ``dumps`` (mesmas imagens), limiar do val, score no test.

    Um único dump = braço individual. Dois ou mais = fusão tardia.
    ``pooled`` escolhe UM limiar no val de TODOS os seeds juntos (é o que o treino faz,
    `threshold_mode='pooled_oof'`) em vez de um limiar por seed. Mais dados de validação
    por trás da escolha ⇒ menos sobreajuste do val.
    Retorna {seed: métrica no test}.
    """
    _checa_alinhamento(dumps)
    base = dumps[0]
    prob_all = np.mean([d.y_prob for d in dumps], axis=0)

    thr_pooled: Optional[float] = None
    if pooled:
        mv = base.splits == "val"
        yv, pv = base.y_true[mv], prob_all[mv]
        if por_paciente:
            # Agrega por (seed, paciente): um mesmo paciente em seeds diferentes é uma
            # observação de validação distinta — senão a média entre seeds entraria aqui,
            # que é a alavanca `ens_seeds`, não o limiar.
            chave = np.array([f"{s}|{p}" for s, p in zip(base.seeds[mv], base.patient_ids[mv])])
            yv, pv = _agrupa_por_paciente(chave, yv, pv)
        thr_pooled = _melhor_limiar(yv, pv, metrica, grade)

    out: Dict[int, float] = {}
    for s in base.seed_list():
        ms = (base.seeds == s)
        prob = prob_all[ms]
        sp, y, pid = base.splits[ms], base.y_true[ms], base.patient_ids[ms]

        yv, pv = y[sp == "val"], prob[sp == "val"]
        yt, pt = y[sp == "test"], prob[sp == "test"]
        if por_paciente:
            yv, pv = _agrupa_por_paciente(pid[sp == "val"], yv, pv)
            yt, pt = _agrupa_por_paciente(pid[sp == "test"], yt, pt)
        thr = thr_pooled if thr_pooled is not None else _melhor_limiar(yv, pv, metrica, grade)
        out[int(s)] = _score(yt, pt, thr, metrica)
    return out


def avalia_ensemble_seeds(dumps: Sequence[PredDump], metrica: str, por_paciente: bool,
                          grade: np.ndarray = GRADE_FINA) -> float:
    """Média das probabilidades ENTRE seeds (e entre dumps), depois pontua — um número só.

    É o modelo que de fato se implanta: não se roda 5 seeds em produção e se tira a média
    das acurácias; roda-se o comitê e decide-se uma vez. Como o resultado é escalar, não
    há IC95 pareado — reporta-se contra a média dos 5 seeds, sem intervalo.
    """
    _checa_alinhamento(dumps)
    base = dumps[0]
    prob = np.mean([d.y_prob for d in dumps], axis=0)

    pont: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for split in ("val", "test"):
        m = base.splits == split
        # Chave = imagem (ou paciente); a média entre seeds acontece na agregação.
        chave = base.patient_ids[m] if por_paciente else base.paths[m]
        pont[split] = _agrupa_por_paciente(chave, base.y_true[m], prob[m])
    yv, pv = pont["val"]
    yt, pt = pont["test"]
    return _score(yt, pt, _melhor_limiar(yv, pv, metrica, grade), metrica)


# --------------------------------------------------------------------------- carga
def carregar_dumps(rdir: Path) -> Dict[Tuple[str, str, str], Dict[str, PredDump]]:
    """{(tag, dataset, backbone): {colorset: PredDump}} varrendo results/**/preds/."""
    grupos: Dict[Tuple[str, str, str], Dict[str, PredDump]] = {}
    for f in sorted(rdir.glob("*/*/preds/*.npz")):
        ds, bb = f.parts[-4], f.parts[-3]
        partes = f.stem.split("__")
        if len(partes) < 3:
            continue
        tag, colorset = partes[0], partes[2]
        try:
            grupos.setdefault((tag, ds, bb), {})[colorset] = PredDump.load(f)
        except Exception as e:                                   # pragma: no cover
            print(f"[late] aviso: falha ao ler {f}: {e}")
    return grupos


# --------------------------------------------------------------------------- A4
def tabela_a4(grupos: Dict) -> List[Dict]:
    """RGB vs cada braço de cor vs a fusão tardia dos dois, por célula/nível/métrica."""
    linhas: List[Dict] = []
    for (tag, ds, bb), porcs in sorted(grupos.items()):
        if "RGB" not in porcs or len(porcs) < 2:
            continue                                  # sem baseline, não há comparação
        braços = [c for c in sorted(porcs) if c != "RGB"]
        for metrica, nivel in itertools.product(METRICAS, ("imagem", "paciente")):
            pac = (nivel == "paciente")
            try:
                rgb = avalia([porcs["RGB"]], metrica, pac)
                for cs in braços:
                    cor = avalia([porcs[cs]], metrica, pac)
                    ens = avalia([porcs["RGB"], porcs[cs]], metrica, pac)
                    seeds = sorted(set(rgb) & set(cor) & set(ens))
                    vr = [rgb[s] for s in seeds]
                    vc = [cor[s] for s in seeds]
                    ve = [ens[s] for s in seeds]
                    d_cor, d_ens = pareado(vc, vr), pareado(ve, vr)
                    linhas.append({
                        "dataset": ds, "backbone": bb, "tag": tag, "colorset": cs,
                        "nivel": nivel, "metrica": metrica, "n_seeds": len(seeds),
                        "rgb": _n(float(np.mean(vr))),
                        "precoce": _n(float(np.mean(vc))),
                        "tardia": _n(float(np.mean(ve))),
                        "delta_precoce": _n(d_cor["delta"]),
                        "delta_tardia": _n(d_ens["delta"]),
                        "ic95_lo_tardia": _n(d_ens["lo"]),
                        "ic95_hi_tardia": _n(d_ens["hi"]),
                        "sinal_tardia": d_ens["sinal"],
                        "_d_tardia": d_ens["delta"], "_d_precoce": d_cor["delta"],
                    })
            except ValueError as e:                            # pragma: no cover
                print(f"[late] pulando {tag}/{ds}/{bb}: {e}")
    return linhas


def resumo_a4(a4: List[Dict]) -> str:
    """Bloco de texto com o veredito da fusão tardia (o resultado principal)."""
    out: List[str] = []
    out.append("| métrica | nível | n células | Δ(tardia−RGB) | IC95 | positivas | "
               "teste de sinais | Δ(precoce−RGB) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for metrica in METRICAS:
        for nivel in ("imagem", "paciente", "ambos"):
            sub = [r for r in a4 if r["metrica"] == metrica
                   and (nivel == "ambos" or r["nivel"] == nivel)]
            if not sub:
                continue
            v = np.array([r["_d_tardia"] for r in sub], float)
            vp = np.array([r["_d_precoce"] for r in sub], float)
            pos, neg = int((v > 0).sum()), int((v < 0).sum())
            media = float(v.mean())
            meia = (float(sps.t.ppf(1 - ALPHA / 2, df=len(v) - 1))
                    * float(v.std(ddof=1)) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
            p = sps.binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else float("nan")
            out.append(f"| {metrica} | {nivel} | {len(v)} | **{_n(media, 4)}** | "
                       f"[{_n(media - meia, 3)}, {_n(media + meia, 3)}] | "
                       f"**{pos}/{pos + neg}** | p={_n(p, 2)} | {_n(float(vp.mean()), 3)} |")
    return "\n".join(out)


# --------------------------------------------------------------------------- A5
def tabela_a5(grupos: Dict) -> List[Dict]:
    """Incremento de cada alavanca de re-scoring, isolada, sobre o baseline publicado.

    Baseline = grade de limiar do treino (passo 0,05), nível-imagem, média das métricas
    por seed — exatamente o que está nos marcadores. Cada linha adiciona UMA alavanca.
    """
    linhas: List[Dict] = []
    for (tag, ds, bb), porcs in sorted(grupos.items()):
        for cs, dump in sorted(porcs.items()):
            for metrica in METRICAS:
                base = avalia([dump], metrica, False, GRADE_TREINO)
                seeds = sorted(base)
                media_base = float(np.mean([base[s] for s in seeds]))
                variantes = {
                    "limiar_fino": avalia([dump], metrica, False, GRADE_FINA),
                    "limiar_pooled": avalia([dump], metrica, False, GRADE_TREINO, pooled=True),
                    "por_paciente": avalia([dump], metrica, True, GRADE_TREINO),
                    "paciente+limiar_fino": avalia([dump], metrica, True, GRADE_FINA),
                    "paciente+limiar_pooled": avalia([dump], metrica, True, GRADE_TREINO,
                                                     pooled=True),
                }
                r: Dict[str, object] = {
                    "dataset": ds, "backbone": bb, "tag": tag, "colorset": cs,
                    "metrica": metrica, "baseline": _n(media_base),
                }
                for nome, v in variantes.items():
                    d = pareado([v[s] for s in seeds], [base[s] for s in seeds])
                    r[f"delta_{nome}"] = _n(d["delta"])
                    r[f"_d_{nome}"] = d["delta"]
                # Ensemble de seeds é ESCALAR (um comitê, não 5 modelos): compara-se com a
                # média dos 5 seeds e não há IC95 pareado — n=1 por construção.
                for nome, pac in (("ens_seeds", False), ("ens_seeds_paciente", True)):
                    esc = avalia_ensemble_seeds([dump], metrica, pac, GRADE_TREINO)
                    ref = media_base if not pac else float(
                        np.mean(list(avalia([dump], metrica, True, GRADE_TREINO).values())))
                    r[f"delta_{nome}"] = _n(esc - ref)
                    r[f"_d_{nome}"] = esc - ref
                linhas.append(r)
    return linhas


ALAVANCAS = ("limiar_fino", "limiar_pooled", "por_paciente", "paciente+limiar_fino",
             "paciente+limiar_pooled", "ens_seeds", "ens_seeds_paciente")


def resumo_a5(a5: List[Dict]) -> str:
    out = ["| métrica | alavanca | n células | Δ médio | positivas | teste de sinais |",
           "|---|---|---|---|---|---|"]
    for metrica in METRICAS:
        sub = [r for r in a5 if r["metrica"] == metrica]
        for nome in ALAVANCAS:
            v = np.array([r[f"_d_{nome}"] for r in sub], float)
            pos, neg = int((v > 0).sum()), int((v < 0).sum())
            p = sps.binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else float("nan")
            out.append(f"| {metrica} | {nome} | {len(v)} | **{_n(float(v.mean()), 3)}** | "
                       f"{pos}/{len(v)} | p={_n(p, 2)} |")
    return "\n".join(out)


# --------------------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="local/runs")
    ap.add_argument("--tables-dir", default="local/generated/tables")
    ap.add_argument("--alavancas", action="store_true",
                    help="também mede o incremento isolado de cada alavanca (A5, mais lento)")
    args = ap.parse_args(argv)

    rdir, tdir = Path(args.results_dir), Path(args.tables_dir)
    grupos = carregar_dumps(rdir)
    n_dumps = sum(len(v) for v in grupos.values())
    print(f"[late] {n_dumps} dumps de predição em {len(grupos)} células")
    comparaveis = {k: v for k, v in grupos.items() if "RGB" in v and len(v) >= 2}
    print(f"[late] {len(comparaveis)} células com baseline RGB + ≥1 braço de cor")

    a4 = tabela_a4(grupos)
    cols_a4 = ["dataset", "backbone", "tag", "colorset", "nivel", "metrica", "n_seeds",
               "rgb", "precoce", "tardia", "delta_precoce", "delta_tardia",
               "ic95_lo_tardia", "ic95_hi_tardia", "sinal_tardia"]
    write_csv(rdir / "A4_fusao_tardia.csv", cols_a4, a4)
    write_md(tdir / "A4_fusao_tardia.md",
             "A4 — Fusão tardia (ensemble de espaços) vs fusão precoce vs RGB",
             cols_a4, a4,
             "`precoce` = o colorset empilhado na entrada e projetado N→3 (o que a campanha "
             "mediu). `tardia` = média das probabilidades do modelo RGB e do modelo do "
             "colorset. Limiar escolhido só no val, por seed, grade de 0,01; métrica no test; "
             "5 seeds pareados. Acurácia em %, F1-macro em [0,1].")

    print("\n" + "=" * 78)
    print("A4 — FUSÃO TARDIA: o resultado principal")
    print("=" * 78)
    print(resumo_a4(a4))

    if args.alavancas:
        a5 = tabela_a5(grupos)
        cols_a5 = (["dataset", "backbone", "tag", "colorset", "metrica", "baseline"]
                   + [f"delta_{a}" for a in ALAVANCAS])
        write_csv(rdir / "A5_alavancas.csv", cols_a5, a5)
        write_md(tdir / "A5_alavancas.md",
                 "A5 — Alavancas de re-scoring offline (custo GPU zero)",
                 cols_a5, a5,
                 "Baseline = grade de limiar do treino (passo 0,05), nível-imagem — o que está "
                 "publicado nos marcadores. Cada coluna adiciona UMA alavanca, todas "
                 "leakage-free (limiar sempre do val).")
        print("\n" + "=" * 78)
        print("A5 — ALAVANCAS DE RE-SCORING")
        print("=" * 78)
        print(resumo_a5(a5))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
