#!/usr/bin/env python3
"""Refaz o HPO de seis pares com semente de sampler independente, e avalia o efeito.

## A pergunta

`docs/PENDENCIAS.md` item 15: os 30 estudos da campanha correram com
`TPESampler(seed=cfg.seed)` e `cfg.seed` era 42 em todos. Verificado no `hpo.db` da
campanha: os **10 primeiros trials dos 30 estudos são idênticos** (uma única assinatura
em 30), e **20 dos 30 vencedores saíram desse bloco partilhado**. Metade do orçamento de
cada estudo foi a mesma lista de candidatos.

Isso não invalida nenhuma alegação --- os hiperparâmetros são procurados no braço RGB e
transferidos aos 14 braços cromáticos, portanto os dois lados de cada Δ correm com a
configuração idêntica e uma escolha subóptima desloca-os juntos. O que fica por saber é
se uma busca mais diversa teria encontrado configurações melhores, e quanto isso moveria
os números que o artigo reporta.

## O desenho

Duas fases, por par:

1. **HPO** com a configuração EXACTA da campanha (`run_hpo_1gpu.sh`: 20 trials, fold 0,
   pruner hyperband, `--colorset RGB`, `adapter_v2`, LoRA, split congelado) e uma única
   diferença --- `--sampler-seed` próprio de cada estudo. É o contrafactual de "30
   explorações independentes".
2. **Avaliação** dos dois braços do par (o cromático e o seu baseline RGB) com os
   hiperparâmetros novos, nas cinco sementes e no split congelado.

O que se lê no fim são três coisas, e a terceira é a que interessa ao artigo: se a
configuração vencedora mudou; se a acurácia de cada braço mudou; e **se o Δ entre braços
mudou**, que é a quantidade sobre a qual todas as alegações são feitas.

## Os seis pares

Os mesmos de `run_explain_cells.py`, para a comparação ser contra células já replicadas:
os dois sistemas em destaque e os quatro extremos da Tabela 5. Cinco deles tiveram o
vencedor no bloco partilhado (t0, t0, t0, t2, t7); o `vit_l_16`/NeoJaundice venceu no
t16, já em fase guiada, e entra como **controlo negativo** --- se o efeito for do bloco
partilhado, este é o par que menos deve mexer.

## Custo

Medido nos `datetime` do `hpo.db` da campanha: 78 min para estes seis estudos. A
avaliação são as mesmas 12 células que o retreino de atribuição correu em 128 min. Total
~3,5 h. Escreve em `evidence/v7_reseed/` e não toca em `evidence/v7/`.

Uso:
    python scripts/run_hpo_reseed.py --dry-run
    nohup bash scripts/run_hpo_reseed.sh &
    python scripts/run_hpo_reseed.py --report
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import DATA_ROOT
RESEED = ROOT / "evidence/v7_reseed"
BEST_NEW = RESEED / "best_params"
BEST_OLD = ROOT / "evidence/v7/best_params"
SEEDS = "42 123 456 7 99"
PYTHON = sys.executable

#: (dataset, backbone, colour set cromático, semente do sampler, trial vencedor na campanha).
#: As sementes são arbitrárias mas fixas, e DISTINTAS entre si --- é isso que a campanha
#: não teve. Nenhuma é 42.
CELLS = [
    ("NJN",         "mobilenetv3_large", "RGB+LAB+HSV",       7001, "t7"),
    ("NJN",         "densenet121",       "RGB+YCrCb+HSV",     7002, "t0"),
    ("NJN",         "dinov3_vits16",     "RGB+YCrCb+HSV",     7003, "t2"),
    ("NeoJaundice", "deit_base",         "RGB+LAB+YCrCb+HSV", 7004, "t0"),
    ("NeoJaundice", "deit_small",        "RGB+LAB+YCrCb+HSV", 7005, "t0"),
    ("NeoJaundice", "vit_l_16",          "RGB+LAB+YCrCb+HSV", 7006, "t16"),  # controlo
]

PARAM_FLAGS = {
    "lr": "--lr", "optimizer": "--optimizer", "n_unfreeze": "--n-unfreeze",
    "batch_size": "--batch-size", "da_strength": "--da-strength", "fusion": "--fusion",
}


def hpo_command(dataset: str, backbone: str, sampler_seed: int, gpu: int) -> list[str]:
    """Os flags de `run_hpo_1gpu.sh` da campanha, mais `--sampler-seed`."""
    cmd = [
        PYTHON, "-m", "src.hpo",
        "--backbone", backbone, "--dataset", dataset, "--colorset", "RGB",
        "--hpo-trials", "20", "--hpo-fold", "0", "--pruner", "hyperband",
        "--sampler-seed", str(sampler_seed),
        "--storage", f"sqlite:///{RESEED}/hpo_reseed.db",
        "--study-name", f"{backbone}_{dataset}_s{sampler_seed}",
        "--fusion", "adapter_v2", "--backbone-mode", "lora", "--frozen-split",
        "--num-workers", "8", "--gpu", str(gpu),
        "--data-root", str(DATA_ROOT), "--results-dir", str(RESEED),
        "--best-params-dir", str(BEST_NEW),
    ]
    cmd += ["--njn-mode", "full_image"] if dataset == "NJN" else ["--wb", "off"]
    return cmd


def train_command(dataset: str, backbone: str, colorset: str, gpu: int) -> list[str]:
    """Avaliação de um braço com os hiperparâmetros NOVOS."""
    path = BEST_NEW / f"{backbone}_{dataset}.json"
    params = json.loads(path.read_text(encoding="utf-8")).get("best_params", {})
    tag = f"rs_{backbone}_njn" if dataset == "NJN" else f"rs_{backbone}_neo_wboff"
    cmd = [
        PYTHON, "-m", "src.train",
        "--backbone", backbone, "--dataset", dataset, "--colors", colorset,
        "--backbone-mode", "lora", "--frozen-split",
        "--seeds", SEEDS, "--tag", tag, "--gpu", str(gpu),
        "--data-root", str(DATA_ROOT), "--results-dir", str(RESEED),
    ]
    cmd += ["--njn-mode", "full_image"] if dataset == "NJN" else ["--wb", "off"]
    for key, flag in PARAM_FLAGS.items():
        if key in params and params[key] is not None:
            cmd += [flag, str(params[key])]
    if params.get("augment"):
        cmd.append("--augment")
    if "--fusion" not in cmd:
        cmd += ["--fusion", "adapter_v2"]
    return cmd


def hpo_done(dataset: str, backbone: str) -> bool:
    return (BEST_NEW / f"{backbone}_{dataset}.json").is_file()


def train_done(dataset: str, backbone: str, colorset: str) -> bool:
    """Marcador da célula já escrito. Existe para a fase 2 ser retomável.

    A corrida de 2026-08-18 perdeu os dois braços do `vit_l_16` por OOM com a máquina
    contendida, e as outras dez células estavam boas. Sem esta guarda, retomar custaria
    2,5 h a refazer trabalho válido --- e refazê-lo com sementes de treino iguais mas
    contenção de GPU diferente introduziria uma segunda fonte de variação no meio da
    comparação.
    """
    return _marker_acc(RESEED, dataset, backbone, colorset) is not None


def run(cmd: list[str], label: str) -> int:
    print(f"\n>>> {label}\n    {shlex.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args(argv)

    if args.report:
        return report()

    BEST_NEW.mkdir(parents=True, exist_ok=True)

    # --- fase 1: HPO --------------------------------------------------------
    jobs1 = [(d, b, s, hpo_command(d, b, s, args.gpu)) for d, b, _, s, _ in CELLS]
    if args.dry_run:
        print("# fase 1 — HPO com semente de sampler independente")
        for d, b, s, c in jobs1:
            print(f"# {d}/{b}  sampler_seed={s}")
            print(shlex.join(c))
        print("\n# fase 2 — avaliação (comandos dependem dos best_params novos)")
        for d, b, cs, _, _ in CELLS:
            print(f"# {d}/{b}: {cs} e RGB")
        return 0

    t0 = time.time()
    print(f"início: {datetime.now().isoformat(timespec='seconds')}", flush=True)
    fails = 0
    for i, (d, b, s, cmd) in enumerate(jobs1, 1):
        if hpo_done(d, b):
            print(f"[{i}/{len(jobs1)}] [skip] best_params já existe: {d}/{b}", flush=True)
            continue
        rc = run(cmd, f"[fase 1 {i}/{len(jobs1)}] HPO {d}/{b} sampler_seed={s}")
        if rc != 0:
            print(f"[FALHA rc={rc}] HPO {d}/{b} — seguindo", flush=True)
            fails += 1

    # --- fase 2: avaliação --------------------------------------------------
    jobs2 = []
    for d, b, cs, _, _ in CELLS:
        if not hpo_done(d, b):
            print(f"[aviso] sem best_params novo p/ {d}/{b} — braços saltados", flush=True)
            continue
        for c in (cs, "RGB"):
            jobs2.append((d, b, c))
    for i, (d, b, c) in enumerate(jobs2, 1):
        if train_done(d, b, c):
            print(f"[fase 2 {i}/{len(jobs2)}] [skip] marcador já existe: {d}/{b}/{c}", flush=True)
            continue
        rc = run(train_command(d, b, c, args.gpu), f"[fase 2 {i}/{len(jobs2)}] {d}/{b}/{c}")
        if rc != 0:
            print(f"[FALHA rc={rc}] treino {d}/{b}/{c} — seguindo", flush=True)
            fails += 1

    print(f"\nfim: {(time.time()-t0)/60:.1f} min | {fails} falhas", flush=True)
    print(f"fim: {datetime.now().isoformat(timespec='seconds')}", flush=True)
    return 0


def oom_by_block() -> dict[str, int]:
    """Quantas vezes cada bloco do log caiu no fallback de OOM do engine.

    Porque isto entra no relatório: o fallback halva o micro-batch e dobra a acumulação,
    portanto o batch EFECTIVO fica em 32 e o gradiente é equivalente. O que não fica
    igual é a `BatchNorm2d` do bloco de fusão, que passa a normalizar sobre 16 amostras
    em vez de 32.

    Isso seria inócuo se atingisse os dois braços de um par por igual --- desloca-os
    juntos e o Δ sobrevive. Mas o braço de quatro espaços entrega 12 canais à fusão onde
    o RGB entrega 3, portanto é o que primeiro esgota memória: um par em que SÓ o braço
    cromático caiu para 16 tem um Δ contaminado. É por isso que se conta por bloco e não
    no total.
    """
    logs = sorted(RESEED.glob("reseed_*.log"))
    counts: dict[str, int] = {}
    for log in logs:
        atual = None
        for line in log.read_text(errors="replace").splitlines():
            m = re.match(r">>> \[fase \d+ \d+/\d+\] (?:HPO )?(.+?)(?: sampler_seed=\d+)?$", line)
            if m:
                atual = m.group(1).strip()
                counts.setdefault(atual, 0)
            elif atual and "OOM ->" in line:
                counts[atual] += 1
    return counts


def _marker_acc(results_dir: Path, dataset: str, backbone: str, colorset: str):
    """Acurácia média do marcador da célula, como o `run_explain_cells.report` faz."""
    d = results_dir / dataset / backbone
    if not d.is_dir():
        return None
    hits = [q for q in d.glob("*.json") if f"__{colorset}__" in q.name]
    if not hits:
        return None
    return json.loads(hits[0].read_text())["test_metrics_mean"]["accuracy"]


def report() -> int:
    """Config antiga contra nova e, o que interessa, o Δ entre braços --- contra ruído.

    O ponto que torna a leitura possível: `run_explain_cells.py` já retreinou estas mesmas
    doze células com os hiperparâmetros da campanha, **inalterados**. A diferença entre o
    Δ dessa réplica e o Δ do registo é ruído puro de re-treino, e é o piso contra o qual
    qualquer efeito da busca tem de ser lido. Sem essa coluna, um Δ que se mexe 2 p.p. não
    se distingue de um Δ que se mexeria de qualquer maneira.
    """
    def f(x):
        return float(str(x).replace(",", "."))

    old_acc = {}
    with (ROOT / "evidence/v7/aggregate.csv").open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            if r["tag"].startswith("f1_"):
                old_acc[(r["dataset"], r["backbone"], r["colorset_id"])] = f(r["acc_mean"])

    EXPLAIN = ROOT / "local/experiments/explain"

    def delta(get, dataset, backbone, colorset):
        a, b = get(dataset, backbone, colorset), get(dataset, backbone, "RGB")
        return (a - b) if (a is not None and b is not None) else float("nan")

    d_camp = lambda ds, bb, cs: old_acc.get((ds, bb, cs))
    d_repl = lambda ds, bb, cs: _marker_acc(EXPLAIN, ds, bb, cs)
    d_rese = lambda ds, bb, cs: _marker_acc(RESEED, ds, bb, cs)

    print(f"{'dataset':12s} {'backbone':18s} {'config':7s} {'venceu':>7s} {'ΔAUC val':>9s} "
          f"{'Δ camp.':>8s} {'Δ repl.':>8s} {'Δ rese.':>8s} {'ruído':>7s} {'busca':>7s}")
    rows = []
    aucs = []
    for ds, bb, cs, seed, won in CELLS:
        pn = BEST_NEW / f"{bb}_{ds}.json"
        if not pn.is_file():
            print(f"{ds:12s} {bb:18s} {'—':7s} {won:>7s}   (HPO por correr)")
            continue
        jn = json.loads(pn.read_text())
        jo = json.loads((BEST_OLD / f"{bb}_{ds}.json").read_text())
        new_p, old_p = jn["best_params"], jo["best_params"]
        changed = "MUDOU" if new_p != old_p else "igual"
        # o objectivo que a própria busca optimiza: AUC de validação do fold 0.
        # É a resposta directa a "uma busca mais diversa teria encontrado melhor?",
        # e não depende de nenhum re-treino nem do ruído das cinco sementes.
        d_auc = jn["best_value"] - jo["best_value"]
        aucs.append(d_auc)

        dc = delta(d_camp, ds, bb, cs)
        dp = delta(d_repl, ds, bb, cs)
        dr = delta(d_rese, ds, bb, cs)
        ruido = dp - dc          # re-treino com a MESMA config
        busca = dr - dc          # re-treino com config de busca independente
        rows.append((ruido, busca))
        print(f"{ds:12s} {bb:18s} {changed:7s} {won:>7s} {d_auc:+9.4f} "
              f"{dc:+8.2f} {dp:+8.2f} {dr:+8.2f} {ruido:+7.2f} {busca:+7.2f}")
        if changed == "MUDOU":
            for k in sorted(set(old_p) | set(new_p)):
                if old_p.get(k) != new_p.get(k):
                    print(f"    {k}: {old_p.get(k)} -> {new_p.get(k)}")

    ooms = oom_by_block()
    quentes = {k: v for k, v in ooms.items() if v}
    if quentes:
        print("\nfallback de OOM (micro-batch 16, batch efectivo 32) por bloco:")
        for k, v in sorted(quentes.items()):
            print(f"    {k}: {v}x")
        # o que invalidaria um Δ: só um dos dois braços do par ter caído
        for ds, bb, cs, _, _ in CELLS:
            a = ooms.get(f"{ds}/{bb}/{cs}", 0)
            b = ooms.get(f"{ds}/{bb}/RGB", 0)
            if (a > 0) != (b > 0):
                print(f"    !! ASSIMÉTRICO em {ds}/{bb}: cromático={a}x RGB={b}x "
                      f"-> este Δ tem confundidor de BatchNorm")
    elif ooms:
        print("\nnenhum bloco caiu no fallback de OOM.")

    # Sem esta tabela, um Δ que encolhe lê-se como "a busca não valeu nada". Pode ser o
    # contrário: uma busca melhor levanta primeiro o braço em que foi feita, e a busca é
    # feita no RGB. A direcção do encolhimento é a informação, não a magnitude.
    braços = {"camp": [], "repl": [], "rese": [], "rgb_c": [], "rgb_r": [],
              "cro_c": [], "cro_r": []}
    for ds, bb, cs, _, _ in CELLS:
        for c in (cs, "RGB"):
            a, b, d = d_camp(ds, bb, c), d_repl(ds, bb, c), d_rese(ds, bb, c)
            if None in (a, b, d):
                continue
            braços["camp"].append(a); braços["repl"].append(b); braços["rese"].append(d)
            k = "rgb" if c == "RGB" else "cro"
            braços[f"{k}_c"].append(a); braços[f"{k}_r"].append(d)
    if braços["camp"]:
        n = len(braços["camp"])
        med = lambda v: sum(v) / len(v)
        print(f"\nacurácia média sobre os {n} braços com marcador nos três lados:")
        print(f"    campanha {med(braços['camp']):.2f}   réplica {med(braços['repl']):.2f}   "
              f"reseed {med(braços['rese']):.2f}")
        if braços["rgb_c"] and braços["cro_c"]:
            gr = med(braços["rgb_r"]) - med(braços["rgb_c"])
            gc = med(braços["cro_r"]) - med(braços["cro_c"])
            print(f"    ganho do reseed sobre a campanha: RGB {gr:+.2f}   "
                  f"cromático {gc:+.2f} p.p.")
            if gr > gc:
                print("    -> a busca levantou MAIS o RGB, e é por isso que o Δ encolhe.")
                print("       É a direcção que a §3.6.2 já prevê: a busca corre no braço RGB,")
                print("       portanto o seu proveito acumula-se aí e o Δ reportado é conservador.")

    if aucs:
        melhor = sum(1 for a in aucs if a > 0)
        print(f"\nbusca independente com MELHOR AUC de validação: {melhor}/{len(aucs)} estudos "
              f"(mediana {statistics.median(aucs):+.4f})")

    # EMPARELHADO: as duas medianas têm de sair das MESMAS células, senão uma falha de
    # treino num par muda a comparação sem que nada de real tenha mudado. A corrida de
    # 2026-08-18 perdeu o `vit_l_16` só do lado da busca, e a versão não-emparelhada
    # deste bloco inflacionava o piso de ruído com uma célula que a busca não tinha.
    par = [(abs(a), abs(b)) for a, b in rows if a == a and b == b]
    if par:
        ru = [a for a, _ in par]
        bu = [b for _, b in par]
        maior = sum(1 for a, b in par if b > a)
        print(f"\nsobre as MESMAS {len(par)} células:")
        print(f"    |movimento do Δ| só por re-treinar (mesma config): "
              f"mediana {statistics.median(ru):.2f}, máximo {max(ru):.2f} p.p.")
        print(f"    |movimento do Δ| com busca independente:           "
              f"mediana {statistics.median(bu):.2f}, máximo {max(bu):.2f} p.p.")
        print(f"    a busca mexeu mais do que o ruído em {maior}/{len(par)} células")
        print("\nA leitura é a comparação das duas linhas. Se a segunda não exceder a "
              "primeira,\no bloco de arranque partilhado não custou nada que o próprio "
              "ruído de treino não custe.")
    descartadas = [1 for a, b in rows if (a != a) or (b != b)]
    if descartadas:
        print(f"({len(descartadas)} célula(s) fora do emparelhamento por falta de "
              f"marcador --- ver as falhas no log.)")
    print("\nΔ = acurácia do braço cromático menos a do seu baseline RGB (Register B).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
