#!/usr/bin/env python3
"""Treina as células que faltam para completar os dumps que o artigo cita.

## Porque é preciso

A campanha v7 produziu 550 marcadores mas guardou **dumps de predição** só de 64
células — o RGB e o braço seleccionado de cada par (dataset, backbone). Sem dump,
uma célula existe apenas como métrica agregada: não se pode re-pontuar por paciente,
mudar limiar, formar ensemble, nem responder a um revisor que peça a curva ROC.

`analysis/coverage_report.py` prioriza as 28 células que o artigo efectivamente cita
e que não têm dump: os sistemas recomendados na T2, as células positivas da T4, e a
melhor configuração de cada backbone (que é o que a T3 conta). As 111 células de
só-croma negativas ficam de fora de propósito — ninguém pede para re-pontuar "LAB
sozinho é mau", e isso já está fechado pela métrica agregada.

## Configuração

Idêntica à da campanha, para as células novas serem comparáveis às antigas. Os
hiperparâmetros vêm de `evidence/v7/best_params/<backbone>_<dataset>.json`, que é o
resultado do HPO da própria campanha — nada é re-optimizado, porque re-optimizar
tornaria as células novas incomparáveis com as 64 existentes.

    NJN          --fusion adapter_v2 --njn-mode full_image --backbone-mode lora
    NeoJaundice  --fusion adapter_v2 --wb off --backbone-mode lora
    ambos        --frozen-split --seeds "42 123 456 7 99"

## Segurança

Idempotente: uma célula cujo dump já exista é saltada. Interromper a qualquer momento
é seguro — a retomada é por marcador, e nada é escrito parcialmente. Escreve num
`results-dir` separado (`local/experiments/extra`), portanto **não toca nos 64 dumps
originais** nem nos marcadores da campanha.

Uso:
    python scripts/run_missing_dumps.py --dry-run     # imprime os comandos, não treina
    nohup bash scripts/run_missing_dumps.sh &         # a sério, pela madrugada
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import DATA_ROOT
RESULTS_DIR = ROOT / "local/experiments/extra"
BEST_PARAMS = ROOT / "evidence/v7/best_params"
COVERAGE = ROOT / "results/coverage.json"
SEEDS = "42 123 456 7 99"
PYTHON = sys.executable

# Flags que vêm do HPO da campanha. `fusion` está aqui porque o best_params pode
# divergir do adapter_v2 por omissão, e o que vale é o que a campanha usou.
#
# `activation` está deliberadamente FORA: aparece no best_params como registo, mas é
# constante "none" no próprio train.py (linha 219) e no hpo.py, que a documenta como
# "fixo (head linear, logits puros para CrossEntropy)". Passá-la como flag rebenta o
# parser, que não a conhece.
PARAM_FLAGS = {
    "lr": "--lr", "optimizer": "--optimizer", "n_unfreeze": "--n-unfreeze",
    "batch_size": "--batch-size", "da_strength": "--da-strength",
    "fusion": "--fusion",
}


def known_flags() -> set[str]:
    """As opções que o train.py aceita de facto, lidas do seu próprio parser.

    Existe porque a primeira versão deste script emitia `--activation`, que o
    best_params regista mas o train.py não conhece — e as 28 células falharam todas
    de imediato. O dry-run não o apanhou por só imprimir comandos sem os validar.
    Ler o parser em vez de confiar numa lista escrita à mão fecha essa porta.
    """
    import contextlib
    import io
    import re

    sys.path.insert(0, str(ROOT))
    import src.train as train_module

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.suppress(SystemExit):
        train_module.main(["--help"])
    return set(re.findall(r"--[a-z0-9\-]+", buffer.getvalue()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Imprime os comandos e sai, sem treinar nada.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0,
                        help="Trata só as N primeiras células (para um teste curto).")
    parser.add_argument("--cells", type=Path, default=None,
                        help="JSON alternativo com a lista de células, no formato de "
                             "coverage.json ('prioritarias' ou lista à cabeça). Serve "
                             "para correr grupos que não são o alvo prioritário — por "
                             "exemplo o controlo de células não-máximas.")
    return parser.parse_args()


def best_params(backbone: str, dataset: str) -> dict:
    path = BEST_PARAMS / f"{backbone}_{dataset}.json"
    if not path.is_file():
        raise SystemExit(f"Sem HPO para {backbone}/{dataset}: {path}")
    return json.loads(path.read_text(encoding="utf-8")).get("best_params", {})


def dump_exists(dataset: str, backbone: str, colorset: str) -> bool:
    """Procura o dump nas duas árvores: a original e a desta execução.

    Na árvore original a pergunta é feita a `analysis.dumps`, que só conta a célula
    canónica (`f1_*` + `adapter_v2`, na condição do dataset). Antes bastava o nome do
    backbone e do colorset baterem, e um dump de `stemcnn` ou `ccat` — outra
    arquitectura de fusão — fazia esta função dizer que a célula já estava feita.
    """
    sys.path.insert(0, str(ROOT))
    from analysis.dumps import canonical_dumps

    if (dataset, backbone, colorset) in canonical_dumps():
        return True
    base = RESULTS_DIR / dataset / backbone / "preds"
    if base.is_dir():
        for path in base.glob("*.npz"):
            parts = path.stem.split("__")
            if len(parts) >= 3 and parts[1] == backbone and parts[2] == colorset:
                return True
    return False


def build_command(dataset: str, backbone: str, colorset: str, gpu: int) -> list[str]:
    params = best_params(backbone, dataset)
    tag = f"f1_{backbone}_njn" if dataset == "NJN" else f"f1_{backbone}_neo_wboff"
    cmd = [
        PYTHON, "-m", "src.train",
        "--backbone", backbone, "--dataset", dataset, "--colors", colorset,
        "--backbone-mode", "lora", "--frozen-split",
        "--seeds", SEEDS, "--tag", tag, "--gpu", str(gpu),
        "--data-root", str(DATA_ROOT), "--results-dir", str(RESULTS_DIR),
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


def main() -> int:
    args = parse_args()
    source = args.cells or COVERAGE
    if not source.is_file():
        raise SystemExit(f"Ficheiro de células não encontrado: {source}\n"
                         "Corra analysis/coverage_report.py primeiro.")
    payload = json.loads(source.read_text(encoding="utf-8"))
    cells = payload["prioritarias"] if isinstance(payload, dict) else payload
    if args.limit:
        cells = cells[: args.limit]

    # Preflight: toda a flag que vamos emitir tem de existir no parser do train.py.
    # Falhar aqui custa segundos; falhar a meio da noite custa a noite inteira.
    valid = known_flags()
    unknown = sorted(set(PARAM_FLAGS.values()) - valid)
    if unknown:
        raise SystemExit(f"Flags que o train.py não conhece: {unknown}")

    pending = [c for c in cells
               if not dump_exists(c["dataset"], c["backbone"], c["colorset"])]
    print(f"fonte: {source}")
    print(f"células na lista: {len(cells)}   já com dump: {len(cells)-len(pending)}   "
          f"a treinar: {len(pending)}")
    if not pending:
        print("nada a fazer.")
        return 0
    print(f"estimativa: ~{len(pending)*8/60:.1f} h a ~8 min por célula\n")

    if args.dry_run:
        for c in pending:
            cmd = build_command(c["dataset"], c["backbone"], c["colorset"], args.gpu)
            bad = sorted({p for p in cmd if p.startswith("--")} - valid)
            if bad:
                raise SystemExit(f"Comando inválido para {c}: flags desconhecidas {bad}")
            print(" ".join(shlex.quote(p) for p in cmd))
            print()
        print(f"DRY RUN: {len(pending)} comandos acima, todos com flags válidas. "
              "Nada foi treinado.")
        return 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[dict] = []
    for index, cell in enumerate(pending, start=1):
        stamp = datetime.now().strftime("%H:%M:%S")
        label = f"{cell['dataset']}/{cell['backbone']}/{cell['colorset']}"
        print(f"[{stamp}] {index}/{len(pending)}  {label}", flush=True)
        # Re-verifica agora: uma execução anterior pode ter tratado esta célula.
        if dump_exists(cell["dataset"], cell["backbone"], cell["colorset"]):
            print("           já tem dump; saltada", flush=True)
            continue
        cmd = build_command(cell["dataset"], cell["backbone"], cell["colorset"], args.gpu)
        start = time.time()
        result = subprocess.run(cmd, cwd=ROOT)
        minutes = (time.time() - start) / 60
        if result.returncode != 0:
            print(f"           FALHOU (código {result.returncode}) após {minutes:.1f} min",
                  flush=True)
            failures.append({**cell, "returncode": result.returncode})
        else:
            print(f"           ok em {minutes:.1f} min", flush=True)

    print(f"\nconcluído. falhas: {len(failures)}")
    for f in failures:
        print(f"  {f['dataset']}/{f['backbone']}/{f['colorset']} -> {f['returncode']}")
    (ROOT / "results/missing_dumps_run.json").write_text(json.dumps({
        "executado_em": datetime.now().isoformat(timespec="seconds"),
        "celulas_tentadas": len(pending), "falhas": failures,
        "results_dir": str(RESULTS_DIR),
        "nota": ("Os dumps novos ficam em local/experiments/extra e não tocam nos 64 originais. "
                 "Depois de verificar, copie-os para evidence/v7/preds/<dataset>/."),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
