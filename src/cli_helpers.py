"""
cli_helpers.py
==============

Pequenos utilitarios chamados pelo ``run.sh`` para expandir argumentos amigaveis
em listas concretas (mantendo a logica de configuracao em Python).

Uso:
    python src/cli_helpers.py targets <familia|modelo|all>
    python src/cli_helpers.py colorsets <all|individual|"RGB,LAB+YCrCb,...">
"""

from __future__ import annotations

import pathlib
import sys

# Permite rodar como script solto: adiciona 'colorspace_experiments/' ao path
# para que 'src' seja importavel como pacote (com seus imports relativos).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    INDIVIDUAL_COLORSPACES,
    colorset_id,
    parse_colorset,
    powerset_colorspaces,
    resolve_targets,
)


def cmd_targets(arg: str) -> None:
    print(" ".join(resolve_targets(arg)))


def cmd_colorsets(arg: str) -> None:
    arg = arg.strip()
    if arg.lower() == "all":
        sets = [colorset_id(c) for c in powerset_colorspaces()]
    elif arg.lower() == "individual":
        sets = [colorset_id(c) for c in INDIVIDUAL_COLORSPACES]
    else:
        # Lista explicita separada por virgula; cada item e um subconjunto (canais com '+').
        sets = [colorset_id(parse_colorset(item)) for item in arg.split(",") if item.strip()]
    for s in sets:
        print(s)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("uso: cli_helpers.py {targets|colorsets} <arg>", file=sys.stderr)
        raise SystemExit(2)
    cmd, arg = sys.argv[1], sys.argv[2]
    if cmd == "targets":
        cmd_targets(arg)
    elif cmd == "colorsets":
        cmd_colorsets(arg)
    else:
        print(f"comando desconhecido: {cmd}", file=sys.stderr)
        raise SystemExit(2)
