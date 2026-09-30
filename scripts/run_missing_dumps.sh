#!/usr/bin/env bash
set -euo pipefail

# Envoltório de madrugada para scripts/run_missing_dumps.py.
#
# Interromper é seguro a qualquer momento: cada célula só conta como feita quando o
# seu dump está escrito, e a próxima execução salta as que já estão. Não toca nos 64
# dumps originais da campanha — escreve em local/experiments/extra.
#
# Uso:
#   nohup bash scripts/run_missing_dumps.sh > results_madrugada.log 2>&1 &
#   tail -f results_madrugada.log
#
# Com uma lista alternativa de células (por exemplo o controlo de não-máximas):
#   nohup bash scripts/run_missing_dumps.sh results/control_cells.json &

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${JAUNDICE_PYTHON:-python}"
LOG="${ROOT}/results/missing_dumps_$(date +%Y%m%d_%H%M%S).log"

cd "${ROOT}"
mkdir -p results

if ! command -v "${PYTHON}" >/dev/null; then
  echo "Python do projecto não encontrado: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -d "${JAUNDICE_DATA_ROOT:-${ROOT}/dataset}" ]]; then
  echo "Dataset não encontrado; o treino precisa das imagens reais." >&2
  exit 2
fi
if ! "${PYTHON}" -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
  echo "CUDA indisponível. Este script treina modelos e precisa de GPU." >&2
  exit 2
fi

echo "início: $(date -Is)" | tee "${LOG}"
"${PYTHON}" -c 'import torch; print("GPU:", torch.cuda.get_device_name(0))' | tee -a "${LOG}"

CELLS="${1:-}"

if [[ -z "${CELLS}" ]]; then
  # Recalcula a cobertura antes de decidir o que falta: se uma execução anterior
  # tratou células, a lista encurta sozinha.
  "${PYTHON}" analysis/coverage_report.py 2>&1 | tee -a "${LOG}"
  "${PYTHON}" scripts/run_missing_dumps.py --gpu 0 2>&1 | tee -a "${LOG}"
  status=$?
else
  if [[ ! -f "${CELLS}" ]]; then
    echo "Lista de células não encontrada: ${CELLS}" >&2
    exit 2
  fi
  echo "lista de células: ${CELLS}" | tee -a "${LOG}"
  "${PYTHON}" scripts/run_missing_dumps.py --gpu 0 --cells "${CELLS}" 2>&1 | tee -a "${LOG}"
  status=$?
fi

echo "fim: $(date -Is) (código ${status})" | tee -a "${LOG}"
echo
echo "Próximo passo, depois de verificar os dumps novos:"
echo "  python analysis/dumps.py --install          # mostra o que vai copiar"
echo "  python analysis/dumps.py --install --apply  # copia para a pasta certa"
echo "  python analysis/coverage_report.py"
echo "  python analysis/pre_submission_revision.py --check"
echo
echo "Não use 'cp local/experiments/extra/*/*/preds/*.npz evidence/v7/preds/<dataset>/':"
echo "esse glob apanha os dois datasets, e correndo-o uma vez por pasta cada dump"
echo "acaba nas duas. O --install tira o destino do conteúdo do ficheiro."
exit ${status}
