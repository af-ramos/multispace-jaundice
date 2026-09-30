#!/usr/bin/env bash
# Captura as versões exactas do ambiente que correu a campanha.
#
# Correr NO CLUSTER, dentro do ambiente que treinou as 2.250 execuções — não na máquina
# onde a análise é feita. A saída é um registro de ambiente separado; não deve
# substituir requirements.txt, que agora contém apenas dependências de análise.
#
#   bash scripts/capture_env.sh > requirements-train.lock
set -euo pipefail

echo "# Ambiente de treino da campanha v7, capturado em $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "# host: $(hostname)"
python -c 'import sys; print("# python ==%d.%d.%d" % sys.version_info[:3])'

for m in torch torchvision timm cv2 optuna numpy scipy pandas PIL; do
  python - "$m" <<'PY'
import importlib, sys
name = sys.argv[1]
dist = {"cv2": "opencv-python", "PIL": "Pillow"}.get(name, name)
try:
    print(f"{dist}=={importlib.import_module(name).__version__}")
except Exception as exc:
    print(f"# {dist}: AUSENTE ({type(exc).__name__})")
PY
done

echo "# --- CUDA e GPU, para o registo ---"
python - <<'PY_CUDA' 2>/dev/null || echo "# torch indisponivel"
import torch
print(f"# torch.version.cuda = {torch.version.cuda}")
print(f"# cudnn = {torch.backends.cudnn.version()}")
print(f"# gpu = {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sem CUDA'}")
PY_CUDA

echo "# --- ambiente completo, para auditoria ---"
python -m pip freeze 2>/dev/null | sed "s/^/# /" || true
