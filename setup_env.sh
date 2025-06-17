#!/usr/bin/env bash
set -euo pipefail

# 1. Crear o recrear el virtualenv
PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-.venv}

echo "→ (Re)creando entorno virtual en $VENV_DIR"
rm -rf "$VENV_DIR"
$PYTHON -m venv "$VENV_DIR"

# 2. Activar e instalar requisitos
echo "→ Instalando dependencias"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 2.b Instalar PyTorch según GPU o CPU
echo "→ Comprobando soporte GPU"
if command -v nvidia-smi &> /dev/null; then
  echo "   • GPU detectada: instalando PyTorch con CUDA"
  pip install --upgrade \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
else
  echo "   • Sin GPU: instalando PyTorch CPU-only"
  pip install --upgrade \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu
fi

# 2.c Instalar el resto de dependencias
echo "→ Instalando dependencias adicionales"
pip install --upgrade pip
pip install -r requirements.txt

# 3. Parchar el activate para LD_LIBRARY_PATH
ACTIVATE="$VENV_DIR/bin/activate"
CU_LIB_PATH='$VIRTUAL_ENV/lib/python'"$( "$PYTHON" - <<'PY'
import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"'/site-packages/nvidia/cudnn/lib:$VIRTUAL_ENV/lib/python'"$( "$PYTHON" - <<'PY'
import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"'/site-packages/nvidia/cublas/lib'

grep -q "## Rutas cuDNN/cuBLAS" "$ACTIVATE" || cat >> "$ACTIVATE" <<EOF

# --- Rutas cuDNN/cuBLAS dentro del venv ---------------------------
export LD_LIBRARY_PATH="$CU_LIB_PATH:\${LD_LIBRARY_PATH:-}"
EOF

echo "→ Descargando modelos Pyannote"
mkdir -p models/pyannote
BASE_URL="https://github.com/cypher-256/whisper-es/releases/download/v1.0"

for MODEL in \
  pyannote_model_segmentation-3.0.bin \
  pyannote_model_wespeaker-voxceleb-resnet34-LM.bin; do
  echo "   – ${MODEL}"
  wget --progress=dot:giga \
       "${BASE_URL}/${MODEL}" \
       -O "models/pyannote/${MODEL}"
done

echo "→ Entorno preparado. Actívalo con: source $VENV_DIR/bin/activate"

