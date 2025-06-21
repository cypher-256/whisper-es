#!/usr/bin/env bash
# grid_benchmark.sh AUDIO  [gpu|cpu|both]
#
# Ejemplos:
#   ./grid_benchmark.sh tests/data/test_audio.wav gpu   # sólo CUDA
#   ./grid_benchmark.sh tests/data/test_audio.wav cpu   # sólo CPU
#   ./grid_benchmark.sh tests/data/test_audio.wav       # ambos (por defecto)

set -euo pipefail

AUDIO=$1
TARGET=${2:-both}           # gpu | cpu | both

MODELS=(tiny base small medium large large-v2 turbo)
CTYPES=(float16 int8)

case "$TARGET" in
  gpu)  DEVICES=(cuda) ;;
  cpu)  DEVICES=(cpu)  ;;
  both) DEVICES=(cuda cpu) ;;
  *)    echo "[ERROR] Segundo argumento debe ser gpu, cpu o both"; exit 1 ;;
esac

for DEVICE in "${DEVICES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for CT in "${CTYPES[@]}"; do

      # --- Reglas de exclusión rápidas -----------------------------
      [[ "$DEVICE" == "cpu"  && "$CT" == "float16" ]] && continue
      [[ "$DEVICE" == "cpu"  && "$MODEL" =~ ^(medium|large|large-v2|turbo)$ ]] && continue
      [[ "$DEVICE" == "cuda" && "$CT" == "float16" \
         && "$MODEL" =~ ^(large|large-v2|turbo)$ ]] && continue
      # --------------------------------------------------------------

      TAG="${MODEL}_${CT}"
      echo "==> device=$DEVICE  model=$MODEL  compute_type=$CT"
      ./monitor_run.sh "$AUDIO" "$DEVICE" "$TAG" \
        --model "$MODEL" --compute-type "$CT"
      echo
    done
  done
done

echo "[INFO] Resumen global:"
python compare_bench.py
python recommend_config.py
