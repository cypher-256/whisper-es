#!/usr/bin/env bash
# monitor_run.sh
set -euo pipefail

AUDIO=$1; shift
DEVICE=${1:-cuda}; shift || true
TAG=${1:-run}; shift || true
EXTRA_FLAGS=("$@")

TS=$(date +"%Y%m%d_%H%M%S")
LOGDIR="logs/${DEVICE}/${TS}_${TAG}"
mkdir -p "$LOGDIR"
echo "[INFO] Guardando métricas en ${LOGDIR}"

if [[ "$DEVICE" == "cuda" ]]; then
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,\
memory.used,temperature.gpu --format=csv,nounits --loop-ms=1000 \
    > "${LOGDIR}/gpu_usage.csv" &
  GPU_MON_PID=$!
fi

pidstat -u -h -p ALL 1 > "${LOGDIR}/cpu_usage.csv" &
CPU_MON_PID=$!

/usr/bin/time -v \
  python main.py "$AUDIO" --device "$DEVICE" "${EXTRA_FLAGS[@]}" \
  > "${LOGDIR}/output.log" 2> "${LOGDIR}/time.log"

kill "$CPU_MON_PID"
[[ -n "${GPU_MON_PID:-}" ]] && kill "$GPU_MON_PID"

echo "[OK] Ejecutado; logs en ${LOGDIR}"
