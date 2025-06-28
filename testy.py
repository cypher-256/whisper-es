#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diarización con métricas de uso CPU/GPU para la pasada de inferencia.
Requiere:
    pip install pynvml psutil
"""

import os, sys, time, torch, torchaudio, psutil, threading, queue
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import Hooks, ProgressHook, TimingHook
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates

# --------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath("./src"))
import utils.monkeypatch_pooling       # noqa: F401  (side-effects)
# --------------------------------------------------------------------------


def poll_gpu(q, handle, stop_evt, interval=0.2):
    while not stop_evt.is_set():
        util = nvmlDeviceGetUtilizationRates(handle).gpu
        q.put(util)
        time.sleep(interval)


# ---------- audio y pipeline ----------
WAV   = "tests/data/test_audio.wav"
CFG   = "models/pyannote/pyannote_diarization_config.yaml"
waveform, sr = torchaudio.load(WAV)
device = torch.device("cuda")
pipe   = Pipeline.from_pretrained(CFG).to(device)
pipe._segmentation.batch_size = 32
pipe._embedding.batch_size    = 32
file = {
    "waveform": waveform,
    "sample_rate": sr,
}


# ---------- inicializa NVML ----------
nvmlInit()
gpu = nvmlDeviceGetHandleByIndex(0)
util_q, stop_evt = queue.Queue(), threading.Event()
threading.Thread(target=poll_gpu,
                 args=(util_q, gpu, stop_evt),
                 daemon=True).start()


# ---------- limpia caché y warm-up ----
torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
with ProgressHook() as warm:
    _ = pipe({"waveform": waveform, "sample_rate": sr}, hook=warm)
torch.cuda.synchronize()

# ---------- medición -------------------
cpu_proc = psutil.Process(os.getpid())
cpu_proc.cpu_percent(None)            # reinicia contador

util_before = nvmlDeviceGetUtilizationRates(gpu)
mem_before  = torch.cuda.memory_allocated()

t0 = time.time()
with Hooks(ProgressHook(), TimingHook(file_key="timing")) as both_hooks:
    _ = pipe(file, hook=both_hooks)
torch.cuda.synchronize()
elapsed = time.time() - t0

util_after = nvmlDeviceGetUtilizationRates(gpu)
mem_after  = torch.cuda.memory_allocated()
cpu_pct    = cpu_proc.cpu_percent(None)     # % medio del intervalo
vram_peak  = torch.cuda.max_memory_allocated() / 1024**2

# ---------- cálculo de delta GPU ------------
stop_evt.set()
samples = list(util_q.queue)
gpu_avg = sum(samples)/len(samples) if samples else 0
gpu_max = max(samples) if samples else 0

# ---------- reporte --------------------
print("\nRESUMEN")
print(f"Tiempo total           : {elapsed:.2f} s")
print(f"CPU proceso (avg)      : {cpu_pct:.1f} %")
print(f"GPU util (avg)         : {gpu_avg:.0f} % (máx {gpu_max:.0f} %)")
print(f"VRAM pico              : {vram_peak:.1f} MiB")

# ---------- desglose de etapas del pipeline ----------
print("\nDesglose por etapa:")
for step_name, duration in file["timing"].items():
    print(f"  • {step_name:18s} → {duration:.3f} s")