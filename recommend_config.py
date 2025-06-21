#!/usr/bin/env python3
"""Sugiere la mejor configuración a partir de logs/summary.csv."""
import pandas as pd, math, sys
from pathlib import Path



csv = Path("logs/summary.csv")
if not csv.exists():
    sys.exit("[ERROR] Primero ejecute compare_bench.py para generar summary.csv")

df = pd.read_csv(csv)

for col in ("GPU_mean%", "VRAM_peak", "RAM_peak", "CPU_mean%", "throughput"):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

def to_sec(t):
    parts = [float(p) for p in t.split(":")]
    while len(parts) < 3: parts.insert(0,0)
    h,m,s = parts
    return h*3600 + m*60 + s

df["wall_s"] = df["wall_time"].apply(to_sec)

for col in ("wall_s","VRAM_peak","RAM_peak"):
    m,M = df[col].min(), df[col].max() or 1
    df[col+"_norm"] = (df[col]-m)/(M-m+1e-9)

df["gpu_eff_norm"] = 1 - (abs(df["GPU_mean%"].fillna(0)-70)/70).clip(0,1)
# === CPU efficiency (ideal 70 %) ================================
if "CPU_mean%" in df.columns:
    df["cpu_eff_norm"] = 1 - (abs(df["CPU_mean%"].fillna(0)-70)/70).clip(0,1)

# === Throughput (cuanto mayor mejor) ============================
if "throughput" in df.columns and df["throughput"].notna().any():
    thr = df["throughput"].fillna(0)
    df["throughput_norm"] = (thr - thr.min())/(thr.max()-thr.min()+1e-9)

# --- Ponderaciones ---------------------------------------------
W = {"wall_s_norm":0.4,
     "gpu_eff_norm":0.25,
     "VRAM_peak_norm":0.15}

if "cpu_eff_norm" in df.columns:
    W["cpu_eff_norm"] = 0.1
if "throughput_norm" in df.columns:
    W["throughput_norm"] = 0.1
    
df["score"] = 0
for k, w in W.items():
    if k in df.columns:
        df["score"] += df[k].fillna(df[k].max()).mul(w)

best = df.sort_values("score").iloc[0]
print("\n========= MEJOR CONFIGURACIÓN =========")
print(best[["run","device","model","score","wall_time","GPU_mean%","VRAM_peak"]]
      .to_markdown())

audio_hint = "<audio.wav>"
cmd = f"python main.py {audio_hint} --device {best.device} --model {best.model}"
if best.device=="cuda" and "int8" in best.run:
    cmd += " --compute-type int8"
print("\nSugerencia de comando:")
print(cmd)
