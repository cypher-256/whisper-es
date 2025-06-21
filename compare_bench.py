#!/usr/bin/env python3
"""Agrega resultados y usa el nombre de la carpeta como respaldo de modelo."""
from pathlib import Path
import pandas as pd, re, subprocess, shlex, sys

def audio_duration(path):
    try:
        cmd = f"ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 {shlex.quote(str(path))}"
        return float(subprocess.check_output(cmd, shell=True, text=True).strip())
    except Exception:
        return 0.0

def find_col(df, key):
    key = key.lower()
    for col in df.columns:
        if key in col.lower().replace(" ", ""):
            return col
    raise KeyError

rows = []
for csv in Path("logs").rglob("gpu_usage.csv"):
    run_dir   = csv.parent
    time_log  = run_dir / "time.log"
    output_log = run_dir / "output.log"

    # --- GPU ---
    gpu_df  = pd.read_csv(csv)
    util_col = find_col(gpu_df, "utilization.gpu")
    mem_col  = find_col(gpu_df, "memory.used")
    mean_gpu = gpu_df[util_col].astype(float).mean()
    vram_pk  = gpu_df[mem_col].astype(float).max()


    # === CPU (pidstat) ==============================================
    cpu_csv = run_dir / "cpu_usage.csv"
    cpu_mean = None
    if cpu_csv.exists():
        cpu_df = pd.read_csv(cpu_csv, sep=r"\s+", header=None, engine="python", on_bad_lines="skip")
        # Columna %CPU es la 8ª (índice 7)
        if cpu_df.shape[1] > 7:
            series = cpu_df.iloc[:, 7].astype(str).str.replace(",", ".", regex=False)
            cpu_mean = pd.to_numeric(series, errors="coerce").mean()


    # --- TIME / RAM ---
    wall = rss = "NA"
    with open(time_log) as f:
        for ln in f:
            if "Elapsed (wall clock)" in ln:
                wall = ln.split(": ",1)[1].strip()
            if "Maximum resident" in ln:
                rss  = int(ln.split(":")[1]) // 1024  # MiB

    # --- Modelo y audio ---
    model = audio = duration = "NA"
    with open(output_log) as f:
        for ln in f:
            if m := re.search(r"model=(\w[-\w]*)", ln):
                model = m.group(1)
            if m := re.search(r"Procesando (.+?\.wav)", ln):
                audio = m.group(1)
    # Respaldo: derivar modelo del nombre de carpeta si sigue patrón *_<model>_<ctype>
    if model == "NA":
        parts = run_dir.name.split("_")
        if len(parts) >= 3:
            model = parts[-2]
    if audio != "NA":
        duration = round(audio_duration(Path(audio)), 1)

    thru = "NA"
    if duration != "NA" and wall != "NA":
        # wall es 'M:SS.xx' o 'H:MM:SS'
        parts = [float(p) for p in wall.split(":")]
        while len(parts) < 3: parts.insert(0,0)
        h,m,s = parts; wall_s = h*3600 + m*60 + s
        if wall_s:
            thru = round(duration / wall_s, 2)  # s audio / s pared

    rows.append({
        "run": run_dir.name,
        "device": run_dir.parent.name,
        "model": model,
        "audio_s": duration,
        "wall_time": wall,
        "RAM_peak": rss,
        "GPU_mean%": round(mean_gpu,1),
        "VRAM_peak": vram_pk,
        "CPU_mean%": round(cpu_mean,1) if cpu_mean is not None else "NA",
        "throughput": thru
    })

if not rows:
    sys.exit("[WARN] No se encontraron logs válidos.")

df = pd.DataFrame(rows)
print(df.to_markdown(index=False))

out = Path("logs/summary.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"\n[OK] Resumen guardado en {out}")
