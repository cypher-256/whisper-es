#./main.py
# --- configuración global Loggins única ------------------------------------
import logging
from rich.logging import RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[RichHandler(markup=False)],
    force=True            # fuerza la reconfiguración si algo ya registró un handler
)

# ❶ Filtro que descarta DEBUG e INFO de speechbrain
class _DropSpeechBrainBelowWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Si el logger pertenece a speechbrain y su nivel es inferior a WARNING, lo descartamos
        if record.name.startswith("speechbrain") and record.levelno < logging.WARNING:
            return False
        return True
logger = logging.getLogger()
for handler in logger.handlers:
    handler.addFilter(_DropSpeechBrainBelowWarning())
# -------------------------------------------------------------------


import types, sys, os
sys.path.insert(0, os.path.abspath("./src"))
import torch

# --- silencia warning molesto ------------------------------------
def _quiet_check(*a, **k):
    pass
mod = types.ModuleType("pyannote.audio.utils.version")
mod.__dict__["check_version"] = _quiet_check
sys.modules["pyannote.audio.utils.version"] = mod
# -------------------------------------------------------------------

import contextlib
import argparse
from pipelines.full_pipeline import run_pipeline
from utils.hooks import ForcedProgressHook
import utils.monkeypatch_pooling

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de transcripción y diarización",
        epilog=(
            "whisper-es v1.0.0\n"
            "Basado en whisperX © 2022–2025 Max Bain et al. (BSD-2-Clause).\n"
            "Modificaciones © 2025 cypher-256.\n"
            "Para más detalles, consulte el archivo LICENSE."
        )
    )

    # — Grupo I/O —
    io_group = parser.add_argument_group("Opciones de I/O")
    io_group.add_argument(
        "audio",
        help="Archivo WAV de entrada"
    )
    io_group.add_argument(
        "-o", "--output",
        default="transcripcion_diarizada.jsonl",
        help="Ruta de salida JSONL"
    )

    # — Grupo ASR —
    asr_group = parser.add_argument_group("Opciones ASR")

    asr_group.add_argument(
    "--model", "-m",
    default="large-v2",
    choices=[
        "tiny",
        "base",
        "small",
        "medium",
        "large",
        "large-v2",
        "turbo"
    ],
    help="Nombre del modelo WhisperX a usar"
    )
    asr_group.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Dispositivo de cómputo ('cuda' o 'cpu')"
    )
    asr_group.add_argument(
        "--asr-batch",
        type=int,
        default=None, #96
        help="Tamaño de lote para transcripción WhisperX (frames por batch). Si no se especifica, será 8 en GPU y 4 en CPU"
    )
    asr_group.add_argument(
        "--compute-type",
        choices=["float16", "float32", "int8"],
        default="float16",
        help="Tipo de cómputo que usará WhisperX (precisión y uso de memoria)"
    )
    asr_group.add_argument(
        "--model-dir",
        default="models/whisper",
        help="Directorio donde están o se descargarán los modelos Whisper"
    )
    asr_group.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Activa TensorFloat-32 en GPU (no recomendado)"
    )

    # — Grupo VAD —
    vad_group = parser.add_argument_group("Opciones VAD")
    vad_group.add_argument(
        "--vad-method",
        default="pyannote",
        choices=["pyannote", "silero"],
        help="Método VAD a utilizar"
    )
    vad_group.add_argument(
        "--vad-onset",
        type=float,
        default=0.50,
        help="Umbral de inicio para VAD (pyannote)"
    )
    vad_group.add_argument(
        "--vad-offset",
        type=float,
        default=0.363,
        help="Umbral de fin para VAD (pyannote)"
    )
    vad_group.add_argument(
        "--chunk-size",
        type=int,
        default=30,
        help="Chunk en segundos al fusionar segmentos VAD"
    )
    
    # — Grupo Alineación / Decodificador —
    align_group = parser.add_argument_group("Alineación / Decodificador")
    align_group.add_argument(
        "--align-model",
        type=str,
        default=None,
        help="Nombre del modelo de alignment a usar (p.ej. WAV2VEC2_ASR_LARGE_LV60K_960H)."
    )
    align_group.add_argument(
        "--align-batch",
        type=int,
        default=None,                 # ← None indica “calcular”
        help="Segmentos por lote durante la alineación; " "si se omite, se ajusta automáticamente según la RAM disponible"
    )
    align_group.add_argument(
        "--no-align",
        action="store_true",
        help="Omite la alineación fonética"
    )
    align_group.add_argument(
        "--return-char-alignments",
        action="store_true",
        help="Devuelve alineaciones a nivel carácter en el JSON"
    )
    align_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperatura de muestreo (0 = greedy/beam)"
    )
    align_group.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam width si temperature==0"
    )
    align_group.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Hilos CPU para Torch (0 = automático)"
    )
    align_group.add_argument(
    "--initial-prompt",
    type=str, default=None,
    help="Texto inicial (prompt) para la primera ventana"
    )

    # — Grupo Diarización —
    diar_group = parser.add_argument_group("Opciones Diarización")
    diar_group.add_argument(
        "--min-speakers",
        type=int,
        default=1,
        help="Número mínimo de oradores a detectar"
    )
    diar_group.add_argument(
        "--max-speakers",
        type=int,
        default=5,
        help="Número máximo de oradores a detectar"
    )

    # — Grupo Utilidades —
    util_group = parser.add_argument_group("Utilidades")
    util_group.add_argument(
        "--show-progress",
        action="store_true",
        help="Muestra la barra de progreso unificada"
    )
    util_group.add_argument(
        "--no-diarize",
        action="store_true",
        help="Omite la fase de diarización, solo transcribe (y alinea si no está activado --no_align)",
    )
    args = parser.parse_args()
    progress_hook = ForcedProgressHook(transient=True) if args.show_progress else None

    if progress_hook:
        for h in list(logger.handlers):
            logger.removeHandler(h)
        logger.addHandler(
            RichHandler(console=progress_hook.console, markup=False, show_path=False)
        )

    # ------------------------------------------------------------------
    # ➊ Ajuste dinámico de --asr-batch
    # ------------------------------------------------------------------
    if args.asr_batch is None:
        if args.device == "cpu":
            cores = os.cpu_count() or 1
            args.asr_batch = max(1, min(8, cores))  # RTX 3090 Ti no usa CPU
        else:
            props       = torch.cuda.get_device_properties(0)
            total_vram  = props.total_memory // 2**20          # MiB
            target_vram = int(total_vram * 0.80)               # 80 % seguro

            # Consumo base (MiB por frame @ FP16) – valores refinados
            base_mem = {
                "tiny":      22,
                "base":      35,
                "small":     55,
                "medium":    90,
                "large":    135,
                "large-v2": 210,
                "turbo":    150,
            }[args.model]

            factor = {"float16": 1.30, "float32": 2.00, "int8": 0.70}[args.compute_type]
            est_per_frame = base_mem * factor

            batch_calc = max(4, int(target_vram // est_per_frame))
            # Redondea al múltiplo de 4 más próximo y nunca sobrepasa 512
            args.asr_batch = min(512, (batch_calc + 3) // 4 * 4)

    logging.info(f"asr_batch dinámico = {args.asr_batch}")

    if args.device == "cpu" and args.compute_type in ("float16", "float32"):
        logging.warning("float16 no soportado en CPU y float32 muy lento → usando int8 en su lugar")
        args.compute_type = "int8"
    if args.device == "cpu":
        logging.warning("Diarización y alineación no soportadas en CPU, solo transcripción")
        logging.warning("Usando batch_size de 4 debido a uso de CPU")
        logging.warning("Usando chunk_size size de 15 debido a uso de CPU")
        logging.warning("Usando vad_method silero debido a uso de CPU")
        logging.warning("Usando modelo tiny debido a uso de CPU")
        args.model        = "tiny"
        #args.no_align = True
        args.threads      = os.cpu_count()
        args.chunk_size   = 15
        args.vad_method   = "silero"


    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.info(f"Procesando {args.audio}")
    logging.info(f"model={args.model}")
    logging.info(f"Procesando {args.audio}")

    # 3. Ejecuta el pipeline dentro del contexto del hook
    if args.threads:
        torch.set_num_threads(args.threads)

    ctx = progress_hook or contextlib.nullcontext()
    with ctx:
        out = run_pipeline(
            model_name    = args.model,
            audio_file    = args.audio,
            output_jsonl  = args.output,
            device        = args.device,
            asr_batch     = args.asr_batch,
            compute_type  = args.compute_type,
            min_speakers  = args.min_speakers,
            max_speakers  = args.max_speakers,
            model_dir     = args.model_dir,
            allow_tf32    = args.allow_tf32,
            progress_hook = progress_hook,
            vad_method  = args.vad_method,
            vad_onset   = args.vad_onset,
            vad_offset  = args.vad_offset,
            chunk_size  = args.chunk_size,
            no_align    = args.no_align,
            align_model_name    = args.align_model,
            align_batch = args.align_batch,
            return_char_alignments = args.return_char_alignments,
            no_diarize    = args.no_diarize,
            temperature = args.temperature,
            beam_size   = args.beam_size,
            initial_prompt = args.initial_prompt,
            
        )

    logging.info(f"→ Transcripción guardada en {out}")

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# Copyright (c) 2022-2025 Max Bain et al. (whisperX)
# License: BSD-2-Clause (véase LICENSE en la raíz)
#
# Modificaciones © 2025 cypher-256
# -----------------------------------------------------------------------------
