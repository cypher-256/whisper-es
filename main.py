#./main.py
import types, sys, os

import torch
sys.path.insert(0, os.path.abspath("./src"))

# --- silencia warning molesto ------------------------------------
def _quiet_check(*a, **k):
    pass
mod = types.ModuleType("pyannote.audio.utils.version")
mod.check_version = _quiet_check
sys.modules["pyannote.audio.utils.version"] = mod
# -------------------------------------------------------------------

import logging
import contextlib
import argparse
from pipelines.full_pipeline import run_pipeline
from utils.hooks import ForcedProgressHook
from rich.logging import RichHandler
import utils.monkeypatch_pooling

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de transcripción y diarización"
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
    default="small",
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


    if args.asr_batch is None:
        if args.device == "cpu":
            # En CPU, limitamos el batch según núcleos disponibles
            # para no saturar la memoria RAM ni los hilos de Torch.
            # Por ejemplo, podemos fijar batch = núcleos // 2, con mínimo 1 y máximo 4.
            cores = os.cpu_count() or 1
            args.asr_batch = max(1, min(4, cores // 2))
        else:
            # En GPU, miramos la VRAM total y dimensionamos batch
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            # Escalamos batch con un factor de 2 lotes por GB de VRAM
            # y lo limitamos entre 4 y 16 para seguridad.
            estimated = int(vram_gb * 2)
            args.asr_batch = max(4, min(16, estimated))


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

    # 1. Inicializa hook de progreso
    progress_hook = ForcedProgressHook(transient=True) if args.show_progress else None

    # 2. Configura logging con la consola de Rich
    if progress_hook:
        logging.basicConfig(
            level=logging.INFO,
            handlers=[RichHandler(console=progress_hook.console, markup=False)],
            format="%(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S"
        )
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

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
            return_char_alignments = args.return_char_alignments,
            no_diarize    = args.no_diarize,
            temperature = args.temperature,
            beam_size   = args.beam_size,
            initial_prompt = args.initial_prompt,
            align_model_name    = args.align_model,
        )

    print(f"→ Transcripción guardada en {out}")

if __name__ == "__main__":
    main()
