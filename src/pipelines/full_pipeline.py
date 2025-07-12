#./src/pipelines/full_pipeline.py

from src.asr.transcriber import Transcriber
from src.diarization.diarizer import Diarizer
from src.formatting.formatter import Formatter
import logging, psutil, pandas as pd, torch
logger = logging.getLogger(__name__)

def run_pipeline(
    model_name: str,
    audio_file: str,
    output_jsonl: str,
    device: str,
    asr_batch: int,
    compute_type: str,
    min_speakers: int,
    max_speakers: int,
    model_dir: str,
    allow_tf32: bool,
    progress_hook,
    vad_method: str,
    vad_onset: float,
    vad_offset: float,
    chunk_size: int,
    no_align: bool,
    no_diarize: bool,
    return_char_alignments: bool,
    temperature: float,
    beam_size: int,
    initial_prompt: str,
    align_model_name: str,
    align_batch: int,
) -> str:
    """
    Ejecuta todo el flujo de trabajo de ASR + alineación + diarización + guardado.
    Todos los parámetros se reciben desde main.py.
    """

    # ----------------------------------------------------------
    # 1.  CÁLCULO AUTOMÁTICO DEL align_batch  (si el usuario no lo fijó)
    # ----------------------------------------------------------
    if align_batch is None:
        avail_mib    = psutil.virtual_memory().available // (1024 * 1024)
        safety_ratio = 0.45          # usar el 40 % de la RAM libre
        est_mib_seg  = 0.25          # MiB aprox. por segmento
        calc_batch   = int((avail_mib * safety_ratio) / est_mib_seg)

        # Sólo mínimo, sin tope:
        align_batch = max(10, calc_batch)

        # —o— mínimo y tope superior de 5000:
        #align_batch = max(10, min(calc_batch, 5000))

        logger.info(
            f"align_batch dinámico = {align_batch} "
            f"(RAM libre ≈ {avail_mib} MiB, calc_batch ≈ {calc_batch})"
    )
    # ----------------------------------------------------------
    # 2.  CREACIÓN DEL TRANSCRIBER
    # ----------------------------------------------------------
    t = Transcriber(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language="es",
        download_root=model_dir,
        allow_tf32=allow_tf32,
        vad_method = vad_method,
        vad_onset  = vad_onset,
        vad_offset = vad_offset,
        chunk_size = chunk_size,
        temperature= temperature,
        beam_size  = beam_size,
        initial_prompt = initial_prompt,
        align_model_name = align_model_name,
        align_batch=align_batch
    )

    # ----------------------------------------------------------
    # 3.  TRANSCRIPCIÓN
    # ----------------------------------------------------------

    batches = t.estimate_batches(audio_file, batch_size=asr_batch)
    if progress_hook:
        adv_transcribe = progress_hook.new_phase("ASR-Transcribe", batches, "lote")
    else:
        adv_transcribe = lambda *_: None

    while True:
        try:
            result = t.transcribe(
                audio_file,
                batch_size=asr_batch,
                on_batch_end=lambda *_: adv_transcribe(1)
            )
            break

        except RuntimeError as e:
            msg = str(e).lower()   
            if "out of memory" in msg and asr_batch > 1:
                old = asr_batch
                asr_batch = max(1, asr_batch // 2)
                print(f"[WARN] OOM con batch={old}, reintentando con batch={asr_batch}")
                torch.cuda.empty_cache()
                # Recalcular estimación y barra
                batches = t.estimate_batches(audio_file, batch_size=asr_batch)
                if progress_hook:
                    adv_transcribe = progress_hook.new_phase("ASR-Transcribe", batches, "lote")
                continue
            else:
                raise


    if progress_hook:
        adv_transcribe() 
        progress_hook.close_phase()

    # ----------------------------------------------------------
    # 4.  ALINEACIÓN
    # ----------------------------------------------------------

    if not no_align:
        steps_align = len(result["segments"])
        adv_align = progress_hook.new_phase("ASR-Align", steps_align, "seg") if progress_hook else (lambda *_: None)
        result = t.align(
            result,
            audio_file,
            device=device,
            return_char_alignments=return_char_alignments,
            on_chunk_end=lambda n: adv_align(n),
        )
        if progress_hook:
            progress_hook.close_phase()


    # --- fase Diarización (si está permitida y hay CUDA) ---
    if not no_diarize and device != "cpu":
        d = Diarizer(
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            device=device,
            #device_index=device_index,
            allow_tf32=allow_tf32,
            progress_hook=progress_hook,
        )
        diarize_df = d.diarize(audio_file)
    else:
        # Omitir diarización → DataFrame vacío
        diarize_df = pd.DataFrame(columns=["start", "end", "speaker"])
    
# --- fase fusión y guardado ----------------------------------------------
    final_update = (
        progress_hook.new_phase("Guardar", 1) if progress_hook else (lambda *_: None)
    )
    fmt = Formatter()
    merged = fmt.assign_speakers(diarize_df, result)
    path = fmt.save_jsonl(merged, output_jsonl)
    final_update(1)
    if progress_hook:
        progress_hook.close_phase()
    return path

# -----------------------------------------------------------------------------
# Copyright (c) 2022-2025 Max Bain et al. (whisperX)
# License: BSD-2-Clause (véase LICENSE en la raíz)
#
# Modificaciones © 2025 cypher-256
# -----------------------------------------------------------------------------
