# src/asr/transcriber.py

from itertools import islice
import whisperx
import logging
import os
import torch
import warnings
from pyannote.audio.utils.reproducibility import ReproducibilityWarning
import math, torchaudio
warnings.filterwarnings("ignore", category=ReproducibilityWarning)

logger = logging.getLogger(__name__)

def batch(iterable, chunk_size):
    """
    Divide `iterable` en listas de longitud <= chunk_size.
    """
    it = iter(iterable)
    while True:
        block = list(islice(it, chunk_size))
        if not block:
            break
        yield block

class Transcriber:
    def __init__(
        self,
        model_name,
        device,
        compute_type,
        language,
        download_root,
        allow_tf32,
        vad_method,
        vad_onset,
        vad_offset,
        chunk_size,
        temperature,
        beam_size,
        initial_prompt,
    ):
        self.allow_tf32 = allow_tf32

        # --- NUEVO: bloquea el fix de reproducibilidad ----
        if self.allow_tf32:
            self._enable_tf32()          # activa flags antes de cualquier import
            
        # --- opciones ASR que SÍ entiende TranscriptionOptions --- 
        asr_opts = {}
        if beam_size is not None:
            asr_opts["beam_size"] = beam_size           # int
        if temperature is not None:
            asr_opts["temperatures"] = [temperature]    # ← lista
        if initial_prompt:
            asr_opts["initial_prompt"] = initial_prompt

        self.model = whisperx.load_model(
            model_name,
            device,
            compute_type=compute_type,
            language=language,
            local_files_only=True,
            download_root=download_root,
            vad_method   = vad_method,

            # ← VAD
            vad_options  = {
                "vad_onset"  : vad_onset,
                "vad_offset" : vad_offset,
                "chunk_size" : chunk_size,
            },
            # ← Opciones de decodificación
            asr_options = asr_opts,
        )
    def _enable_tf32(self):
        """
        Anula el fix de reproducibilidad de Pyannote y fuerza TF32
        antes de cualquier pipeline (VAD o align).
        """
        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32     = True

    def transcribe(self, audio_path: str, batch_size: int, on_batch_end=lambda *_: None, ) -> dict:
        """
        Ejecuta la transcripción y devuelve el dict con keys: language, segments, etc.
        """
        if self.allow_tf32:
            self._enable_tf32()

        result = self.model.transcribe(
            audio_path,
            batch_size=batch_size,
        )

        on_batch_end(1)
        if self.allow_tf32:
            self._enable_tf32()
            logging.warning("CUIDADO: TF32 forzado en VAD (WhisperX)")
        else:
            logging.info("TF32 deshabilitado para máxima reproducibilidad en VAD (WhisperX)")
        return result
    

    def align(self, result, audio_path, device, return_char_alignments=False, on_chunk_end=lambda *_: None) -> dict:
        """
        Reemplaza result["segments"] por segmentos alineados palabra a palabra.
        """
        if self.allow_tf32:
            self._enable_tf32()
        align_model, meta = whisperx.load_align_model(
            result["language"],
            device=device,
            model_dir="models/align/w2v_spanish",
        )
        segs = [seg for seg in result["segments"] if seg["text"].strip()]
        aligned = []
        for chunk in batch(segs, 25):
            out = whisperx.align(
                chunk,
                align_model,
                meta,
                audio_path,
                device=device,
                return_char_alignments=return_char_alignments
            )
            aligned.extend(out["segments"])
            on_chunk_end(len(chunk)) 
        result["segments"] = aligned
        return result


    def estimate_batches(self, audio_path: str, batch_size: int = 96) -> int:
        """
        Devuelve cuántos lotes (aprox.) procesará WhisperX.

        Cálculo simple: duración del audio / 30 s  ÷  batch_size.
        Ajusta '30' si usas otra ventana (valores típicos: 30 s o 15 s).
        """
        info = torchaudio.info(audio_path)
        duration = info.num_frames / info.sample_rate           # en segundos
        frames = math.ceil(duration / 30)
        return max(1, math.ceil(frames / batch_size))