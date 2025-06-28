#./src/diarization/diarizer.py
import pandas as pd
import torchaudio
from pyannote.core import Annotation
from src.diarization.pipeline_loader import load_local_pipeline
from typing import Optional
from pyannote.audio.pipelines.utils.hook import ProgressHook

class Diarizer:
    def __init__(
        self,
        min_speakers,
        max_speakers,
        device,
        models_root: str = "models/pyannote",
        allow_tf32: bool = False,
        progress_hook: Optional[ProgressHook] = None,
    ):
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.use_cuda = device
        self.models_root = models_root
        self.allow_tf32 = allow_tf32
        self.progress_hook = progress_hook

    def _run_diarization(self, audio_path: str) -> Annotation:
        """
        Devuelve objeto Annotation con los segmentos por hablante.
        Usa YAML y pesos locales desde models/pyannote.
        """
        pipeline = load_local_pipeline(
            models_root=self.models_root,
            use_cuda=self.use_cuda,
            allow_tf32=self.allow_tf32
        )
        waveform, sample_rate = torchaudio.load(audio_path)
        return pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
            hook=self.progress_hook
        )

    def diarize(self, audio_path: str) -> pd.DataFrame:
        annotation = self._run_diarization(audio_path)
        records = [
            {
                "start": seg.start,
                "end": seg.end,
                "speaker": speaker,
            }
            for seg, _, speaker in annotation.itertracks(yield_label=True) # type: ignore
        ]
        return pd.DataFrame.from_records(records)
