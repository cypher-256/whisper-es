#./src/diarization/pipeline_loader.py
from pathlib import Path
from pyannote.audio import Pipeline
import torch, logging
import os
import warnings
from pyannote.audio.utils.reproducibility import ReproducibilityWarning

warnings.filterwarnings("ignore", category=ReproducibilityWarning)

def load_local_pipeline(
    cfg_name: str = "pyannote_diarization_config.yaml",
    models_root: str = "models/pyannote",
    use_cuda: bool = True,
    allow_tf32: bool = False,
) -> Pipeline:
    """
    Carga un pipeline de diarización desde config local.

    Parámetros:
      - cfg_name: nombre del YAML de configuración.
      - models_root: ruta raíz donde están el YAML y los pesos.
      - use_cuda: si es False, fuerza CPU.
      - allow_tf32: si es True, habilita TF32 para optimizar rendimiento,
        aunque los resultados pueden variar ligeramente.

    Retorna:
      Instancia de pyannote.audio.Pipeline en CPU o GPU.
    """
    cfg = Path(models_root) / cfg_name
    pipeline = Pipeline.from_pretrained(cfg)

    # Decide si usar GPU o no
    if not use_cuda:
        logging.info("Forzando CPU por opción del usuario")
        return pipeline

    if not torch.cuda.is_available():
        logging.info("CUDA no disponible → se usará CPU")
        return pipeline

    try:
        idx = 0
        device = torch.device(f"cuda:{idx}")
        torch.cuda.set_device(device)
        pipeline.to(device)
        logging.info("Pipeline movido a %s (%s)", device, torch.cuda.get_device_name(idx))

        if allow_tf32:
            os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32     = True
            logging.warning("CUIDADO: TF32 forzado en Speaker Diarization")
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32     = False
            logging.info("TF32 deshabilitado para máxima reproducibilidad en Speaker Diarization")
    except Exception as err:
        logging.warning("Fallo al activar GPU, continúo en CPU (%s)", err)

    return pipeline
