# src/formatting/formatter.py

import json
from pathlib import Path
from whisperx.diarize import assign_word_speakers

class Formatter:
    def assign_speakers(self, diarize_df, asr_result: dict) -> dict:
        """
        Fusiona DataFrame de dia­rización con segmentos ASR y devuelve el dict resultante.
        """
        return assign_word_speakers(diarize_df, asr_result)

    def save_jsonl(self, merged: dict, output_jsonl: str) -> str:
        """
        Guarda merged['segments'] en formato JSONL y retorna la ruta de salida.
        """
        out_path = Path(output_jsonl)
        with out_path.open("w", encoding="utf-8") as f:
            for seg in merged["segments"]:
                text = seg["text"].strip()
                if not text:
                    continue
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")
        return str(out_path)
