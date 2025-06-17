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

    def save_srt(self, merged: dict, output_srt: str) -> str:
        """Guarda los segmentos en formato SRT y retorna la ruta."""

        def fmt(ts: float) -> str:
            hours, rem = divmod(ts, 3600)
            minutes, seconds = divmod(rem, 60)
            millis = round((seconds - int(seconds)) * 1000)
            return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{millis:03d}"

        out_path = Path(output_srt)
        with out_path.open("w", encoding="utf-8") as f:
            for i, seg in enumerate(merged["segments"], start=1):
                text = seg["text"].strip()
                if not text:
                    continue
                speaker = seg.get("speaker")
                if speaker:
                    text = f"[{speaker}] {text}"
                f.write(f"{i}\n")
                start = fmt(seg["start"])
                end = fmt(seg["end"])
                f.write(f"{start} --> {end}\n{text}\n\n")
        return str(out_path)
