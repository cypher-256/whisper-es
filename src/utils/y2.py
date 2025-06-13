import argparse
import yt_dlp

# descarga yt_url al mismo directorio desde donde corre el script
def download_audio(yt_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'postprocessor_args': [
            '-ar', '16000',  # frecuencia de muestreo 16 kHz
            '-ac', '1',      # 1 canal (mono)
        ],
        'ignoreerrors': True,
        'retries': 3,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([yt_url])

def main():
    parser = argparse.ArgumentParser(
        description="Descarga Audio de youtube"
    )
    parser.add_argument(
        "yt_url",
        type=str,
        help="Ruta al archivo de audio (p. ej. 'Audios/seminario1.mp3')",
    )
    
    
    args = parser.parse_args()
    download_audio(args.yt_url)

if __name__ == "__main__":
    main()