# Proyecto: Pipeline de Transcripción y Diarización

## Descripción

Este proyecto proporciona un **pipeline** de transcripción automática y diarización de hablantes sobre archivos de audio WAV, utilizando WhisperX para ASR (detección y transcripción de voz) y Pyannote/Silero para VAD y diarización. Incluye:

* Transcripción ASR en lotes y alineación palabra a palabra.
* Diarización de hablantes con modelo local o Silero.
* Interfaz CLI flexible con múltiples opciones.
* Barra de progreso unificada con Rich.

## Características

* **ASR configurable**: tamaño de lote, tipo de cómputo, TF32 opcional.
* **VAD ajustable**: método (pyannote/silero), umbrales de inicio/fin, chunk.
* **Alineación opcional**: activar/desactivar, retornos de alineación carácter.
* **Decodificación**: temperatura, beam size, prompt inicial.
* **Diarización**: número mínimo/máximo de oradores.
* **Barra de progreso**: fases unificadas (transcribe, align, diarize, guardar).

## Requisitos

* Python 3.8+
* CUDA (opcional, para GPU)
* Crear entorno virtual e instalar dependencias:

  ```bash
  git lfs install
  git clone https://github.com/cypher-256/whisper-es.git
  cd whisper-es
  python -m venv .venv
  source .venv/bin/activate  # o activate en Windows
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

## Uso

```bash
python main.py [OPCIONES] audio.wav
```

### Ejemplos

Transcribir con VAD Silero y sin alineación:

```bash
python main.py tests/data/test_audio.wav \
  --vad-method silero --vad-onset 0.4 --vad-offset 0.3 \
  --no-align -o salida.jsonl --show-progress
```

Transcribir con prompt inicial y beam size:

```bash
python main.py audio.wav \
  --initial-prompt "Hola mundo" --beam-size 2 \
  --temperature 0.8 -o resultado.jsonl
```

## Opciones de la CLI

Ver todas las banderas disponibles:

```bash
python main.py -h
```

> Incluye secciones para:
>
> * I/O
> * ASR
> * VAD
> * Alineación/Decodificador
> * Diarización
> * Utilidades

## Pruebas

Ejecuta la suite de tests con:

```bash
pytest tests/
```

## Contribución

1. Haz **fork** del repositorio.
2. Crea una **branch**: `git checkout -b feature/nombre`.
3. Realiza tus cambios y haz **commit**:

   ```bash
   git commit -m "feat: descripción breve"
   ```
4. Publica tu branch:

   ```bash
   git push origin feature/nombre
   ```
5. Abre un **PR** en GitHub.

## Licencia

Este proyecto está licenciado bajo la **MIT License**. Consulte el archivo `LICENSE` para más detalles.

---

## 💸 Donaciones

Si este dataset te ha sido útil y deseas apoyar su desarrollo, puedes contribuir con una donación en Bitcoin o Monero. Estas contribuciones ayudan a mantener y expandir recursos abiertos para el procesamiento del lenguaje en español chileno.

### Bitcoin (BTC – On-chain)

**Dirección:**  
`bc1p2t0gfxe3c0yw3rm3mdpgkrqwrpphgklhdt55lus3lmp4e86ljhnq4qkmp6`

> Puedes usar cualquier billetera compatible con direcciones Taproot (P2TR), como Muun, Sparrow o BlueWallet.

<img src="https://raw.githubusercontent.com/cypher-256/emotional-dataset-chile/main/assets/donacion_btc.png" alt="BTC QR" width="200"/>

---

### Monero (XMR – Anónimo)

**Dirección:**  
`44dS68RZrCY2cnoEWZYhtJJ3DXY52x75D2kCTLqffhpTWCFJUcty89W2VVUKCqE4J4WH8dnUJHCT1XQsXFkEKNyvQzqx8ar`

> Puedes usar billeteras como Feather Wallet, Monerujo o Cake Wallet.

<img src="https://raw.githubusercontent.com/cypher-256/emotional-dataset-chile/main/assets/donacion_xmr.png" alt="XMR QR" width="200"/>

---

Gracias por apoyar el software libre y la investigación abierta.

