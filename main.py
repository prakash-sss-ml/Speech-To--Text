"""Local near-real-time microphone transcription with FastAPI and faster-whisper."""
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

logger = logging.getLogger('speech_demo')
logging.basicConfig(level=logging.INFO)

BASE = Path(__file__).resolve().parent
SAMPLE_RATE = 16000
CHUNK_SECONDS = float(os.getenv('CHUNK_SECONDS', '3.5'))
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)
MODEL_SIZE = os.getenv('MODEL_SIZE', 'base')
DEVICE = os.getenv('DEVICE', 'cpu')
COMPUTE_TYPE = os.getenv('COMPUTE_TYPE', 'int8')
MAX_RECORDING_SECONDS = 30 * 60

model = None
inference_lock = asyncio.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info('Loading faster-whisper model %s on %s (%s)', MODEL_SIZE, DEVICE, COMPUTE_TYPE)
    model = await asyncio.to_thread(WhisperModel, MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    logger.info('Model ready')
    yield

app = FastAPI(title='Live Speech-to-Text POC', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=BASE / 'static'), name='static')

@app.get('/')
def index():
    return FileResponse(BASE / 'static' / 'index.html')

@app.get('/health')
def health():
    return {'status': 'ready' if model is not None else 'loading', 'model': MODEL_SIZE, 'device': DEVICE}


def recognize(samples: np.ndarray, language: str | None):
    # An energy threshold prevents Whisper inventing words on quiet input.
    if samples.size < SAMPLE_RATE // 3 or np.sqrt(np.mean(samples * samples)) < 0.004:
        return ''
    segments, _ = model.transcribe(
        samples.astype(np.float32, copy=False),
        language=language,
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
    )
    return ' '.join(s.text.strip() for s in segments).strip()


@app.websocket('/ws/transcribe')
async def transcribe_ws(ws: WebSocket):
    await ws.accept()
    buffer = np.empty(0, dtype=np.float32)
    received_samples = 0
    language = None
    try:
        while True:
            message = await ws.receive()
            if message.get('type') == 'websocket.disconnect':
                break
            if message.get('text') is not None:
                try:
                    command = json.loads(message['text'])
                except json.JSONDecodeError:
                    await ws.send_json({'type': 'error', 'message': 'Invalid JSON command'})
                    continue
                if command.get('type') == 'config':
                    requested = command.get('language', 'auto')
                    if requested not in ('auto', 'en', 'ta', 'ko', 'hi'):
                        await ws.send_json({'type': 'error', 'message': 'Unsupported language'})
                        continue
                    language = None if requested == 'auto' else requested
                    await ws.send_json({'type': 'ready'})
                elif command.get('type') == 'stop':
                    if buffer.size:
                        async with inference_lock:
                            transcript = await asyncio.to_thread(recognize, buffer, language)
                        if transcript:
                            await ws.send_json({'type': 'segment', 'text': transcript})
                    await ws.send_json({'type': 'done'})
                    await ws.close()
                    return
                continue
            raw = message.get('bytes')
            if raw is None:
                continue
            if len(raw) % 4:
                await ws.send_json({'type': 'error', 'message': 'Invalid float32 audio buffer'})
                continue
            samples = np.frombuffer(raw, dtype='<f4')
            received_samples += samples.size
            if received_samples > SAMPLE_RATE * MAX_RECORDING_SECONDS:
                await ws.send_json({'type': 'error', 'message': 'Maximum recording length reached'})
                await ws.close()
                return
            buffer = np.concatenate((buffer, samples))
            while buffer.size >= CHUNK_SAMPLES:
                chunk = buffer[:CHUNK_SAMPLES].copy()
                buffer = buffer[CHUNK_SAMPLES:]
                async with inference_lock:
                    transcript = await asyncio.to_thread(recognize, chunk, language)
                if transcript:
                    await ws.send_json({'type': 'segment', 'text': transcript})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception('WebSocket transcription error')
        try:
            await ws.send_json({'type': 'error', 'message': f'Transcription failed: {type(exc).__name__}'})
            await ws.close()
        except Exception:
            pass
