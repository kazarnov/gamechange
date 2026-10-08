import logging
from contextlib import asynccontextmanager

import torch

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from transformers.audio_utils import load_audio

from .asr import ASREngine, resample
from .config import ROOT, settings
from .llm import LLM
from .session import Models, VoiceSession
from .tts import SAMPLE_RATE as TTS_RATE
from .tts import TTSEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("voice-agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.ollama_api_key:
        log.warning("OLLAMA_API_KEY is not set; LLM calls to Ollama cloud will fail")
    log.info("Loading models...")
    asr = ASREngine(settings.asr_model, settings.asr_device, getattr(torch, settings.asr_dtype),
                    settings.asr_language, settings.asr_lookahead)
    ref_text = settings.tts_ref_text
    if settings.tts_ref_audio and not ref_text:
        # Transcribe the voice reference with our ASR instead of OmniVoice's Whisper (saves VRAM)
        ref_text = asr.transcribe(load_audio(settings.tts_ref_audio, sampling_rate=asr.sample_rate))
        log.info("Voice reference transcript: %r", ref_text)
    app.state.models = Models(
        asr=asr,
        tts=TTSEngine(settings.tts_model, settings.tts_device, getattr(torch, settings.tts_dtype),
                      settings.tts_num_step, settings.tts_speed,
                      settings.tts_voice_instruct, settings.tts_ref_audio, ref_text,
                      transcribe=lambda audio: asr.transcribe(resample(audio, TTS_RATE, asr.sample_rate))),
        llm=LLM(settings.ollama_host, settings.ollama_api_key, settings.llm_model, settings.llm_think),
    )
    log.info("Ready")
    yield


app = FastAPI(title="Voice Agent", lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True, "llm": settings.llm_model, "asr_mode": settings.asr_mode}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("client connected")
    await VoiceSession(ws, app.state.models).run()
    log.info("client disconnected")


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
