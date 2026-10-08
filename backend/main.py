import logging
from contextlib import asynccontextmanager

import torch

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from transformers.audio_utils import load_audio

from .asr import ASREngine, resample
from .browser import Browser
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
    if settings.browser_enabled:
        browser = Browser(settings.website_url, settings.browser_headless, ROOT / settings.browser_profile_dir)
        try:
            await browser.start()
            app.state.models.browser = browser
            app.state.models.browser_llm = LLM(
                settings.ollama_host, settings.ollama_api_key,
                settings.browser_llm_model or settings.llm_model, settings.browser_llm_think,
            )
        except Exception as exc:
            # Keep the voice agent usable without website tools
            log.error("Browser failed to start, website tools disabled: %s", str(exc).splitlines()[0])
            log.error("If Chromium is missing system libraries, run: sudo .venv/bin/playwright install-deps chromium")
            await browser.close()
    log.info("Ready")
    yield
    if app.state.models.browser:
        await app.state.models.browser.close()


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
