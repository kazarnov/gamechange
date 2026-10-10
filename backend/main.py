import asyncio
import logging
from contextlib import asynccontextmanager

import torch

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from transformers.audio_utils import load_audio

from .asr import ASREngine, resample
from .browser import Browser
from .comfyui import ComfyClient, ComfyError
from .config import ROOT, settings
from .conversations import ConversationStore
from .llm import LLM
from .session import Models, VoiceSession
from .tts import SAMPLE_RATE as TTS_RATE
from .tts import TTSEngine
from .voicestudio import VoiceStudioTTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for name in ("httpx", "httpx2"):  # every model download request otherwise (httpx2: huggingface_hub)
    logging.getLogger(name).setLevel(logging.WARNING)
log = logging.getLogger("voice-agent")

MEDIA_DIR = ROOT / settings.media_dir
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


async def check_comfy(comfy: ComfyClient):
    try:
        await comfy.refresh()
        log.info("ComfyUI at %s: %s", comfy.base_url,
                 ", ".join(f"{w.name}{'' if w.ready else ' (installing)'}" for w in comfy.workflows.values())
                 or "no workflows")
    except ComfyError as exc:
        log.warning("ComfyUI at %s: %s (generation tools will report errors until it is up)", comfy.base_url, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.ollama_api_key:
        log.warning("OLLAMA_API_KEY is not set; LLM calls to Ollama cloud will fail")
    log.info("Loading models...")
    asr = ASREngine(settings.asr_model, settings.asr_device, getattr(torch, settings.asr_dtype),
                    settings.asr_language, settings.asr_lookahead)
    ref_text = settings.tts_ref_text
    if settings.tts_ref_audio and not ref_text and not settings.tts_url:
        # Transcribe the voice reference with our ASR instead of OmniVoice's Whisper (saves VRAM)
        ref_text = asr.transcribe(load_audio(settings.tts_ref_audio, sampling_rate=asr.sample_rate))
        log.info("Voice reference transcript: %r", ref_text)
    if settings.tts_url:
        tts = VoiceStudioTTS(settings.tts_url, settings.tts_api_key, settings.tts_voice, settings.tts_speed)
    else:
        tts = TTSEngine(settings.tts_model, settings.tts_device, getattr(torch, settings.tts_dtype),
                        settings.tts_num_step, settings.tts_speed,
                        settings.tts_voice_instruct, settings.tts_ref_audio, ref_text,
                        transcribe=lambda audio: asr.transcribe(resample(audio, TTS_RATE, asr.sample_rate)))
    app.state.models = Models(
        asr=asr,
        tts=tts,
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
    if settings.comfyui_url:
        comfy = ComfyClient(settings.comfyui_url, settings.comfyui_api_key, ROOT / settings.comfyui_descriptions)
        app.state.models.comfy = comfy
        app.state.comfy_check = asyncio.create_task(check_comfy(comfy))  # don't hold up startup
    else:
        log.info("COMFYUI_URL is not set: picture/video generation is off")
    app.state.models.conversations = ConversationStore(
        ROOT / settings.data_dir / "conversations", app.state.models.comfy, MEDIA_DIR, ROOT / settings.fonts_dir,
        settings.comfyui_timeout_minutes * 60)
    log.info("Ready")
    yield
    if app.state.models.browser:
        await app.state.models.browser.close()
    if app.state.models.comfy:
        await app.state.models.comfy.close()


app = FastAPI(title="Voice Agent", lifespan=lifespan)


@app.get("/health")
def health():
    comfy = app.state.models.comfy
    return {"ok": True, "llm": settings.llm_model, "asr_mode": settings.asr_mode,
            "tts": "voicestudio" if settings.tts_url else "local", "flowai": settings.flowai_url or "off",
            "comfyui": "off" if not comfy else (comfy.error or f"ok, {len(comfy.workflows)} workflows")}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("client connected")
    await VoiceSession(ws, app.state.models).run()
    log.info("client disconnected")


app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")
app.mount("/fonts", StaticFiles(directory=ROOT / settings.fonts_dir), name="fonts")  # the page uses them too
app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
