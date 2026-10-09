"""Speech from a VoiceStudio server instead of this PC's GPU.

VoiceStudio runs the same OmniVoice model behind an OpenAI-style API (POST /v1/audio/speech),
e.g. on a RunPod GPU. Asked for "pcm" it returns raw 24 kHz 16-bit mono, the format the web UI
plays, so a sentence goes straight through. On an L40 a five-second sentence takes about 0.4 s,
against about 4 s with OmniVoice on a GTX 1660 Ti. It also means the agent can run on a machine
without a GPU for speech (FLOWAI_INTEGRATION.md, "Where the agent runs").
"""

import logging
import time

import httpx
import numpy as np

log = logging.getLogger(__name__)


class VoiceStudioTTS:
    """Same interface as tts.TTSEngine: synthesize(text) -> 24 kHz float32 audio (called in a thread)."""

    def __init__(self, url: str, api_key: str, voice: str, speed: float = 1.0, model: str = "omnivoice"):
        self.http = httpx.Client(base_url=url.rstrip("/"), timeout=60,
                                 headers={"Authorization": f"Bearer {api_key}"} if api_key else None)
        self.voice, self.speed, self.model = voice, speed, model
        t0 = time.perf_counter()
        self._speech("Warming up.")  # the server loads the model on its first request (about a minute)
        log.info("TTS: VoiceStudio at %s, ready in %.1f s", url, time.perf_counter() - t0)

    def _speech(self, text: str) -> np.ndarray:
        r = self.http.post("/v1/audio/speech", json={"model": self.model, "input": text, "voice": self.voice,
                                                     "speed": self.speed, "response_format": "pcm"})
        r.raise_for_status()
        return np.frombuffer(r.content, dtype="<i2").astype(np.float32) / 32768.0

    def synthesize(self, text: str) -> np.ndarray:
        for attempt in (1, 2):
            try:
                return self._speech(text)
            except httpx.HTTPError as exc:
                log.warning("VoiceStudio speech failed (attempt %d): %s", attempt, exc)
        return np.zeros(0, dtype=np.float32)  # the words still show on screen
