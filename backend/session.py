"""One WebSocket conversation: mic audio -> VAD -> ASR -> LLM (+tools) -> TTS -> speaker.

Wire protocol
  client -> server
    binary: PCM16 little-endian mono 16 kHz mic audio
    json:   {"type": "text", "text": "..."}   typed user message
            {"type": "interrupt"}              stop the assistant
            {"type": "reset"}                  clear conversation history
  server -> client
    binary: 4-byte little-endian turn id + PCM16 mono 24 kHz assistant speech
    json:   ready | vad | transcript | assistant_delta | assistant_done |
            tool_call | tool_result | interrupt | error
"""

import asyncio
import json
import logging
import struct
import time
from dataclasses import dataclass

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from . import tools
from .asr import ASREngine, StreamingTranscription
from .config import settings
from .llm import LLM
from .tts import SAMPLE_RATE as TTS_RATE
from .tts import SentenceChunker, TTSEngine
from .vad import TurnDetector

log = logging.getLogger(__name__)


@dataclass
class Models:
    asr: ASREngine
    tts: TTSEngine
    llm: LLM


class VoiceSession:
    def __init__(self, ws: WebSocket, models: Models):
        self.ws = ws
        self.m = models
        self.loop = asyncio.get_running_loop()
        self.vad = TurnDetector(settings.vad_threshold, settings.vad_min_silence_ms,
                                settings.vad_min_speech_ms, settings.vad_preroll_ms)
        self.messages: list = [{"role": "system", "content": settings.system_prompt}]
        self.send_lock = asyncio.Lock()
        self.turn = 0
        self.response: asyncio.Task | None = None
        self.speaking_until = 0.0  # estimated end of client-side playback
        self.stream: StreamingTranscription | None = None
        self.utterance: list[np.ndarray] = []
        self.background: set[asyncio.Task] = set()

    # --- sending ------------------------------------------------------------------

    async def send(self, **msg):
        async with self.send_lock:
            await self.ws.send_text(json.dumps(msg))

    async def send_audio(self, turn: int, audio: np.ndarray):
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        async with self.send_lock:
            await self.ws.send_bytes(struct.pack("<I", turn) + pcm)

    def send_threadsafe(self, **msg):
        asyncio.run_coroutine_threadsafe(self.send(**msg), self.loop)

    # --- main loop ----------------------------------------------------------------

    async def run(self):
        await self.send(type="ready", tts_sample_rate=TTS_RATE, asr_mode=settings.asr_mode)
        try:
            while True:
                msg = await self.ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    await self.on_audio(msg["bytes"])
                elif msg.get("text"):
                    await self.on_message(json.loads(msg["text"]))
        except WebSocketDisconnect:
            pass
        finally:
            await self.interrupt(notify=False)
            if self.stream:
                await asyncio.to_thread(self.stream.finish)

    async def on_message(self, msg: dict):
        match msg.get("type"):
            case "text" if msg.get("text", "").strip():
                text = msg["text"].strip()
                await self.send(type="transcript", text=text, final=True)
                await self.start_response(text)
            case "interrupt":
                await self.interrupt()
            case "reset":
                await self.interrupt()
                self.messages = self.messages[:1]

    # --- user speech --------------------------------------------------------------

    async def on_audio(self, data: bytes):
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        for ev in self.vad.process(samples):
            if ev.kind == "start":
                await self.on_speech_start()
            elif ev.kind == "audio":
                if self.stream:
                    self.stream.feed(ev.audio)
                else:
                    self.utterance.append(ev.audio)
            elif ev.kind == "end":
                await self.on_speech_end()

    async def on_speech_start(self):
        await self.send(type="vad", speaking=True)
        if self.assistant_active():
            await self.interrupt()  # barge-in
        self.utterance = []
        if settings.asr_mode == "streaming":
            self.stream = self.m.asr.stream(
                lambda text: self.send_threadsafe(type="transcript", text=text, final=False)
            )

    async def on_speech_end(self):
        await self.send(type="vad", speaking=False)
        stream, self.stream = self.stream, None
        audio, self.utterance = self.utterance, []
        # Finalize off the receive loop so mic audio keeps flowing
        task = asyncio.create_task(self.finalize(stream, audio))
        self.background.add(task)
        task.add_done_callback(self.background.discard)

    async def finalize(self, stream: StreamingTranscription | None, audio: list[np.ndarray]):
        t0 = time.perf_counter()
        try:
            if stream:
                text = await asyncio.to_thread(stream.finish)
            else:
                text = await asyncio.to_thread(self.m.asr.transcribe, np.concatenate(audio)) if audio else ""
        except Exception as exc:
            log.exception("ASR failed")
            await self.send(type="error", message=f"ASR failed: {exc}")
            return
        log.info("ASR %.0f ms: %r", (time.perf_counter() - t0) * 1000, text)
        await self.send(type="transcript", text=text, final=True)
        if text:
            await self.start_response(text)

    # --- assistant ----------------------------------------------------------------

    def assistant_active(self) -> bool:
        busy = self.response is not None and not self.response.done()
        return busy or time.monotonic() < self.speaking_until

    async def interrupt(self, notify: bool = True):
        if self.response and not self.response.done():
            self.response.cancel()
            try:
                await self.response
            except (asyncio.CancelledError, Exception):
                pass
        was_speaking = time.monotonic() < self.speaking_until
        self.response = None
        self.speaking_until = 0.0
        self.turn += 1  # client drops any audio from older turns
        if notify:
            await self.send(type="interrupt", turn=self.turn, was_speaking=was_speaking)

    async def start_response(self, text: str):
        await self.interrupt(notify=self.assistant_active())
        self.response = asyncio.create_task(self.respond(text, self.turn))

    async def respond(self, user_text: str, turn: int):
        self.messages.append({"role": "user", "content": user_text})
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        speaker = asyncio.create_task(self.speak(sentences, turn))
        t0 = time.perf_counter()
        first_token = True
        try:
            for _ in range(settings.llm_max_tool_rounds):
                content, calls, chunker = "", [], SentenceChunker()
                try:
                    async for msg in self.m.llm.stream(self.messages, tools.schemas()):
                        if msg.content:
                            if first_token:
                                log.info("LLM first token %.0f ms", (time.perf_counter() - t0) * 1000)
                                first_token = False
                            content += msg.content
                            await self.send(type="assistant_delta", turn=turn, text=msg.content)
                            for s in chunker.push(msg.content):
                                sentences.put_nowait(s)
                        if msg.tool_calls:
                            calls.extend(msg.tool_calls)
                finally:
                    # Keep what was said in history, even if the user cut us off
                    if content or calls:
                        self.messages.append({"role": "assistant", "content": content, "tool_calls": calls or None})
                for s in chunker.flush():
                    sentences.put_nowait(s)
                if not calls:
                    break
                await self.run_tools(calls, turn)

            sentences.put_nowait(None)
            await speaker
            await self.send(type="assistant_done", turn=turn)
        except asyncio.CancelledError:
            speaker.cancel()
            raise
        except Exception as exc:
            speaker.cancel()
            log.exception("response failed")
            await self.send(type="error", message=f"Assistant failed: {exc}")

    async def run_tools(self, calls: list, turn: int):
        pending = list(calls)
        try:
            while pending:
                call = pending[0]
                name, args = call.function.name, dict(call.function.arguments or {})
                await self.send(type="tool_call", turn=turn, name=name, arguments=args)
                result = await tools.run(name, args)
                log.info("tool %s(%s) -> %s", name, args, result[:200])
                await self.send(type="tool_result", turn=turn, name=name, result=result)
                self.messages.append({"role": "tool", "tool_name": name, "content": result})
                pending.pop(0)
        finally:
            # Every tool call needs a result in history, or the next request is invalid
            for call in pending:
                self.messages.append({"role": "tool", "tool_name": call.function.name,
                                      "content": json.dumps({"error": "interrupted by user"})})

    async def speak(self, sentences: asyncio.Queue, turn: int):
        while (text := await sentences.get()) is not None:
            t0 = time.perf_counter()
            audio = await asyncio.to_thread(self.m.tts.synthesize, text)
            dur = len(audio) / TTS_RATE
            log.info("TTS %.0f ms for %.1f s audio: %r", (time.perf_counter() - t0) * 1000, dur, text)
            if turn != self.turn:
                return
            await self.send_audio(turn, audio)
            self.speaking_until = max(time.monotonic(), self.speaking_until) + dur
