"""One WebSocket conversation: mic audio -> VAD -> ASR -> LLM (+tools) -> TTS -> speaker.

Wire protocol
  client -> server
    binary: PCM16 little-endian mono 16 kHz mic audio
    json:   {"type": "text", "text": "..."}   typed user message
            {"type": "interrupt"}              stop the assistant
            {"type": "cancel_task"}            stop the running browser task
            {"type": "upload", "name": "...", "data": "data:image/png;base64,..."}
                                               attach an image/video/audio file
            {"type": "upload", "name": "x.json", "workflow": {...}}
                                               attach a ComfyUI workflow file
            {"type": "reset"}                  clear conversation history
  server -> client
    binary: 4-byte little-endian turn id + PCM16 mono 24 kHz assistant speech
    json:   ready | vad | transcript | assistant_delta | assistant_done |
            tool_call | tool_result | interrupt | error |
            browser_task (started/done/failed/cancelled) | browser_step |
            media_job (generate/install: started/running/done/failed/cancelled) |
            media (a generated or attached file, served at /media/...)
"""

import asyncio
import json
import logging
import struct
import time
from dataclasses import dataclass, field

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from . import tools
from .asr import ASREngine, StreamingTranscription
from .browser import Browser
from .browser_agent import BrowserAgent
from .comfyui import ComfyClient, ComfyError
from .config import ROOT, settings
from .llm import LLM
from .media import MediaSession
from .skills import catalogue, load_skills
from .tts import SAMPLE_RATE as TTS_RATE
from .tts import SentenceChunker, TTSEngine
from .vad import TurnDetector

log = logging.getLogger(__name__)


@dataclass
class Models:
    asr: ASREngine
    tts: TTSEngine
    llm: LLM
    browser: Browser | None = None
    browser_llm: LLM | None = None
    comfy: ComfyClient | None = None


@dataclass
class BrowserJob:
    id: int
    instruction: str
    task: asyncio.Task | None = None
    steps: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)
    result: str | None = None


class VoiceSession:
    def __init__(self, ws: WebSocket, models: Models):
        self.ws = ws
        self.m = models
        self.loop = asyncio.get_running_loop()
        self.vad = TurnDetector(settings.vad_threshold, settings.vad_min_silence_ms,
                                settings.vad_min_speech_ms, settings.vad_preroll_ms)
        self.media = self.new_media()
        self.messages: list = [{"role": "system", "content": self.system_prompt()}]
        self.send_lock = asyncio.Lock()
        self.turn = 0
        self.response: asyncio.Task | None = None
        self.speaking_until = 0.0  # estimated end of client-side playback
        self.stream: StreamingTranscription | None = None
        self.utterance: list[np.ndarray] = []
        self.background: set[asyncio.Task] = set()
        self.user_speaking = False
        self.job: BrowserJob | None = None
        self.job_count = 0
        self.notes: list[str] = []  # events (task results, uploads) the LLM hasn't seen yet
        self.announce = False  # a note needs a spoken reply even if the user says nothing

    def new_media(self) -> MediaSession | None:
        if not self.m.comfy:
            return None
        return MediaSession(self.m.comfy, ROOT / settings.media_dir, settings.comfyui_timeout_minutes * 60,
                            self.try_send, self.add_note, lambda text: self.add_note(text, announce=False))

    def system_prompt(self) -> str:
        prompt = settings.system_prompt
        if self.m.browser:
            prompt += (
                f"\n\nYou can operate the website {settings.website_url} in a web browser by calling "
                "browser_task with a complete instruction. The task runs in the background: after starting "
                "it, say one short sentence such as 'On it', and never say it is done until you are given "
                "its result. If the user asks how it is going, use browser_task_status; if they want to "
                "stop it, use cancel_browser_task. The browser agent has these skills:\n"
                + catalogue(load_skills(ROOT / settings.skills_dir))
            )
        if self.m.comfy:
            prompt += (
                "\n\nYou can make pictures and videos with ComfyUI workflows by calling generate_media. It runs "
                "in the background: say one short sentence such as 'Sure, making it now', and never say it is "
                "ready until you are told. When it is, it is already on the user's screen, so just say so "
                "briefly. Pictures take seconds; videos take several minutes, so tell the user. Write the prompt "
                "in English as a rich visual description (subject, setting, style, lighting, camera, and motion "
                "for videos), expanding on what the user asked for. Every picture, video and attached file has a "
                "number the user can refer to; to change or animate a picture, use a workflow that needs an image "
                "and pass its number (the latest one is used by default). Use list_media_workflows when asked "
                "which workflows exist, media_status for progress, and cancel_media to stop one. To add a "
                "workflow, find it with search_workflow_templates (or use a workflow file the user attached, or "
                "a URL they give), confirm the choice with the user, then call add_media_workflow. "
                "Workflows on the pod:\n" + self.m.comfy.catalogue()
            )
        return prompt

    # --- sending ------------------------------------------------------------------

    async def send(self, **msg):
        async with self.send_lock:
            await self.ws.send_text(json.dumps(msg))

    async def try_send(self, **msg):
        """For background work that may outlive the connection."""
        try:
            await self.send(**msg)
        except Exception:
            pass

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
            await self.cancel_browser_task()
            if self.media:
                await self.media.close()
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
            case "cancel_task":
                await self.cancel_browser_task()
            case "upload":
                await self.on_upload(msg)
            case "reset":
                await self.interrupt()
                await self.cancel_browser_task()
                if self.media:
                    await self.media.close()
                    self.media = self.new_media()
                self.notes.clear()
                self.announce = False
                self.messages = [{"role": "system", "content": self.system_prompt()}]

    async def on_upload(self, msg: dict):
        if not self.media:
            await self.send(type="error", message="Attachments are used for image generation, "
                                                  "which is off (set COMFYUI_URL).")
            return
        try:
            await self.media.upload(msg.get("name", ""), msg.get("data"), msg.get("workflow"))
        except ComfyError as exc:
            await self.send(type="error", message=f"Upload failed: {exc}")

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
        self.user_speaking = True
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
        text = ""
        try:
            if stream:
                text = await asyncio.to_thread(stream.finish)
            else:
                text = await asyncio.to_thread(self.m.asr.transcribe, np.concatenate(audio)) if audio else ""
            log.info("ASR %.0f ms: %r", (time.perf_counter() - t0) * 1000, text)
            await self.send(type="transcript", text=text, final=True)
            if text:
                await self.start_response(text)
        except Exception as exc:
            log.exception("ASR failed")
            await self.send(type="error", message=f"ASR failed: {exc}")
        finally:
            self.user_speaking = self.vad.in_speech  # a new utterance may have started
            if not text:
                self.maybe_announce()

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

    async def respond(self, user_text: str | None, turn: int):
        """Answers user_text, or with None just reacts to pending notes (task results)."""
        if self.m.comfy:
            self.m.comfy.maybe_refresh()  # the workflow list in the prompt; picked up next turn
        self.messages[0]["content"] = self.system_prompt()
        for note in self.notes:
            self.messages.append({"role": "system", "content": note})
        self.notes.clear()
        self.announce = False
        if user_text:
            self.messages.append({"role": "user", "content": user_text})
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        speaker = asyncio.create_task(self.speak(sentences, turn))
        t0 = time.perf_counter()
        first_token = True
        try:
            for _ in range(settings.llm_max_tool_rounds):
                content, calls, chunker = "", [], SentenceChunker()
                try:
                    exclude = (set() if self.m.browser else tools.BROWSER_TOOLS) | \
                              (set() if self.media else tools.MEDIA_TOOLS)
                    async for msg in self.m.llm.stream(self.messages, tools.schemas(exclude)):
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
        # A task may have finished while we were talking
        self.loop.call_soon(self.maybe_announce)

    async def run_tools(self, calls: list, turn: int):
        pending = list(calls)
        try:
            while pending:
                call = pending[0]
                name, args = call.function.name, dict(call.function.arguments or {})
                await self.send(type="tool_call", turn=turn, name=name, arguments=args)
                result = await tools.run(name, args, session=self)
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

    # --- browser tasks ------------------------------------------------------------
    # A task runs in the background, independent of the spoken response, so the user
    # can keep talking (or barge in) without killing it. When it ends, its result is
    # queued as a note and the assistant announces it at the next quiet moment.

    async def start_browser_task(self, instruction: str) -> dict:
        if not self.m.browser:
            return {"error": "the browser is not available"}
        if self.job and not self.job.task.done():
            return {"error": f"task {self.job.id} ({self.job.instruction!r}) is still running; "
                             "wait for it or cancel it first"}
        self.job_count += 1
        job = BrowserJob(self.job_count, instruction)
        job.task = asyncio.create_task(self.run_browser_job(job))
        self.job = job
        return {"task_id": job.id, "status": "started"}

    async def run_browser_job(self, job: BrowserJob):
        await self.try_send(type="browser_task", id=job.id, status="started", text=job.instruction)
        browser = self.m.browser

        async def on_step(step: int, action: str, args: dict, result: str):
            job.steps.append(f"{action} {json.dumps(args, ensure_ascii=False)}: {result[:150]}")
            shot = None
            if settings.browser_screenshots:
                try:
                    shot = await browser.screenshot_b64()
                except Exception:
                    pass
            await self.try_send(type="browser_step", id=job.id, step=step, action=action,
                                args=args, result=result[:300], url=browser.page.url, screenshot=shot)

        try:
            async with browser.lock:  # one task at a time in the shared window
                agent = BrowserAgent(browser, self.m.browser_llm, load_skills(ROOT / settings.skills_dir),
                                     settings.website_url, settings.browser_max_steps, on_step)
                job.result = await agent.run(job.instruction)
            status = "done"
        except asyncio.CancelledError:
            job.result = "cancelled"
            await self.try_send(type="browser_task", id=job.id, status="cancelled")
            raise
        except Exception as exc:
            log.exception("browser task %d failed", job.id)
            job.result = f"it failed with an error: {exc}"
            status = "failed"
        log.info("browser task %d %s in %.1f s: %s", job.id, status, time.monotonic() - job.started, job.result)
        await self.try_send(type="browser_task", id=job.id, status=status, text=job.result)
        self.add_note(f"Browser task {job.id} ({job.instruction!r}) has finished. Result: {job.result}\n"
                      "Tell the user the outcome in one or two short spoken sentences.")

    def browser_task_status(self) -> dict:
        job = self.job
        if not job:
            return {"status": "no task has been started"}
        if job.task.done():
            return {"task_id": job.id, "status": "finished", "result": job.result}
        return {"task_id": job.id, "status": "running", "instruction": job.instruction,
                "seconds": round(time.monotonic() - job.started), "recent_steps": job.steps[-3:]}

    async def cancel_browser_task(self) -> dict:
        job = self.job
        if not job or job.task.done():
            return {"error": "no browser task is running"}
        job.task.cancel()
        try:
            await job.task
        except (asyncio.CancelledError, Exception):
            pass
        return {"task_id": job.id, "status": "cancelled"}

    def add_note(self, text: str, announce: bool = True):
        """Tells the LLM about an event on its next turn; announce=True also makes it speak up."""
        self.notes.append(text)
        if announce:
            self.announce = True
            self.maybe_announce()

    def maybe_announce(self):
        """Starts a response for pending notes once nobody is talking."""
        if not self.announce or self.user_speaking:
            return  # the user's next turn will pick the notes up
        if self.response and not self.response.done():
            return  # respond() calls us again when it ends
        wait = self.speaking_until - time.monotonic()
        if wait > 0:  # let the current audio finish playing
            self.loop.call_later(wait + 0.3, self.maybe_announce)
            return
        self.response = asyncio.create_task(self.respond(None, self.turn))
