# Voice Agent

A real-time conversational agent: you talk, it listens, thinks, can call tools
(later: drive your website with Playwright), and answers out loud.

```
browser mic ──PCM16 16k──▶ WebSocket ──▶ Silero VAD ──▶ Nemotron 3.5 ASR (streaming, local GPU)
                                                              │ transcript
browser speaker ◀──PCM16 24k── OmniVoice TTS (local GPU) ◀── GLM 5.3 Flash (Ollama cloud) ⇄ tools
```

| Stage | Model | Runs on |
|---|---|---|
| ASR | `nvidia/nemotron-3.5-asr-streaming-0.6b` (cache-aware streaming RNNT) | this PC |
| LLM | `glm-5.3-flash` via Ollama cloud API | ollama.com |
| TTS | `k2-fsa/OmniVoice` | this PC |
| VAD | Silero VAD | this PC (CPU) |

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv -r requirements.txt
cp .env.example .env   # then put your key from https://ollama.com/settings/keys in OLLAMA_API_KEY
```

## Run

```bash
.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 and click **Start talking**. The first start downloads the
models (~6 GB) and designs the assistant's voice once (cached in `voices/`). Voice design is checked
with the ASR and retried with new seeds if the result isn't intelligible.

Use headphones, or untick **Allow interrupting by voice**; otherwise the assistant
can hear itself through your speakers and interrupt itself.

## How a turn works

1. The browser streams mic audio continuously over `/ws`.
2. Silero VAD detects speech start, which opens a streaming ASR session. Partial
   transcripts are sent back live. If the assistant was talking, it stops (barge-in).
3. After `VAD_MIN_SILENCE_MS` of silence the transcript is finalized and sent to the LLM.
4. LLM tokens stream back. Every complete sentence goes to TTS immediately, so the
   assistant starts speaking before the LLM has finished.
5. If the LLM calls tools, they run (`backend/tools.py`), results go back to the LLM,
   and it continues.

Audio frames from the server carry a turn id, so audio from an interrupted turn is dropped.

## Adding tools (e.g. website actions)

```python
# backend/tools.py
@tool("click_button", "Click a button on the website by its visible label.", {
    "type": "object",
    "properties": {"label": {"type": "string"}},
    "required": ["label"],
})
async def click_button(label: str):
    await page.get_by_role("button", name=label).click()
    return {"clicked": label}
```

## Layout

```
backend/
  main.py      FastAPI app, model loading, /ws endpoint, serves web/
  session.py   per-connection pipeline: VAD → ASR → LLM/tools → TTS, barge-in
  vad.py       Silero VAD turn detector
  asr.py       Nemotron 3.5 ASR (streaming + offline fallback)
  llm.py       Ollama cloud client
  tts.py       OmniVoice + sentence chunker
  tools.py     tool registry
web/           test UI (index.html, app.js, mic-worklet.js)
```

## Tuning (`.env`)

- `ASR_LOOKAHEAD`: 0 / 3 / 6 / 13, which is 80 / 320 / 560 / 1120 ms of ASR latency. Higher is more accurate.
- `ASR_MODE=offline` transcribes the whole utterance after you stop talking (no partials).
- `VAD_MIN_SILENCE_MS`: how long a pause ends your turn. Lower is snappier but may cut you off.
- `TTS_NUM_STEP`: OmniVoice diffusion steps. The default of 8 ran at about 0.67× realtime on a GTX 1660 Ti, with no intelligibility
  loss vs 16 in tests. Raise it if you have a faster GPU.
- `TTS_DTYPE`: keep `float32` on GTX 16xx cards. They have no tensor cores, so fp16 is about 6× slower there, and
  OmniVoice produces garbled audio in fp16. `float16` is fine on RTX cards.
- `LLM_THINK`: keep `low`. With `false`, GLM 5.3 Flash leaks its reasoning into the reply and it gets spoken.
- `TTS_REF_AUDIO` / `TTS_REF_TEXT`: clone a specific voice instead of the designed one.
