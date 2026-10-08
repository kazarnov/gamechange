# Voice Agent

A real-time conversational agent: you talk, it listens, thinks, can operate your
website in a real browser (Playwright), and answers out loud.

```
browser mic ──PCM16 16k──▶ WebSocket ──▶ Silero VAD ──▶ Nemotron 3.5 ASR (streaming, local GPU)
                                                              │ transcript
browser speaker ◀──PCM16 24k── OmniVoice TTS (local GPU) ◀── GLM 5.3 Flash (Ollama cloud) ⇄ tools
                                                                                   │ browser_task
                                                         browser agent (LLM loop) ◀┘ + skills/*.md
                                                                   │ click / type / read
                                                              Chromium (Playwright)
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
.venv/bin/playwright install chromium
sudo .venv/bin/playwright install-deps chromium   # system libraries Chromium needs (once)
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

## Browser agent

The voice assistant doesn't click things itself. It hands a whole task in plain language to a
**browser agent** with `browser_task("add eggs and bread to my todo list")`:

1. The task runs **in the background**. The assistant says "On it" and you can keep talking,
   ask something else, or interrupt its voice without stopping the task. Say "how's it going?"
   (`browser_task_status`) or "stop that" (`cancel_browser_task`), or press **Cancel task**.
2. The browser agent is its own LLM loop (`backend/browser_agent.py`). Each turn it sees the page
   as text: the URL, every visible interactive element with a number, and the page text:
   ```
   [4] checkbox "Toggle Todo" in row "buy milk" unchecked
   [5] button "Delete" in row "buy milk"
   ```
   It then calls one of its tools: `click`, `type`, `select`, `hover`, `press`, `scroll`, `goto`,
   `back`, `read_page`, `load_skill`, `finish`. Only the newest page state is kept in its context.
3. When it calls `finish(result)`, the result goes back to the voice assistant, which tells you
   the outcome as soon as nobody is talking.

The web UI shows the task, each step, and a screenshot after every step. With
`BROWSER_HEADLESS=false` you also see the real Chromium window. Its profile is kept in
`browser-profile/`, so if your site needs a login, log in once in that window and it sticks.

The agent stops and asks instead of acting when it would do something irreversible the
task didn't ask for (paying, deleting, sending), or when it needs information it doesn't have.

### Skills

A skill teaches the agent how to do something on **your** site. It's a markdown file in `skills/`:

```markdown
---
name: add-todos
description: Add one or more items to the todo list on the TodoMVC demo site.
---
1. If the page is not the TodoMVC app, open https://demo.playwright.dev/todomvc/.
2. For each item, type it into the textbox "What needs to be done?" with submit set to true.
3. Check that every item now appears in the list, then finish and say how many were added.
```

The agent always sees every skill's name and description, and loads the steps with `load_skill` when
a task matches. The voice assistant sees the list too, so it can say what it's able to do. Skills
are re-read for every task, so you can edit them without restarting. Start from `skills/_template.md`.
Files starting with `_` are ignored.

Write the steps using the labels visible on the page, and mention anything tricky, such as controls
that only appear on hover, confirmation dialogs, or how to tell it worked. The agent can usually
manage without a skill, but a skill makes it faster and more reliable.

### Pointing it at your website

Set `WEBSITE_URL` in `.env`, replace the two TodoMVC skills with skills for your site, and restart.

## Adding plain tools

Quick actions that don't need a browser go straight in `backend/tools.py`:

```python
@tool("get_weather", "Get the weather for a city.", {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
})
async def get_weather(city: str):
    ...
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
  tools.py     tool registry (incl. browser_task / status / cancel)
  browser.py   shared Playwright browser + text view of the page
  browser_agent.py  the browser agent's LLM loop and tools
  skills.py    loads skills/*.md
skills/        browser agent skills (markdown)
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
- `BROWSER_LLM_MODEL`: a different (e.g. bigger) model for the browser agent; each step is
  one LLM call, about 1 s with `glm-5.3-flash`.
- `BROWSER_MAX_STEPS`: the agent gives up after this many steps.
