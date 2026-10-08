# Voice Agent

A real-time conversational agent: you talk, it listens, thinks, can operate your
website in a real browser (Playwright), makes pictures and videos with ComfyUI on a
RunPod GPU pod, and answers out loud.

```
browser mic ──PCM16 16k──▶ WebSocket ──▶ Silero VAD ──▶ Nemotron 3.5 ASR (streaming, local GPU)
                                                              │ transcript
browser speaker ◀──PCM16 24k── OmniVoice TTS (local GPU) ◀── GLM 5.3 Flash (Ollama cloud) ⇄ tools
                                                                                   │ browser_task
                                                         browser agent (LLM loop) ◀┘ + skills/*.md
                                                                   │ click / type / read      │ generate_media
                                                              Chromium (Playwright)           ▼
                                                                          ComfyUI manager API on a RunPod pod
                                                                          (images/videos shown in the web UI)
```

| Stage | Model | Runs on |
|---|---|---|
| ASR | `nvidia/nemotron-3.5-asr-streaming-0.6b` (cache-aware streaming RNNT) | this PC |
| LLM | `glm-5.3-flash` via Ollama cloud API | ollama.com |
| TTS | `k2-fsa/OmniVoice` | this PC |
| VAD | Silero VAD | this PC (CPU) |
| Pictures / video | ComfyUI workflows (Z-Image Turbo, FLUX.2 Klein, Wan 2.2, ...) | RunPod GPU pod (`comfyui/`) |

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

## Pictures and videos (ComfyUI on RunPod)

The `comfyui/` folder is the RunPod kit: a small API (`manager.py`) in front of ComfyUI that installs
workflows and runs them from simple JSON (see `comfyui/API.md` and `comfyui/RUNPOD_SETUP.md`).
The voice assistant talks to that API directly. It doesn't use the browser for this, which is faster
and more reliable than clicking through the ComfyUI editor.

### Connecting it

1. Start the pod (`bash /workspace/comfy-kit/setup.sh`, see `comfyui/RUNPOD_SETUP.md`).
2. In `.env`, set `COMFYUI_URL=https://<pod-id>-8000.proxy.runpod.net` (it changes with every new pod)
   and `COMFYUI_API_KEY` to `MANAGER_API_KEY` from `comfyui/pod.env` (that one doesn't change).
3. Restart the server. The log says `ComfyUI at ...: z_image_turbo, flux2_klein_edit, ...`, and
   http://localhost:8000/health shows `"comfyui": "ok, 4 workflows"`.

With `COMFYUI_URL` empty, the feature is off and the assistant doesn't offer it.

### What you can say

| You say | What happens |
|---|---|
| "Make me a picture of a lighthouse at dusk" | `generate_media(z_image_turbo, <a detailed English prompt>)`, about 10 s |
| "Make it night time" / "Edit picture two: add snow" | `flux2_klein_edit` with that picture as input (the latest by default) |
| "Animate it" / "Make a three second video of a koi pond" | `wan_i2v` / `wan_t2v`, several minutes; you can keep talking meanwhile |
| "How's the video going?" / "Stop it" | `media_status` / `cancel_media` |
| "Which workflows do you have?" | `list_media_workflows` |
| "Add the Wan 2.2 14B image to video template" | `search_workflow_templates`, then (after you confirm) `add_media_workflow`. Custom nodes and model downloads run on the pod in the background; you're told when it's ready |
| Attach a workflow `.json`, then "add this as my_flux" | `add_media_workflow(file=<its number>)`. A URL of a workflow JSON works too |

Generation runs in the background like browser tasks: the assistant says "On it", and when the
files are ready they appear in the chat and it tells you. Every picture, video and attached file
gets a number (`#3`) that you can refer to ("animate number three"). Attach your own pictures
with 📎, or by pasting or dropping them on the page. Large pictures are scaled down to 2048 px first.

Results are downloaded to `media/` (the pod's disk is not permanent) and served at `/media/...`.

### Telling the assistant what each workflow is for

`comfyui/descriptions.yaml` has one line per workflow (what it does, how long it takes, input quirks such as
Wan's 4n+1 frame counts). The assistant sees these lines with each workflow's inputs and defaults. Workflows
it adds get a line automatically. Workflows without one are described from their inputs ("Image to video.").
The list is refreshed from the pod every minute, so workflows saved in the ComfyUI editor show up too.

**Cancelling** needs the `POST /cancel/{prompt_id}` endpoint, which was added to `comfyui/manager.py` with this
integration. Run `deploy.ps1` once so the pod gets it. Until then, "stop" only stops waiting, and the pod
finishes the job.

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
  comfyui.py   client for the ComfyUI manager API on the pod
  media.py     per-conversation generation jobs, numbered media, uploads, adding workflows
skills/        browser agent skills (markdown)
comfyui/       RunPod kit (manager.py, setup scripts, workflow bundles) + descriptions.yaml
media/         generated and attached files (served at /media)
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
- `COMFYUI_TIMEOUT_MINUTES`: how long to wait for one generation (a video queued behind others can take a while).
