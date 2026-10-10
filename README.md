# Voice Agent

A real-time conversational agent: you talk, it listens, thinks, can operate your
website in a real browser (Playwright), makes pictures and videos with ComfyUI on a
RunPod GPU pod, makes and edits Instagram and X posts as drafts, saves and schedules them in
FlowAI (`content-generator/`), and answers out loud. It has its own page, where all of that is
done by talking.

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
| TTS | `k2-fsa/OmniVoice` | this PC, or a VoiceStudio server (`TTS_URL`) |
| VAD | Silero VAD | this PC (CPU) |
| Pictures / video | ComfyUI workflows (Z-Image Turbo, FLUX.2 Klein, Wan 2.2, ...) | RunPod GPU pod (`comfyui/`) |

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv -r requirements.txt
.venv/bin/playwright install chromium
sudo .venv/bin/playwright install-deps chromium   # system libraries Chromium needs (once)
cp .env.example .env   # then put your key from https://ollama.com/settings/keys in OLLAMA_API_KEY
```

## Run

```bash
.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 (the agent page, see [below](#the-agent-page-and-flowai)) and tap the
mic. http://localhost:8000/console.html is the test console: it shows every event, tool calls and
browser steps included. The first start downloads the models (~6 GB) and designs the assistant's
voice once (cached in `voices/`). Voice design is checked with the ASR and retried with new seeds if
the result isn't intelligible.

Use headphones, or untick **Interrupt by voice**; otherwise the assistant can hear itself
through your speakers and interrupt itself.

## Docker

The same app as an image, to run as a service. Settings come from `.env`, as above. FlowAI runs
the same image as its `assistant` service; see [Inside FlowAI](#inside-flowai).

```bash
docker compose up -d --build        # NVIDIA GPU: recognition and OmniVoice on the card
docker compose logs -f agent        # wait for "Ready"
```

The agent page is on http://localhost:8000 (`AGENT_PORT` to change). The first start downloads
the models (5.8 GB) into the `models` volume, which takes a few minutes. After that, a start
takes seconds. Drawn and attached files are kept in the `media` volume, conversations in `data`,
and OmniVoice's designed voice in `voices`.

- **FlowAI from the container.** Inside it, `localhost` is the container itself. For the local
  FlowAI stack, set `FLOWAI_URL=http://host.docker.internal:8002`.
- **No GPU.** `docker compose --profile cpu up -d --build agent-cpu` builds a CPU-only image
  and runs recognition on the CPU. Set `TTS_URL` so speech comes from VoiceStudio, because
  OmniVoice on a CPU takes seconds per sentence.
- **Website tools** are off in the container: the image has no Chromium. Build with
  `--build-arg BROWSER=true` and set `BROWSER_ENABLED` to `true` in `docker-compose.yml` to
  use them.
- **Under a path.** The page, its media links and its WebSocket are all relative, so it works
  behind a proxy at `/assistant/` (nginx settings in `FLOWAI_INTEGRATION.md`, change 4).

| Image | Size | Recognition, after you stop talking | Memory |
|---|---|---|---|
| `flowai-agent` (GPU, RTX 5060 Ti) | 12.6 GB | 0.55 s | not measured |
| `flowai-agent:cpu` | 2.8 GB | 2.0 to 2.3 s, using about one core | under 0.5 GB of RAM |

Measured by speaking a 6-second request into the WebSocket the way the page does. The time
includes the 0.6 s of silence that ends a turn. The GPU image's PyTorch (CUDA 12.8) runs on
GTX 16xx up to RTX 50xx cards.

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

## Content studio (posts for Instagram and X)

The assistant makes posts as **drafts** and changes them when you ask. A draft is shaped like a
FlowAI post (`content-generator/`): one platform and placement, a caption, and slides. A slide is
a picture or video with texts drawn on top. Every change makes a new version. The version is
drawn, checked against the platform's specs (the same table and check as FlowAI's pre-export
check, plus one of ours: no `[placeholder]` left in), and shown on the page with its problems.

| You say | What happens |
|---|---|
| "Make an Instagram post for our linen shirt launch" | one `create_draft` with the caption, the texts and a picture prompt; the picture is made in the draft's 4:5 shape and drops in when ready |
| "Remove the cup on the table" | `generate_media(flux2_klein_edit, draft=1)`: edits that slide's current picture and puts the result back in the draft |
| (paint over the cup on the slide) "remove this" / "make it gold" / "put a vase there" | `edit_area`: only the painted part changes, the rest of the picture stays exactly as it was (see [Changing part of a picture](#changing-part-of-a-picture)) |
| "Make it a reel with a voiceover and an Order now button" | `make_video`: the pictures move, your voice reads a line over each, captions, music, an end card (see [Videos with the voice](#videos-with-the-voice)) |
| "Make the headline yellow and move it to the bottom" / "Drop the second line" | `edit_text` / `remove_text` |
| (click a text on the page) "make this bigger" | the click tells the assistant which text "this" is |
| "Shorter caption, fewer hashtags" | `update_draft(caption=...)` |
| "Go back" | `undo_draft`, as many steps as you like (or the Undo button) |
| "Now the same for X" | `create_draft(platform=x, from_draft=1)`, re-cropped to 16:9, caption rewritten for 280 characters |
| "A five-slide carousel on how to care for linen" | one `create_draft` with a text set and a picture prompt per slide, the same style on every slide |

Text is drawn with Pillow, never by the image model, so it is spelled right and easy to change.
Texts wrap and shrink to fit. Texts at the same position stack instead of overlapping. On stories
and reels they keep clear of Instagram's buttons. Rendered files go to `media/` like everything
else. Text on videos isn't drawn yet.

### Fonts

24 fonts in seven styles, listed in `fonts/fonts.yaml`:

| Style | Fonts |
|---|---|
| Modern | `bold` (Geist, the default), `semibold`, `montserrat`, `poppins`, `syne` |
| Poster | `anton`, `bebas`, `archivo` |
| Elegant | `serif` (Instrument Serif), `playfair`, `dm-serif`, `cormorant`, `fraunces` |
| Handwritten | `caveat`, `pacifico`, `dancing`, `great-vibes` |
| Playful | `fredoka`, `lilita` |
| Retro | `righteous`, `abril`, `typewriter` |
| Mono | `mono` (Geist Mono), `space-mono` |

- **What each suits:** every font has a line saying what it's for ("high-contrast serif: luxury,
  beauty, editorial"). The assistant gets the list by style, so "more elegant" or "like a sale
  poster" picks a font from the right style, at most two per post. A font can also be named by its
  label ("Playfair Display").
- **Same size in every font:** each font has a scale, so a size looks about the same in any of
  them. Thin scripts are drawn bigger, tall condensed fonts a little smaller.
- **On the page:** the text editor's Font picker has a tab per style. Each font is previewed in the
  text's own words, from the same file the assistant draws with (served at `/fonts`).
- **Source and licences:** all but Geist and Instrument Serif come from Google Fonts. The
  licences sit next to the files: `OFL-*.txt` for the SIL Open Font License, and
  `LICENSE-SpecialElite.txt` for Apache 2.0.
- **Adding a font:** drop its `.ttf`/`.otf` in `fonts/` and add a line to `fonts.yaml`. Any file
  there can also be used by its file name. Changes apply without a restart.

### Colours and readability

Every slide comes with its picture's colours: a pale tint of its main colour, up to three of its
vivid colours made bright enough to read, and a deep shade for panels (`picture_colours` in
`studio.py`). The assistant sees them as `picture_colours`, and the page shows them first in the
text's colour picker, before a curated set (cream, sand, ink, sun, coral, sky, forest, navy...).

Each text also gets a readability score where it sits: a contrast ratio of its colour against the
darkest and lightest parts of the picture behind it, or against its panel, with credit for an
outline or a shadow. Under 2.2:1 the draft's check warns ("hard to read on the picture behind")
and the assistant is told. The page works the score out the same way, so the colour picker marks
the colours that would be hard to read on that spot before you pick them. It also offers a few
ready-made looks that read well there (a tint on a deep panel, ink on a pale panel, a vivid colour
outlined...), one click each.

### Changing part of a picture

On the page, **Paint to change part of it** under a slide puts a brush on the picture. Paint over
what to change (S/M/L brush, eraser, undo), then pick **Remove**, **Replace** (with what),
**Change** (how) or **Improve**, or just say it ("remove this", "make it navy blue"). After each
stroke the page sends the painted area (`{"type": "mask"}`), and the assistant is told where it is
("the upper left, 9% of the picture"), so "this" means that area.

`edit_area` (`retouch.py`) never sends the whole picture to the editing model, which redraws all
of it and shifts what it keeps. It cuts out the painted area with some room around it, sends that
alone (the model also works at a higher resolution there), and pastes the result back through the
painted area with a soft edge. Outside it, the picture is pixel for pixel what it was. The model
shifts colours a little, so the difference along the area's edge is measured and carried smoothly
into it, and the patch meets the old picture without a seam. To remove or replace something, the
area is filled with flat magenta first, so the model fills in that patch from what is around it.
Told only to "remove the thing in the middle", FLUX.2 Klein kept the thing, or took half of it, in
our tests. The result is a new numbered picture that replaces the slide's own, so Undo goes back.
Results vary from run to run: if a removal leaves a smudge, run it again or paint a little wider.

### Videos with the voice

`make_video` (`reel.py`) turns a draft's pictures, one or several, into a short video:

- **Voice:** the assistant writes a line per picture and reads it in the user's chosen voice (the
  TTS, VoiceStudio or local). A line of several sentences is a shot per sentence.
- **Motion:** each shot moves the whole time it's on screen (a slow zoom in or out, or a pan), so a
  still looks filmed. A new picture fades in over the last one. The same picture again comes in
  with a straight cut and a new framing, like an edit.
- **On screen:** the slide's texts fade in over it, and captions show the spoken words in the
  lower third (or above the texts, if they're there).
- **End card:** optionally, the call to action on a button in the post's colour that pops in
  ("Order now"), over the last picture blurred, with the voice saying it ("Order yours today, the
  link is in our bio").
- **Music:** one of FlowAI Sound's six moods (`SOUND_URL`): Golden hour unless asked otherwise, or
  none. It steps aside whenever the voice speaks (sidechain compression), and the mix is levelled to
  -14 LUFS for social video.

Frames are drawn with Pillow and encoded by ffmpeg (`imageio-ffmpeg` brings a static build) as
H.264 + AAC at the placement's size (1080 × 1920 for a reel). A 15-second video takes about
15 seconds. It's made in the background into a draft of its own (an Instagram reel by default,
story or feed on request; an X post for X), which passes the platform check (size and length are
known). Asked again, of either draft ("other music", "say it shorter", "the button says Shop
now"), that same video draft is made again with the changes, and Undo goes back to the previous
video. On the page, the draft's **Video with voice** section does the same by hand: a line per
picture, the end card, music, motion, captions and format, with a progress bar while it's made.

### Content skills

`skills/content/*.md` say how to make each kind of post: an Instagram post, carousel and story,
and an X post. While they total under 8000 characters (about 2,000 tokens), they go in the prompt
whole, which saves the assistant a round of reading one before every new post. Past that, it sees
each skill's name and description and loads the steps with `load_skill`, like the browser agent's
skills. Copy `skills/content/_template.md` to add one (a brand's rules, a recurring format, a
client's house style). Edits apply on the next turn, without a restart.

## The agent page and FlowAI

`web/index.html` is the agent's own page, in FlowAI's dashboard style. FlowAI has its own version,
the Assistant tab (see [Inside FlowAI](#inside-flowai)).
- **Left:** the conversation, with the mic.
- **Middle:** the current draft as it will look on Instagram or X: carousel arrows, story bars,
  X's picture grid, a caption past the limit highlighted, and the platform check.
- **Right:** the session's drafts, the latest FlowAI posts and the numbered pictures.

On a phone the panes stack and the mic stays at the bottom.

With `FLOWAI_URL` set, the assistant works in FlowAI. At the start of a session it reads the
user's Instagram and X accounts and each account's voice (profile and memory), and writes in it.

| You say or do | What happens |
|---|---|
| "Save it" / **Save to FlowAI** | `save_draft`: the slides are uploaded as assets and the post is saved with status draft; saving again updates the same post |
| "What do we have scheduled?" / "Find the linen post" | `find_posts` |
| "Open it" / click a post in the list | `open_post`: the post becomes a draft here; saving updates it |
| "Use the autumn photo from the gallery" | `find_assets`, `use_assets`: gallery pictures get numbers to use in drafts |
| "Open the autumn launch campaign" | `find_campaigns`, `open_campaign`: each of its posts becomes a draft, linked to the campaign (see [Campaigns](#campaigns)) |
| "Schedule it for Friday at 9" | `schedule_post`: an approval card appears; only **Approve** books it |
| click a text on the picture | that text is selected: "make this gold" means it |
| **Undo** / **Save** / a post in the list | done directly, in a fraction of a second, without the assistant; it is told what happened |

Nothing is scheduled without a person's click. Changing a scheduled post makes it a draft
again, and the approval card comes back for the same time. On this page, the assistant signs in
to FlowAI as `FLOWAI_EMAIL` with `FLOWAI_PASSWORD`. Use the local stack's demo user
(`demo@flowai.test` / `password`). `FLOWAI_DASHBOARD_URL` must be one of FlowAI's
`SANCTUM_STATEFUL_DOMAINS`.

### Conversations

Every conversation is kept, so it can be listed, switched to and picked up again later, or after
a restart. Each one has its own history, numbered pictures, drafts (with their undo steps),
saved posts, campaign links and approvals. `backend/conversations.py` writes it to
`DATA_DIR/conversations/<owner>/<id>.json`, with an `index.json` per owner for the list. The
owner is the FlowAI user (`local` when `FLOWAI_URL` is empty). While FlowAI can't say who the
user is, nothing is kept, so no one's conversations end up in a shared folder.

- **Reconnecting.** The connection shows one conversation at a time, and the conversation
  outlives it. A page that reconnects with the conversation's id in its `hello` gets it back
  whole, in one `conversation` event.
- **Switching.** What the assistant is saying stops, but what it's making doesn't: a picture still
  being made lands in its draft in the conversation it was asked for. That conversation stays in
  memory until the picture is in and saved.
- **Messages.** `new`, `open`, `rename`, `delete` and `conversations` (listed at the top of
  `backend/session.py`). This repo's page starts a new conversation on each load; FlowAI's tab
  has the list.
- **Deleting** removes its file, its pictures and every drawn version of its slides. Drawn files
  are named after the conversation for that reason.

### Campaigns

`open_campaign` (or **Open a campaign** in FlowAI's tab) brings a FlowAI campaign into the
conversation. Its brief and big idea go into the assistant's instructions, and each of its posts
becomes a draft, linked to the campaign rather than to a single FlowAI post:
- one draft for each account's version of a post (FlowAI's *variants*), with that version's
  caption, placement and pictures;
- before the versions exist (while the plan is still being made), one draft per post, showing
  the post itself.

Saving a linked draft changes the campaign through FlowAI's own campaign endpoints:
- **A version:** its caption and placement. Its slides are uploaded only if they changed, and
  become that account's own pictures.
- **A post without versions:** its title and caption, and its pictures.

FlowAI then sends a changed version back to review (gate 6B). The assistant asks on screen for
the person's approval, and approving it updates the times already booked for that version too.
A version that is approved can get another time with `schedule_post`, also approved on screen.
The assistant's token can't approve anything itself.

### Voice

Every sentence is spoken in the same voice. OmniVoice makes up a new voice for each request
unless it is given a recording to clone, so the agent always passes one:
- **Locally**, the voice designed once from `TTS_VOICE_INSTRUCT` (or `TTS_REF_AUDIO`), which is
  the only one.
- **With VoiceStudio**, a voice profile on the server. The user picks one on the page: the
  profiles ready to use (a fresh pod has its demo voice), or one of the server's 1,126 catalog
  voices, described by gender, age, pitch and accent and searched by those words. A catalog voice
  becomes a profile the first time it is picked (about 3 s), and picking it again gives the same
  profile.

The pick is kept per user in `prefs.json`, next to their conversations, so it's the voice in all
of them and after a reload. If the pod is replaced and the profile is gone, the agent sets the
catalog voice up again on the next connection, and it sounds the same, because VoiceStudio
renders catalog voices with a fixed seed. Until the user picks, it's `TTS_VOICE`. The messages
are `voices` (what to pick from, with a search), `voice` (pick one) and `preview_voice` (play
its sample, which stops what is being said).

### Inside FlowAI

FlowAI (`content-generator/`) has a copy of this repo in `assistant/`. It runs as its `assistant`
service: `docker compose --profile assistant up -d`, or `--profile assistant-cpu` without a GPU.
Its page is the dashboard's **Assistant** tab (`frontend/src/dashboard/assistant/`), served
through the Vite proxy at `/assistant/`. Beyond this page, it adds:
- **Sign-in as the person who opened it.** The tab asks FlowAI for a token
  (`POST /api/assistant/session`) and sends it in its `hello`. That token can save drafts but
  not schedule, and reaches only the routes the tools use. **Approve** books the post with the
  person's own session.
- **Editing by hand without the model:** captions, titles, placement, text words and styles,
  slide order, removing slides, adding pictures. These are the `action` messages listed at the
  top of `backend/session.py`. The assistant is told about each change.
- **A calendar:** drop a draft on a slot to book it there.
- **A conversation that survives page changes:** it stays open from page to page, with a dock to
  come back.
- **Conversations and campaigns:** a list over the conversation panel (new, open, rename,
  delete, search), **Open a campaign**, and an **Open in the assistant** button on each campaign
  in FlowAI's Campaigns page. A draft from a campaign shows which post and account it is, where
  it stands at gate 6B, and its booked times. The last conversation opens again after a reload.

After changing the agent here, copy it over and rebuild:

```bash
rsync -a --delete --exclude __pycache__ --exclude comfyui/outputs --exclude comfyui/pod.env \
  backend web skills fonts comfyui requirements.txt Dockerfile .dockerignore .env.example .gitignore README.md \
  content-generator/assistant/
cd content-generator && docker compose --profile assistant up -d --build assistant
```

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
Dockerfile     the agent as an image (GPU, or CPU with TORCH=cpu)
docker-compose.yml  runs it: agent (GPU) or agent-cpu, with volumes for models and media
backend/
  main.py      FastAPI app, model loading, /ws endpoint, serves web/
  session.py   per-connection pipeline: VAD → ASR → LLM/tools → TTS, barge-in
  vad.py       Silero VAD turn detector
  asr.py       Nemotron 3.5 ASR (streaming + offline fallback)
  llm.py       Ollama cloud client
  tts.py       OmniVoice + sentence chunker
  voicestudio.py  speech from a VoiceStudio server instead (TTS_URL), its voices and catalog
  tools.py     tool registry (incl. browser_task / status / cancel)
  browser.py   shared Playwright browser + text view of the page
  browser_agent.py  the browser agent's LLM loop and tools
  skills.py    loads skills/*.md
  comfyui.py   client for the ComfyUI manager API on the pod
  media.py     per-conversation generation jobs, numbered media, uploads, adding workflows
  studio.py    post drafts: slides, texts drawn with Pillow, colours and readability, versions and undo
  retouch.py   changing only the painted part of a picture (cut out, edited, pasted back seamlessly)
  reel.py      a draft's pictures as a video: motion, the voice, captions, end card, music, ffmpeg
  platforms.py Instagram and X specs and the pre-export check (mirrors FlowAI's)
  flowai.py    FlowAI: accounts and voice, saving, finding, opening and scheduling posts, campaigns
  conversations.py  conversations kept on disk: list, reopen, switch, delete
skills/        browser agent skills (markdown)
  content/     content skills: how to make each kind of post
fonts/         fonts for text on pictures, by style (fonts.yaml), with their licences
comfyui/       RunPod kit (manager.py, setup scripts, workflow bundles) + descriptions.yaml
media/         generated and attached files (served at media/, next to the page)
data/          conversations and each user's voice (not served)
web/           the agent page (index.html, agent.js), the test console (console.html, app.js),
               the protocol client both use (voice.js), mic-worklet.js
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
- `LLM_MODEL`: Ollama cloud models compared as the content-studio agent (October 2026). Each ran the same scripted
  seven-turn conversation three times, through the real agent loop with stand-in pictures: a post, removing an
  object, recolouring the headline, a shorter caption, undo, the X version, and a carousel.

  | Model | `LLM_THINK` | Tasks right (of 6) | Malformed tool calls | Spoken reply | Notes |
  |---|---|---|---|---|---|
  | `deepseek-v4.1-flash` | `low` | 5, 6, 6 | none | ~200 chars | steadiest time to first word (worst 2 s); best captions |
  | `gemma4:31b` | `false` | 5, 5, 6 | none | ~120 chars | shortest replies, the best fit for speech; sometimes changes both lines when asked for "the headline" |
  | `glm-5.3-flash` | `low` | 6, 6, 6 | 1 to 5 a run | ~190 chars | most precise edits; recovers from its bad calls, which costs a round each |
  | `gpt-oss:120b` | `low` | — | — | — | not usable here: described drafts it never made, left captions empty |

  Qwen3.6 35B-A3B isn't on Ollama cloud. Locally it is a 22.6 GB download, more than a 16 GB card holds next
  to the ASR and TTS models.

  Since then a post is made in one `create_draft` call and the content skills sit in the prompt, so every
  request in the script takes one tool round (two model calls), where a new post used to take five. Rerun on
  that code, both models got all seven requests right with no tool errors. Time for the reply itself:

  | Model | New post | Edits | Carousel |
  |---|---|---|---|
  | `gemma4:31b` | 2.1 s | 0.5 to 1.4 s | 4.8 s |
  | `deepseek-v4.1-flash` | 4.8 s | 1.0 to 3.3 s | 8.2 s |

  Gemma 4 is now the faster fit for a voice agent; DeepSeek still writes the stronger captions. DeepSeek
  writes `&` as `&amp;` in tool calls, which the studio turns back into `&`. Prompt length barely matters: on
  both models, going from 2k to 15k tokens of prompt added 0 to 0.4 s before the first word.
- `TTS_REF_AUDIO` / `TTS_REF_TEXT`: clone a specific voice instead of the designed one.
- `TTS_URL`, `TTS_API_KEY`, `TTS_VOICE`: speech from a VoiceStudio server (the same OmniVoice model, e.g. on a RunPod
  GPU) instead of this PC. On an L40, a 5.8-second sentence came back in about 0.4 s, against about 4 s with
  OmniVoice on a GTX 1660 Ti. The first request after VoiceStudio starts loads the model (about a minute); the agent
  makes it at startup. `TTS_VOICE` is the voice until the user picks one on the page (see [Voice](#voice)): a
  VoiceStudio profile's name or id, or a catalog voice's id such as `feat_04_the_neighbor`; empty means the
  server's oldest profile. Don't use an OpenAI name like `alloy`: VoiceStudio maps those to OmniVoice's default,
  which is a new random voice for every request, so the voice changed from one sentence to the next (measured:
  233, 121, 123 and 110 Hz median pitch over four sentences, against 197, 198, 200 and 179 Hz with a profile).
- `BROWSER_LLM_MODEL`: a different (e.g. bigger) model for the browser agent; each step is
  one LLM call, about 1 s with `glm-5.3-flash`.
- `BROWSER_MAX_STEPS`: the agent gives up after this many steps.
- `COMFYUI_TIMEOUT_MINUTES`: how long to wait for one generation (a video queued behind others can take a while).
- `SOUND_URL`: FlowAI Sound (e.g. `http://sound:8000`, set in FlowAI's compose) for music under the videos
  `make_video` makes. Empty: the videos have only the voice.
- `LLM_MAX_TOOL_ROUNDS`: tool-calling rounds per reply (default 12). A post or a carousel takes one; the rest is
  room for longer chains (find a post, open it, change it, save it).
