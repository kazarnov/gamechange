# Connecting the agent to FlowAI

The voice agent in this repo makes Instagram and X posts as drafts, edits them when asked
("remove the cup", "make the headline yellow", "now the same for X"), and checks them against
each platform's specs. FlowAI (`content-generator/`) is the studio those posts end up in. This
document is the plan for joining the two once the website is complete.

**Nothing in `content-generator/` has been changed.** Everything FlowAI needs is listed under
[Changes in FlowAI](#changes-in-flowai), to be done later.

## How the two line up

The agent's drafts were built to map onto FlowAI's models one to one, so saving a draft is a
straight translation:

| In the agent | In FlowAI | API that exists today |
|---|---|---|
| Draft (`backend/studio.py`) | `Post` with `status: draft` | `POST /api/posts`, `PATCH /api/posts/{id}` |
| Draft fields: platform, placement, caption, title | `platforms`, `placement`, `body`, `title` | same names; placements are the same (`feed`, `story`, `reel`, `post`) |
| A slide as drawn (picture + texts) | `Asset`, attached in order (`post_assets.position`) | `POST /api/assets` (multipart `files[]`), then `asset_ids` on the post |
| Numbered picture or video in the conversation | `Asset` | `GET /api/assets` |
| `generate_media` (ComfyUI on RunPod) | `Generation` (Higgsfield, Google, gateway) | `POST /api/generations`, `GET /api/generations/{id}` |
| `backend/platforms.py` table | `config/platforms.php` | `GET /api/platform-specs` |
| `platforms.check()` | `PlatformSpecs::check()` | `POST /api/checks` |
| Content skills (`skills/content/*.md`) | Account profile + memory (`Voice::context()`) | `GET /api/accounts`, `GET /api/accounts/{id}/voice` |

The agent stays a separate Python service: it holds the microphone, speech, and the
conversation. It gets **its own page** in the dashboard, where the user does the work by talking:
make a post, change it, save it, find and reopen saved posts, and schedule them with one click
of approval. The page talks to the agent over the agent's WebSocket protocol, which is documented
at the top of `backend/session.py`. The agent talks to FlowAI's API as the signed-in user.

```
FlowAI dashboard, /dashboard/assistant ──WebSocket /assistant/ws──▶ agent (this repo)
        ▲   │ Approve: PATCH /api/posts/{id}                        │  drafts, texts, pictures
        │   │ (the user's own session)                              │  save, find, open, ask to schedule
        │   ▼                                                       ▼
        └── FlowAI API (Laravel) ◀── assistant token: assets, posts (drafts only), accounts, voice
```

**A working version of that page is in this repo** (`web/index.html`, `web/agent.js`,
`web/voice.js`), running against FlowAI's API today with the sign-in described under
[the agent's part](#the-agents-part). Porting it means re-drawing it in React with FlowAI's
components; the protocol, the events and the behaviour stay the same.

## Order of work

1. **FlowAI changes 1 to 4** below: tokens, drafts only, the page, the route. After this, the
   agent can be opened from the dashboard and act for the user.
2. **Agent side**: done in this repo (see [the agent's part](#the-agents-part)), except
   swapping its stand-in sign-in for the token from change 1.
3. **FlowAI changes 5 to 7**: provenance, editable drafts, story captions. Saved drafts can then
   be reopened with their texts still editable.
4. **FlowAI change 8**: ComfyUI as a FlowAI media provider, so every picture is in one Gallery.
5. **FlowAI changes 9 and 10** whenever usage calls for them.

## Changes in FlowAI

Paths are relative to `content-generator/`.

### 1. Let the agent act for a signed-in user

**Why.** The dashboard signs in with Sanctum's session cookie. The agent is another service and
needs a bearer token for the same user. The `personal_access_tokens` table already exists
(`database/migrations/2026_10_02_203858_create_personal_access_tokens_table.php`), but `User`
doesn't use `HasApiTokens`, so no token can be issued.

**What.**
- `backend/app/Models/User.php`: add `use Laravel\Sanctum\HasApiTokens;` to the traits.
- New route `POST /api/assistant/session` (inside `auth:sanctum` + `verified`). It returns a
  token with the abilities `assistant`, `posts:draft`, `assets`, `generations`, `checks` and
  `accounts:read`, expiring after about 12 hours. Delete the user's assistant tokens on logout.
- Don't use the `/api/agent/*` prefix: that is already the phone-automation API, with its own
  `agent` middleware and token.

The dashboard sends the token as the first WebSocket message (`{"type": "auth", "token": "…"}`),
never in the URL, so it doesn't end up in logs.

### 2. Drafts only, unless a person approves

**Why.** In `PostController::fill()`, a post saved with any status other than `draft` counts as
the person's approval (`approved_at`, `approved_by`). If the agent's token could set `scheduled`,
FlowAI would record a post the agent wrote as approved by the user. That breaks the two-gate rule
in `DECISIONS.md`.

**What.** In `backend/app/Http/Requests/SavePostRequest.php`, when the request's token has the
`assistant` ability, allow only `status: draft` and reject `queue` and `scheduled_at`. The user
schedules from the Composer, which stays the approval step.

### 3. The assistant's own page

**What.** A page at `/dashboard/assistant`, ported from this repo's working page:

| This repo | In FlowAI |
|---|---|
| `web/voice.js`: the protocol (mic audio up as PCM16 16 kHz, speech down as PCM16 24 kHz behind a turn id, JSON events), barge-in, attachments | a hook, e.g. `frontend/src/dashboard/assistant/useAssistant.ts` |
| `web/agent.js` + `web/index.html`: the page | `frontend/src/dashboard/pages/Assistant.tsx` |

- **Layout.** Conversation and mic on the left. In the middle, the draft as it will look on the
  platform. On the right, the session's drafts, the latest FlowAI posts and the numbered pictures.
  On a phone the panes stack, with the mic fixed at the bottom.
- **The draft.** It shows every slide, not just the first one: Instagram's carousel arrows and
  dots, the story bars, X's picture grid. It also highlights a caption over the platform's limit,
  and lists the platform check. `PostPreview.tsx` only shows a post's first asset, so either give
  it a `slides` prop or keep the page's own preview.
- **Selection.** Each `draft` event carries where every text was drawn (`slides[].texts[].box`,
  as fractions of the slide). The page puts a clickable box over each text. A click sends
  `{"type": "focus", "draft", "slide", "text"}`, so "make this gold" works without naming the
  text.
- **Buttons that skip the assistant.** Undo, Save, and opening a post from the list send
  `{"type": "action", ...}`. They take a fraction of a second instead of a model round, and the
  assistant is told what happened.
- **Approval.** `schedule_post` sends an `approval` event with the post, the time (UTC) and the
  post's fields. Approve must do `PATCH /api/posts/{id}` with `status: scheduled` and that time
  **with the user's own session**, then send `{"type": "approve", "id", "done": true}`. That way
  the person is the approver, and the assistant's token stays drafts-only (change 2). The page in
  this repo has no FlowAI session, so it sends `approve` without `done` and the agent books the
  post itself.
- **Navigation.** Add an entry in `frontend/src/dashboard/nav.ts` (under Create) and the route in
  `Dashboard.tsx`. Saved posts link to `/dashboard/create?post={id}`, as the page here already
  does.

### 4. Serve the agent from the same origin

**Why.** On the same origin there is no CORS to configure, and the WebSocket rides the site's
HTTPS.

**What.** The agent ships as a Docker image (`Dockerfile` in this repo; README, "Docker").
Copy this repo into `assistant/`, next to `sound/`, then add it as a service. The page, its
media links and its WebSocket are all relative, so it works under `/assistant/` unchanged.

- `docker-compose.yml` (local stack), a service and its three volumes:

  ```yaml
  # The voice assistant (assistant/). Its page is /assistant/ through the Vite proxy.
  # Recognition and OmniVoice run on the GPU (needs the NVIDIA container toolkit).
  assistant:
    build: ./assistant
    env_file: ./assistant/.env
    environment:
      FLOWAI_URL: http://api:8000
      BROWSER_ENABLED: "false"
    ports:
      - "${ASSISTANT_PORT:-8003}:8000"
    volumes:
      - assistant-models:/cache
      - assistant-media:/app/media
      - assistant-voices:/app/voices
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    depends_on:
      - api
  ```

- `frontend/vite.config.ts`: proxy `/assistant` with WebSockets, and drop the prefix:

  ```ts
  const assistant = { target: process.env.ASSISTANT_ORIGIN ?? 'http://localhost:8003', ws: true,
                      rewrite: (p: string) => p.replace(/^\/assistant/, '') }
  // in server.proxy:
  '/assistant': assistant,
  ```

- `deploy/docker-compose.yml`: the same service, built for the CPU, because the VPS has no GPU.
  Speech comes from VoiceStudio (`TTS_URL`). See [Open decisions](#open-decisions) for how
  recognition does on a CPU.

  ```yaml
  # The voice assistant. Private like PHP-FPM: nginx sends /assistant/ here.
  assistant:
    build:
      context: ../assistant
      args:
        TORCH: cpu
    env_file: assistant.env   # the agent's settings (OLLAMA_API_KEY, TTS_URL, ...), apart from Laravel's
    environment:
      FLOWAI_URL: http://web
      ASR_DEVICE: cpu
      ASR_DTYPE: float32
      BROWSER_ENABLED: "false"
    volumes:
      - assistant-models:/cache
      - assistant-media:/app/media
    depends_on:
      - web
    restart: unless-stopped
  ```

- `deploy/nginx.conf`, next to the API's location:

  ```nginx
  # A relative redirect, so it keeps the port (nginx's own redirect would drop :8090)
  location = /assistant {
      absolute_redirect off;
      return 301 /assistant/;
  }
  # The voice assistant's page and its WebSocket. Conversations stay open for a long time.
  location /assistant/ {
      proxy_pass http://assistant:8000/;
      proxy_http_version 1.1;
      proxy_set_header Upgrade $http_upgrade;
      proxy_set_header Connection "upgrade";
      proxy_read_timeout 3600s;
  }
  ```

Until change 1, the agent signs in with a cookie, as `FLOWAI_EMAIL`. That works in the local
stack, where the session cookie is neither `Secure` nor tied to a domain. In production, where it
is likely both, the agent needs the token from change 1.

### 5. Uploads that remember where they came from

**Why.** `POST /api/assets` takes files only, so drawn slides arrive as plain uploads. Nothing
links them to the draft, the prompt, or the picture they were made from.

**What.** In `backend/app/Http/Controllers/AssetController.php` `store()`, accept an optional
`source` (`generated`, or a new `assistant` value) and `meta` (prompt, workflow, parent asset id).
Pass them through to `AssetStore` (`fromUpload` takes `$source` already; add `$meta`). If you add
`assistant`, also add it to the `source` filter in `index()` and to the `Asset['source']` type in
`frontend/src/lib/api.ts`.

### 6. Keep a draft's layers so it stays editable

**Why.** A FlowAI post stores only flattened files. To reopen a saved post with the agent and
"move the headline down", the agent needs the slide layout: which picture is under each slide,
and each text's words, position, size, colour, style and font.

**What.** Add a nullable JSON `composition` column to `posts` (new migration). Then add it to
`Post`'s `#[Fillable]` and `casts()`, add a `nullable|array` rule in `SavePostRequest`, and
return it from `PostResource`. Only the agent writes or reads it. Everything else keeps using
the flattened assets, which stay the source of truth for publishing.

### 7. Story drafts without a caption

**Why.** `SavePostRequest` has `'body' => ['required', ...]`, but Instagram stories show no caption
(`caption => 0` in `config/platforms.php`). A story draft from the agent therefore fails
validation.

Tested on the local stack: a story draft with an empty body gets `422 The body field is required.`

**What.** Make `body` required unless the placement is `story`, and store `''` when it is
missing: the column is `text`, not nullable, so no migration is needed. Places that show the
post's name already fall back from `title` to `body` (`PostController::afterSave`).

### 8. ComfyUI as a media provider

**Why.** FlowAI makes pictures with Higgsfield, Google and gateway models. The agent uses its
ComfyUI pod, which costs less per picture and has the picture-editing workflow
(`flux2_klein_edit`). As a FlowAI provider, every ComfyUI result lands in the Gallery with its
prompt, cost and retry button. The Composer, Creative Lab and the campaign media team could use
it too, and the agent could make all its pictures through `POST /api/generations`.

**What.**
- `backend/config/ai.php`, under `providers`: add a `comfyui` entry with `reach`, `url` (the pod's
  manager API, `COMFYUI_URL`), `key` (`COMFYUI_API_KEY`) and `models`. List each workflow as a
  model: `z_image_turbo` (image, no inputs), `flux2_klein_edit` (image, `requires_image`,
  `max_inputs: 1`) and `wan_i2v` (video, `requires_image`), each with `capabilities` like the
  Higgsfield entries.
- `backend/app/Services/Ai/Models/ModelRegistry.php`: add `comfyui` to `PROVIDERS` and a
  `comfyui()` lister to `all()`. It should report models available when the URL is set and
  `GET {url}/health` answers. Also add the provider to `configured()`, the hint in `providers()`,
  and a case in `test()`.
- New `backend/app/Services/Ai/Media/ComfyUiProvider.php` implementing `MediaProvider`:
  - `submit()`: `POST {url}/run` with `name`, `prompt` and `wait: false`. Turn the aspect ratio
    into `width`/`height` near the workflow's default pixel count (the agent's `_size()` in
    `backend/media.py` does this). Send the input asset as a data URL in `image`: the pod accepts
    data URLs, so no public link is needed. Return `prompt_id` as `external_id`.
  - `poll()`: `GET {url}/result/{prompt_id}`. When `done`, return each file as
    `{url: {url}{file.url}, mime, headers: {X-API-Key: key}}`. `GenerationRunner::finish()`
    already sends per-output `headers` when downloading.
- `backend/app/Services/Ai/Media/GenerationRunner.php` `provider()`: add the `'comfyui'` case.
- Reference: the pod's API is documented in this repo's `comfyui/API.md`. Calls must come from the
  backend, never the browser, because the key would be exposed.

### 9. The account's voice as text

**Why.** The agent should write in the account's voice, from the same rules as FlowAI's own
writers. `GET /api/accounts/{id}/voice` returns the profile and memory as data, but the text
FlowAI puts in its prompts is built by `Voice::context()`.

**What.** Add `context` (the `Voice::context($account)` string) to the response of
`AccountVoiceController::show()`.

### 10. Rate limits for a working session

**Why.** Generation routes use `throttle:intake`, which allows 20 per minute and 300 per day per
user. A seven-slide carousel with a few rounds of edits fits, but a long session may not.

**What.** Add an `assistant` limiter in `AppServiceProvider` for requests made with the
assistant token, or raise `intake` for them. The agent's own writing doesn't count against
`throttle:ai`, because it calls its language model directly and not `/api/ai/write`.

## The agent's part

Done in this repo and tested against the local stack:

- `backend/flowai.py`: the FlowAI client and the conversation's link to it.
- **At the start of a session:** the agent reads the user and their Instagram and X accounts,
  plus each account's profile and memory, written the way `Voice::context()` writes them
  (until change 9 sends that text). All of this goes in the system prompt, so the agent writes
  in the account's voice without a tool call.
- **Tools:**
  - `save_draft`: uploads the drawn slides (`POST /api/assets`), then creates the post as a
    draft or updates it, and deletes the slides of the previous save.
  - `find_posts` and `open_post`: an opened post becomes a draft here; saving it updates the
    same post. A post that is already out opens as a copy.
  - `find_assets` and `use_assets`: pictures from the Gallery, for drafts.
  - `schedule_post`: asks for approval on the page, as described in change 3.
- **Rules:**
  - Scheduling is refused while the draft's check fails.
  - Saving is refused while the caption is over the platform's limit, since FlowAI rejects it
    even on a draft.
  - Saving changes to a scheduled post makes it a draft again. FlowAI does this, and the agent
    then asks the user right away to approve the same time again, so nothing goes out
    unapproved.
- **Stand-ins until FlowAI's changes exist:**
  - Sign-in: the agent signs in with `FLOWAI_EMAIL` / `FLOWAI_PASSWORD` through Sanctum's cookie
    flow, with the dashboard's address as `Referer`. Replace this with the token from change 1,
    sent by the page as `{"type": "auth", "token"}`; `FlowAIClient` already takes a `token`.
  - Story bodies: a story's body is sent as its title, until change 7.
  - Slides are uploaded as plain uploads until change 5, and texts are flattened into the
    pictures until change 6.
- When FlowAI's `config/platforms.php` changes, update `backend/platforms.py` (or load it from
  `GET /api/platform-specs` at start). The check here also refuses `[placeholders]` left in a
  caption or text. That one is ours, not FlowAI's.
- Once change 8 exists, switch the pictures to `POST /api/generations`.

## Tested against the local stack

Run on 2026-10-09 against `docker compose up` with the demo seed, as `demo@flowai.test`. Every
post and asset the tests made was deleted afterwards.

| What | Result |
|---|---|
| `GET /api/platform-specs` against `backend/platforms.py` | Identical for Instagram feed, reel and story, and the X post. |
| `POST /api/checks` against `platforms.check()`, 7 cases (right sizes, wrong shape, too small, 3 files, no media, story caption) | Same statuses and the same messages in every case. One wording difference, on purpose: FlowAI says "Instagram storys don’t show a caption" (see below). |
| Uploading drawn slides (`POST /api/assets`) | Works. Width and height are read correctly, `source` is `upload` (change 5). |
| A draft post with an account and a slide (`POST /api/posts`, `status: draft`) | `201`, `approved_at` is null, the asset is attached. |
| A story draft with no caption | `422` (change 7). |
| An X draft over 280 characters | `422`, even as a draft (see the agent's part). |
| Picture and video models in FlowAI | None available: no Higgsfield or Gemini key is set. Until one is, FlowAI can't make pictures at all on this stack, which makes change 8 more useful. |
| `POST /api/assistant/session` | `404`, as expected (change 1 isn't done yet). |
| The agent's FlowAI tools, end to end | Save, save again (same post, old slides deleted), no-change save, find, open (a scheduled post), gallery pictures into a draft, schedule, approve, change after scheduling, re-approval, decline, story with no caption, X caption over 280, queue with no slots: all as described above. |
| The page in a browser (headless Chromium, DeepSeek V4.1 Flash) | "Make an Instagram post…" → draft on screen; click the headline, "make this gold" → only that text changed; Save → post in FlowAI; "schedule it for next Friday at 9" → approval card → Approve → post `scheduled`, `approved_at` set. |
| The agent in Docker (both images), FlowAI at `host.docker.internal:8002` | A spoken request over the WebSocket: recognised, `find_posts`, a spoken reply. Behind nginx at `/assistant/` with the change 4 settings, the page loaded, the WebSocket connected, the drawn slide loaded from `/assistant/media/`, and Save made a FlowAI post. `/assistant` without the slash redirects to `/assistant/` and keeps the port. |

Found along the way, not needed for the agent:

- **The queue worker dies on the first start.** It waits for `vendor/` and `.env`, but not for
  migrations. On a fresh database it asked for the `cache` table before the API had migrated
  (`no such table: cache`). It has no `restart:` policy, so it stayed down, and with it every
  queued job (campaign agents, media, publishing). Fix in `docker-compose.yml`: give `worker` and
  `scheduler` `restart: unless-stopped`, or give `api` a healthcheck that passes after
  `migrate` and make them `depends_on: api: condition: service_healthy`.
- **"Instagram storys".** `PlatformSpecs::check()` builds the caption warning as
  `"{$name}s don’t show a caption"`, which gives "Instagram storys" and "Facebook storys".
  Ours says "An Instagram story shows no caption".
- **Demo accounts have no voice.** `@maisoncire` on both platforms has an empty profile and memory,
  so anything written "in the account's voice" has nothing to go on. A few profile fields
  in `DemoSeeder` would make demos of the agent (and FlowAI's own writers) more convincing.

## Open decisions

- **Where the agent runs.** Speech can come from a VoiceStudio server (`TTS_URL`,
  `backend/voicestudio.py`): on an L40, a five-second sentence is ready in about 0.4 s.
  Recognition (Nemotron) runs inside the agent. In Docker it takes 0.55 s on a GPU. The
  CPU-only image (2.8 GB, under 0.5 GB of RAM, about one core) takes 2.0 to 2.3 s, so each
  turn waits about 1.5 s longer. That was on a 16-core desktop; a small VPS may be slower.
  VoiceStudio also offers streaming transcription over a WebSocket
  (`/v1/audio/transcriptions/stream`). Moving recognition there would remove that wait on a
  machine without a GPU. The options are:
  - Run the agent on a GPU machine and route `/assistant/` to it.
  - Run it next to FlowAI on the VPS (the CPU image in change 4), with speech on a VoiceStudio
    GPU. Recognition stays on the CPU until it moves to VoiceStudio.
  - Swap them for FlowAI Sound's Whisper and Kokoro (`sound/`, CPU) when the agent runs next to
    FlowAI.
  - Offer the website panel as typed chat first. The protocol already accepts
    `{"type": "text", ...}` without a microphone.
- **Text on video.** The agent draws text on pictures only. For reels, FlowAI's reel renderer
  (`Services/Sound/ReelRenderer.php`, `POST /api/generations` with `kind: reel`) already does
  captions and titles. The agent could call that instead of drawing text itself.
- **Brand assets.** Logos and brand fonts per account would let the agent put a logo on every
  slide. FlowAI has no place for them yet. One option is an account `brand` field holding asset
  ids, read alongside the voice.
