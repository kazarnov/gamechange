# ComfyUI Manager API

An HTTP API in front of ComfyUI, running on a RunPod GPU pod. You register a workflow once, and the
manager installs its custom nodes, downloads its models and works out its inputs. After that you
call it with simple JSON such as `{"name": "...", "prompt": "...", "image": "<url>"}`.

- [Basics](#basics)
- [Generating: run, poll, download](#generating)
- [Workflows installed today](#installed-workflows)
- [Adding a new workflow](#adding-a-new-workflow)
- [Endpoint reference](#endpoint-reference)
- [Client code (Python, JavaScript)](#client-code)
- [Errors and gotchas](#errors-and-gotchas)

---

## Basics

| | |
|---|---|
| Base URL | `https://<pod-id>-8000.proxy.runpod.net`. The pod id changes when a new pod is created (see RUNPOD_SETUP.md). |
| Auth | Header `X-API-Key: <key>`, or `?key=<key>` in the URL (useful for `<img src>` / download links). Every endpoint needs it except `/health`. |
| Body | JSON (`Content-Type: application/json`). |
| Time limit | RunPod's proxy cuts any request at about **100 s**. Anything slower (all video) must use `"wait": false` and poll. |
| Queue | One GPU, one queue. Jobs run one at a time in order, so a 5-minute video delays images queued behind it. |
| CORS | None. Call the API from your backend, never from a browser, because that would expose the key. |

In the examples below, `$BASE` and `$KEY` stand for the base URL and the API key.

---

## Generating

### 1. Submit: `POST /run`

```bash
curl -X POST "$BASE/run" -H "X-API-Key: $KEY" -H "Content-Type: application/json" -d '{
  "name": "wan_i2v",
  "prompt": "The woman turns her head and smiles, gentle wind in her hair",
  "image": "https://example.com/portrait.jpg",
  "length": 81,
  "wait": false
}'
```

| Field | Meaning |
|---|---|
| `name` | Registered workflow to run (see `GET /workflows`). |
| *input keys* | `prompt`, `negative_prompt`, `seed`, `width`, `height`, `steps`, ... Each workflow has its own list (see `GET /workflows/{name}`). Unknown keys aren't an error; they come back in `ignored_keys`. |
| `image`, `image_2`, ... | Image inputs, filled in node order. `images: [..]` also works. `video`/`video_2` and `audio`/`audio_2` work the same way. Each value is an **https URL** the pod can download, a **data URL** (`data:image/png;base64,...`), **raw base64**, or the name of a file already in ComfyUI's `input/` folder. |
| `seed` | Optional. A random seed is used if you leave it out, and the response always shows the seed that was used. |
| `overrides` | Optional raw edits for anything not exposed as an input: `{"<node_id>.<input>": value}`, e.g. `{"3.sampler_name": "euler"}`. Node ids are in `GET /workflows/{name}/api`. |
| `wait` | Default `true`: the server waits up to `timeout` seconds and returns the files. Use `false` for anything slow: it returns right away with a `prompt_id` to poll. |
| `timeout` | Seconds to wait when `wait` is true (default 90; keep it under 100 because of the proxy). |
| `return_base64` | `true` adds a `base64` field to each file, so you don't have to download it. |

Instead of `name` you can also send `"template": "<template name>"`, which installs a built-in template on
first use, or `"workflow": {...}`, which runs a full workflow JSON once without registering it (its models must already be on the pod).

**Response with `"wait": false`:**
```json
{"prompt_id": "5b4fba2d-...", "applied": ["image -> 56.image", "prompt", "length"],
 "ignored_keys": [], "seed": 157594889799461, "status": "queued", "poll": "/result/5b4fba2d-..."}
```

**Response with `"wait": true`**, when the job finishes within the timeout:
```json
{"prompt_id": "5b4fba2d-...", "applied": ["prompt", "width", "height"], "ignored_keys": [],
 "seed": 157594889799461, "status": "done",
 "files": [{"filename": "z-image-turbo_00004_.png", "subfolder": "", "type": "output",
            "kind": "images", "node": "9", "metadata": {"width": 768, "height": 768},
            "url": "/view?filename=z-image-turbo_00004_.png&subfolder=&type=output"}]}
```
If it isn't finished by then you get `"status": "pending"` plus `poll`, and you continue as below.

### 2. Poll: `GET /result/{prompt_id}`

Poll every 2–3 s for images and every 10 s for video.

| `status` | Meaning |
|---|---|
| `pending` | Queued or running. |
| `done` | Finished. `files` lists the outputs. |
| `error` | ComfyUI failed. `messages` has the reason (e.g. out of memory). |

`/result` only knows jobs from ComfyUI's current session. If the pod restarts, older ids stay `pending`
forever, so give your polling loop a timeout (e.g. 15 min for video).

### 3. Download: `GET /view?...`

Fetch `$BASE + file.url` with the API key, either as the header or by appending `&key=$KEY`.
Images are PNG. Videos are MP4 (H.264) with `subfolder=video` and `metadata` holding `width`, `height`, `duration` and `fps`.
**Store results on your side**, because the pod's disk is not permanent.

---

## Installed workflows

| Name | Does | Inputs (defaults) | Time on A40 |
|---|---|---|---|
| `z_image_turbo` | Text to image (Z-Image Turbo) | `prompt`, `width` 1024, `height` 1024, `steps` 8, `cfg` 1, `seed`, `batch_size` 1, `denoise` 1 | ~8 s |
| `flux2_klein_edit` | Image + prompt → edited image (FLUX.2 Klein 4B) | **`image`**, `prompt`, `megapixels` 1, `steps` 4, `seed`, `batch_size` | ~7 s |
| `wan_t2v` | Text to video (Wan 2.2 5B) | `prompt`, `negative_prompt`, `width` 1280, `height` 704, `length` 121, `fps` 24, `steps` 20, `cfg` 5, `seed`, `denoise`, `batch_size` | ~5 min |
| `wan_i2v` | Image + prompt → video (Wan 2.2 5B) | **`image`**, `prompt`, `negative_prompt`, `megapixels` 0.9, `length` 121, `fps` 24, `steps` 20, `cfg` 5, `seed`, `denoise`, `batch_size` | ~5 min |

- Edit and image-to-video outputs keep the input image's aspect ratio. `megapixels` sets the size.
- Wan `length` is the frame count and must be 4n+1. At 24 fps, 49 frames is about 2 s, 81 is about 3.4 s and 121 is about 5 s. Fewer frames render faster.

---

## Adding a new workflow

The manager stores every workflow in **API format**: a JSON object of nodes, keyed by node id.
```json
{"6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["38", 0]}}}
```
A value of `["38", 0]` is a link (output 0 of node 38). Ids inside subgraphs look like `"75:80"`.

### Recipe A: a built-in ComfyUI template (easiest)

ComfyUI ships hundreds of ready-made workflows (Wan, LTX, Hunyuan, Flux, Qwen, ...).

```bash
# 1. find one. Skip names starting with api_ : those call paid cloud services
curl "$BASE/templates?q=wan" -H "X-API-Key: $KEY"

# 2. install it under your own name
curl -X POST "$BASE/templates/video_wan2_2_14B_i2v/install" -H "X-API-Key: $KEY" \
     -H "Content-Type: application/json" -d '{"as": "wan14_i2v"}'
# -> {"workflow": "wan14_i2v", "job_id": "03189faa412b", "status_url": "/jobs/03189faa412b"}

# 3. wait for the setup job (custom nodes + model downloads, can take minutes)
curl "$BASE/jobs/03189faa412b" -H "X-API-Key: $KEY"
#    status queued -> running -> done | error. Check that "failed", "missing_node_types" and
#    "missing_models_no_url" are empty.

# 4. see which inputs were detected
curl "$BASE/workflows/wan14_i2v" -H "X-API-Key: $KEY"

# 5. test
curl -X POST "$BASE/run" -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"name": "wan14_i2v", "prompt": "...", "image": "https://...", "wait": false}'
```

Check the result of step 4 before you rely on the workflow:
- **An input is missing.** Add it with a mapping (see below).
- **There are two `prompt` or two `seed` entries.** The template contains more than one pipeline (e.g. edit and text-to-image). Trim it with Recipe C.
- **There's no `images` entry although the template is image-to-video.** The image loader is *bypassed* in the template, so conversion dropped it. Rebuild it with Recipe C (this is how `wan_i2v` was made), or enable the node in the ComfyUI editor and save it (Recipe B).

### Recipe B: your own workflow from the ComfyUI editor

Build and test it in the editor (`https://<pod-id>-8188.proxy.runpod.net`), then either:
- **Save it in the editor** (Workflow → Save as `my_name`). The manager picks up saved workflows automatically within about 10 s and registers them under the file name. Or
- **Send the file**: `POST /workflows/my_name` with the saved JSON (normal Save / Export, *not* "Export (API)") as the body.

Custom nodes in it are installed and models are downloaded *if* the workflow has their URLs
(templates do; for your own loaders, add them as shown in Recipe C under `models`).

### Recipe C: an API-format workflow or "bundle" (full control)

`POST /workflows/{name}` also accepts a **bundle**:
```json
{
  "api":     { "...API-format workflow (Export (API), or GET /workflows/<x>/api, edited)..." },
  "models":  [{"name": "wan2.2_vae.safetensors", "directory": "vae",
               "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors"}],
  "mapping": {"megapixels": ["60.megapixels"]},
  "ui":      { "...optional: the editor version, only needed to auto-install custom nodes..." }
}
```
- `models`: files to download if missing. `directory` is the folder under `ComfyUI/models` (`diffusion_models`, `text_encoders`, `vae`, `loras`, `checkpoints`, ...). Gated Hugging Face models work once you've accepted their license with the account of the pod's HF token.
- With only `api` and no `ui`, custom nodes can't be auto-installed; core nodes are fine.
- With both, the `api` graph is what runs; from the `ui` only the download URLs of models the `api` uses are taken.
- To modify an existing workflow: `GET /workflows/<x>/api`, edit the JSON (delete nodes, add nodes, change defaults), then POST it as a bundle under a new name.

Registering is idempotent: POSTing the same name again replaces the workflow. `GET /workflows/{name}/bundle`
returns exactly this format. It's also how workflows move to the next pod: whenever a workflow becomes ready,
or its mapping changes, the manager saves its bundle to the network volume (`/workspace/comfy-kit/workflows/`),
and a new pod registers everything found there (see RUNPOD_SETUP.md). `DELETE` removes it there too.

### Inputs and mappings

Inputs are detected automatically by node type:

| Key | Detected from |
|---|---|
| `prompt` / `negative_prompt` | text-encoder nodes (polarity is followed to the sampler's positive/negative input) |
| `image`, `video`, `audio` | `LoadImage`, `LoadVideo`, `LoadAudio` (+ VHS loaders); 2nd and later ones become `image_2`, ... |
| `seed` | `seed` / `noise_seed` on any node |
| `width`, `height`, `length`, `batch_size` | latent / "Empty..." / "...ToVideo" nodes |
| `steps`, `cfg`, `denoise` | sampler / scheduler nodes |
| `guidance`, `fps` | guidance nodes, video save nodes |

For anything else, add a **mapping**: an input name of your choice pointing to one or more `node_id.input` slots.
```bash
curl -X PUT "$BASE/workflows/wan14_i2v/mapping" -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"megapixels": ["60.megapixels"], "lora_strength": ["101.strength_model", "102.strength_model"]}'
```
- A mapping key overrides an auto-detected one of the same name (e.g. point `prompt` to a different node).
- `PUT` replaces the whole mapping, so always send all your custom keys.
- To declare media inputs use the plural key (`{"images": ["12.image", "15.image"]}`). Callers then send `image`, `image_2`.

---

## Endpoint reference

| Method & path | Body / params | Returns |
|---|---|---|
| `GET /health` | (no auth) | `{"comfyui_running": true, "setup_running": false}` |
| **Generation** | | |
| `POST /run` | see [above](#1-submit-post-run) | queued / done / pending; **202** `{"status":"setting_up","job_id"}` if the workflow is still installing |
| `GET /result/{prompt_id}` | `?return_base64=true` | `{"status":"pending\|done\|error","files":[...],"messages"?}` |
| `POST /cancel/{prompt_id}` | | `{"status":"removed\|interrupted\|not_queued"}`. Removes a queued job or stops a running one (its `/result` then says `error`). A removed job's `/result` stays `pending`. |
| `GET /view` | `filename`, `subfolder`, `type` (`output`/`input`) | the file bytes |
| **Workflows** | | |
| `GET /workflows` | | `[{"name","ready","inputs":[...]}]` |
| `GET /workflows/{name}` | | `{"name","inputs":{key:[{"node","input","class_type","default"}]},"example_request"}` |
| `GET /workflows/{name}/api` | | the API-format JSON |
| `GET /workflows/{name}/bundle` | | `{"api","models","ui"?,"mapping"?,"models_without_url"}`, which can be POSTed back as is |
| `POST /workflows/{name}` | UI workflow, API workflow, or bundle | `{"workflow","job_id","status_url"}` |
| `PUT /workflows/{name}/mapping` | `{"key": ["node.input", ...]}` | `{"inputs":[...]}` |
| `DELETE /workflows/{name}` | | `{"deleted","files"}`, also removes its bundle from the volume. Model files stay (others may share them). |
| `POST /convert` | UI workflow | the API-format version (no registering) |
| **Templates** | | |
| `GET /templates` | `?q=search&refresh=true` | `[{"name","title","description","media_type","category","installed"}]` |
| `POST /templates/{template}/install` | `{"as": "my_name", "mapping"?: {...}}` (optional) | `{"workflow","job_id","status_url"}` |
| **Setup jobs** | | |
| `GET /jobs` | | `[{"id","name","source","status"}]` |
| `GET /jobs/{id}` | | `{"status","progress":{model: %},"downloaded","failed","missing_node_types","missing_models_no_url","inputs","log"}` |

Workflow names: letters, digits, `_ - .`, up to 120 characters. Prefer `lower_snake_case`.

---

## Client code

### Python
```python
import time
import requests

BASE = "https://<pod-id>-8000.proxy.runpod.net"
HEADERS = {"X-API-Key": "<api key>"}


def generate(name, timeout=900, **inputs):
    """Runs a workflow and returns (list of file bytes, seed). Works for images and videos."""
    job = requests.post(f"{BASE}/run", headers=HEADERS, timeout=60,
                        json={"name": name, "wait": False, **inputs})
    job.raise_for_status()
    job = job.json()
    if "prompt_id" not in job:  # 202 setting_up: models still downloading
        raise RuntimeError(f"not ready: {job}")
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = requests.get(f"{BASE}/result/{job['prompt_id']}", headers=HEADERS, timeout=30).json()
        if res["status"] == "done":
            files = [requests.get(BASE + f["url"], headers=HEADERS, timeout=300).content for f in res["files"]]
            return files, job.get("seed")
        if res["status"] == "error":
            raise RuntimeError(res.get("messages"))
        time.sleep(3)
    raise TimeoutError(job["prompt_id"])


# images, seed = generate("z_image_turbo", prompt="a red fox in the snow", width=1216, height=832)
# edited, _   = generate("flux2_klein_edit", prompt="make it night time", image="https://.../photo.jpg")
# video, _    = generate("wan_i2v", prompt="the camera orbits slowly", image=base64_string, length=81)
# open("out.mp4", "wb").write(video[0])
```

### JavaScript (Node 18+, server-side)
```js
const BASE = "https://<pod-id>-8000.proxy.runpod.net";
const KEY = process.env.COMFY_API_KEY;
const headers = { "X-API-Key": KEY, "Content-Type": "application/json" };

async function generate(name, inputs = {}, timeoutMs = 15 * 60_000) {
  const job = await (await fetch(`${BASE}/run`, {
    method: "POST", headers, body: JSON.stringify({ name, wait: false, ...inputs }),
  })).json();
  if (!job.prompt_id) throw new Error(`not ready: ${JSON.stringify(job)}`);
  const end = Date.now() + timeoutMs;
  while (Date.now() < end) {
    const res = await (await fetch(`${BASE}/result/${job.prompt_id}`, { headers })).json();
    if (res.status === "done") return res.files.map((f) => `${BASE}${f.url}&key=${KEY}`);
    if (res.status === "error") throw new Error(JSON.stringify(res.messages));
    await new Promise((r) => setTimeout(r, 3000));
  }
  throw new Error(`timeout ${job.prompt_id}`);
}
// const [url] = await generate("wan_t2v", { prompt: "a koi pond in the rain", length: 81 });
// Those URLs contain the key: download them on your server, don't hand them to browsers.
```

---

## Errors and gotchas

| HTTP | When |
|---|---|
| 401 | Wrong or missing API key. |
| 400 | Bad JSON, a media input on a workflow without one, too many images, a bad `overrides` key, or ComfyUI rejecting a value (the `detail` field has ComfyUI's message, e.g. a width that's not allowed). |
| 404 | Unknown workflow, template or file. |
| 202 | `/run` on a workflow whose setup job is still running. Retry later. |
| 502/504 | RunPod proxy: the pod is restarting, or a request took longer than ~100 s (use `"wait": false`). |

- Check `ignored_keys` in the response. A misspelled input name doesn't fail, it just gets ignored.
- A workflow saved in the ComfyUI editor under the name of an API-registered workflow **replaces** it. Use different names.
- Outputs pile up on the pod (`ComfyUI/output`). Download them and treat the pod as temporary.
- Don't publish the port-8188 URL (the ComfyUI editor): it has no password. Only port 8000 checks the key.
