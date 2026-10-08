#!/usr/bin/env python3
"""
ComfyUI headless manager for RunPod  (v2)

- Runs ComfyUI as a child process (restarts it after new custom nodes are installed)
- Register workflows by API or from built-in templates -> missing custom nodes are installed,
  models downloaded, and the workflow is converted to API format server-side (no UI needed)
- Auto-detects the workflow's inputs (prompt, negative prompt, images, video, audio, seed,
  size, steps...) so you can call /run with {"prompt": "...", "image": "<url|base64>"}
- Still watches workflows saved in the ComfyUI UI, if you ever use it

Needs the custom node "comfyui-workflow-to-api-converter-endpoint" (start.sh installs it).
"""
import base64
import json
import mimetypes
import os
import random
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlencode, urlparse

import requests
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response

WS = Path(os.environ.get("WORKSPACE", "/workspace"))
COMFY_DIR = Path(os.environ.get("COMFY_DIR", WS / "ComfyUI"))
MODELS_DIR = COMFY_DIR / "models"
INPUT_DIR = COMFY_DIR / "input"
CUSTOM_NODES = COMFY_DIR / "custom_nodes"
UI_WF_DIR = COMFY_DIR / "user" / "default" / "workflows"
STORE = WS / "workflow_store"
# Every ready workflow is also saved there as <name>.json (a bundle, see GET /workflows/{name}/bundle).
# start.sh points it at the network volume, so a new pod re-creates all of them.
BUNDLE_DIR = Path(os.environ["BUNDLE_DIR"]) if os.environ.get("BUNDLE_DIR") else None
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
API_PORT = int(os.environ.get("MANAGER_PORT", "8000"))
COMFY = f"http://127.0.0.1:{COMFY_PORT}"
API_KEY = os.environ.get("MANAGER_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CIVITAI_TOKEN = os.environ.get("CIVITAI_TOKEN", "")
COMFY_CLI = str(Path(sys.executable).parent / "comfy")

MODEL_EXT = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft", ".onnx")
UI_ONLY_NODES = {"Note", "MarkdownNote", "Reroute", "PrimitiveNode", "GroupNode"}
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
NAME_RE = re.compile(r"[\w\-. ]{1,120}")

# input auto-detection
TEXT_KEYS = ("text", "prompt", "text_g", "text_l", "t5xxl", "clip_l")
MEDIA_LOADERS = {  # class_type -> (kind, input name)
    "LoadImage": ("images", "image"),
    "LoadImageMask": ("images", "image"),
    "LoadVideo": ("videos", "file"),
    "VHS_LoadVideo": ("videos", "video"),
    "LoadAudio": ("audios", "audio"),
    "VHS_LoadAudioUpload": ("audios", "audio"),
}
MEDIA_SINGLE = {"images": "image", "videos": "video", "audios": "audio"}
SIZE_KEYS = ("width", "height", "length", "batch_size")
CONTROL_KEYS = {"name", "template", "workflow", "overrides", "wait", "timeout",
                "return_base64", "images", "videos", "audios", "image", "video", "audio"}

STORE.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="ComfyUI headless manager")
jobs: dict = {}
setup_lock = threading.Lock()
seen_files: dict = {}


# =========================================================== ComfyUI process
class Comfy:
    proc = None
    lock = threading.Lock()

    @classmethod
    def start(cls):
        cls.proc = subprocess.Popen(
            [sys.executable, "main.py", "--listen", "0.0.0.0", "--port", str(COMFY_PORT)],
            cwd=COMFY_DIR)
        cls.wait_ready()

    @classmethod
    def wait_ready(cls, timeout=900):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if requests.get(f"{COMFY}/system_stats", timeout=3).ok:
                    return True
            except requests.RequestException:
                pass
            time.sleep(2)
        return False

    @classmethod
    def wait_idle(cls, timeout=3600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                q = requests.get(f"{COMFY}/queue", timeout=5).json()
                if not q.get("queue_running") and not q.get("queue_pending"):
                    return
            except requests.RequestException:
                return
            time.sleep(5)

    @classmethod
    def restart(cls):
        with cls.lock:
            if cls.proc and cls.proc.poll() is None:
                cls.proc.terminate()
                try:
                    cls.proc.wait(30)
                except subprocess.TimeoutExpired:
                    cls.proc.kill()
            cls.start()


def available_node_types() -> set:
    try:
        return set(requests.get(f"{COMFY}/object_info", timeout=120).json())
    except requests.RequestException:
        return set()


def convert_ui_to_api(ui: dict) -> dict:
    """Server-side 'Export (API)' via the converter custom node."""
    r = requests.post(f"{COMFY}/workflow/convert", json=ui, timeout=120)
    if r.status_code == 404:
        raise RuntimeError("converter node not installed (comfyui-workflow-to-api-converter-endpoint)")
    r.raise_for_status()
    return r.json()


# =========================================================== workflow parsing
def is_ui(wf) -> bool:
    return isinstance(wf, dict) and isinstance(wf.get("nodes"), list)


def is_api(wf) -> bool:
    return isinstance(wf, dict) and bool(wf) and all(
        isinstance(v, dict) and "class_type" in v for v in wf.values())


def scan(obj, models: dict, files: set, types: set):
    """Collect model download info, referenced model files and node types (incl. subgraphs)."""
    if isinstance(obj, dict):
        if isinstance(obj.get("models"), list):
            for m in obj["models"]:
                if isinstance(m, dict) and m.get("name") and m.get("url"):
                    models[m["name"]] = m
        if isinstance(obj.get("type"), str) and "id" in obj and "mode" in obj:
            types.add(obj["type"])
        if isinstance(obj.get("class_type"), str):
            types.add(obj["class_type"])
        for k, v in obj.items():
            if k != "models":
                scan(v, models, files, types)
    elif isinstance(obj, list):
        for v in obj:
            scan(v, models, files, types)
    elif isinstance(obj, str) and obj.lower().endswith(MODEL_EXT):
        files.add(obj.replace("\\", "/"))


def model_index() -> set:
    idx = set()
    for p in MODELS_DIR.rglob("*"):
        if p.is_file():
            rel = p.relative_to(MODELS_DIR)
            idx.add(p.name)
            if len(rel.parts) > 1:
                idx.add("/".join(rel.parts[1:]))
    return idx


def missing_types(types: set, available: set) -> list:
    return sorted(t for t in types
                  if t not in available and t not in UI_ONLY_NODES and not UUID_RE.match(t))


def is_link(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[1], int) and isinstance(v[0], (str, int))


def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def id_key(nid: str):
    return tuple(int(p) if p.isdigit() else 0 for p in str(nid).split(":"))


def detect_schema(wf: dict) -> dict:
    """Find the user-facing inputs of an API-format workflow."""
    consumers = defaultdict(list)
    for nid, node in wf.items():
        for inp, v in node.get("inputs", {}).items():
            if is_link(v):
                consumers[str(v[0])].append((nid, inp))

    def polarity(start):
        labels, seen, queue = set(), {start}, [start]
        while queue:
            n = queue.pop(0)
            for c, inp in consumers.get(n, []):
                il = inp.lower()
                if "negative" in il:
                    labels.add("negative")
                elif "positive" in il:
                    labels.add("positive")
                elif c not in seen:
                    seen.add(c)
                    queue.append(c)
        return "negative_prompt" if labels == {"negative"} else "prompt"

    def literal_slot(nid, inp):
        v = wf[nid]["inputs"].get(inp)
        if isinstance(v, str):
            return [nid, inp]
        if is_link(v):  # e.g. a PrimitiveString node feeding the encoder
            src = str(v[0])
            s_in = wf.get(src, {}).get("inputs", {})
            for k in ("value", "text", "string", "prompt"):
                if isinstance(s_in.get(k), str):
                    return [src, k]
        return None

    schema = defaultdict(list)

    def add(key, slot):
        if slot and slot not in schema[key]:
            schema[key].append(slot)

    for nid, node in sorted(wf.items(), key=lambda kv: id_key(kv[0])):
        ct, ins = node.get("class_type", ""), node.get("inputs", {})
        if ct in MEDIA_LOADERS:
            kind, inp = MEDIA_LOADERS[ct]
            add(kind, [nid, inp])
            continue
        tkeys = [k for k in TEXT_KEYS if k in ins]
        if tkeys and ("clip" in ins or "Encode" in ct):
            key = polarity(nid)
            for k in tkeys:
                add(key, literal_slot(nid, k))
        for k in ("seed", "noise_seed"):
            if is_num(ins.get(k)):
                add("seed", [nid, k])
        if re.search(r"Latent|ToVideo|Empty", ct):
            for k in SIZE_KEYS:
                if is_num(ins.get(k)):
                    add(k, [nid, k])
        if "Sampler" in ct or "Scheduler" in ct:
            for k in ("steps", "cfg", "denoise"):
                if is_num(ins.get(k)):
                    add(k, [nid, k])
        if "Guidance" in ct and is_num(ins.get("guidance")):
            add("guidance", [nid, "guidance"])
        for k in ("fps", "frame_rate"):
            if is_num(ins.get(k)):
                add("fps", [nid, k])
    return dict(schema)


def describe_schema(wf: dict, schema: dict) -> dict:
    out = {}
    for key, slots in schema.items():
        out[key] = [{"node": s[0], "input": s[1], "class_type": wf.get(s[0], {}).get("class_type"),
                     "default": wf.get(s[0], {}).get("inputs", {}).get(s[1])} for s in slots]
    return out


def load_schema(name: str) -> dict:
    p = STORE / f"{name}.schema.json"
    schema = json.loads(p.read_text()) if p.exists() else {}
    m = STORE / f"{name}.map.json"  # user-defined mapping wins
    if m.exists():
        for key, slots in json.loads(m.read_text()).items():
            schema[key] = [s.split(".", 1) if isinstance(s, str) else s for s in slots]
    return schema


# =========================================================== built-in templates
_templates_cache = None


def scan_templates():
    global _templates_cache
    files, meta = {}, {}
    for base in map(Path, sys.path):
        if not base.is_dir():
            continue
        for pkg in base.glob("comfyui_workflow_templates*"):
            for tdir in pkg.rglob("templates"):
                if not tdir.is_dir():
                    continue
                for idx in sorted(tdir.glob("index*.json"), key=lambda p: p.name != "index.json"):
                    try:
                        data = json.loads(idx.read_text())
                        cats = data if isinstance(data, list) else data.get("categories", [])
                        for cat in cats:
                            for t in cat.get("templates", []):
                                if t.get("name") and t["name"] not in meta:
                                    meta[t["name"]] = {"title": t.get("title"),
                                                       "description": t.get("description"),
                                                       "media_type": t.get("mediaType"),
                                                       "category": cat.get("title")}
                    except Exception:
                        pass
                for f in tdir.glob("*.json"):
                    if not f.name.startswith("index"):
                        files[f.stem] = f
    _templates_cache = (files, meta)
    return _templates_cache


def get_template(name: str):
    files, _ = _templates_cache or scan_templates()
    if name in files:
        return json.loads(files[name].read_text())
    try:  # fallback: ComfyUI may serve them over HTTP
        r = requests.get(f"{COMFY}/templates/{name}.json", timeout=20)
        if r.ok and is_ui(r.json()):
            return r.json()
    except Exception:
        pass
    return None


# =========================================================== setup job
def log(job, msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(f"[setup:{job['name']}] {line}", flush=True)
    job["log"].append(line)


def download(job, name, url, dest: Path):
    headers = {}
    if "huggingface.co" in url and HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    if "civitai.com" in url and CIVITAI_TOKEN:
        url += ("&" if "?" in url else "?") + f"token={CIVITAI_TOKEN}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    log(job, f"↓ downloading {name} -> {dest.relative_to(MODELS_DIR)}")
    with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        total, done, last = int(r.headers.get("content-length", 0)), 0, 0.0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and time.time() - last > 3:
                    job["progress"][name] = round(100 * done / total, 1)
                    last = time.time()
    tmp.rename(dest)
    job["progress"][name] = 100.0
    log(job, f"✓ {name} done ({done / 1e9:.2f} GB)")


def install_nodes(job, ui: dict) -> list:
    CUSTOM_NODES.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in CUSTOM_NODES.iterdir()}
    tmp = STORE / f".tmp-{job['id']}.json"
    tmp.write_text(json.dumps(ui))
    try:
        r = subprocess.run([COMFY_CLI, "--skip-prompt", f"--workspace={COMFY_DIR}",
                            "node", "install-deps", f"--workflow={tmp}"],
                           capture_output=True, text=True, timeout=3600)
        log(job, "install-deps output:\n" + (r.stdout + r.stderr)[-2000:].strip())
    finally:
        tmp.unlink(missing_ok=True)
    return sorted({p.name for p in CUSTOM_NODES.iterdir()} - before)


def setup_workflow(job, ui, api, extra_models):
    name = job["name"]
    with setup_lock:
        job["status"] = "running"
        try:
            Comfy.wait_ready()
            models, files, types = {}, set(), set()
            scan(api or ui, models, files, types)
            if ui and api:  # api is what runs: from the ui take only the download URLs of files api uses
                ui_models, used = {}, {Path(f).name for f in files}
                scan(ui, ui_models, set(), set())
                models.update({n: m for n, m in ui_models.items() if n in used and n not in models})
            for m in extra_models or []:
                if m.get("name") and m.get("url"):
                    models[m["name"]] = m

            # 1) custom nodes
            missing = missing_types(types, available_node_types())
            new_nodes = []
            if missing and ui:
                log(job, f"missing node types: {missing} -> installing custom nodes")
                new_nodes = install_nodes(job, ui)
            elif missing:
                log(job, "missing node types but no UI-format workflow given; can't resolve packs")

            # 2) models
            have = model_index()
            for mname, m in models.items():
                directory = m.get("directory") or "checkpoints"
                dest = (MODELS_DIR / directory / mname).resolve()
                if not str(dest).startswith(str(MODELS_DIR.resolve())):
                    log(job, f"skipping unsafe path {directory}/{mname}")
                    continue
                if dest.exists() or mname in have:
                    log(job, f"✓ {mname} already present")
                    continue
                try:
                    download(job, mname, m["url"], dest)
                    job["downloaded"].append(mname)
                except Exception as e:
                    job["failed"].append({"name": mname, "error": repr(e)})
                    log(job, f"✗ {mname} failed: {e!r}")
            have = model_index()
            job["missing_models_no_url"] = sorted(
                f for f in files if f not in have and Path(f).name not in have)

            # 3) restart so ComfyUI loads new nodes
            if new_nodes:
                log(job, f"new custom nodes {new_nodes}; restarting ComfyUI when idle")
                Comfy.wait_idle()
                Comfy.restart()
            job["missing_node_types"] = missing_types(types, available_node_types())

            # 4) API format: converted server-side unless you supplied one
            if api is None and ui is not None:
                try:
                    api = convert_ui_to_api(ui)
                    log(job, f"converted to API format ({len(api)} nodes)")
                except Exception as e:
                    log(job, f"✗ API conversion failed: {e!r}")
            if api:
                (STORE / f"{name}.api.json").write_text(json.dumps(api))
                schema = detect_schema(api)
                (STORE / f"{name}.schema.json").write_text(json.dumps(schema))
                job["inputs"] = describe_schema(api, load_schema(name))
                log(job, f"detected inputs: {sorted(job['inputs'])}")
                save_bundle(name, job)
            job["status"] = "done" if api else "error"
            log(job, "setup finished" if api else "setup finished WITHOUT a runnable API workflow")
        except Exception as e:
            job["status"] = "error"
            log(job, f"ERROR: {e!r}")


def active_job(name):
    for j in jobs.values():
        if j["name"] == name and j["status"] in ("queued", "running"):
            return j
    return None


def start_job(name, ui=None, api=None, models=None, source="api") -> dict:
    existing = active_job(name)
    if existing:
        return existing
    jid = uuid.uuid4().hex[:12]
    job = {"id": jid, "name": name, "source": source, "status": "queued", "log": [],
           "downloaded": [], "failed": [], "progress": {}, "created": time.time()}
    jobs[jid] = job
    threading.Thread(target=setup_workflow, args=(job, ui, api, models), daemon=True).start()
    return job


def register(name, ui=None, api=None, models=None, mapping=None, source="api") -> dict:
    if not NAME_RE.fullmatch(name):
        raise HTTPException(400, "invalid workflow name")
    if ui is not None:
        (STORE / f"{name}.ui.json").write_text(json.dumps(ui))
        UI_WF_DIR.mkdir(parents=True, exist_ok=True)
        ui_path = UI_WF_DIR / f"{name}.json"
        ui_path.write_text(json.dumps(ui))
        seen_files[str(ui_path)] = ui_path.stat().st_mtime
    if api is not None:
        if not is_api(api):
            raise HTTPException(400, "'api' must be an API-format workflow")
        (STORE / f"{name}.api.json").write_text(json.dumps(api))
    if mapping:
        (STORE / f"{name}.map.json").write_text(json.dumps(mapping))
    if models:  # kept so GET /workflows/{name}/bundle can re-create it on another pod
        (STORE / f"{name}.models.json").write_text(json.dumps(models))
    return start_job(name, ui, api, models, source)


# =========================================================== UI watcher
def watch_ui_workflows():
    if UI_WF_DIR.exists():
        for p in UI_WF_DIR.rglob("*.json"):
            seen_files[str(p)] = p.stat().st_mtime
    while True:
        time.sleep(10)
        if Comfy.proc and Comfy.proc.poll() is not None and not Comfy.lock.locked():
            print("[manager] ComfyUI exited, restarting", flush=True)
            threading.Thread(target=Comfy.restart, daemon=True).start()
        if not UI_WF_DIR.exists():
            continue
        for p in UI_WF_DIR.rglob("*.json"):
            try:
                mt = p.stat().st_mtime
            except FileNotFoundError:
                continue
            if seen_files.get(str(p)) == mt or time.time() - mt < 3:
                continue
            seen_files[str(p)] = mt
            try:
                wf = json.loads(p.read_text())
            except Exception:
                continue
            if is_ui(wf) and NAME_RE.fullmatch(p.stem):
                (STORE / f"{p.stem}.ui.json").write_text(json.dumps(wf))
                start_job(p.stem, ui=wf, source="ui-save")


# =========================================================== media inputs
def sniff_ext(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return ".wav"
    if data[:4] == b"GIF8":
        return ".gif"
    if data[4:8] == b"ftyp":
        return ".mp4"
    if data[:4] == b"\x1aE\xdf\xa3":
        return ".webm"
    if data[:3] == b"ID3" or data[:2] == b"\xff\xfb":
        return ".mp3"
    if data[:4] == b"fLaC":
        return ".flac"
    return ".bin"


def media_to_input(value: str) -> str:
    """URL / data-URL / raw base64 / existing input filename -> filename in ComfyUI/input."""
    if not isinstance(value, str) or not value:
        raise HTTPException(400, "media inputs must be a URL, base64 string or input filename")
    if len(value) < 260 and "://" not in value and not value.startswith("data:") \
            and (INPUT_DIR / value).is_file():
        return value
    ext = ""
    if value.startswith(("http://", "https://")):
        r = requests.get(value, timeout=300)
        r.raise_for_status()
        data = r.content
        ext = Path(urlparse(value).path).suffix.lower()
    else:
        if value.startswith("data:"):
            header, value = value.split(",", 1)
            ext = mimetypes.guess_extension(header[5:].split(";")[0]) or ""
        try:
            data = base64.b64decode(value, validate=False)
        except Exception:
            raise HTTPException(400, "could not decode base64 media")
    if not ext or ext in (".jpe", ".bin"):
        ext = sniff_ext(data)
    fname = f"api_{uuid.uuid4().hex[:12]}{ext}"
    r = requests.post(f"{COMFY}/upload/image", files={"image": (fname, data)},
                      data={"overwrite": "true", "type": "input"}, timeout=300)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def apply_inputs(wf: dict, schema: dict, body: dict) -> dict:
    report = {"applied": [], "ignored_keys": []}

    def set_slots(slots, value):
        for nid, inp in slots:
            wf[nid]["inputs"][inp] = value

    # media: "image" / "image_2" / "images": [...]
    for kind, single in MEDIA_SINGLE.items():
        slots = schema.get(kind, [])
        vals = list(body.get(kind) or [])
        if not vals:
            vals = [body.get(single if i == 0 else f"{single}_{i + 1}") for i in range(max(len(slots), 1))]
        given = [v for v in vals if v]
        if given and not slots:
            raise HTTPException(400, f"this workflow has no {kind[:-1]} input")
        if len(vals) > len(slots) and any(vals[len(slots):]):
            raise HTTPException(400, f"workflow takes {len(slots)} {kind}, got more")
        for slot, v in zip(slots, vals):
            if v:
                set_slots([slot], media_to_input(v))
                report["applied"].append(f"{single} -> {slot[0]}.{slot[1]}")

    # seed: random each call unless given (otherwise ComfyUI may just return the cached result)
    if schema.get("seed"):
        seed = int(body["seed"]) if "seed" in body else random.randint(0, 2 ** 50)
        set_slots(schema["seed"], seed)
        report["seed"] = seed

    # everything else: prompt, negative_prompt, width, steps, ... and custom mapped keys
    for key, value in body.items():
        if key in CONTROL_KEYS or key == "seed" or re.fullmatch(r"(image|video|audio)_\d+", key):
            continue
        if key in schema:
            set_slots(schema[key], value)
            report["applied"].append(key)
        else:
            report["ignored_keys"].append(key)

    # raw overrides: {"6.text": "..."}
    for key, val in (body.get("overrides") or {}).items():
        try:
            nid, inp = key.split(".", 1)
            wf[nid]["inputs"][inp] = val
        except (ValueError, KeyError):
            raise HTTPException(400, f"bad override '{key}' (use 'node_id.input_name')")
    return report


# =========================================================== HTTP API
def auth(x_api_key: str = Header(default=""), key: str = ""):
    if API_KEY and API_KEY not in (x_api_key, key):
        raise HTTPException(401, "invalid API key")


@app.get("/health")
def health():
    return {"comfyui_running": bool(Comfy.proc and Comfy.proc.poll() is None),
            "setup_running": setup_lock.locked()}


# ---- templates
@app.get("/templates", dependencies=[Depends(auth)])
def list_templates(refresh: bool = False, q: str = ""):
    files, meta = scan_templates() if refresh or not _templates_cache else _templates_cache
    out = []
    for n in sorted(files):
        m = meta.get(n, {})
        if q and q.lower() not in f"{n} {m.get('title', '')} {m.get('description', '')}".lower():
            continue
        out.append({"name": n, **m, "installed": (STORE / f"{n}.api.json").exists()})
    return out


@app.post("/templates/{template}/install", dependencies=[Depends(auth)])
def install_template(template: str, body: dict = None):
    ui = get_template(template)
    if ui is None:
        raise HTTPException(404, "template not found (see GET /templates)")
    name = (body or {}).get("as") or template
    job = register(name, ui=ui, mapping=(body or {}).get("mapping"), source=f"template:{template}")
    return {"workflow": name, "job_id": job["id"], "status_url": f"/jobs/{job['id']}"}


# ---- workflows
@app.post("/workflows/{name}", dependencies=[Depends(auth)])
def add_workflow(name: str, body: dict):
    """Body: UI-format workflow (normal Save) | API-format workflow |
    {"ui": {...}, "api": {...}, "models": [{"name","url","directory"}], "mapping": {"prompt": ["6.text"]}}"""
    ui, api = body.get("ui"), body.get("api")
    if ui is None and api is None:
        if is_ui(body):
            ui = body
        elif is_api(body):
            api = body
        else:
            raise HTTPException(400, "body is not a ComfyUI workflow")
    job = register(name, ui, api, body.get("models"), body.get("mapping"))
    return {"workflow": name, "job_id": job["id"], "status_url": f"/jobs/{job['id']}"}


@app.get("/workflows", dependencies=[Depends(auth)])
def list_workflows():
    names = sorted({p.name.split(".")[0] for p in STORE.glob("*.json") if not p.name.startswith(".")})
    return [{"name": n, "ready": (STORE / f"{n}.api.json").exists(),
             "inputs": sorted(load_schema(n))} for n in names]


@app.get("/workflows/{name}", dependencies=[Depends(auth)])
def get_workflow(name: str):
    p = STORE / f"{name}.api.json"
    if not p.exists():
        raise HTTPException(404, "workflow not ready")
    wf = json.loads(p.read_text())
    schema = load_schema(name)
    example = {"name": name}
    for k in schema:
        if k in MEDIA_SINGLE:
            example[MEDIA_SINGLE[k]] = "<url or base64>"
        elif k in ("prompt", "negative_prompt"):
            example[k] = "..."
    return {"name": name, "inputs": describe_schema(wf, schema), "example_request": example}


@app.get("/workflows/{name}/api", dependencies=[Depends(auth)])
def export_api(name: str):
    """The API-format export (same as 'Export (API)' in the UI)."""
    p = STORE / f"{name}.api.json"
    if not p.exists():
        raise HTTPException(404, "workflow not ready")
    return JSONResponse(json.loads(p.read_text()))


@app.get("/workflows/{name}/bundle", dependencies=[Depends(auth)])
def export_bundle(name: str):
    """{"api", "ui"?, "mapping"?, "models"}: POST it to /workflows/{name} on another pod to re-create it.
    The ui (when there is one) lets that pod install the custom nodes the workflow needs."""
    p = STORE / f"{name}.api.json"
    if not p.exists():
        raise HTTPException(404, "workflow not ready")
    api = json.loads(p.read_text())
    models, files = {}, set()
    scan(api, {}, files, set())
    for src in (STORE / f"{name}.ui.json", STORE / f"{name}.models.json"):
        if src.exists():
            data = json.loads(src.read_text())
            scan(data if isinstance(data, dict) else {"models": data}, models, set(), set())
    used = {Path(f).name for f in files}
    bundle = {"api": api, "models": [m for n, m in models.items() if n in used]}
    for key, suffix in (("ui", "ui"), ("mapping", "map")):
        f = STORE / f"{name}.{suffix}.json"
        if f.exists():
            bundle[key] = json.loads(f.read_text())
    bundle["models_without_url"] = sorted(used - {Path(n).name for n in models})
    return bundle


def save_bundle(name, job=None):
    if not BUNDLE_DIR:
        return
    try:
        BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
        (BUNDLE_DIR / f"{name}.json").write_text(json.dumps(export_bundle(name)))
        if job:
            log(job, f"saved bundle to {BUNDLE_DIR / name}.json")
    except Exception as e:
        print(f"[bundle] saving {name} failed: {e!r}", flush=True)


@app.delete("/workflows/{name}", dependencies=[Depends(auth)])
def delete_workflow(name: str):
    """Removes the workflow (model files stay on disk, other workflows may share them)."""
    if not NAME_RE.fullmatch(name):
        raise HTTPException(400, "invalid workflow name")
    paths = [STORE / f"{name}.{s}.json" for s in ("api", "ui", "schema", "map", "models")]
    paths.append(UI_WF_DIR / f"{name}.json")
    if BUNDLE_DIR:
        paths.append(BUNDLE_DIR / f"{name}.json")
    removed = [p.name for p in paths if p.exists()]
    if not removed:
        raise HTTPException(404, "unknown workflow")
    for p in paths:
        p.unlink(missing_ok=True)
    return {"deleted": name, "files": removed}


@app.put("/workflows/{name}/mapping", dependencies=[Depends(auth)])
def set_mapping(name: str, mapping: dict):
    """Rename/extend inputs, e.g. {"prompt": ["6.text"], "style": ["44.text"]}"""
    (STORE / f"{name}.map.json").write_text(json.dumps(mapping))
    if (STORE / f"{name}.api.json").exists():
        save_bundle(name)
    return {"inputs": sorted(load_schema(name))}


@app.post("/convert", dependencies=[Depends(auth)])
def convert(body: dict):
    """UI-format workflow in -> API-format workflow out."""
    if not is_ui(body):
        raise HTTPException(400, "expected a UI-format workflow")
    try:
        return JSONResponse(convert_ui_to_api(body))
    except Exception as e:
        raise HTTPException(500, str(e))


# ---- jobs
@app.get("/jobs", dependencies=[Depends(auth)])
def list_jobs():
    return [{k: j[k] for k in ("id", "name", "source", "status")} for j in jobs.values()]


@app.get("/jobs/{jid}", dependencies=[Depends(auth)])
def get_job(jid: str):
    if jid not in jobs:
        raise HTTPException(404, "unknown job")
    j = dict(jobs[jid])
    j["log"] = j["log"][-60:]
    return j


# ---- generation
@app.post("/run", dependencies=[Depends(auth)])
def run(body: dict):
    """{"name": "my_wf" | "template": "<template name>" | "workflow": {...},
        "prompt": "...", "negative_prompt": "...", "image": "<url|base64>", "image_2": ...,
        "video": ..., "audio": ..., "seed": 1, "width": 1024, "steps": 20, ...,
        "overrides": {"6.text": "..."}, "wait": true, "timeout": 90, "return_base64": false}"""
    schema = {}
    if body.get("workflow") is not None:
        wf = body["workflow"]
        if is_ui(wf):
            wf = convert_ui_to_api(wf)
        if not is_api(wf):
            raise HTTPException(400, "'workflow' is not a ComfyUI workflow")
        schema = detect_schema(wf)
    else:
        name = body.get("name") or body.get("template")
        if not name or not NAME_RE.fullmatch(name):
            raise HTTPException(400, "give 'name', 'template' or 'workflow'")
        p = STORE / f"{name}.api.json"
        if not p.exists():
            job = active_job(name)
            if not job and body.get("template"):
                ui = get_template(name)
                if ui is None:
                    raise HTTPException(404, "template not found (see GET /templates)")
                job = register(name, ui=ui, source=f"template:{name}")
            if job:
                return JSONResponse(status_code=202, content={
                    "status": "setting_up", "job_id": job["id"], "status_url": f"/jobs/{job['id']}",
                    "detail": "installing nodes / downloading models; retry when the job is done"})
            raise HTTPException(404, "unknown workflow (register it or pass 'template')")
        wf = json.loads(p.read_text())
        schema = load_schema(name)

    report = apply_inputs(wf, schema, body)
    r = requests.post(f"{COMFY}/prompt", json={"prompt": wf, "client_id": "manager"}, timeout=60)
    if not r.ok:
        raise HTTPException(r.status_code, r.text)
    pid = r.json()["prompt_id"]
    base = {"prompt_id": pid, **report}
    if not body.get("wait", True):
        return {**base, "status": "queued", "poll": f"/result/{pid}"}
    deadline = time.time() + float(body.get("timeout", 90))  # RunPod proxy cuts at ~100 s
    while time.time() < deadline:
        res = result(pid, body.get("return_base64", False))
        if res["status"] != "pending":
            return {**base, **res}
        time.sleep(1)
    return {**base, "status": "pending", "poll": f"/result/{pid}"}


@app.get("/result/{pid}", dependencies=[Depends(auth)])
def result(pid: str, return_base64: bool = False):
    h = requests.get(f"{COMFY}/history/{pid}", timeout=10).json()
    if pid not in h:
        return {"prompt_id": pid, "status": "pending"}
    entry = h[pid]
    st = entry.get("status", {}) or {}
    files = []
    for node_id, out in entry.get("outputs", {}).items():
        for kind, items in out.items():
            if not isinstance(items, list):
                continue
            for f in items:
                if not (isinstance(f, dict) and "filename" in f) or f.get("type") == "temp":
                    continue
                params = {"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                          "type": f.get("type", "output")}
                item = {**f, "node": node_id, "kind": kind, "url": f"/view?{urlencode(params)}"}
                if return_base64:
                    v = requests.get(f"{COMFY}/view", params=params, timeout=300)
                    item["base64"] = base64.b64encode(v.content).decode()
                files.append(item)
    status = "error" if st.get("status_str") == "error" else "done"
    res = {"prompt_id": pid, "status": status, "files": files}
    if status == "error":
        res["messages"] = st.get("messages", [])[-5:]
    return res


@app.post("/cancel/{pid}", dependencies=[Depends(auth)])
def cancel(pid: str):
    """Removes a queued job or interrupts it while it runs (its /result then reports an error)."""
    q = requests.get(f"{COMFY}/queue", timeout=10).json()
    if any(item[1] == pid for item in q.get("queue_pending", [])):
        requests.post(f"{COMFY}/queue", json={"delete": [pid]}, timeout=10).raise_for_status()
        return {"prompt_id": pid, "status": "removed"}
    if any(item[1] == pid for item in q.get("queue_running", [])):
        # Older ComfyUI ignores prompt_id and interrupts whatever runs, which we just checked is pid
        requests.post(f"{COMFY}/interrupt", json={"prompt_id": pid}, timeout=10).raise_for_status()
        return {"prompt_id": pid, "status": "interrupted"}
    return {"prompt_id": pid, "status": "not_queued"}  # already finished, or unknown


@app.get("/view", dependencies=[Depends(auth)])
def view(filename: str, subfolder: str = "", type: str = "output"):
    r = requests.get(f"{COMFY}/view", params={"filename": filename, "subfolder": subfolder,
                                             "type": type}, timeout=300)
    return Response(r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/octet-stream"))


if __name__ == "__main__":
    threading.Thread(target=Comfy.start, daemon=True).start()
    threading.Thread(target=watch_ui_workflows, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
