"""Client for the ComfyUI manager API on the RunPod pod (comfyui/manager.py, see comfyui/API.md).

Workflows are registered on the pod once, then run with simple JSON such as
{"name": "z_image_turbo", "prompt": "..."}. This wraps the calls the voice agent needs and
keeps a cached list of the workflows for the assistant's system prompt.

What each workflow is for comes from a small YAML file (COMFYUI_DESCRIPTIONS), one line per
workflow. Workflows without a line are described from their inputs ("Text to video.", ...).
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml

log = logging.getLogger(__name__)

REFRESH_SECONDS = 60
TEMPLATES_SECONDS = 600
MEDIA_INPUTS = {"images": "image", "videos": "video", "audios": "audio"}
NOT_LISTED = {"seed"}  # random by default; not worth the assistant's attention
EXTENSIONS = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".gif"},
    "video": {".mp4", ".webm", ".mov", ".mkv"},
    "audio": {".wav", ".mp3", ".flac", ".ogg", ".m4a"},
}
SEARCH_STOPWORDS = {"a", "an", "the", "to", "for", "with", "and", "of", "in", "workflow", "template", "model"}


class ComfyError(Exception):
    """A failure with a message the assistant can pass on to the user."""


def file_kind(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return next((kind for kind, exts in EXTENSIONS.items() if ext in exts), "file")


def describe_error(messages) -> str:
    """ComfyUI's execution messages ([[type, data], ...]) -> one readable line."""
    for item in reversed(messages or []):
        if not (isinstance(item, list) and len(item) == 2 and isinstance(item[1], dict)):
            continue
        kind, data = item
        if kind == "execution_interrupted":
            return "it was cancelled"
        if kind == "execution_error":
            lines = str(data.get("exception_message") or "").strip().splitlines()
            return f"ComfyUI failed in {data.get('node_type') or 'a node'}: {lines[0] if lines else 'unknown error'}"
    return "ComfyUI reported an error"


@dataclass
class Workflow:
    name: str
    ready: bool
    inputs: dict[str, object] = field(default_factory=dict)  # input -> default (None if unknown)
    slots: dict[str, int] = field(default_factory=dict)  # input -> how many nodes it sets
    description: str = ""

    @property
    def media_inputs(self) -> list[str]:
        """Media it takes, e.g. ["image"]."""
        return [single for plural, single in MEDIA_INPUTS.items() if plural in self.inputs]

    @property
    def output(self) -> str:
        """What it makes, guessed from its inputs: video workflows have a frame count or fps."""
        return "video" if {"length", "fps"} & set(self.inputs) else "image"

    def about(self) -> str:
        if self.description:
            return self.description
        takes = self.media_inputs
        if "image" in takes:
            return "Image to video." if self.output == "video" else "Edits or transforms an image."
        if takes:
            return f"{takes[0].capitalize()} to {self.output}."
        return f"Text to {self.output}."

    def summary(self) -> str:
        if not self.ready:
            return f"{self.name}: still being installed, not usable yet."
        parts = []
        for key, default in self.inputs.items():
            if key in NOT_LISTED or key in MEDIA_INPUTS:
                continue
            is_num = isinstance(default, (int, float)) and not isinstance(default, bool)
            parts.append(f"{key}={default}" if is_num else key)
        s = f"{self.name}: {self.about()}"
        if self.media_inputs:
            s += f" Needs: {', '.join(self.media_inputs)}."
        return s + (f" Inputs: {', '.join(parts)}." if parts else "")

    def info(self) -> dict:
        # The prompts saved in a workflow are just examples, not defaults worth showing
        return {"name": self.name, "ready": self.ready, "makes": self.output, "needs": self.media_inputs,
                "description": self.about(),
                "inputs": {k: None if k in ("prompt", "negative_prompt") else v for k, v in self.inputs.items()
                           if k not in NOT_LISTED and k not in MEDIA_INPUTS}}


def load_descriptions(path: Path) -> dict[str, str]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        log.warning("could not read %s: %s", path, exc)
        return {}
    return {str(k): " ".join(str(v).split()) for k, v in data.items()} if isinstance(data, dict) else {}


def save_description(path: Path, name: str, text: str):
    """Sets one workflow's line, keeping the rest of the file (and its comments) as is."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    lines = [line for line in lines if not line.startswith(f"{name}:")]
    lines.append(f"{name}: {json.dumps(' '.join(text.split()), ensure_ascii=False)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _detail(r: httpx.Response) -> str:
    try:
        detail = r.json().get("detail")
        if detail:
            return detail if isinstance(detail, str) else json.dumps(detail)[:300]
    except Exception:
        pass
    return re.sub(r"<[^>]+>|\s+", " ", r.text).strip()[:200] or r.reason_phrase


class ComfyClient:
    def __init__(self, base_url: str, api_key: str, descriptions: Path):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.AsyncClient(base_url=self.base_url, headers={"X-API-Key": api_key} if api_key else {},
                                      timeout=httpx.Timeout(60, connect=15), follow_redirects=True)
        self.descriptions = descriptions
        self.workflows: dict[str, Workflow] = {}
        self.error: str | None = "not contacted yet"  # why the last refresh failed
        self.refreshed = 0.0
        self._refresh_task: asyncio.Task | None = None
        self._templates: tuple[float, list] | None = None

    async def close(self):
        await self.http.aclose()

    # --- HTTP -----------------------------------------------------------------------

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            r = await self.http.request(method, path, **kwargs)
        except httpx.TimeoutException:
            raise ComfyError("the ComfyUI pod did not answer in time") from None
        except httpx.HTTPError as exc:
            raise ComfyError(f"the ComfyUI pod is not reachable ({type(exc).__name__}); is it running?") from None
        if r.status_code == 401:
            raise ComfyError("the ComfyUI pod rejected the API key; check COMFYUI_API_KEY")
        if r.status_code in (502, 503, 504) or (r.status_code == 404 and "json" not in r.headers.get("content-type", "")):
            # RunPod's proxy answers for a pod that is stopped, restarting, or not this URL
            raise ComfyError(f"the ComfyUI pod is not reachable (HTTP {r.status_code}); "
                             "is it running and is COMFYUI_URL right?")
        if r.status_code >= 400:
            raise ComfyError(f"ComfyUI said: {_detail(r)}")
        return r

    async def _json(self, method: str, path: str, **kwargs):
        r = await self._request(method, path, **kwargs)
        try:
            return r.json()
        except ValueError:
            raise ComfyError(f"unexpected answer from the ComfyUI pod for {path}") from None

    async def health(self) -> dict:
        return await self._json("GET", "/health")

    async def run(self, name: str, inputs: dict) -> dict:
        """Queues a job. Returns {"prompt_id", "seed", "ignored_keys", ...} or {"status": "setting_up"}."""
        return await self._json("POST", "/run", json={**inputs, "name": name, "wait": False})

    async def result(self, prompt_id: str) -> dict:
        return await self._json("GET", f"/result/{prompt_id}")

    async def download(self, url: str) -> bytes:
        return (await self._request("GET", url, timeout=300)).content

    async def cancel(self, prompt_id: str) -> dict:
        try:
            return await self._json("POST", f"/cancel/{prompt_id}")
        except ComfyError as exc:
            if "Not Found" in str(exc):
                raise ComfyError("the pod's manager has no cancel endpoint yet (redeploy comfyui/manager.py), "
                                 "so the pod will finish the job anyway") from None
            raise

    async def workflow(self, name: str) -> dict:
        return await self._json("GET", f"/workflows/{name}")

    async def register(self, name: str, body: dict) -> dict:
        return await self._json("POST", f"/workflows/{name}", json=body)

    async def install_template(self, template: str, name: str) -> dict:
        return await self._json("POST", f"/templates/{template}/install", json={"as": name})

    async def setup_job(self, job_id: str) -> dict:
        return await self._json("GET", f"/jobs/{job_id}")

    # --- workflow list --------------------------------------------------------------

    async def refresh(self):
        """Reloads the workflow list (with input defaults). Raises ComfyError."""
        try:
            listing = await self._json("GET", "/workflows")
            details = await asyncio.gather(
                *(self.workflow(w["name"]) for w in listing if w.get("ready")), return_exceptions=True)
        except ComfyError as exc:
            self.error = str(exc)
            raise
        detail = {d["name"]: d for d in details if isinstance(d, dict) and "name" in d}
        notes = load_descriptions(self.descriptions)
        workflows = {}
        for w in listing:
            wf = Workflow(w["name"], bool(w.get("ready")), {k: None for k in w.get("inputs", [])},
                          description=notes.get(w["name"], ""))
            for key, slots in (detail.get(w["name"], {}).get("inputs") or {}).items():
                wf.inputs[key] = slots[0].get("default") if slots else None
                wf.slots[key] = len(slots)
            workflows[wf.name] = wf
        self.workflows, self.error, self.refreshed = workflows, None, time.monotonic()

    def maybe_refresh(self):
        """Refreshes in the background when the list is stale; never waits."""
        stale = time.monotonic() - self.refreshed > REFRESH_SECONDS
        if stale and not (self._refresh_task and not self._refresh_task.done()):
            self._refresh_task = asyncio.create_task(self._refresh_quietly())

    async def _refresh_quietly(self):
        try:
            await self.refresh()
        except ComfyError as exc:
            log.warning("ComfyUI: %s", exc)
            self.refreshed = time.monotonic()  # don't retry on every turn while the pod is down

    def catalogue(self) -> str:
        lines = [f"- {w.summary()}" for w in self.workflows.values()]
        if self.error:
            lines.append(f"(The ComfyUI pod could not be reached just now: {self.error}. "
                         + ("This list may be out of date.)" if lines else "No workflows are known yet.)"))
        return "\n".join(lines) or "(no workflows registered on the pod yet)"

    # --- templates ------------------------------------------------------------------

    async def search_templates(self, query: str, limit: int = 10) -> list[dict]:
        """Built-in ComfyUI templates ranked by how many query words they match."""
        if not self._templates or time.monotonic() - self._templates[0] > TEMPLATES_SECONDS:
            self._templates = (time.monotonic(), await self._json("GET", "/templates"))
        words = [w for w in re.findall(r"[a-z0-9.]+", query.lower()) if w not in SEARCH_STOPWORDS]
        scored = []
        for t in self._templates[1]:
            if t["name"].startswith("api_"):  # paid cloud services, not run on the pod
                continue
            hay = " ".join(str(t.get(k) or "") for k in ("name", "title", "description", "media_type")).lower()
            score = sum(w in hay for w in words)
            if score:
                scored.append((score, t))
        scored.sort(key=lambda st: (-st[0], st[1]["name"]))
        return [{"name": t["name"], "title": t.get("title"), "description": (t.get("description") or "")[:160],
                 "makes": t.get("media_type"), "installed": t.get("installed", False)} for _, t in scored[:limit]]
