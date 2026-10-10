"""Image and video generation for one conversation: jobs, numbered media, uploads.

Like browser tasks, a generation runs in the background: the assistant starts it and keeps
talking, and when the files are downloaded they appear in the web UI and the assistant is
told (a note) so it can say so. Every picture, video or attached file gets a number the user
and the assistant can refer to ("animate number three"). Files are kept in MEDIA_DIR, served
at /media, because the pod's disk is not permanent.

Without a pod (COMFYUI_URL empty) the numbered files still work: attachments and pictures from
the FlowAI gallery can go into post drafts; only generating is off.
"""

import asyncio
import base64
import binascii
import json
import logging
import mimetypes
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .comfyui import MEDIA_INPUTS, ComfyClient, ComfyError, describe_error, file_kind, save_description

log = logging.getLogger(__name__)

MAX_RUNNING = 10  # generations per conversation at once, e.g. a whole carousel (the pod runs them one by one)
MAX_UPLOAD = 15 * 1024 * 1024
MAX_WORKFLOW_DOWNLOAD = 10 * 1024 * 1024
POLL_ERRORS = 5  # consecutive failed polls before giving up on a job
NAME_RE = re.compile(r"[A-Za-z0-9_\-.]{1,120}")
MEDIA_KEY_RE = re.compile(r"(image|video|audio)(_\d+)?")
CONTROL_KEYS = {"name", "template", "workflow", "wait", "timeout", "return_base64", "images", "videos", "audios"}


@dataclass
class MediaItem:
    id: int
    kind: str  # image | video | audio | file | workflow
    name: str  # original or generated file name
    source: str  # generated | upload
    path: Path | None = None
    workflow: str = ""
    prompt: str = ""
    data: dict | None = None  # the JSON of an attached workflow file

    def event(self, job: int | None = None) -> dict:
        return {"type": "media", "id": self.id, "kind": self.kind, "name": self.name, "source": self.source,
                "url": f"media/{self.path.name}" if self.path else None,
                "workflow": self.workflow, "prompt": self.prompt, "job": job}


@dataclass
class MediaJob:
    id: int
    kind: str  # generate | install
    workflow: str
    prompt: str = ""
    task: asyncio.Task | None = None
    status: str = "running"  # running | done | failed | cancelled
    started: float = field(default_factory=time.monotonic)
    prompt_id: str | None = None  # ComfyUI's id, once queued
    setup_id: str | None = None  # the manager's setup job, for installs
    progress: str = ""
    result: str = ""
    # Called with the files once made (e.g. to put them in a post draft); returns a sentence for the note
    then: Callable[[list["MediaItem"]], Awaitable[str]] | None = None

    def info(self) -> dict:
        d = {"job": self.id, "kind": self.kind, "workflow": self.workflow, "status": self.status,
             "seconds": round(time.monotonic() - self.started)}
        if self.progress and self.status == "running":
            d["progress"] = self.progress
        if self.result:
            d["result"] = self.result
        return d


def _a(word: str) -> str:
    return ("an " if word[0] in "aeiou" else "a ") + word


def _size(ratio: float, width, height) -> tuple[int, int]:
    """A width and height of this shape with about as many pixels as the workflow's defaults."""
    numbers = all(isinstance(v, (int, float)) and v > 0 for v in (width, height))
    w = ((width * height if numbers else 1024 * 1024) * ratio) ** 0.5
    return max(64, round(w / 64) * 64), max(64, round(w / ratio / 64) * 64)


def _seconds(s: float) -> str:
    return f"{s:.0f} seconds" if s < 90 else f"{s / 60:.1f} minutes"


def _workflow_json(data) -> dict:
    """Accepts what POST /workflows takes: an editor workflow, an API workflow, or a bundle."""
    if isinstance(data, dict) and (
        isinstance(data.get("nodes"), list) or "api" in data or "ui" in data
        or (data and all(isinstance(v, dict) and "class_type" in v for v in data.values()))
    ):
        return data
    raise ComfyError("that file is not a ComfyUI workflow")


class MediaSession:
    def __init__(self, comfy: ComfyClient | None, media_dir: Path, timeout_s: float,
                 send: Callable[..., Awaitable[None]], notify: Callable[[str], None],
                 remember: Callable[[str], None]):
        self.comfy = comfy  # None: no pod, so no generating
        self.dir = media_dir
        self.timeout_s = timeout_s
        self.send = send  # to the web UI; must not raise
        self.notify = notify  # a result the assistant should announce
        self.remember = remember  # context the assistant needs, without a reply
        self.items: dict[int, MediaItem] = {}
        self.jobs: dict[int, MediaJob] = {}
        self.item_count = 0
        self.job_count = 0

    # --- media ----------------------------------------------------------------------

    async def _store(self, data: bytes, name: str, kind: str, source: str, **extra) -> MediaItem:
        self.item_count += 1
        stem = re.sub(r"[^\w\-]+", "_", Path(name).stem).strip("_")[:40] or kind
        suffix = Path(name).suffix.lower() or ".bin"
        path = self.dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}-{stem}{suffix}"
        await asyncio.to_thread(path.write_bytes, data)
        item = MediaItem(self.item_count, kind, name, source, path, **extra)
        self.items[item.id] = item
        return item

    async def add(self, data: bytes, name: str, kind: str, source: str) -> MediaItem:
        """A file from elsewhere (e.g. the FlowAI gallery), numbered and shown like the others."""
        item = await self._store(data, name, kind, source)
        await self.send(**item.event())
        return item

    def _latest(self, kind: str) -> MediaItem | None:
        return next((i for i in reversed(self.items.values()) if i.kind == kind), None)

    def _resolve(self, ref, kind: str) -> tuple[str, MediaItem | None]:
        """A media number, "latest" or a URL -> what the pod accepts (a URL or a data URL)."""
        text = str(ref).strip()
        if text.startswith(("https://", "http://", "data:")):
            return text, None
        if text.lower().lstrip("#") in ("", "last", "latest", "previous", "current", "this", "it"):
            item = self._latest(kind)
            if not item:
                raise ComfyError(f"there is no {kind} in this conversation yet; "
                                 f"make one first, or ask the user to attach one")
        else:
            number = re.search(r"\d+", text)
            item = self.items.get(int(number.group())) if number else None
            if not item:
                raise ComfyError(f"there is no number {text} in this conversation")
        if item.kind != kind:
            raise ComfyError(f"number {item.id} is {_a(item.kind)}, not {_a(kind)}")
        mime = mimetypes.guess_type(item.path.name)[0] or "application/octet-stream"
        return f"data:{mime};base64,{base64.b64encode(item.path.read_bytes()).decode()}", item

    async def upload(self, name: str, data: str | None = None, workflow=None) -> MediaItem:
        """A file the user attached in the web UI: a data URL, or a parsed workflow JSON."""
        name = Path(str(name or "upload")).name[:120]
        if workflow is not None:
            item = await self._store(json.dumps(_workflow_json(workflow)).encode(), name, "workflow", "upload")
            item.data = workflow
            note = (f"The user attached a ComfyUI workflow file, number {item.id} ({name}). To add it, call "
                    f"add_media_workflow with file={item.id}.")
        else:
            try:
                header, b64 = str(data).split(",", 1)
                raw = base64.b64decode(b64, validate=True)
            except (ValueError, binascii.Error):
                raise ComfyError("the attached file could not be read") from None
            if len(raw) > MAX_UPLOAD:
                raise ComfyError(f"{name} is too large (max {MAX_UPLOAD // 2**20} MB)")
            mime = header.removeprefix("data:").split(";")[0]
            kind = mime.split("/")[0] if mime.split("/")[0] in MEDIA_INPUTS.values() else file_kind(name)
            if kind == "file":
                raise ComfyError(f"{name} is not an image, video or audio file")
            if not Path(name).suffix:
                name += mimetypes.guess_extension(mime) or ""
            item = await self._store(raw, name, kind, "upload")
            note = f"The user attached {kind} number {item.id} ({name})."
        await self.send(**item.event())
        self.remember(note)
        return item

    # --- generation -----------------------------------------------------------------

    def _pod(self) -> ComfyClient:
        if not self.comfy:
            raise ComfyError("picture and video generation is off (COMFYUI_URL is not set)")
        return self.comfy

    def default_workflow(self) -> str | None:
        """The text-to-image workflow, for pictures asked for without one."""
        found = [w.name for w in self.comfy.workflows.values()
                 if w.ready and w.output == "image" and not w.media_inputs] if self.comfy else []
        return found[0] if found else None

    async def _workflow(self, name: str):
        self._pod()
        wf = self.comfy.workflows.get(name)
        if wf is None:  # maybe added since the last refresh
            await self.comfy.refresh()
            wf = self.comfy.workflows.get(name)
        if wf is None:
            known = ", ".join(self.comfy.workflows) or "none"
            raise ComfyError(f"there is no workflow named {name!r}. Available: {known}")
        if not wf.ready:
            raise ComfyError(f"{name} is still being installed on the pod")
        return wf

    async def generate(self, workflow: str, prompt: str, image=None, options=None, *, ratio: float | None = None,
                       source: str | None = None, then=None) -> dict:
        """ratio: the shape to make (width ÷ height) when the workflow has a size the options don't set.
        source: the picture an editing workflow edits when no image is given (instead of the latest).
        then: see MediaJob.then."""
        try:
            wf = await self._workflow(str(workflow))
            if isinstance(options, str):
                try:
                    options = json.loads(options or "{}")
                except ValueError:
                    raise ComfyError("options must be a JSON object") from None
            inputs = {k: v for k, v in (options or {}).items() if k not in CONTROL_KEYS}
            if image not in (None, ""):
                inputs["image"] = image
            elif source is not None and "image" in wf.media_inputs:
                inputs["image"] = source
            if ratio and {"width", "height"} <= wf.inputs.keys() and not {"width", "height"} & inputs.keys():
                inputs["width"], inputs["height"] = _size(ratio, wf.inputs["width"], wf.inputs["height"])
            if prompt:
                inputs["prompt"] = prompt

            used = []
            for key in [k for k in inputs if MEDIA_KEY_RE.fullmatch(k)]:
                kind = MEDIA_KEY_RE.fullmatch(key).group(1)
                if kind not in wf.media_inputs:
                    takes = [w.name for w in self.comfy.workflows.values() if w.ready and kind in w.media_inputs]
                    raise ComfyError(f"{wf.name} does not take {_a(kind)}. Workflows that do: "
                                     f"{', '.join(takes) or 'none'}")
                inputs[key], item = self._resolve(inputs[key], kind)
                used.append(item)
            for kind in wf.media_inputs:  # required input left out: use the latest one
                if kind not in inputs:
                    inputs[kind], item = self._resolve("latest", kind)
                    used.append(item)

            running = [j for j in self.jobs.values() if j.kind == "generate" and j.status == "running"]
            if len(running) >= MAX_RUNNING:
                raise ComfyError(f"{len(running)} generations are already running; wait for one to finish")
        except ComfyError as exc:
            return {"error": str(exc)}

        self.job_count += 1
        job = MediaJob(self.job_count, "generate", wf.name, prompt, then=then)
        self.jobs[job.id] = job
        job.task = asyncio.create_task(self._run_generation(job, inputs, wf.output))
        reply = {"job": job.id, "status": "started", "makes": wf.output,
                 "expected_time": "several minutes" if wf.output == "video" else "a few seconds"}
        if [i for i in used if i]:
            reply["input"] = ", ".join(f"{i.kind} number {i.id}" for i in used if i)
        return reply

    async def _run_generation(self, job: MediaJob, inputs: dict, output: str):
        await self.send(type="media_job", id=job.id, kind="generate", status="started",
                        workflow=job.workflow, prompt=job.prompt, makes=output)
        try:
            queued = await self.comfy.run(job.workflow, inputs)
            if queued.get("status") == "setting_up":
                raise ComfyError(f"{job.workflow} is still being installed on the pod; try again in a few minutes")
            job.prompt_id = queued["prompt_id"]
            res = await self._wait_for(job)
            files = [f for f in res.get("files") or [] if f.get("url")]
            if not files:
                raise ComfyError("it finished without producing any file")
            made = []
            for f in files:
                data = await self.comfy.download(f["url"])
                item = await self._store(data, f.get("filename") or "output", file_kind(f.get("filename", "")),
                                         "generated", workflow=job.workflow, prompt=job.prompt)
                made.append(item)
                await self.send(**item.event(job.id))
            job.status = "done"
            job.result = ", ".join(f"{i.kind} number {i.id}" for i in made)
            then = ""
            if job.then:
                try:
                    then = await job.then(made)
                except Exception:
                    log.exception("generation %d: follow-up failed", job.id)
            took = _seconds(time.monotonic() - job.started)
            await self.send(type="media_job", id=job.id, kind="generate", status="done", text=f"took {took}")
            note = (f"Generation {job.id} ({job.workflow}) finished in {took}: {job.result} is now shown on the "
                    "user's screen.")
            if then:
                note += f" {then}"
            if queued.get("ignored_keys"):
                note += f" The workflow ignored these options: {', '.join(queued['ignored_keys'])}."
            note += " Tell the user it is ready in one short sentence."
        except asyncio.CancelledError:
            job.status, job.result = "cancelled", "cancelled"
            await self.send(type="media_job", id=job.id, kind="generate", status="cancelled")
            raise
        except Exception as exc:
            if not isinstance(exc, ComfyError):
                log.exception("generation %d failed", job.id)
            job.status, job.result = "failed", str(exc)
            await self.send(type="media_job", id=job.id, kind="generate", status="failed", text=str(exc))
            note = f"Generation {job.id} ({job.workflow}) failed: {exc}. Tell the user briefly."
        log.info("generation %d %s in %.1f s: %s", job.id, job.status, time.monotonic() - job.started, job.result)
        self.notify(note)

    async def _wait_for(self, job: MediaJob) -> dict:
        deadline = time.monotonic() + self.timeout_s
        delay, errors = 1.5, 0
        while time.monotonic() < deadline:
            await asyncio.sleep(delay)
            delay = min(delay * 1.25, 5)
            try:
                res = await self.comfy.result(job.prompt_id)
                errors = 0
            except ComfyError:
                errors += 1  # a network blip shouldn't lose a five-minute video
                if errors >= POLL_ERRORS:
                    raise
                continue
            if res.get("status") == "done":
                return res
            if res.get("status") == "error":
                raise ComfyError(describe_error(res.get("messages")))
        raise ComfyError(f"no result after {_seconds(self.timeout_s)}; the pod may have restarted")

    # --- workflows ------------------------------------------------------------------

    async def list_workflows(self) -> dict:
        try:
            await self._pod().refresh()
        except ComfyError as exc:
            return {"error": str(exc)}
        return {"workflows": [w.info() for w in self.comfy.workflows.values()]}

    async def search_templates(self, query: str) -> dict:
        try:
            found = await self._pod().search_templates(query)
        except ComfyError as exc:
            return {"error": str(exc)}
        return {"templates": found} if found else {"templates": [], "hint": "nothing matched; try other words"}

    async def add_workflow(self, name: str, description: str = "", template: str | None = None,
                           file=None, url: str | None = None, replace: bool = False) -> dict:
        try:
            name = str(name).strip()
            if not NAME_RE.fullmatch(name):
                raise ComfyError("the name may only use letters, digits, _ - and . (e.g. wan14_i2v)")
            self._pod()
            sources = [s for s in (template, file, url) if s not in (None, "")]
            if len(sources) != 1:
                raise ComfyError("give exactly one of template, file or url")
            try:
                await self.comfy.refresh()
            except ComfyError:
                pass  # the request below reports it
            if name in self.comfy.workflows and not replace:
                raise ComfyError(f"a workflow named {name} already exists; pick another name, "
                                 "or set replace if the user wants to overwrite it")
            if template:
                started = await self.comfy.install_template(str(template), name)
            elif url:
                started = await self.comfy.register(name, await self._fetch_workflow(str(url)))
            else:
                number = re.search(r"\d+", str(file))
                item = self.items.get(int(number.group())) if number else None
                if not item or item.kind != "workflow":
                    raise ComfyError(f"there is no attached workflow file number {file}")
                started = await self.comfy.register(name, item.data)
        except ComfyError as exc:
            return {"error": str(exc)}
        if description:
            save_description(self.comfy.descriptions, name, description)
        self.job_count += 1
        job = MediaJob(self.job_count, "install", name, setup_id=started["job_id"])
        self.jobs[job.id] = job
        job.task = asyncio.create_task(self._run_install(job))
        return {"job": job.id, "status": "installing",
                "detail": "custom nodes and models are being installed on the pod; this can take several "
                          "minutes. You will be told when it is ready."}

    async def _fetch_workflow(self, url: str) -> dict:
        # GitHub page links -> the raw file
        url = re.sub(r"^https://github\.com/([^/]+/[^/]+)/blob/", r"https://raw.githubusercontent.com/\1/", url)
        if not url.startswith(("https://", "http://")):
            raise ComfyError("the workflow url must start with https://")
        try:  # a separate client: the pod's API key must not go to other sites
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as http:
                r = await http.get(url)
                r.raise_for_status()
        except httpx.HTTPError as exc:
            raise ComfyError(f"could not download the workflow: {exc}") from None
        if len(r.content) > MAX_WORKFLOW_DOWNLOAD:
            raise ComfyError("that file is too large to be a workflow")
        try:
            return _workflow_json(r.json())
        except ValueError:
            raise ComfyError("that url is not a JSON workflow file") from None

    async def _run_install(self, job: MediaJob):
        await self.send(type="media_job", id=job.id, kind="install", status="started", workflow=job.workflow)
        try:
            setup, errors = {}, 0
            deadline = time.monotonic() + 3 * 3600  # big video models are tens of GB
            while setup.get("status") not in ("done", "error"):
                if time.monotonic() > deadline:
                    raise ComfyError("it was still installing after three hours")
                await asyncio.sleep(5)
                try:
                    setup = await self.comfy.setup_job(job.setup_id)
                    errors = 0
                except ComfyError:
                    errors += 1
                    if errors >= POLL_ERRORS:
                        raise
                    continue
                downloading = [f"{n} {p:.0f}%" for n, p in (setup.get("progress") or {}).items() if p < 100]
                progress = f"downloading {', '.join(downloading)}" if downloading else setup.get("status", "")
                if progress != job.progress:
                    job.progress = progress
                    await self.send(type="media_job", id=job.id, kind="install", status="running", text=progress)

            problems = []
            if setup.get("failed"):
                problems.append("downloads failed for " + ", ".join(f.get("name", "?") for f in setup["failed"]))
            if setup.get("missing_node_types"):
                problems.append("missing node types " + ", ".join(setup["missing_node_types"]))
            if setup.get("missing_models_no_url"):
                problems.append("models without a download URL " + ", ".join(setup["missing_models_no_url"]))
            if setup.get("status") == "error":
                last = next((line for line in reversed(setup.get("log") or []) if "ERROR" in line or "✗" in line), "")
                raise ComfyError("; ".join(filter(None, [last.split(" ", 1)[-1], *problems])) or "setup failed")

            await self.comfy.refresh()
            wf = self.comfy.workflows.get(job.workflow)
            if wf is None or not wf.ready:
                raise ComfyError("setup finished but the workflow is not runnable")
            if wf.slots.get("prompt", 0) > 1:
                problems.append(f"it has {wf.slots['prompt']} prompt inputs, so it may contain more than one "
                                "pipeline (see comfyui/API.md, Recipe A)")
            job.status = "done"
            job.result = wf.summary() + (f" Problems: {'; '.join(problems)}." if problems else "")
            await self.send(type="media_job", id=job.id, kind="install", status="done", text=job.result)
            note = (f"The workflow {job.workflow} has been added and is ready: {job.result} "
                    "Tell the user briefly, mentioning any problem.")
        except asyncio.CancelledError:
            job.status = "cancelled"  # only stops watching: the pod keeps installing
            raise
        except Exception as exc:
            if not isinstance(exc, ComfyError):
                log.exception("install %d failed", job.id)
            job.status, job.result = "failed", str(exc)
            await self.send(type="media_job", id=job.id, kind="install", status="failed", text=str(exc))
            note = f"Adding the workflow {job.workflow} failed: {exc}. Tell the user briefly."
        self.notify(note)

    # --- kept with the conversation (backend/conversations.py) -------------------------

    @property
    def busy(self) -> bool:
        return any(j.status == "running" for j in self.jobs.values())

    def dump(self) -> dict:
        """The numbered files, to reopen the conversation later. Jobs aren't kept: a restart ends them."""
        return {"count": self.item_count, "jobs": self.job_count,
                "items": [{"id": i.id, "kind": i.kind, "name": i.name, "source": i.source,
                           "file": i.path.name if i.path else None, "workflow": i.workflow, "prompt": i.prompt,
                           **({"data": i.data} if i.data is not None else {})}
                          for i in self.items.values()]}

    def restore(self, data: dict):
        self.item_count, self.job_count = data.get("count", 0), data.get("jobs", 0)
        for i in data.get("items", []):
            name = i.get("file")
            path = self.dir / name if name and Path(name).name == name else None
            if path and not path.is_file():
                continue  # gone from MEDIA_DIR: its number stays taken
            self.items[i["id"]] = MediaItem(i["id"], i["kind"], i["name"], i["source"], path,
                                            i.get("workflow", ""), i.get("prompt", ""), i.get("data"))

    def running(self) -> list[dict]:
        """Generations still going, as the page shows them when the conversation is reopened."""
        def makes(name: str) -> str:
            wf = self.comfy.workflows.get(name) if self.comfy else None
            return wf.output if wf else "image"
        return [{"type": "media_job", "id": j.id, "kind": "generate", "status": "started", "workflow": j.workflow,
                 "prompt": j.prompt, "makes": makes(j.workflow), "seconds": round(time.monotonic() - j.started)}
                for j in self.jobs.values() if j.kind == "generate" and j.status == "running"]

    # --- status / cancel ------------------------------------------------------------

    def status(self) -> dict:
        jobs = list(self.jobs.values())
        shown = [j for j in jobs if j.status == "running"] + [j for j in jobs if j.status != "running"][-3:]
        return {"jobs": [j.info() for j in shown]} if shown else {"jobs": [], "note": "nothing has been started"}

    async def cancel(self, job_id=None) -> dict:
        running = [j for j in self.jobs.values() if j.kind == "generate" and j.status == "running"]
        if job_id not in (None, ""):
            number = re.search(r"\d+", str(job_id))
            job = self.jobs.get(int(number.group())) if number else None
        else:
            job = running[-1] if running else None
        if not job or job.status != "running":
            return {"error": "no such generation is running"}
        if job.kind == "install":
            return {"error": "installing a workflow can't be cancelled; it keeps going on the pod"}
        return {"job": job.id, "status": "cancelled", "pod": await self._stop(job)}

    async def _stop(self, job: MediaJob) -> str:
        pod = "not queued on the pod yet"
        if job.prompt_id:
            try:
                pod = (await self.comfy.cancel(job.prompt_id)).get("status", "?")
            except ComfyError as exc:
                pod = str(exc)
        job.task.cancel()
        try:
            await job.task
        except (asyncio.CancelledError, Exception):
            pass
        return pod

    async def close(self):
        """Conversation over: stop generations on the pod (nobody will see them) and stop watching installs."""
        running = [j for j in self.jobs.values() if j.status == "running"]
        await asyncio.gather(*(self._stop(j) for j in running), return_exceptions=True)
