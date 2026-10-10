"""Conversations: each has its own history, pictures, drafts and FlowAI links, kept on disk so the
page can list them, switch between them and pick one up again later, or after a restart.

A conversation stays in memory while a connection shows it, or while a picture or video is still
being made for it (it goes into its draft even if the user has moved on); otherwise only its file
is kept: DATA_DIR/conversations/<owner>/<id>.json, plus index.json per owner for the list. The
owner is the FlowAI user it belongs to ("local" when FlowAI is off). When FlowAI can't say who the
user is, the conversation isn't kept at all, so nobody's conversations end up in a shared folder.
"""

import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .media import MediaSession
from .studio import StudioSession

if TYPE_CHECKING:
    from .comfyui import ComfyClient
    from .session import VoiceSession

log = logging.getLogger(__name__)

ID_RE = re.compile(r"[a-f0-9]{12}")
OWNER_RE = re.compile(r"[a-z0-9-]{1,40}")
SAVE_DELAY = 1.0  # seconds: a burst of changes (a whole carousel) is one write
STATE_EVENTS = {"draft", "media", "saved", "approval", "linked"}  # sent when something worth keeping changed
HISTORY_SHOWN = 200  # log lines the page gets when a conversation is reopened
TITLE_CHARS = 60


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cut(text: str, n: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= n:
        return text
    return text[:n].rsplit(" ", 1)[0].rstrip(",.;:") + "…"


def _call(c) -> tuple[str, dict]:
    """A tool call as the LLM made it (an ollama object), or as it was read back from disk (a dict)."""
    f = c.get("function", {}) if isinstance(c, dict) else c.function
    name = f.get("name", "") if isinstance(f, dict) else f.name
    args = f.get("arguments", {}) if isinstance(f, dict) else f.arguments
    return name, dict(args or {})


def _jsonable(m: dict) -> dict:
    calls = m.get("tool_calls")
    if not calls:
        return m
    return {**m, "tool_calls": [{"function": {"name": n, "arguments": a}} for n, a in map(_call, calls)]}


def _error(result: str) -> str | None:
    try:
        data = json.loads(result)
    except (TypeError, ValueError):
        return None
    return data.get("error") if isinstance(data, dict) else None


class Conversation:
    def __init__(self, store: "ConversationStore", owner: str | None, cid: str | None = None):
        self.store = store
        self.owner = owner  # None: not kept
        self.id = cid or uuid.uuid4().hex[:12]
        self.title = ""
        self.created = self.updated = _now()
        self.campaign: dict | None = None  # the FlowAI campaign it was opened for: id, name, stage, brief
        self.sessions: set["VoiceSession"] = set()  # the connections showing it
        self.messages: list[dict] = []  # for the LLM, without the system prompt (made fresh every turn)
        self.notes: list[str] = []  # events the LLM hasn't seen yet
        self.announce = False  # a note needs a spoken reply even if the user says nothing
        self.focus: dict = {}  # what the user selected on screen
        self.focus_told: dict = {}  # the selection the LLM was last told about
        # Its FlowAI links (backend/flowai.py): draft id -> Saved and -> Linked, approvals by id,
        # and the assets it uploaded (the only ones it may delete)
        self.saved: dict = {}
        self.linked: dict = {}
        self.approvals: dict = {}
        self.uploaded: set[int] = set()
        self.media = MediaSession(store.comfy, store.media_dir, store.timeout_s, self.send, self.notify, self.remember)
        self.studio = StudioSession(self.media, store.media_dir, store.fonts_dir, self.send, f"{self.id}-")
        self.dirty = False
        self.restoring = False
        self._saving: asyncio.Task | None = None

    @property
    def key(self) -> tuple[str | None, str]:
        return self.owner, self.id

    @property
    def empty(self) -> bool:
        return not (self.messages or self.studio.drafts or self.media.items or self.campaign)

    # --- what happens in it ---------------------------------------------------------

    async def send(self, **msg):
        """To every connection showing it; never raises. State changes are saved soon after."""
        for s in list(self.sessions):
            await s.try_send(**msg)
        if msg.get("type") in STATE_EVENTS:
            self.changed()

    def notify(self, text: str):
        """Something the assistant should say (a picture is ready), now if it's on screen, else when reopened."""
        self.notes.append(text)
        self.announce = True
        self.changed()
        for s in list(self.sessions):
            s.maybe_announce()

    def remember(self, text: str):
        """Context for the assistant, without a reply."""
        self.notes.append(text)
        self.changed()

    def heard(self, text: str) -> bool:
        """The user said or typed something: the first thing they say names the conversation (True)."""
        named = not self.title
        if named:
            self.title = _cut(text, TITLE_CHARS)
        self.changed()
        return named

    def changed(self):
        if self.restoring:
            return
        self.updated = _now()
        self.dirty = True
        if self.owner and not (self._saving and not self._saving.done()):
            self._saving = asyncio.create_task(self._save_later())

    async def _save_later(self):
        while self.dirty:
            await asyncio.sleep(SAVE_DELAY)
            self.dirty = False
            await self.store.save(self)
        self.store.release(self)

    async def flush(self):
        """Saved now (switching away, disconnecting)."""
        if self.dirty and self.owner:
            self.dirty = False
            await self.store.save(self)

    def about(self) -> dict:
        """Its name and campaign, as the page shows them above the conversation."""
        return {"type": "about", "id": self.id, "title": self.title, "kept": bool(self.owner),
                "campaign": {k: self.campaign.get(k) for k in ("id", "name", "stage")} if self.campaign else None}

    # --- on disk --------------------------------------------------------------------

    def summary(self) -> dict:
        """Its line in the list."""
        last = next((m["content"] for m in reversed(self.messages)
                     if m.get("role") in ("user", "assistant") and m.get("content")), "")
        return {"id": self.id, "title": self.title or "New conversation", "created": self.created,
                "updated": self.updated, "drafts": len(self.studio.drafts),
                "pictures": sum(i.kind in ("image", "video") for i in self.media.items.values()),
                "campaign": {"id": self.campaign["id"], "name": self.campaign.get("name")} if self.campaign else None,
                "preview": _cut(last, 90)}

    def dump(self) -> dict:
        from .flowai import dump_links  # flowai imports studio, which this module imports too

        return {"id": self.id, "title": self.title, "created": self.created, "updated": self.updated,
                "campaign": self.campaign, "messages": [_jsonable(m) for m in self.messages], "notes": self.notes,
                "media": self.media.dump(), "studio": self.studio.dump(),
                "flowai": dump_links(self.saved, self.linked, self.approvals, self.uploaded)}

    async def restore(self, data: dict):
        from .flowai import restore_links

        self.restoring = True  # redrawing a lost slide isn't a change worth a new date
        self.title, self.campaign = data.get("title", ""), data.get("campaign")
        self.created, self.updated = data.get("created", self.created), data.get("updated", self.updated)
        self.messages, self.notes = data.get("messages", []), data.get("notes", [])
        self.media.restore(data.get("media", {}))
        await self.studio.restore(data.get("studio", {}))
        self.saved, self.linked, self.approvals, self.uploaded = restore_links(data.get("flowai", {}))
        self.restoring = False

    def history(self) -> list[dict]:
        """What was said and done, as the page's log shows it."""
        out, pending = [], []
        for m in self.messages:
            role = m.get("role")
            if role == "user" and m.get("content"):
                out.append({"kind": "user", "text": m["content"]})
            elif role == "assistant":
                if m.get("content"):
                    out.append({"kind": "bot", "text": m["content"]})
                for c in m.get("tool_calls") or []:
                    name, args = _call(c)
                    act = {"kind": "act", "name": name, "arguments": args, "status": "done"}
                    out.append(act)
                    pending.append(act)
            elif role == "tool":
                act = next((a for a in pending if a["name"] == m.get("tool_name")), None)
                if act:
                    pending.remove(act)
                    if why := _error(m.get("content")):
                        act.update(status="failed", why=why)
        return out[-HISTORY_SHOWN:]


class ConversationStore:
    """Every user's conversations: the ones in memory, and the files of the others."""

    def __init__(self, root: Path, comfy: "ComfyClient | None", media_dir: Path, fonts_dir: Path, timeout_s: float):
        self.root = root
        self.comfy = comfy
        self.media_dir, self.fonts_dir, self.timeout_s = media_dir, fonts_dir, timeout_s
        self.live: dict[tuple[str, str], Conversation] = {}
        self.lock = asyncio.Lock()  # one write at a time: index.json is read, changed and written

    def _dir(self, owner: str) -> Path:
        if not OWNER_RE.fullmatch(owner):
            raise ValueError(f"bad owner {owner!r}")
        return self.root / owner

    def new(self, owner: str | None) -> Conversation:
        conv = Conversation(self, owner)
        if owner:
            self.live[conv.key] = conv
        return conv

    def adopt(self, conv: Conversation, owner: str):
        """A conversation started before the user was known (FlowAI was still answering) is theirs."""
        if conv.owner:
            return
        conv.owner = owner
        self.live[conv.key] = conv
        if not conv.empty:
            conv.changed()

    async def open(self, owner: str, cid: str) -> Conversation | None:
        if not ID_RE.fullmatch(str(cid)):
            return None
        if conv := self.live.get((owner, cid)):
            return conv
        path = self._dir(owner) / f"{cid}.json"
        try:
            data = json.loads(await asyncio.to_thread(path.read_text))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            log.error("conversation %s/%s could not be read: %s", owner, cid, exc)
            return None
        conv = Conversation(self, owner, cid)
        await conv.restore(data)
        self.live[conv.key] = conv
        return conv

    async def save(self, conv: Conversation):
        if not conv.owner or conv.empty:
            return
        folder = self._dir(conv.owner)
        data, summary = conv.dump(), conv.summary()
        try:
            async with self.lock:
                await asyncio.to_thread(self._write, folder, conv.id, data, summary)
        except OSError as exc:
            log.error("conversation %s could not be saved: %s", conv.id, exc)

    @staticmethod
    def _write(folder: Path, cid: str, data: dict, summary: dict | None):
        folder.mkdir(parents=True, exist_ok=True)
        if data is not None:
            _write_json(folder / f"{cid}.json", data)
        index = _read_json(folder / "index.json")
        if summary is None:
            index.pop(cid, None)
        else:
            index[cid] = summary
        _write_json(folder / "index.json", index)

    def release(self, conv: Conversation):
        """Out of memory once nothing shows it, nothing is being made for it and it's saved."""
        if not conv.sessions and not conv.media.busy and not conv.dirty and self.live.get(conv.key) is conv:
            del self.live[conv.key]

    async def list(self, owner: str) -> list[dict]:
        """Newest first. Ones in memory are newer than their files."""
        index = await asyncio.to_thread(_read_json, self._dir(owner) / "index.json")
        for (o, cid), conv in self.live.items():
            if o == owner and not conv.empty:
                index[cid] = conv.summary()
        return sorted(index.values(), key=lambda c: c.get("updated", ""), reverse=True)

    async def rename(self, owner: str, cid: str, title: str) -> Conversation | None:
        conv = await self.open(owner, cid)
        if conv:
            conv.title = _cut(title, TITLE_CHARS) or conv.title
            conv.changed()
            await conv.flush()
        return conv

    async def delete(self, owner: str, cid: str):
        """Its file, its line in the list, its pictures, and every version of its drawn slides."""
        conv = await self.open(owner, cid)
        if not conv:
            return
        conv.owner = None  # never saved again, even by a generation still running
        self.live.pop((owner, cid), None)
        await conv.media.close()
        files = [i.path for i in conv.media.items.values() if i.path] + \
                [p for d in conv.studio.drafts.values() for p, _ in d.files] + \
                list(self.media_dir.glob(f"{cid}-draft*"))
        async with self.lock:
            await asyncio.to_thread(self._write, self._dir(owner), cid, None, None)
        await asyncio.to_thread(_remove, [self._dir(owner) / f"{cid}.json", *files])


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log.error("%s could not be read: %s", path, exc)
        return {}


def _write_json(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    os.replace(tmp, path)  # whole or not at all


def _remove(paths: list[Path]):
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("could not delete %s: %s", p, exc)
