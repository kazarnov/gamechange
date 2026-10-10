"""FlowAI (content-generator/): the website where posts are saved, scheduled and published.

The assistant's drafts (backend/studio.py) become FlowAI posts: every slide is uploaded as an
asset, then the post is saved with status draft, where the team sees it in the Composer and the
calendar. Saved posts and gallery pictures can be brought back into the conversation to change.

Scheduling needs the person. FlowAI counts whoever saves a post with any status other than
draft as the one who approved it (PostController::fill), so schedule_post only asks: the page
shows an Approve button, and only that click ({"type": "approve"} on the WebSocket) books the
post. In FlowAI's own page, the page makes that call itself with the user's session
(FLOWAI_INTEGRATION.md).

FlowAI's Assistant page hands over the signed-in user's token (POST /api/assistant/session, sent
in the page's hello), so the assistant acts as whoever opened it. Without one (this repo's own
page), the client signs in the way the dashboard does: Sanctum's session cookie, as FLOWAI_EMAIL.
"""

import asyncio
import logging
import mimetypes
import re
import urllib.parse
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from . import platforms
from .studio import Draft, StudioError, _number, _numbers

if TYPE_CHECKING:
    from .media import MediaSession
    from .studio import StudioSession

log = logging.getLogger(__name__)

VOICE_LABELS = {"tone": "Tone", "topics": "Topics", "style": "Style", "do": "Always", "avoid": "Never",
                "language": "Language", "hashtags": "Hashtags"}  # FlowAI's Voice::LABELS
LIVE = {"publishing", "submitted", "published"}  # out already: changes become a new post
QUEUE_WORDS = ("queue", "slot")  # "when" meaning the account's next free posting time


class FlowAIError(Exception):
    """A problem with a message the assistant can pass on to the user."""

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


def _message(r: httpx.Response) -> str:
    """FlowAI's validation messages, or its error message."""
    try:
        data = r.json()
    except ValueError:
        return f"FlowAI answered {r.status_code}"
    errors = data.get("errors") if isinstance(data, dict) else None
    if isinstance(errors, dict) and errors:
        return " ".join(dict.fromkeys(m for messages in errors.values() for m in messages))
    return (data.get("message") if isinstance(data, dict) else "") or f"FlowAI answered {r.status_code}"


class FlowAIClient:
    """FlowAI's API as one user."""

    def __init__(self, base_url: str, dashboard_url: str, email: str = "", password: str = "", token: str = ""):
        self.base_url = base_url.rstrip("/")
        self.dashboard_url = dashboard_url.rstrip("/")
        self.email, self.password, self.token = email, password, token
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        else:  # Sanctum treats requests from the dashboard's address as a signed-in browser
            headers.update(Referer=self.dashboard_url + "/", Origin=self.dashboard_url)
        self.http = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=60)
        self.signed_in = bool(token)
        self.sign_in_lock = asyncio.Lock()

    def _csrf(self) -> dict:
        token = self.http.cookies.get("XSRF-TOKEN")
        return {"X-XSRF-TOKEN": urllib.parse.unquote(token)} if token else {}

    async def _sign_in(self):
        if not (self.email and self.password):
            raise FlowAIError("signing in to FlowAI is not set up (FLOWAI_EMAIL and FLOWAI_PASSWORD)")
        await self.http.get("/sanctum/csrf-cookie")
        r = await self.http.post("/api/auth/login", json={"email": self.email, "password": self.password},
                                 headers=self._csrf())
        if r.status_code >= 400:
            raise FlowAIError(f"FlowAI sign-in failed: {_message(r)}", r.status_code)
        self.signed_in = True

    async def request(self, method: str, path: str, **kwargs):
        try:
            for attempt in (1, 2):
                if not self.signed_in:
                    async with self.sign_in_lock:
                        if not self.signed_in:
                            await self._sign_in()
                r = await self.http.request(method, path, headers=self._csrf(), **kwargs)
                if r.status_code in (401, 419) and not self.token and attempt == 1:
                    self.signed_in = False  # the session ran out: sign in again once
                    continue
                break
        except httpx.HTTPError as exc:
            raise FlowAIError(f"FlowAI is not reachable at {self.base_url} ({type(exc).__name__})") from None
        if r.status_code >= 400:
            raise FlowAIError(_message(r), r.status_code)
        if not r.content:
            return None
        return r.json() if "json" in r.headers.get("content-type", "") else r.content

    async def get(self, path: str, **params):
        return await self.request("GET", path, params={k: v for k, v in params.items() if v not in (None, "")})

    def link(self, post_id: int) -> str:
        """The post in FlowAI's Composer."""
        return f"{self.dashboard_url}/dashboard/create?post={post_id}"

    async def close(self):
        await self.http.aclose()


@dataclass
class Saved:
    """A draft's post in FlowAI."""
    post: int
    account: int | None
    version: int  # the draft's version that was saved
    status: str  # the post's status in FlowAI
    assets: list[int]  # its slides as uploaded
    payload: dict  # what was saved; sent again, with a time, when the user approves scheduling
    scheduled_at: str | None = None


@dataclass
class Approval:
    """Something the user has to approve on screen before it happens."""
    id: int
    draft: int
    version: int  # the draft as the user saw it when asked
    post: int
    when: datetime | None  # UTC; None = the queue's next free time
    text: str
    status: str = "waiting"  # waiting | approved | declined | failed | outdated
    detail: str = ""


def _cut(text: str, n: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def _voice(voice: dict) -> str:
    """An account's profile and memory as text, the way FlowAI's Voice::context() writes it."""
    profile = voice.get("profile") or {}
    lines = [f"{label}: {profile[key]}" for key, label in VOICE_LABELS.items() if str(profile.get(key) or "").strip()]
    memory = voice.get("memory") or {}
    if memory.get("instruction"):
        lines.append("Instructions from the operator: " + "; ".join(_cut(m["content"], 300) for m in memory["instruction"]))
    if memory.get("example"):
        lines.append("Posts the operator liked (match their feel, don't copy them): "
                     + " | ".join(_cut(m["content"], 400) for m in memory["example"][:4]))
    if memory.get("history"):
        lines.append("Recent posts on this account (don't repeat them): "
                     + " | ".join(_cut(m["content"], 200) for m in memory["history"][:5]))
    return "\n".join(lines)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "slide"


class FlowAISession:
    """One conversation's link to FlowAI: the user's accounts, the drafts' posts, approvals."""

    def __init__(self, client: FlowAIClient, studio: "StudioSession", media: "MediaSession",
                 send: Callable[..., Awaitable[None]], notify: Callable[[str], None],
                 remember: Callable[[str], None]):
        self.client = client
        self.send = send  # to the page; must not raise
        self.notify = notify  # something the assistant should say
        self.remember = remember  # context for the assistant, without a reply
        self.user: dict = {}
        self.accounts: dict[int, dict] = {}
        self.voices: dict[int, str] = {}
        self.error = ""  # why FlowAI could not be read
        self.tz = timezone.utc  # the user's, from the page
        self.lock = asyncio.Lock()  # one save at a time: the Save button and the assistant may race
        self.reset(studio, media)

    def reset(self, studio: "StudioSession", media: "MediaSession"):
        """A new conversation: the user and accounts stay, drafts and approvals go."""
        self.studio, self.media = studio, media
        self.saved: dict[int, Saved] = {}  # draft id ->
        self.approvals: dict[int, Approval] = {}
        self.uploaded: set[int] = set()  # assets this conversation uploaded: the only ones it deletes

    async def use_token(self, token: str):
        """The page's own sign-in (FlowAI's POST /api/assistant/session): act as that user from now on."""
        old, self.client = self.client, FlowAIClient(self.client.base_url, self.client.dashboard_url, token=token)
        await old.close()

    def set_timezone(self, name: str):
        try:
            self.tz = ZoneInfo(str(name))
        except (ZoneInfoNotFoundError, ValueError):
            pass

    # --- what the assistant knows ---------------------------------------------------

    async def load(self):
        """The user, their Instagram and X accounts, and each account's voice."""
        try:
            self.user = await self.client.get("/api/user") or {}
            accounts = await self.client.get("/api/accounts") or []
            self.accounts = {a["id"]: a for a in accounts if a.get("platform") in platforms.SPECS}
            voices = await asyncio.gather(*(self.client.get(f"/api/accounts/{i}/voice") for i in self.accounts))
            self.voices = {i: _voice(v or {}) for i, v in zip(self.accounts, voices)}
            self.error = ""
        except FlowAIError as exc:
            self.error = str(exc)
            log.warning("FlowAI: %s", exc)
        await self.send(type="flowai", error=self.error, dashboard_url=self.client.dashboard_url,
                        user={"name": self.user.get("name"), "email": self.user.get("email")},
                        accounts=[{"id": a["id"], "platform": a["platform"], "handle": a["handle"], "label": a["label"],
                                   "name": a.get("name")}
                                  for a in self.accounts.values()])
        if not self.error:
            await self.send_posts()

    def prompt(self) -> str:
        if self.error and not self.accounts:
            return (f"\n\nFlowAI, where posts are saved, can't be reached right now ({self.error}). Drafts can "
                    "still be made, but not saved or scheduled; tell the user if they ask.")
        accounts = []
        for a in self.accounts.values():
            voice = self.voices.get(a["id"])
            accounts.append(f"- account {a['id']}: {a['label']}" + (f" ({a['name']})" if a.get("name") else "")
                            + (":\n  " + voice.replace("\n", "\n  ") if voice else ""))
        return (
            "\n\nYou work inside FlowAI, the agency's studio" + (f", with {self.user['name']}" if self.user.get("name")
                                                                  else "") + ". "
            "When the user likes a draft or wants to keep it, save it with save_draft: it becomes a FlowAI post, "
            "saved as a draft, which the team sees in the Composer and the calendar; save again after more changes. "
            "find_posts and open_post bring back saved posts to change them, and find_assets and use_assets bring "
            "pictures from the FlowAI gallery into this conversation. To schedule a post, call schedule_post with "
            "the time the user wants: they approve it on screen with one click and nothing is booked before that, "
            "so never say it is scheduled until you are told it was approved. Write captions in the account's "
            "voice. Accounts (a draft goes to the one on its platform):\n"
            + ("\n".join(accounts) or "- none: the user has to add an Instagram or X account in FlowAI first")
        )

    # --- tools ----------------------------------------------------------------------

    async def save(self, ref=None, account=None, reschedule: bool = True) -> dict:
        """Saves a draft as a FlowAI post (status draft), or its changes to the same post."""
        try:
            d = self.studio.get(ref)
            saved = self.saved.get(d.id)
            acct = self._account(d.platform, account if account not in (None, "") else saved and saved.account)
            if any(c["key"] == "caption" and c["status"] == "fail" for c in d.check.get("checks", [])):
                raise StudioError(f"the caption is too long for {d.a_label}, and FlowAI refuses it; shorten it first")
        except (StudioError, FlowAIError) as exc:
            return {"error": str(exc)}
        if saved and saved.version == d.version and saved.account == acct["id"]:
            return {"saved": f"draft {d.id} is already saved as post {saved.post}, with no changes since"}
        # Saving as a draft takes a scheduled post off the calendar until a person approves it again
        was = saved.scheduled_at if saved and saved.status == "scheduled" else None
        async with self.lock:
            try:
                saved = await self._save(d, acct)
            except FlowAIError as exc:
                return {"error": f"FlowAI did not save it: {exc}"}
        reply = {"saved": f"draft {d.id} version {d.version} is post {saved.post} in FlowAI, as a draft",
                 "account": acct["label"]}
        if d.check.get("checks") and not d.check.get("ok"):
            reply["problems"] = [c["detail"] for c in d.check["checks"] if c["status"] == "fail"]
        was = was and datetime.fromisoformat(was.replace("Z", "+00:00"))
        if reschedule and was and was > datetime.now(timezone.utc):
            ask = await self._ask(d, saved, was)
            reply["schedule"] = (f"it was scheduled for {self._local(was)}; with the change it is a draft again, "
                                 f"and the user is asked on screen to approve the same time ({ask['asks']})")
        return reply

    async def find_posts(self, query=None, status=None, platform=None) -> dict:
        try:
            page = await self.client.get("/api/posts", q=query, status=status, platform=platform, per_page=8)
        except FlowAIError as exc:
            return {"error": str(exc)}
        posts = [self._post_info(p) for p in page.get("data", [])]
        return {"posts": posts} if posts else {"posts": [], "note": "nothing found"}

    async def open_post(self, post) -> dict:
        """A FlowAI post as a new draft here. Saving it updates the same post, unless it is already out."""
        number = _number(post)
        same = [i for i, s in self.saved.items() if s.post == number and i in self.studio.drafts]
        if same:  # two drafts saving to one post would undo each other
            result = await self.studio.show(same[-1])
            return {**result, "opened": f"post {number} is already open here as draft {same[-1]}"}
        try:
            p = await self.client.get(f"/api/posts/{number}")
            p = p.get("data", p)
            platform = (p.get("account") or {}).get("platform") or (p.get("platforms") or [""])[0]
            if platform not in platforms.SPECS:
                raise StudioError(f"post {number} is for {platform or 'no platform'}; only Instagram and X posts "
                                  "can be opened here")
            placement = p.get("placement") if p.get("placement") in platforms.placements(platform) \
                else platforms.DEFAULT_PLACEMENT[platform]
            numbers, aspect = [], None
            for a in p.get("assets", []):
                if a["kind"] not in ("image", "video"):
                    continue
                data = await self.client.request("GET", a["url"])
                numbers.append((await self.media.add(data, a["name"], a["kind"], "flowai")).id)
                if aspect is None and a.get("width") and a.get("height") and \
                        platforms.fits(platform, placement, a["width"] / a["height"]):
                    aspect = f"{a['width']}:{a['height']}"
        except (StudioError, FlowAIError) as exc:
            return {"error": str(exc)}
        caption = p.get("body", "") if platforms.spec(platform, placement)["caption"] else ""
        result = await self.studio.create(platform, placement, aspect, p.get("title") or "", caption, numbers)
        if "error" in result:
            return result
        d = self.studio.get(result["draft"])
        live = p["status"] in LIVE
        if not live:
            account = (p.get("account") or {}).get("id")
            ids = [a["id"] for a in p.get("assets", [])]
            self.saved[d.id] = Saved(p["id"], account, d.version, p["status"], ids,
                                     self._payload(d, account, ids), p.get("scheduled_at"))
            await self._send_saved(d)
        result["opened"] = (f"post {p['id']} ({p['status']}" + (f", {self._local(p['scheduled_at'])}"
                                                              if p.get("scheduled_at") else "") + f") is draft {d.id}")
        result["note"] = ("it is already out, so saving makes a new post" if live else
                          "saving updates the same post") + "; words already on its pictures are part of them, " \
                                                            "new texts go on top"
        return result

    async def find_assets(self, query=None, kind=None) -> dict:
        try:
            page = await self.client.get("/api/assets", q=query, kind=kind)
        except FlowAIError as exc:
            return {"error": str(exc)}
        assets = [{"asset": a["id"], "name": a["name"], "kind": a["kind"],
                   **({"size": f"{a['width']}x{a['height']}"} if a.get("width") else {}),
                   "added": (a.get("created_at") or "")[:10]}
                  for a in page.get("data", []) if a["kind"] in ("image", "video")][:12]
        return {"assets": assets} if assets else {"assets": [], "note": "nothing found"}

    async def use_assets(self, assets) -> dict:
        """Gallery files into this conversation, numbered like the others."""
        try:
            ids = _numbers(assets)
            found = {a["id"]: a for a in (await self.client.get("/api/assets", ids=",".join(map(str, ids))))["data"]}
            added, numbers = [], []
            for i in ids:
                a = found.get(i)
                if not a or a["kind"] not in ("image", "video"):
                    raise StudioError(f"there is no picture or video {i} in the FlowAI gallery")
                item = await self.media.add(await self.client.request("GET", a["url"]), a["name"], a["kind"], "flowai")
                added.append(f"asset {i} is {item.kind} number {item.id}")
                numbers.append(item.id)
        except (StudioError, FlowAIError) as exc:
            return {"error": str(exc)}
        return {"added": added, "numbers": numbers,
                "note": "use these numbers in drafts, e.g. media in create_draft or update_draft"}

    async def schedule(self, ref=None, when=None) -> dict:
        """Asks the user to approve publishing a draft at a time; saves it first if needed."""
        try:
            d = self.studio.get(ref)
            fails = [c["detail"] for c in d.check.get("checks", []) if c["status"] == "fail"]
            if fails:
                raise StudioError(f"{d.a_label} can't be scheduled until this is fixed: {'; '.join(fails)}")
            at = self._when(when)
        except StudioError as exc:
            return {"error": str(exc)}
        saved = self.saved.get(d.id)
        if not saved or saved.version != d.version:
            reply = await self.save(d.id, reschedule=False)
            if "error" in reply:
                return reply
            saved = self.saved[d.id]
        if at is None:
            try:
                slots = await self.client.get("/api/queue-slots")
            except FlowAIError as exc:
                return {"error": str(exc)}
            if not slots.get("next_free"):
                return {"error": "the queue has no free posting times; the user can add some under Automations in "
                                 "FlowAI, or give a time"}
        return await self._ask(d, saved, at)

    async def decide(self, approval_id, approve: bool, done: bool = False):
        """The user pressed Approve or Decline on screen. done: the page already booked the post itself,
        with the user's own session (how FlowAI's page does it, FLOWAI_INTEGRATION.md change 2)."""
        a = self.approvals.get(_number(approval_id))
        if not a or a.status != "waiting":
            return
        d, saved = self.studio.drafts.get(a.draft), self.saved.get(a.draft)
        if not approve:
            a.status = "declined"
            await self._send_approval(a)
            self.remember(f"The user declined on screen: {a.text}. Nothing was scheduled.")
            return
        if not d or not saved or not d.version == saved.version == a.version:
            a.status, a.detail = "outdated", "The draft changed after this was asked; ask again."
            await self._send_approval(a)
            return
        data = {**saved.payload, "status": "scheduled"}
        if a.when:
            data["scheduled_at"] = a.when.isoformat()
        else:
            data["queue"] = True
        try:
            async with self.lock:
                post = await self.client.get(f"/api/posts/{a.post}") if done else \
                    await self.client.request("PATCH", f"/api/posts/{a.post}", json=data)
        except FlowAIError as exc:
            a.status, a.detail = "failed", str(exc)
            await self._send_approval(a)
            self.notify(f"Scheduling post {a.post} failed: {exc}. Tell the user briefly.")
            return
        post = post.get("data", post)
        if post["status"] != "scheduled":
            a.status, a.detail = "failed", f"post {a.post} is {post['status']}, not scheduled"
            await self._send_approval(a)
            return
        saved.status, saved.scheduled_at = post["status"], post.get("scheduled_at")
        a.status, a.detail = "approved", f"Scheduled for {self._local(saved.scheduled_at)}"
        await self._send_approval(a)
        await self._send_saved(d)
        await self.send_posts()
        self.notify(f"The user approved it on screen: post {a.post} ({d.label}) is scheduled for "
                    f"{self._local(saved.scheduled_at)}. Tell them in a few words.")

    # --- helpers --------------------------------------------------------------------

    def _account(self, platform: str, ref=None) -> dict:
        if ref not in (None, ""):
            a = self.accounts.get(_number(ref))
            if not a or a["platform"] != platform:
                raise StudioError(f"there is no {platforms.LABELS[platform]} account {ref}; accounts: "
                                  + ", ".join(f"{i} {a['label']}" for i, a in self.accounts.items()))
            return a
        found = [a for a in self.accounts.values() if a["platform"] == platform]
        if not found:
            raise StudioError(self.error or f"there is no {platforms.LABELS[platform]} account in FlowAI; the user "
                                             "can add one under Accounts")
        if len(found) > 1:
            raise StudioError("which account? " + ", ".join(f"{a['id']} {a['label']}" for a in found))
        return found[0]

    def _payload(self, d: Draft, account: int | None, asset_ids: list[int]) -> dict:
        kinds = {f.kind for _, f in d.files}
        caption = d.caption
        if not caption and not platforms.spec(d.platform, d.placement)["caption"]:
            caption = d.title or d.label  # FlowAI wants a body even for stories (FLOWAI_INTEGRATION.md, change 7)
        return {"title": d.title or None, "body": caption,
                "format": "video" if "video" in kinds else "image" if kinds else "text",
                "platforms": [d.platform], "placement": d.placement, "account_id": account,
                "status": "draft", "asset_ids": asset_ids}

    async def _save(self, d: Draft, acct: dict) -> Saved:
        files = []
        for i, (path, f) in enumerate(d.files, 1):
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            files.append(("files[]", (f"{_slug(d.title or d.label)}-{i}{path.suffix}",
                                      await asyncio.to_thread(path.read_bytes), mime)))
        uploaded = await self.client.request("POST", "/api/assets", files=files) if files else []
        ids = [a["id"] for a in uploaded]
        self.uploaded.update(ids)
        data = self._payload(d, acct["id"], ids)
        before = self.saved.get(d.id)
        try:
            try:
                post = await self.client.request("PATCH" if before else "POST",
                                                 f"/api/posts/{before.post}" if before else "/api/posts", json=data)
            except FlowAIError as exc:
                if not before or exc.status != 404:
                    raise
                post = await self.client.request("POST", "/api/posts", json=data)  # deleted in FlowAI meanwhile
        except FlowAIError:
            await self._delete(ids)
            raise
        post = post.get("data", post)
        if before:  # the slides the last save uploaded are replaced
            await self._delete([i for i in before.assets if i in self.uploaded and i not in ids])
        saved = Saved(post["id"], acct["id"], d.version, post["status"], ids, data, post.get("scheduled_at"))
        self.saved[d.id] = saved
        await self._send_saved(d)
        await self.send_posts()
        return saved

    async def _delete(self, asset_ids: list[int]):
        for i in asset_ids:
            try:
                await self.client.request("DELETE", f"/api/assets/{i}")
            except FlowAIError as exc:
                log.warning("FlowAI: could not delete asset %d: %s", i, exc)

    async def _ask(self, d: Draft, saved: Saved, at: datetime | None) -> dict:
        for old in self.approvals.values():  # one question per draft at a time
            if old.draft == d.id and old.status == "waiting":
                old.status, old.detail = "outdated", "Replaced by a newer request."
                await self._send_approval(old)
        acct = self.accounts.get(saved.account) or {}
        what = f"“{d.title}”" if d.title else f"draft {d.id}"
        where = f"{d.label}" + (f", @{acct['handle']}" if acct.get("handle") else "")
        text = f"Schedule {what} ({where}) for {self._local(at) if at else 'the next free time in the queue'}"
        a = Approval(len(self.approvals) + 1, d.id, d.version, saved.post, at, text)
        self.approvals[a.id] = a
        await self._send_approval(a)
        return {"approval": a.id, "status": "waiting for the user to approve it on screen", "asks": text,
                "note": "nothing is booked until the user presses Approve; you will be told"}

    def _when(self, when) -> datetime | None:
        text = str(when or "").strip()
        if any(w in text.lower() for w in QUEUE_WORDS):
            return None
        try:
            at = datetime.fromisoformat(text.replace("Z", "+00:00").replace(" ", "T", 1))
        except ValueError:
            raise StudioError(f"{when!r} is not a date and time; give it like 2026-10-16T09:00, "
                              "or \"queue\" for the next free posting time") from None
        if at.tzinfo is None:
            at = at.replace(tzinfo=self.tz)
        at = at.astimezone(timezone.utc)
        if at < datetime.now(timezone.utc) + timedelta(minutes=1):
            raise StudioError(f"{self._local(at)} has already passed; pick a time in the future")
        return at

    def _local(self, at: datetime | str | None) -> str:
        if not at:
            return ""
        if isinstance(at, str):
            at = datetime.fromisoformat(at.replace("Z", "+00:00"))
        t = at.astimezone(self.tz)
        return f"{t:%A} {t.day} {t:%B}, {t.hour}:{t:%M}"

    def _post_info(self, p: dict) -> dict:
        platform = (p.get("account") or {}).get("platform") or (p.get("platforms") or ["?"])[0]
        placement = p.get("placement") or platforms.DEFAULT_PLACEMENT.get(platform)
        info = {"post": p["id"], "title": p.get("title") or _cut(p.get("body", ""), 50),
                "kind": platforms.label(platform, placement) if platform in platforms.SPECS else platform,
                "status": p["status"]}
        if p.get("account"):
            info["account"] = f"@{p['account']['handle']}"
        if p.get("scheduled_at"):
            info["when"] = self._local(p["scheduled_at"])
        info["slides"] = len(p.get("assets", []))
        info["caption"] = _cut(p.get("body", ""), 90)
        return info

    # --- the page -------------------------------------------------------------------

    async def send_posts(self):
        """The latest posts, for the page's list."""
        try:
            page = await self.client.get("/api/posts", per_page=8)
        except FlowAIError as exc:
            log.warning("FlowAI: %s", exc)
            return
        await self.send(type="posts", posts=[{**self._post_info(p), "url": self.client.link(p["id"])}
                                              for p in page.get("data", [])])

    async def _send_saved(self, d: Draft):
        s = self.saved[d.id]
        await self.send(type="saved", draft=d.id, post=s.post, version=s.version, status=s.status,
                        url=self.client.link(s.post), when=self._local(s.scheduled_at) if s.scheduled_at else "")

    async def _send_approval(self, a: Approval):
        """when (UTC, or None for the queue) and payload let a page book the post itself."""
        saved = self.saved.get(a.draft)
        await self.send(type="approval", id=a.id, draft=a.draft, post=a.post, text=a.text, status=a.status,
                        detail=a.detail, when=a.when.isoformat() if a.when else None,
                        payload=saved.payload if saved and a.status == "waiting" else None)
