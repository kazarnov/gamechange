"""Post drafts for Instagram and X: what the assistant makes, then changes when asked.

A draft mirrors a FlowAI post (content-generator/backend/app/Models/Post.php): one platform and
placement, a caption, and media in order. Each piece of media is a slide: a picture or video
from this conversation (a MediaSession number) with texts drawn on top. Texts are drawn here
with Pillow, never by the image model, so they are crisp, spelled right and easy to change.

Every change makes a new version: it is rendered, checked against the platform's specs
(platforms.py) and sent to the web UI. undo steps back through earlier versions. Rendered
files are kept in MEDIA_DIR next to the pictures, served at /media.
"""

import asyncio
import copy
import html
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont, ImageOps

from . import platforms
from .platforms import File

if TYPE_CHECKING:
    from .media import MediaItem, MediaSession

log = logging.getLogger(__name__)

MAX_HISTORY = 30  # versions kept for undo, per draft
MAX_TEXT = 300
POSITIONS = ["top-left", "top", "top-right", "left", "center", "right", "bottom-left", "bottom", "bottom-right"]
SIZES = {"small": 0.045, "medium": 0.065, "large": 0.09, "huge": 0.13}  # of the picture's shorter side
STYLES = ["shadow", "outline", "box", "plain"]
# Short names for the fonts in FONTS_DIR; any other .ttf/.otf there is used by its file name
FONT_ALIASES = {"bold": "geist-extrabold", "semibold": "geist-semibold", "serif": "instrumentserif-italic",
                "mono": "geistmono-medium"}
LOOK = ("position", "size", "color", "style", "box_color", "font")  # what _style takes
FALLBACK_FONT = "DejaVuSans-Bold.ttf"  # looked up in the system's font folders
BACKGROUND = "#1c1c1c"  # behind slides that have no picture yet
PLACEHOLDER = re.compile(r"\[[^\[\]\n]{1,30}\]")  # "[date]", "[link]": must not go out


class StudioError(Exception):
    """A problem with a message the assistant can pass on to the user."""


def _number(value) -> int | None:
    m = re.search(r"\d+", str(value))
    return int(m.group()) if m else None


def _numbers(value) -> list[int]:
    parts = value if isinstance(value, (list, tuple)) else re.findall(r"\d+", str(value))
    numbers = [_number(p) for p in parts]
    if None in numbers:
        raise StudioError(f"{value!r} is not a list of picture numbers")
    return numbers


def _color(value: str) -> tuple[int, int, int]:
    try:
        return ImageColor.getrgb(str(value).strip())[:3]
    except ValueError:
        raise StudioError(f"{value!r} is not a colour; use a name like white or a hex code like #ffd400") from None


def _plain(text) -> str:
    """As the user should read it: some models write & as &amp; and line breaks as \\n."""
    return html.unescape(str(text or "")).replace("\\n", "\n").strip()


def _words(text) -> str:
    text = _plain(text)
    if not text:
        raise StudioError("the text is empty")
    if len(text) > MAX_TEXT:
        raise StudioError(f"that is {len(text)} characters; text on a picture should be short (max {MAX_TEXT})")
    return text


def _state_out(state: dict) -> dict:
    """A draft's state (Draft.state) as JSON."""
    return {**state, "slides": [{"media": s.media, "texts": [asdict(t) for t in s.texts]} for s in state["slides"]]}


def _state_in(data: dict) -> dict:
    state = {k: data[k] for k in Draft.EDITABLE if k in data}
    state["slides"] = [Slide(s.get("media"), [Text(**t) for t in s.get("texts", [])]) for s in data.get("slides", [])]
    return state


@dataclass
class Text:
    id: int
    text: str
    position: str = "bottom"
    size: str = "large"  # a SIZES name, or pixels
    color: str = "white"
    style: str = "shadow"
    box_color: str = "black"  # the panel for style box, the outline for style outline
    font: str = "bold"

    def info(self) -> dict:
        return {"text": self.id, "words": self.text, "position": self.position, "size": self.size,
                "color": self.color, "style": self.style, "font": self.font}


@dataclass
class Slide:
    media: int | None = None  # picture or video number in this conversation; None = plain background
    texts: list[Text] = field(default_factory=list)


@dataclass
class Draft:
    id: int
    platform: str
    placement: str
    aspect: str | None = None  # e.g. "1:1"; None = the placement's own shape
    title: str = ""
    caption: str = ""
    background: str = BACKGROUND
    slides: list[Slide] = field(default_factory=list)
    text_count: int = 0
    version: int = 0
    history: list[dict] = field(default_factory=list)  # earlier states, newest last
    files: list[tuple[Path, File]] = field(default_factory=list)  # one per slide, as rendered
    boxes: list[list[dict]] = field(default_factory=list)  # per slide: where each text was drawn
    check: dict = field(default_factory=dict)

    EDITABLE = ("platform", "placement", "aspect", "title", "caption", "background", "slides", "text_count")

    def state(self) -> dict:
        return copy.deepcopy({k: getattr(self, k) for k in self.EDITABLE})

    def restore(self, state: dict):
        for k, v in copy.deepcopy(state).items():
            setattr(self, k, v)

    @property
    def size(self) -> tuple[int, int]:
        ratio = platforms.parse_aspect(self.aspect) if self.aspect else None
        return platforms.canvas_size(self.platform, self.placement, ratio)

    @property
    def label(self) -> str:
        return platforms.label(self.platform, self.placement)

    @property
    def a_label(self) -> str:
        return f"an {self.label}"  # Instagram…, X… (said "ex")

    def find_text(self, ref) -> tuple[Slide, Text]:
        number = _number(ref)
        for slide in self.slides:
            for t in slide.texts:
                if t.id == number:
                    return slide, t
        known = ", ".join(str(t.id) for s in self.slides for t in s.texts)
        raise StudioError(f"draft {self.id} has no text {ref}" + (f"; its texts are {known}" if known else
                                                                   "; it has no texts yet"))


class StudioSession:
    def __init__(self, media: "MediaSession | None", media_dir: Path, fonts_dir: Path,
                 send: Callable[..., Awaitable[None]], prefix: str = ""):
        self.media = media  # None when picture generation is off
        self.dir = media_dir
        self.prefix = prefix  # of the drawn files' names: the conversation's, to delete them with it
        self.fonts_dir = fonts_dir
        self.send = send  # to the web UI; must not raise
        self.drafts: dict[int, Draft] = {}
        self.count = 0
        self.lock = asyncio.Lock()  # one change at a time, tools and finished generations alike

    # --- tools ----------------------------------------------------------------------

    async def create(self, platform: str, placement=None, aspect=None, title=None, caption=None, media=None,
                     from_draft=None, background=None, texts=None, pictures=None, workflow=None) -> dict:
        """A whole post in one go: texts as add_text takes them (a list of dicts), and pictures as one
        prompt per slide, made in the background and put in when ready."""
        try:
            platform = {"twitter": "x"}.get(str(platform).strip().lower(), str(platform).strip().lower())
            if platform not in platforms.SPECS:
                raise StudioError(f"drafts are for {' or '.join(platforms.LABELS.values())}, not {platform}")
            source = self.get(from_draft) if from_draft not in (None, "") else None
        except StudioError as exc:
            return {"error": str(exc)}

        def fill(d: Draft):
            if source:
                d.title, d.caption, d.background = source.title, source.caption, source.background
                d.slides, d.text_count = copy.deepcopy(source.slides), source.text_count
            self._set(d, placement, aspect, title, caption, media, background)
            if source and texts:  # given texts replace the copied ones rather than doubling them
                for slide in d.slides:
                    slide.texts = []
            for t in texts or []:
                if not isinstance(t, dict) or "text" not in t:
                    raise StudioError("each of texts needs its words in text, e.g. {\"text\": \"Summer sale\"}")
                t = dict(t)
                self._add_text(d, t.pop("text"), t.pop("slide", None), **{k: v for k, v in t.items() if k in LOOK})
            if pictures:  # slides for them, so a carousel has its shape before the pictures arrive
                self._slide(d, len(pictures))

        self.count += 1
        draft = Draft(self.count, platform, platforms.DEFAULT_PLACEMENT[platform])
        result = await self._apply(draft, fill, record=False)
        if "error" in result:
            self.count -= 1
            return result
        self.drafts[draft.id] = draft
        if pictures:
            result["pictures"] = await self._pictures(draft, pictures, workflow)
        return result

    async def _pictures(self, d: Draft, prompts: list, workflow=None) -> str:
        if not self.media or not self.media.comfy:
            return "not made: picture generation is off (COMFYUI_URL is not set)"
        workflow = workflow or self.media.default_workflow()
        if not workflow:
            return "not made: there is no text-to-image workflow on the pod"
        started, failed = [], []
        for n, prompt in enumerate(prompts, 1):
            if str(prompt or "").strip():
                reply = await self.generate(d.id, n, workflow, str(prompt))
                (failed.append(f"slide {n}: {reply['error']}") if "error" in reply else started.append(str(n)))
        said = f"making slide{'s' if len(started) > 1 else ''} {', '.join(started)} with {workflow}; each goes " \
               "in by itself when ready" if started else ""
        return "; ".join(filter(None, [said, *failed]))

    async def update(self, ref=None, placement=None, aspect=None, title=None, caption=None, media=None,
                     background=None) -> dict:
        return await self._change(ref, lambda d: self._set(d, placement, aspect, title, caption, media, background))

    async def add_text(self, ref=None, text="", slide=None, **style) -> dict:
        return await self._change(ref, lambda d: self._add_text(d, text, slide, **style))

    async def edit_text(self, ref=None, text_id=None, text=None, slide=None, **style) -> dict:
        def change(d: Draft):
            current, t = d.find_text(text_id)
            if text not in (None, ""):
                t.text = _words(text)
            self._style(t, **style)
            if slide not in (None, ""):
                target = self._slide(d, slide)
                if target is not current:
                    current.texts.remove(t)
                    target.texts.append(t)
        return await self._change(ref, change)

    async def remove_text(self, ref=None, text_id=None) -> dict:
        def change(d: Draft):
            current, t = d.find_text(text_id)
            current.texts.remove(t)
        return await self._change(ref, change)

    async def arrange(self, ref=None, order=None) -> dict:
        """Slides in a new order, or some dropped (order: the slide numbers to keep, as they should
        be); each slide keeps its texts."""
        def change(d: Draft):
            numbers = _numbers(order)
            if not numbers or len(set(numbers)) != len(numbers) or not all(1 <= n <= len(d.slides) for n in numbers):
                raise StudioError(f"order takes slide numbers from 1 to {len(d.slides)}, each once")
            d.slides = [d.slides[n - 1] for n in numbers]
        return await self._change(ref, change)

    async def place(self, ref=None, media=None, slide=None) -> dict:
        """A picture or video on one slide (its texts stay), or as a new last slide without `slide`."""
        def change(d: Draft):
            item = self._item(_number(media))
            if slide in (None, ""):
                d.slides.append(Slide())
            target = d.slides[-1] if slide in (None, "") else self._slide(d, slide)
            if item.kind == "video" and target.texts:
                raise StudioError("text can't be drawn on videos yet; remove that slide's texts first")
            target.media = item.id
        return await self._change(ref, change)

    async def undo(self, ref=None) -> dict:
        try:
            d = self.get(ref)
        except StudioError as exc:
            return {"error": str(exc)}
        if not d.history:
            return {"error": f"draft {d.id} has no earlier version"}
        state = d.history[-1]
        result = await self._apply(d, lambda d: d.restore(state), record=False)
        if "error" not in result:
            d.history.pop()
        return result

    async def show(self, ref=None) -> dict:
        if ref in (None, "") and len(self.drafts) != 1:
            if not self.drafts:
                return {"drafts": [], "note": "no drafts yet"}
            return {"drafts": [{"draft": d.id, "title": d.title, "post": d.label, "version": d.version}
                               for d in self.drafts.values()]}
        try:
            d = self.get(ref)
        except StudioError as exc:
            return {"error": str(exc)}
        await self.send(**self._event(d))
        return self.summary(d)

    async def generate(self, ref, slide, workflow: str, prompt: str, image=None, options=None) -> dict:
        """A picture or video for one slide: an editing workflow edits the slide's current picture."""
        if not self.media or not self.media.comfy:
            return {"error": "picture generation is off (COMFYUI_URL is not set)"}
        try:
            d = self.get(ref)
            n = 1 if slide in (None, "") else _number(slide)
            most = platforms.spec(d.platform, d.placement)["items"][1]
            if not n or n > most:
                raise StudioError(f"{d.a_label} has slides 1 to {most}, not {slide}")
        except StudioError as exc:
            return {"error": str(exc)}
        current = d.slides[n - 1].media if n <= len(d.slides) else None
        w, h = d.size
        reply = await self.media.generate(workflow, prompt, image, options, ratio=w / h,
                                          source=None if current is None else str(current),
                                          then=lambda items: self._made(d.id, n, items))
        if "error" not in reply:
            reply.update(draft=d.id, slide=n, note="the result goes into the draft by itself when it is ready")
        return reply

    async def _made(self, draft_id: int, n: int, items: list["MediaItem"]) -> str:
        """A generation for a draft finished: put it on its slide. Returns a sentence for the assistant."""
        d = self.drafts.get(draft_id)
        item = next((i for i in items if i.kind in ("image", "video")), None)
        if not d or not item:
            return ""
        result = await self._apply(d, lambda d: setattr(self._slide(d, n), "media", item.id))
        if "error" in result:
            return f"It could not go into draft {d.id}: {result['error']}"
        note = f"It is now slide {n} of draft {d.id} (version {d.version}), shown on screen."
        if item.kind == "video" and d.slides[n - 1].texts:
            note += " The texts on that slide are not drawn on the video."
        if result["problems"] != "none":
            note += f" Platform check: {'; '.join(result['problems'])}."
        return note

    # --- changes --------------------------------------------------------------------

    def get(self, ref=None) -> Draft:
        if not self.drafts:
            raise StudioError("there are no drafts yet; make one with create_draft")
        if ref in (None, "") or str(ref).strip().lower() in ("latest", "last", "this", "it", "current"):
            return self.drafts[max(self.drafts)]
        n = _number(ref)
        if n not in self.drafts:
            raise StudioError(f"there is no draft {ref}; the drafts are {', '.join(map(str, self.drafts))}")
        return self.drafts[n]

    def _item(self, number: int) -> "MediaItem":
        item = self.media.items.get(number) if self.media else None
        if not item:
            raise StudioError(f"there is no picture number {number} in this conversation" if self.media else
                              "there are no pictures in this conversation (picture generation is off)")
        return item

    def _slide(self, d: Draft, number) -> Slide:
        """Slide `number` (default 1), adding empty slides up to it."""
        n = 1 if number in (None, "") else _number(number)
        if not n:
            raise StudioError(f"there is no slide {number}")
        while len(d.slides) < n:
            d.slides.append(Slide())  # _validate checks how many the placement takes
        return d.slides[n - 1]

    def _set(self, d: Draft, placement=None, aspect=None, title=None, caption=None, media=None, background=None):
        if placement not in (None, ""):
            placement = str(placement).strip().lower()
            if placement not in platforms.placements(d.platform):
                raise StudioError(f"{platforms.LABELS[d.platform]} placements are "
                                  f"{', '.join(platforms.placements(d.platform))}, not {placement}")
            if placement != d.placement and aspect in (None, ""):
                d.aspect = None  # the old shape may not suit the new placement
            d.placement = placement
        if aspect not in (None, ""):
            d.aspect = str(aspect).strip()
        if title is not None:
            d.title = _plain(title)[:120]
        if caption is not None:
            d.caption = _plain(caption)
        if background not in (None, ""):
            _color(background)
            d.background = str(background).strip()
        if media is not None:  # the new list of pictures; slides keep their texts by position
            numbers = _numbers(media)
            d.slides = [*d.slides[:len(numbers)], *(Slide() for _ in range(len(numbers) - len(d.slides)))]
            for slide, number in zip(d.slides, numbers):
                slide.media = number

    def _add_text(self, d: Draft, text, slide=None, **style):
        target = self._slide(d, slide)
        if target.media is not None and self._item(target.media).kind == "video":
            raise StudioError("text can't be drawn on videos yet; put it in the caption or on a picture slide")
        d.text_count += 1
        t = Text(d.text_count, _words(text))
        self._style(t, **style)
        target.texts.append(t)

    def _style(self, t: Text, position=None, size=None, color=None, style=None, box_color=None, font=None):
        if position not in (None, ""):
            p = re.sub(r"[\s_]+", "-", str(position).strip().lower())
            p = {"middle": "center", "centre": "center", "top-center": "top", "bottom-center": "bottom",
                 "center-left": "left", "center-right": "right"}.get(p, p)
            if p not in POSITIONS:
                raise StudioError(f"position {position!r} is not one of {', '.join(POSITIONS)}")
            t.position = p
        if size not in (None, ""):
            s = str(size).strip().lower().removesuffix("px")
            if s.isdigit() and 10 <= int(s) <= 600:
                t.size = s
            elif s in SIZES:
                t.size = s
            else:
                raise StudioError(f"size {size!r} is not one of {', '.join(SIZES)} or a pixel size from 10 to 600")
        if color not in (None, ""):
            _color(color)
            t.color = str(color).strip()
        if box_color not in (None, ""):
            _color(box_color)
            t.box_color = str(box_color).strip()
        if style not in (None, ""):
            if str(style).strip().lower() not in STYLES:
                raise StudioError(f"style {style!r} is not one of {', '.join(STYLES)}")
            t.style = str(style).strip().lower()
        if font not in (None, ""):
            name = str(font).strip().lower()
            if name not in self.fonts() and name not in FONT_ALIASES:
                named = [f for f in self.fonts() if f not in FONT_ALIASES.values()]  # aliases, not their files
                raise StudioError(f"there is no font {font!r}; fonts: {', '.join(named) or ', '.join(FONT_ALIASES)}")
            t.font = name

    def _validate(self, d: Draft):
        s = platforms.spec(d.platform, d.placement)
        if d.aspect:
            try:
                ratio = platforms.parse_aspect(d.aspect)
            except ValueError as exc:
                raise StudioError(str(exc)) from None
            if not platforms.fits(d.platform, d.placement, ratio):
                lo, hi = s["ratio"]
                raise StudioError(f"{d.aspect} doesn't fit {d.a_label}: it takes shapes from {lo:g}:1 to {hi:g}:1, "
                                  f"best {s['ratio_hint']}")
        most = s["items"][1]
        if len(d.slides) > most:
            raise StudioError(f"{d.a_label} takes at most {most} {'picture' if most == 1 else 'pictures'}, this would "
                              f"have {len(d.slides)}; give media with the ones to keep")
        for slide in d.slides:
            if slide.media is not None and self._item(slide.media).kind not in ("image", "video"):
                raise StudioError(f"number {slide.media} is not a picture or a video")

    async def _change(self, ref, change: Callable[[Draft], None]) -> dict:
        try:
            d = self.get(ref)
        except StudioError as exc:
            return {"error": str(exc)}
        return await self._apply(d, change)

    async def _apply(self, d: Draft, change: Callable[[Draft], None], record: bool = True) -> dict:
        """Change, check, render and show a draft; on any problem it stays as it was."""
        async with self.lock:
            before = d.state()
            try:
                change(d)
                self._validate(d)
                files, boxes = await asyncio.to_thread(self._render, d, d.version + 1)
            except StudioError as exc:
                d.restore(before)
                return {"error": str(exc)}
            except Exception as exc:
                log.exception("draft %d could not be rendered", d.id)
                d.restore(before)
                return {"error": f"the draft could not be drawn: {exc}"}
            if record and d.version:
                d.history = [*d.history, before][-MAX_HISTORY:]
            d.version += 1
            d.files, d.boxes = files, boxes
            d.check = platforms.check(d.platform, d.placement, d.caption, [f for _, f in files])
            # Ours, not FlowAI's: a "[date]" left in would be published as is
            holes = sorted({h for words in [d.caption, *(t.text for s in d.slides for t in s.texts)]
                            for h in PLACEHOLDER.findall(words)})
            if holes:
                d.check["ok"] = False
                d.check["checks"].append({"key": "placeholder", "status": "fail", "label": "Placeholder",
                                          "detail": f"{', '.join(holes)} still to be filled in."})
        await self.send(**self._event(d))
        return self.summary(d)

    # --- kept with the conversation (backend/conversations.py) -------------------------

    def dump(self) -> dict:
        return {"count": self.count, "drafts": [
            {"id": d.id, "version": d.version, **_state_out(d.state()),
             "history": [_state_out(h) for h in d.history],
             "files": [{"file": path.name, **asdict(f)} for path, f in d.files],
             "boxes": d.boxes, "check": d.check}
            for d in self.drafts.values()]}

    async def restore(self, data: dict):
        self.count = data.get("count", 0)
        for raw in data.get("drafts", []):
            d = Draft(raw["id"], raw["platform"], raw["placement"])
            d.restore(_state_in(raw))
            d.version, d.check, d.boxes = raw.get("version", 0), raw.get("check", {}), raw.get("boxes", [])
            d.history = [_state_in(h) for h in raw.get("history", [])]
            d.files = [(self.dir / Path(f.pop("file")).name, File(**f)) for f in map(dict, raw.get("files", []))]
            self.drafts[d.id] = d
            if not all(path.is_file() for path, _ in d.files):  # MEDIA_DIR lost them: draw them again
                await self._apply(d, lambda d: None, record=False)

    # --- output ---------------------------------------------------------------------

    def summary(self, d: Draft) -> dict:
        """What the assistant is told after every change."""
        s = platforms.spec(d.platform, d.placement)
        w, h = d.size
        slides = []
        for i, slide in enumerate(d.slides, 1):
            item = self.media.items.get(slide.media) if self.media and slide.media is not None else None
            info = {"slide": i, "picture": f"{item.kind} {item.id}" if item else "none yet (plain background)"}
            if slide.texts:
                info["texts"] = [t.info() for t in slide.texts]
            slides.append(info)
        problems = [f"{c['status']}, {c['label'].lower()}: {c['detail']}" for c in d.check.get("checks", [])
                    if c["status"] != "pass"]
        return {"draft": d.id, "version": d.version, "post": d.label, "size": f"{w}x{h}", "title": d.title,
                "caption": d.caption,
                "caption_length": f"{len(d.caption)} of {s['caption']}" if s["caption"] else "this placement has no caption",
                "slides": slides or "none (text only)", "problems": problems or "none"}

    def events(self) -> list[dict]:
        """Every draft, as the web UI shows it."""
        return [self._event(d) for d in self.drafts.values()]

    def _event(self, d: Draft) -> dict:
        """The draft as the web UI shows it."""
        return {"type": "draft", "id": d.id, "version": d.version, "title": d.title, "label": d.label,
                "platform": d.platform, "placement": d.placement, "size": list(d.size), "caption": d.caption,
                "caption_limit": platforms.spec(d.platform, d.placement)["caption"],
                "placements": platforms.placements(d.platform),
                "max_slides": platforms.spec(d.platform, d.placement)["items"][1],
                "slides": [{"url": f"media/{path.name}", "kind": f.kind, "media": slide.media, "texts": boxes}
                           for (path, f), slide, boxes in zip(d.files, d.slides, d.boxes)],
                "check": d.check}

    def fonts(self) -> dict[str, Path]:
        found = {p.stem.lower(): p for p in sorted(self.fonts_dir.glob("*"))
                 if p.suffix.lower() in (".ttf", ".otf")} if self.fonts_dir.is_dir() else {}
        return {**{alias: found[f] for alias, f in FONT_ALIASES.items() if f in found}, **found}

    def _render(self, d: Draft, version: int) -> tuple[list[tuple[Path, File]], list[list[dict]]]:
        """Each slide as it would be posted, and where its texts are (runs in a thread)."""
        size = d.size
        safe = platforms.spec(d.platform, d.placement)["safe"]
        fonts = self.fonts()
        out, layout = [], []
        for i, slide in enumerate(d.slides, 1):
            item = self._item(slide.media) if slide.media is not None else None
            if item and item.kind == "video":
                out.append((item.path, File("video", None, None, item.path.stat().st_size)))
                layout.append([])
                continue
            img = Image.new("RGB", size, _color(d.background))
            if item:
                with Image.open(item.path) as src:
                    img = ImageOps.fit(ImageOps.exif_transpose(src).convert("RGB"), size, Image.LANCZOS)
            taken: dict[str, float] = {}  # texts at the same position stack instead of overlapping
            boxes = {}
            bottom = [t for t in reversed(slide.texts) if t.position.startswith("bottom")]  # last one lowest
            for t in [t for t in slide.texts if t not in bottom] + bottom:
                img, height, (x, y, w, h) = draw_text(img, t, fonts.get(t.font), safe, taken.get(t.position, 0))
                taken[t.position] = taken.get(t.position, 0) + height
                boxes[t.id] = {"id": t.id, "words": t.text,  # as fractions of the slide, for the page
                               "box": [round(x / size[0], 4), round(y / size[1], 4),
                                       round(w / size[0], 4), round(h / size[1], 4)],
                               **{k: getattr(t, k) for k in LOOK}}
            path = self.dir / f"{self.prefix}draft{d.id}-v{version}-{i}-{uuid.uuid4().hex[:6]}.jpg"
            img.save(path, "JPEG", quality=90, optimize=True)
            out.append((path, File("image", size[0], size[1], path.stat().st_size)))
            layout.append([boxes[t.id] for t in slide.texts])
        return out, layout


# --- drawing text ---------------------------------------------------------------------

@lru_cache(maxsize=64)
def _font(path: str | None, size: int) -> ImageFont.FreeTypeFont:
    for candidate in filter(None, (path, FALLBACK_FONT)):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _wrap(text: str, font, max_width: float) -> list[str]:
    """Greedy wrap, then as narrow as the same number of lines allows, so no line ends up alone."""
    lines = _greedy(text, font, max_width)
    lo, hi = max(font.getlength(w) for w in text.split()), max_width
    while len(lines) > 1 and hi - lo > 4:
        mid = (lo + hi) / 2
        if len(_greedy(text, font, mid)) == len(lines):
            hi = mid
        else:
            lo = mid
    return _greedy(text, font, hi) if len(lines) > 1 else lines


def _greedy(text: str, font, max_width: float) -> list[str]:
    lines = []
    for para in text.split("\n"):
        words = para.split()
        line = words[0] if words else ""
        for word in words[1:]:
            if font.getlength(f"{line} {word}") <= max_width:
                line += f" {word}"
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def draw_text(img: Image.Image, t: Text, font_path: Path | None, safe: tuple[float, float],
              shift: float = 0) -> tuple[Image.Image, float, tuple[float, float, float, float]]:
    """Draws one text on the picture, wrapped to fit, kept out of the platform's covered edges.
    shift: room taken by earlier texts at the same position, which this one goes past (below them at
    the top, above them at the bottom, under them elsewhere). Returns the picture, the room it took,
    and where it is: x, y, width and height in pixels, panel included."""
    W, H = img.size
    size = int(t.size) if t.size.isdigit() else round(SIZES[t.size] * min(W, H))
    side, top, bottom = round(0.06 * W), round(safe[0] * H), round(safe[1] * H)
    max_w, max_h = W - 2 * side, (H - top - bottom) * 0.6  # a long text shrinks rather than fill the picture
    while True:  # shrink until it fits
        pad = size * 0.4 if t.style == "box" else 0  # the panel around a boxed text
        font = _font(str(font_path) if font_path else None, size)
        lines = _wrap(t.text, font, max_w - 2 * pad)
        widths = [font.getlength(line) for line in lines]
        _, ascent, _, descent = font.getbbox("Hg", anchor="ls")  # relative to the baseline: ascent < 0
        step = round(size * 1.15)
        block_w, block_h = max(widths), step * (len(lines) - 1) + descent - ascent
        if (block_w + 2 * pad <= max_w and block_h + 2 * pad <= max_h) or size <= 12:
            break
        size = max(12, int(size * 0.9))

    v = "top" if t.position.startswith("top") else "bottom" if t.position.startswith("bottom") else "middle"
    h = "left" if t.position.endswith("left") else "right" if t.position.endswith("right") else "center"
    x0 = side + pad if h == "left" else W - side - pad - block_w if h == "right" else (W - block_w) / 2
    y0 = top + pad + shift if v == "top" else H - bottom - pad - block_h - shift if v == "bottom" \
        else (H - block_h) / 2 + shift

    def lines_on(draw: ImageDraw.ImageDraw, fill, dy: float = 0, **kwargs):
        for i, (line, w) in enumerate(zip(lines, widths)):
            x = x0 if h == "left" else x0 + block_w - w if h == "right" else x0 + (block_w - w) / 2
            draw.text((x, y0 - ascent + i * step + dy), line, font=font, fill=fill, anchor="ls", **kwargs)

    img = img.convert("RGBA")
    if t.style == "box":
        panel = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(panel).rounded_rectangle((x0 - pad, y0 - pad, x0 + block_w + pad, y0 + block_h + pad),
                                                radius=size * 0.25, fill=(*_color(t.box_color), 210))
        img = Image.alpha_composite(img, panel)
    elif t.style == "shadow":
        shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        lines_on(ImageDraw.Draw(shadow), (0, 0, 0, 170), dy=max(2, size * 0.05))
        img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(max(2, size * 0.08))))
    outline = {"stroke_width": max(2, round(size * 0.06)), "stroke_fill": _color(t.box_color)} \
        if t.style == "outline" else {}
    lines_on(ImageDraw.Draw(img), _color(t.color), **outline)
    box = (x0 - pad, y0 - pad, block_w + 2 * pad, block_h + 2 * pad)
    return img.convert("RGB"), block_h + 2 * pad + size * 0.35, box
