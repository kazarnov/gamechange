"""Post drafts for Instagram and X: what the assistant makes, then changes when asked.

A draft mirrors a FlowAI post (content-generator/backend/app/Models/Post.php): one platform and
placement, a caption, and media in order. Each piece of media is a slide: a picture or video
from this conversation (a MediaSession number) with texts drawn on top. Texts are drawn here
with Pillow, never by the image model, so they are crisp, spelled right and easy to change.

Every change makes a new version: it is rendered, checked against the platform's specs
(platforms.py) and sent to the web UI. undo steps back through earlier versions. Rendered
files are kept in MEDIA_DIR next to the pictures, served at /media.

Each slide also comes with the colours of its picture (for texts that match it), and each text with
how readable it is where it sits; a text that is hard to read is a warning in the draft's check.
Part of a slide's picture can be changed alone: the user paints over it on the page (a mask) and
edit_area sends just that part to the editing model (retouch.py). And a draft can become a video
with the assistant's voice over its pictures (make_video, reel.py), kept as a draft of its own.
"""

import asyncio
import base64
import colorsys
import copy
import html
import io
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont, ImageOps

from . import platforms, reel, retouch
from .platforms import File

if TYPE_CHECKING:
    from .media import MediaItem, MediaSession

log = logging.getLogger(__name__)

MAX_HISTORY = 30  # versions kept for undo, per draft
MAX_TEXT = 300
POSITIONS = ["top-left", "top", "top-right", "left", "center", "right", "bottom-left", "bottom", "bottom-right"]
SIZES = {"small": 0.045, "medium": 0.065, "large": 0.09, "huge": 0.13}  # of the picture's shorter side
STYLES = ["shadow", "outline", "box", "plain"]
LOOK = ("position", "size", "color", "style", "box_color", "font")  # what _style takes
FALLBACK_FONT = "DejaVuSans-Bold.ttf"  # looked up in the system's font folders
BACKGROUND = "#1c1c1c"  # behind slides that have no picture yet
PLACEHOLDER = re.compile(r"\[[^\[\]\n]{1,30}\]")  # "[date]", "[link]": must not go out
HARD_TO_READ = 2.2  # readability (a contrast ratio, see readability()) under which a text gets a warning
EDIT_MODES = {"change": "Changing the painted area", "remove": "Removing what's painted",
              "replace": "Replacing what's painted", "improve": "Improving the painted area"}
EDIT_PROMPTS = {
    # Remove and replace get the area in flat magenta (retouch.mark): the model fills it in
    "remove": "Remove the magenta shape. Where it was, show only what was behind it: continue the surroundings "
              "seamlessly (the same surfaces, texture, light and focus), with no object, figure or silhouette "
              "there and nothing magenta left. Keep everything else exactly the same.",
    "replace": "Replace the magenta shape with {what}, fitting the scene's light, perspective and focus. Around it, "
               "continue the surroundings seamlessly, with nothing magenta left. Keep everything else exactly the same.",
    "change": "{what}. Keep everything else exactly the same.",
    "improve": "Improve the quality of this picture{what}: sharper and finer detail, cleaner light, more realistic. "
               "Keep the same subject, shape, colours, position and framing.",
}
VIDEO_PLACEMENT = {"instagram": "reel", "x": "post"}  # where a video with a voice goes by default
MAX_LINE = 400  # characters a picture's line may have


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
    video: dict | None = None  # a video made with make_video: how (its lines, end card, music...) and from which draft
    version: int = 0
    history: list[dict] = field(default_factory=list)  # earlier states, newest last
    files: list[tuple[Path, File]] = field(default_factory=list)  # one per slide, as rendered
    boxes: list[list[dict]] = field(default_factory=list)  # per slide: where each text was drawn
    palettes: list[list[str]] = field(default_factory=list)  # per slide: its picture's colours
    check: dict = field(default_factory=dict)

    EDITABLE = ("platform", "placement", "aspect", "title", "caption", "background", "slides", "text_count", "video")

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
        # The area the user painted over on a slide (one at a time): draft, slide, the PNG, the draft's
        # size then, and where it is in words. seq goes up with every change, so the LLM is told once.
        self.mask: dict | None = None
        self.mask_seq = 0

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

    # --- part of a picture: the area the user painted over (retouch.py) -----------------

    def set_mask(self, draft=None, slide=None, data: str | None = None) -> str | None:
        """The page's painted area (a PNG data URL), or none. Returns a problem to show, if any."""
        if not data:
            if self.mask:
                self.mask, self.mask_seq = None, self.mask_seq + 1
            return None
        try:
            d = self.get(draft)
            n = _number(slide) or 1
            if n > len(d.slides) or d.slides[n - 1].media is None or self._item(d.slides[n - 1].media).kind != "image":
                raise StudioError("only a slide with a picture can be painted on")
            png = base64.b64decode(str(data).split(",", 1)[-1], validate=True)
            area = retouch.read_mask(png, d.size)
        except StudioError as exc:
            return str(exc)
        except Exception:
            return "the painted area couldn't be read"
        share = retouch.coverage(area)
        if share < 0.0005:
            return self.set_mask()
        self.mask = {"draft": d.id, "slide": n, "png": png, "size": d.size, "share": share,
                     "where": retouch.where(area)}
        self.mask_seq += 1
        return None

    def mask_note(self) -> str | None:
        m = self.mask
        if not m:
            return None
        return (f"On screen, the user painted over an area of slide {m['slide']} of draft {m['draft']} ({m['where']} "
                f"of the picture, {m['share']:.0%} of it). “This” or “here” means that area: change, remove, replace "
                "or improve just it with edit_area, and the rest of the picture stays as it is.")

    async def edit_area(self, ref=None, slide=None, mode="change", prompt="") -> dict:
        """Changes only the painted part of a slide's picture: the result replaces the slide's picture."""
        if not self.media or not self.media.comfy:
            return {"error": "picture editing is off (COMFYUI_URL is not set)"}
        m = self.mask
        try:
            if not m:
                raise StudioError("nothing is painted: ask the user to paint over the area on the slide (the brush "
                                  "button on the picture), or change the whole picture with generate_media")
            d = self.get(m["draft"] if ref in (None, "") else ref)
            n = m["slide"] if slide in (None, "") else _number(slide)
            if (d.id, n) != (m["draft"], m["slide"]):
                raise StudioError(f"the painted area is on slide {m['slide']} of draft {m['draft']}")
            if tuple(m["size"]) != d.size:
                raise StudioError("the draft changed shape since the area was painted; ask the user to paint it again")
            item = self._item(d.slides[n - 1].media)
            mode = {"delete": "remove", "erase": "remove", "enhance": "improve", "fix": "improve", "fill": "replace",
                    "add": "replace"}.get(str(mode or "change").strip().lower(), str(mode or "change").strip().lower())
            if mode not in EDIT_MODES:
                raise StudioError(f"mode is one of {', '.join(EDIT_MODES)}")
            what = _plain(prompt).rstrip(".")
            if mode in ("change", "replace") and not what:
                raise StudioError("say what the painted area should become")
            workflow = self.media.edit_workflow()
            if not workflow:
                raise StudioError("there is no picture-editing workflow on the pod")
        except StudioError as exc:
            return {"error": str(exc)}

        def prepare():
            with Image.open(item.path) as src:
                base = ImageOps.fit(ImageOps.exif_transpose(src).convert("RGB"), d.size, Image.LANCZOS)
            area = retouch.read_mask(m["png"], d.size)
            if mode in ("remove", "replace"):  # what is taken away goes a little past the strokes
                area = retouch.grow(area, 0.02)
            box = retouch.region(area)
            sent = (retouch.mark(base, area) if mode in ("remove", "replace") else base).crop(box)
            buf = io.BytesIO()
            sent.save(buf, "PNG")
            return base, area, box, "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        base, area, box, image = await asyncio.to_thread(prepare)

        def paste(data: bytes) -> tuple[bytes, str]:  # the edited part back into the whole picture
            with Image.open(io.BytesIO(data)) as edited:
                done = retouch.stitch(base, box, edited, area)
            buf = io.BytesIO()
            done.save(buf, "PNG", optimize=True)
            return buf.getvalue(), f"{mode}-area-{item.id}.png"

        instruction = EDIT_PROMPTS[mode].format(what=f": {what}" if mode == "improve" and what else what)
        reply = await self.media.generate(workflow, instruction, image=image, label=EDIT_MODES[mode],
                                          transform=paste, then=lambda items: self._made(d.id, n, items))
        if "error" not in reply:
            reply.update(draft=d.id, slide=n, area=m["where"],
                         note="only the painted area changes; the new picture goes on the slide by itself when ready")
        return reply

    # --- a video with the assistant's voice (reel.py) -----------------------------------

    async def make_video(self, ref=None, lines=None, cta=None, cta_line=None, music=None, captions=None,
                         motion=None, placement=None, color=None, *, speak, compose=None, moods=()) -> dict:
        """The draft's pictures as a video, its lines read in the user's voice. It goes into a draft of its
        own; asked again (of either draft), that same video draft is made again with the changes.
        speak(text) -> 24 kHz audio; compose(mood, seconds) -> WAV bytes or None (FlowAI Sound)."""
        try:
            d = self.get(ref)
            if d.video:  # the video draft itself: made again from its pictures' draft
                target, source = d, self.drafts.get(d.video.get("from"))
                if not source:
                    raise StudioError(f"draft {d.video.get('from')}, which this video was made from, is gone")
            else:
                source = d
                target = next((x for x in reversed(self.drafts.values())
                               if x.video and x.video.get("from") == d.id), None)
            recipe = self._recipe(source, dict(target.video) if target else {}, lines, cta, cta_line, music,
                                  captions, motion, placement, color, moods, compose is not None)
        except StudioError as exc:
            return {"error": str(exc)}

        platform, where = source.platform, recipe["placement"]
        spec = platforms.spec(platform, where)
        ratio = source.size[0] / source.size[1]
        size = source.size if platforms.fits(platform, where, ratio) else platforms.canvas_size(platform, where)
        safe, fonts = spec["safe"], self.fonts()
        slides = copy.deepcopy(source.slides)
        pictures = [self._item(sl.media).path if sl.media is not None else None for sl in slides]
        background = source.background
        accent = recipe["color"] or self._accent(source)

        def work(progress) -> tuple[bytes, str, dict]:  # in a thread
            shots = []
            for i, (slide, path, line) in enumerate(zip(slides, pictures, recipe["lines"])):
                picture = Image.open(path).convert("RGB") if path else None
                overlay = reel.layer(size, lambda img, sl=slide: self._draw_texts(img, sl, fonts, safe)[0]) \
                    if slide.texts else None
                # A shot per sentence, each moving its own way: one picture still looks filmed
                for k, part in enumerate(_sentences(line) or [""]):
                    n = len(shots)
                    kind = {"auto": reel.AUTO_MOTIONS[n % len(reel.AUTO_MOTIONS)],
                            "pan": ("pan-right", "pan-left")[n % 2]}.get(recipe["motion"], recipe["motion"])
                    shots.append(reel.Shot(picture, overlay, part, speak(part) if part else None, kind, enter=k == 0))
                progress(0.12 * (i + 1) / len(slides))
            if recipe["cta"]:
                said = recipe["cta_line"] or recipe["cta"]
                shots.append(reel.Shot(None, line=said, voice=speak(said), card=recipe["cta"]))
            seconds = reel.length(shots)
            most = spec.get("duration", (0, 600))[1]
            if seconds > most:
                raise StudioError(f"the voiceover makes it {seconds:.0f} seconds, and {platforms.label(platform, where)} "
                                  f"takes up to {most:g}; shorter lines would fit")
            tune = compose(recipe["music"], seconds + 1) if recipe["music"] and compose else None
            look = reel.Look(size, fonts["bold"].path if "bold" in fonts else None, background, recipe["captions"],
                             accent, safe)
            data, meta = reel.render(shots, look, tune, lambda share: progress(0.15 + 0.85 * share))
            return data, f"video-draft{source.id}.mp4", meta

        async def placed(items: list["MediaItem"]) -> str:
            return await self._place_video(source.id, target.id if target else None, recipe, items[0])

        said = sum(len(x) for x in recipe["lines"])
        reply = self.media.render(f"Making the video of draft {source.id}", " / ".join(filter(None, recipe["lines"])),
                                  work, then=placed)
        if "error" not in reply:
            reply.update(source=source.id, into=f"draft {target.id}, made again" if target else "a new draft",
                         post=platforms.label(platform, where), size=f"{size[0]}x{size[1]}",
                         expected_time=f"about {max(15, said // 12)} seconds",
                         note="it is made in the background and shown when ready; you'll be told")
        return reply

    def _recipe(self, source: Draft, recipe: dict, lines, cta, cta_line, music, captions, motion, placement,
                color, moods, has_music) -> dict:
        """How the video is made: what was asked now over what it was made with before."""
        if not source.slides:
            raise StudioError(f"draft {source.id} has no pictures to make a video of")
        for i, sl in enumerate(source.slides, 1):
            if sl.media is not None and self._item(sl.media).kind != "image":
                raise StudioError(f"slide {i} of draft {source.id} is a video; a video with a voice is made from pictures")
        n = len(source.slides)
        if lines not in (None, ""):
            lines = [_plain(x) for x in (lines if isinstance(lines, list) else [lines])]
            if len(lines) > n:  # more lines than pictures: the last picture takes the rest
                lines = [*lines[:n - 1], " ".join(lines[n - 1:])]
            recipe["lines"] = [*lines, *[""] * (n - len(lines))]
        lines = recipe.get("lines") or []
        lines = [*lines[:n], *[""] * (n - len(lines))]  # slides added or dropped since
        if not any(lines):
            raise StudioError("write what the voice says: lines, one per picture")
        if any(len(x) > MAX_LINE for x in lines):
            raise StudioError(f"a picture's line should be a sentence or two (under {MAX_LINE} characters)")
        recipe["lines"] = lines
        if cta is not None:
            recipe["cta"] = _plain(cta)[:40]
        if cta_line is not None:
            recipe["cta_line"] = _plain(cta_line)[:MAX_LINE]
        if music is not None:
            mood = re.sub(r"[\s_]+", "-", str(music).strip().lower())
            if mood in ("", "none", "no", "off", "false", "silence"):
                recipe["music"] = None
            elif moods and mood not in moods:
                raise StudioError(f"music is one of {', '.join(moods)}, or none")
            else:
                recipe["music"] = mood
        elif "music" not in recipe:
            recipe["music"] = "golden-hour" if has_music else None
        if captions is not None:
            recipe["captions"] = str(captions).strip().lower() not in ("false", "0", "no", "off")
        recipe.setdefault("captions", True)
        if motion not in (None, ""):
            m = re.sub(r"[\s_]+", "-", str(motion).strip().lower())
            if m not in ("auto", "pan", *reel.MOTIONS):
                raise StudioError(f"motion is auto, pan or one of {', '.join(reel.MOTIONS)}")
            recipe["motion"] = m
        recipe.setdefault("motion", "auto")
        if placement not in (None, ""):
            where = str(placement).strip().lower()
            if where not in platforms.placements(source.platform):
                raise StudioError(f"{platforms.LABELS[source.platform]} placements are "
                                  f"{', '.join(platforms.placements(source.platform))}")
            if "video" not in platforms.spec(source.platform, where)["media"]:
                raise StudioError(f"{platforms.label(source.platform, where)} doesn't take videos")
            recipe["placement"] = where
        recipe.setdefault("placement", VIDEO_PLACEMENT[source.platform])
        if color not in (None, ""):
            _color(color)
            recipe["color"] = str(color).strip()
        recipe.setdefault("color", None)
        recipe.setdefault("cta", "")
        recipe.setdefault("cta_line", "")
        recipe["from"] = source.id
        return recipe

    def _accent(self, d: Draft) -> str:
        """The end card's button: the post's own text colour if it has one, else its picture's."""
        for slide in d.slides:
            for t in slide.texts:
                if t.color.lower() not in ("white", "black", "#fff", "#ffffff", "#000", "#000000"):
                    return t.color
        for colours in d.palettes:
            if len(colours) > 2:
                return colours[2]
        return "#ffffff"

    async def _place_video(self, source_id: int, target_id: int | None, recipe: dict, item: "MediaItem") -> str:
        """A finished video into its draft: the one made before, or a new one."""
        source, target = self.drafts.get(source_id), self.drafts.get(target_id) if target_id else None
        meta = item.meta or {}
        aspect = f"{meta['width']}:{meta['height']}" if meta.get("width") else None

        def fill(d: Draft):
            d.placement, d.aspect, d.video = recipe["placement"], aspect, recipe
            d.slides = [Slide(item.id)]
        if target:
            result = await self._apply(target, fill)
        else:
            self.count += 1
            target = Draft(self.count, source.platform if source else "instagram", recipe["placement"])
            if source:
                target.title = f"{source.title or f'Draft {source.id}'} · video"[:120]
                target.caption, target.background = source.caption, source.background
            result = await self._apply(target, fill, record=False)
            if "error" in result:
                self.count -= 1
            else:
                self.drafts[target.id] = target
        if "error" in result:
            return f"It could not go into a draft: {result['error']}"
        if source:  # it lists its videos; quiet: the page stays on the video
            await self.send(**self._event(source), quiet=True)
        note = f"It is draft {target.id}, {target.a_label} of {meta.get('duration', '?')} seconds (version {target.version})."
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
            name = find_font(self.fonts_dir, font)
            if not name:
                raise StudioError(f"there is no font {font!r}; fonts: {', '.join(font_catalog(self.fonts_dir)['fonts'])}")
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

    async def _apply(self, d: Draft, change: Callable[[Draft], None], record: bool = True, redraw: bool = False) -> dict:
        """Change, check, render and show a draft; on any problem it stays as it was. redraw: the same
        version drawn again (its files were lost, or it predates something the page now shows), which
        isn't a change: a draft saved to FlowAI stays saved."""
        async with self.lock:
            before = d.state()
            try:
                change(d)
                self._validate(d)
                files, boxes, palettes = await asyncio.to_thread(self._render, d, d.version + (0 if redraw else 1))
            except StudioError as exc:
                d.restore(before)
                return {"error": str(exc)}
            except Exception as exc:
                log.exception("draft %d could not be rendered", d.id)
                d.restore(before)
                return {"error": f"the draft could not be drawn: {exc}"}
            if record and d.version and not redraw:
                d.history = [*d.history, before][-MAX_HISTORY:]
            d.version += 0 if redraw else 1
            d.files, d.boxes, d.palettes = files, boxes, palettes
            d.check = platforms.check(d.platform, d.placement, d.caption, [f for _, f in files])
            hard = [b for slide in boxes for b in slide if b.get("readability", 99) < HARD_TO_READ]
            if hard:  # ours too: FlowAI doesn't look at the pictures
                d.check["checks"].append({"key": "readability", "status": "warn", "label": "Readability", "detail": (
                    f"{_texts(hard)} {'is' if len(hard) == 1 else 'are'} hard to read on the picture behind; "
                    "a panel behind (style box) or another colour would help.")})
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
             "boxes": d.boxes, "palettes": d.palettes, "check": d.check}
            for d in self.drafts.values()]}

    async def restore(self, data: dict):
        self.count = data.get("count", 0)
        for raw in data.get("drafts", []):
            d = Draft(raw["id"], raw["platform"], raw["placement"])
            d.restore(_state_in(raw))
            d.version, d.check, d.boxes = raw.get("version", 0), raw.get("check", {}), raw.get("boxes", [])
            d.palettes = raw.get("palettes", [])
            d.history = [_state_in(h) for h in raw.get("history", [])]
            d.files = [(self.dir / Path(f.pop("file")).name, File(**f)) for f in map(dict, raw.get("files", []))]
            self.drafts[d.id] = d
            # MEDIA_DIR lost them, or they were drawn before slides had colours: draw them again
            if not all(path.is_file() for path, _ in d.files) or len(d.palettes) != len(d.slides):
                await self._apply(d, lambda d: None, record=False, redraw=True)

    # --- output ---------------------------------------------------------------------

    def summary(self, d: Draft) -> dict:
        """What the assistant is told after every change."""
        s = platforms.spec(d.platform, d.placement)
        w, h = d.size
        slides = []
        for i, slide in enumerate(d.slides, 1):
            item = self.media.items.get(slide.media) if self.media and slide.media is not None else None
            info = {"slide": i, "picture": f"{item.kind} {item.id}" if item else "none yet (plain background)"}
            if i <= len(d.palettes) and d.palettes[i - 1]:
                info["picture_colours"] = d.palettes[i - 1]
            if slide.texts:
                drawn = {b["id"]: b for b in d.boxes[i - 1]} if i <= len(d.boxes) else {}
                info["texts"] = [{**t.info(), **({"hard_to_read": True} if drawn.get(t.id, {}).get("readability", 99)
                                                 < HARD_TO_READ else {})} for t in slide.texts]
            slides.append(info)
        problems = [f"{c['status']}, {c['label'].lower()}: {c['detail']}" for c in d.check.get("checks", [])
                    if c["status"] != "pass"]
        video = {"made_from": f"draft {d.video['from']}", **{k: d.video[k] for k in ("lines", "cta", "cta_line",
                 "music", "captions", "motion") if d.video.get(k) not in (None, "")}} if d.video else None
        return {"draft": d.id, "version": d.version, "post": d.label, "size": f"{w}x{h}", "title": d.title,
                **({"video": video} if video else {}),
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
                "slides": [{"url": f"media/{path.name}", "kind": f.kind, "media": slide.media, "texts": boxes,
                            "palette": colours, **({"duration": f.duration} if f.duration else {})}
                           for (path, f), slide, boxes, colours in zip(d.files, d.slides, d.boxes,
                                                                       [*d.palettes, *[[]] * len(d.slides)])],
                "video": {k: v for k, v in d.video.items()} if d.video else None,
                "videos": [x.id for x in self.drafts.values() if x.video and x.video.get("from") == d.id],
                "check": d.check}

    def fonts(self) -> dict[str, "Font"]:
        """Every font texts can use: the catalog's by name, any other file by its name."""
        return font_catalog(self.fonts_dir)["all"]

    def _render(self, d: Draft, version: int) -> tuple[list[tuple[Path, File]], list[list[dict]], list[list[str]]]:
        """Each slide as it would be posted, where its texts are (and how readable), and its picture's
        colours (runs in a thread)."""
        size = d.size
        safe = platforms.spec(d.platform, d.placement)["safe"]
        fonts = self.fonts()
        out, layout, palettes = [], [], []
        for i, slide in enumerate(d.slides, 1):
            item = self._item(slide.media) if slide.media is not None else None
            if item and item.kind == "video":
                meta = item.meta or {}
                out.append((item.path, File("video", meta.get("width"), meta.get("height"), item.path.stat().st_size,
                                            meta.get("duration"))))
                layout.append([])
                palettes.append([])
                continue
            img = Image.new("RGB", size, _color(d.background))
            if item:
                with Image.open(item.path) as src:
                    img = ImageOps.fit(ImageOps.exif_transpose(src).convert("RGB"), size, Image.LANCZOS)
            plain = img
            img, boxes = self._draw_texts(img, slide, fonts, safe)
            for t in slide.texts:  # how readable each one is, on the picture behind it
                b = boxes[t.id]
                x, y, w, h = b["box"]
                behind = _behind(plain, (x * size[0], y * size[1], (x + w) * size[0], (y + h) * size[1]))
                b["bg"] = behind
                b["readability"] = round(readability(t, behind), 2)
            path = self.dir / f"{self.prefix}draft{d.id}-v{version}-{i}-{uuid.uuid4().hex[:6]}.jpg"
            img.convert("RGB").save(path, "JPEG", quality=90, optimize=True)
            out.append((path, File("image", size[0], size[1], path.stat().st_size)))
            layout.append([boxes[t.id] for t in slide.texts])
            palettes.append(picture_colours(plain) if item else [])
        return out, layout, palettes

    def _draw_texts(self, img: Image.Image, slide: Slide, fonts: dict, safe) -> tuple[Image.Image, dict[int, dict]]:
        """A slide's texts drawn on img, and where each one is (fractions of the slide, for the page)."""
        size = img.size
        taken: dict[str, float] = {}  # texts at the same position stack instead of overlapping
        boxes = {}
        bottom = [t for t in reversed(slide.texts) if t.position.startswith("bottom")]  # last one lowest
        for t in [t for t in slide.texts if t not in bottom] + bottom:
            font = fonts.get(t.font)
            img, height, (x, y, w, h) = draw_text(img, t, font.path if font else None, safe, taken.get(t.position, 0),
                                                  font.scale if font else 1.0)
            taken[t.position] = taken.get(t.position, 0) + height
            boxes[t.id] = {"id": t.id, "words": t.text,
                           "box": [round(x / size[0], 4), round(y / size[1], 4),
                                   round(w / size[0], 4), round(h / size[1], 4)],
                           **{k: getattr(t, k) for k in LOOK}}
        return img, boxes


# --- colours and readability ------------------------------------------------------------

def _luminance(rgb) -> np.ndarray | float:
    """Relative luminance (WCAG) of sRGB colours, 0 to 1."""
    c = np.asarray(rgb, dtype=np.float32) / 255
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return c[..., 0] * 0.2126 + c[..., 1] * 0.7152 + c[..., 2] * 0.0722


def _behind(img: Image.Image, box) -> list[float]:
    """How light the picture is behind a text: its darker and lighter parts (10th and 90th percentile
    luminance), since a text is only as readable as the worst spot it crosses."""
    x0, y0, x1, y1 = (max(0, round(box[0])), max(0, round(box[1])), min(img.width, round(box[2])),
                      min(img.height, round(box[3])))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return [0.0, 0.0]
    part = img.crop((x0, y0, x1, y1)).convert("RGB")
    part.thumbnail((64, 64))
    lum = _luminance(np.asarray(part))
    return [round(float(np.percentile(lum, 10)), 3), round(float(np.percentile(lum, 90)), 3)]


def _ratio(a: float, b: float) -> float:
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def readability(t: Text, behind: list[float]) -> float:
    """A contrast ratio (1 to 21) for the text where it is: its colour against the panel (style box,
    82% opaque) or the picture's darkest and lightest spots behind it; an outline or a shadow helps.
    The page works it out the same way (DraftView.tsx) to show it for any colour before it's picked."""
    lo, hi = behind
    ink = float(_luminance(_color(t.color)))
    if t.style == "box":
        panel = float(_luminance(_color(t.box_color)))
        return min(_ratio(ink, 0.82 * panel + 0.18 * lo), _ratio(ink, 0.82 * panel + 0.18 * hi))
    worst = min(_ratio(ink, lo), _ratio(ink, hi))
    if t.style == "outline":
        return max(worst, 0.8 * _ratio(ink, float(_luminance(_color(t.box_color)))))
    return worst * 1.3 if t.style == "shadow" else worst


def _hex(h: float, s: float, v: float) -> str:
    return "#" + "".join(f"{round(c * 255):02x}" for c in colorsys.hsv_to_rgb(h, s, v))


def picture_colours(img: Image.Image) -> list[str]:
    """Text colours that go with a picture: a pale tint of its main colour, up to three of its vivid
    colours made bright enough to read, and a deep shade of the main colour (for panels)."""
    small = img.convert("RGB")
    small.thumbnail((96, 96))
    q = small.quantize(colors=10, method=Image.Quantize.MEDIANCUT)
    palette = q.getpalette() or []
    found = [(n, colorsys.rgb_to_hsv(*(c / 255 for c in palette[3 * i:3 * i + 3])))
             for n, i in sorted(q.getcolors() or [], reverse=True)]
    if not found:
        return []
    main = next((c for _, c in found if c[1] > 0.12 and c[2] > 0.15), None)
    tint = _hex(main[0], min(main[1], 0.16), 0.97) if main else "#f6f4f0"
    deep = _hex(main[0], min(0.6, max(main[1], 0.3)), 0.17) if main else "#151515"
    accents, hues = [], []
    for _, (h, s, v) in sorted(found, key=lambda nc: -nc[0] * nc[1][1] ** 1.5 * (0.4 + nc[1][2])):
        if s < 0.28 or v < 0.3 or any(min(abs(h - x), 1 - abs(h - x)) < 0.07 for x in hues):
            continue
        hues.append(h)
        accents.append(_hex(h, max(s, 0.55), max(v, 0.9)))
        if len(accents) == 3:
            break
    return list(dict.fromkeys([tint, *accents, deep]))


def _sentences(text: str) -> list[str]:
    """A line in sentences; very short ones stay with the one before ("Yes. It's here.")."""
    out = []
    for part in re.split(r"(?<=[.!?…])\s+", " ".join(str(text or "").split())):
        if out and (len(part) < 12 or len(out[-1]) < 12):
            out[-1] += f" {part}"
        elif part:
            out.append(part)
    return out


def _texts(boxes: list[dict]) -> str:
    named = [f"{b['id']} (“{b['words'][:30]}”)" for b in boxes]
    return f"Text {named[0]}" if len(named) == 1 else f"Texts {', '.join(named[:-1])} and {named[-1]}"


# --- fonts ------------------------------------------------------------------------------

@dataclass
class Font:
    name: str
    path: Path
    label: str
    style: str = ""  # a style from fonts.yaml; "" for a file that isn't in it
    suits: str = ""
    scale: float = 1.0


_catalogs: dict[Path, tuple[tuple, dict]] = {}


def font_catalog(fonts_dir: Path) -> dict:
    """fonts.yaml read with the folder: {"styles": {id: label}, "fonts": {name: Font} (the catalog's,
    in order), "all": the catalog's plus any other .ttf/.otf by its lower-case file name}. Read
    again when the folder or the file changes."""
    files = sorted(p for p in fonts_dir.glob("*") if p.suffix.lower() in (".ttf", ".otf")) if fonts_dir.is_dir() else []
    meta = fonts_dir / "fonts.yaml"
    key = (tuple(files), meta.stat().st_mtime if meta.is_file() else 0)
    cached = _catalogs.get(fonts_dir)
    if cached and cached[0] == key:
        return cached[1]
    try:
        data = yaml.safe_load(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
    except Exception as exc:
        log.warning("could not read %s: %s", meta, exc)
        data = {}
    data = data if isinstance(data, dict) else {}
    named = {}
    for name, f in (data.get("fonts") or {}).items():
        path = fonts_dir / str((f or {}).get("file", ""))
        if not path.is_file():
            log.warning("fonts.yaml: %s has no file %s", name, path.name)
            continue
        named[str(name)] = Font(str(name), path, str(f.get("label") or name), str(f.get("style") or ""),
                                str(f.get("suits") or ""), float(f.get("scale") or 1.0))
    others = {p.stem.lower(): Font(p.stem.lower(), p, p.stem) for p in files}
    catalog = {"styles": {str(k): str(v) for k, v in (data.get("styles") or {}).items()}, "fonts": named,
               "all": {**others, **named}}
    _catalogs[fonts_dir] = (key, catalog)
    return catalog


def find_font(fonts_dir: Path, wanted) -> str | None:
    """A font's name from what was asked: its name, its label ("Playfair Display") or its file."""
    fonts = font_catalog(fonts_dir)["all"]
    key = re.sub(r"[\s_\-]+", "", str(wanted).strip().lower())
    for f in fonts.values():
        if key in (re.sub(r"[\s_\-]+", "", f.name), re.sub(r"[\s_\-]+", "", f.label.lower())):
            return f.name
    return None


def fonts_event(fonts_dir: Path) -> dict:
    """The fonts as the page shows them: by style, with where to load each one from."""
    c = font_catalog(fonts_dir)
    return {"type": "fonts", "styles": [{"id": k, "label": v} for k, v in c["styles"].items()],
            "fonts": [{"name": f.name, "label": f.label, "style": f.style, "suits": f.suits,
                       "url": f"fonts/{f.path.name}"} for f in c["fonts"].values()]}


def fonts_prompt(fonts_dir: Path) -> str:
    """The fonts by style, for the assistant to pick from."""
    c = font_catalog(fonts_dir)
    lines = []
    for style, label in c["styles"].items():
        fonts = [f for f in c["fonts"].values() if f.style == style]
        if fonts:
            lines.append(f"- {label}: " + "; ".join(f"{f.name} ({f.label}, {f.suits})" for f in fonts))
    return "\n".join(lines)


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
              shift: float = 0, scale: float = 1.0) -> tuple[Image.Image, float, tuple[float, float, float, float]]:
    """Draws one text on the picture, wrapped to fit, kept out of the platform's covered edges.
    shift: room taken by earlier texts at the same position, which this one goes past (below them at
    the top, above them at the bottom, under them elsewhere). scale: the font's (fonts.yaml), so a
    named size looks the same in every font. Returns the picture, the room it took, and where it is:
    x, y, width and height in pixels, panel included."""
    W, H = img.size
    size = int(t.size) if t.size.isdigit() else round(SIZES[t.size] * min(W, H) * scale)
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
