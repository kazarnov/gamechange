"""What Instagram and X accept, per placement, and the check every post draft is held against.

Mirrors FlowAI's config/platforms.php and PlatformSpecs::check (content-generator/backend), so a
draft that passes here passes the studio's pre-export check. Two additions of our own: `size`,
the picture size drafts are rendered at, and `safe`, the share of the height at the top and
bottom that the app's own buttons cover (text is kept out of it). Once the agent works through
FlowAI, GET /api/platform-specs and POST /api/checks take over (see FLOWAI_INTEGRATION.md).
"""

import re
from dataclasses import dataclass

LABELS = {"instagram": "Instagram", "x": "X"}

SPECS = {
    "instagram": {
        "feed": {"label": "Feed post", "media": ["image", "video"], "items": (1, 10),
                 "ratio": (0.8, 1.91), "ratio_hint": "1080 × 1350 (4:5)", "ratio_strict": True,
                 "min_width": 1080, "image_mb": 8, "video_mb": 650, "duration": (3, 60),
                 "caption": 2200, "hashtags": 30, "size": (1080, 1350), "safe": (0.06, 0.06)},
        "reel": {"label": "Reel", "media": ["video"], "items": (1, 1),
                 "ratio": (0.54, 0.58), "ratio_hint": "1080 × 1920 (9:16)", "ratio_strict": True,
                 "min_width": 720, "video_mb": 1000, "duration": (3, 90),
                 "caption": 2200, "hashtags": 30, "size": (1080, 1920), "safe": (0.13, 0.2)},
        "story": {"label": "Story", "media": ["image", "video"], "items": (1, 1),
                  "ratio": (0.54, 0.58), "ratio_hint": "1080 × 1920 (9:16)", "ratio_strict": True,
                  "min_width": 720, "image_mb": 30, "video_mb": 250, "duration": (1, 60),
                  "caption": 0, "hashtags": 10, "size": (1080, 1920), "safe": (0.13, 0.2)},
    },
    "x": {
        "post": {"label": "Post", "media": ["image", "video"], "items": (0, 4),
                 "ratio": (0.5, 2.0), "ratio_hint": "1600 × 900 (16:9)", "ratio_strict": False,
                 "min_width": 600, "image_mb": 5, "video_mb": 512, "duration": (0.5, 140),
                 "caption": 280, "hashtags_soft": 2, "size": (1600, 900), "safe": (0.06, 0.06)},
    },
}

DEFAULT_PLACEMENT = {"instagram": "feed", "x": "post"}
HASHTAG_RE = re.compile(r"(?<![\w#])#\w+")
ASPECT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[:x/×]\s*(\d+(?:\.\d+)?)")


@dataclass
class File:
    """One piece of media as it would be posted."""
    kind: str  # image | video
    width: int | None
    height: int | None
    size: int  # bytes
    duration: float | None = None  # seconds, for videos


def spec(platform: str, placement: str) -> dict:
    return SPECS[platform][placement]


def label(platform: str, placement: str) -> str:
    return f"{LABELS[platform]} {spec(platform, placement)['label'].lower()}"


def placements(platform: str) -> list[str]:
    return list(SPECS.get(platform, {}))


def parse_aspect(text: str) -> float:
    """"4:5", "16x9", "1.91:1" or "square" -> width ÷ height."""
    text = str(text).strip().lower()
    if text == "square":
        return 1.0
    m = ASPECT_RE.fullmatch(text)
    if not m or float(m.group(2)) == 0:
        raise ValueError(f"{text!r} is not a shape; write it like 4:5, 1:1 or 16:9")
    return float(m.group(1)) / float(m.group(2))


def fits(platform: str, placement: str, ratio: float) -> bool:
    lo, hi = spec(platform, placement)["ratio"]
    return lo - 0.005 <= ratio <= hi + 0.005


def canvas_size(platform: str, placement: str, ratio: float | None = None) -> tuple[int, int]:
    """The size a draft is rendered at: the placement's own, or `ratio` at the same long side."""
    w, h = spec(platform, placement)["size"]
    if ratio is None:
        return w, h
    long = max(w, h)
    return (long, round(long / ratio)) if ratio >= 1 else (round(long * ratio), long)


def check(platform: str, placement: str, caption: str, files: list[File]) -> dict:
    """The pre-export check: a failure blocks publishing, a warning is shown but doesn't."""
    s = spec(platform, placement)
    name = label(platform, placement)
    checks = []

    def add(key: str, status: str, what: str, detail: str = ""):
        checks.append({"key": key, "status": status, "label": what, "detail": detail})

    lo, hi = s["items"]
    n = len(files)
    if n < lo or n > hi:
        want = f"exactly {lo}" if lo == hi else f"up to {hi}" if lo == 0 else f"{lo} to {hi}"
        add("items", "fail", "Media count", f"{name} takes {want} {'file' if hi == 1 else 'files'}, this has {n}.")
    else:
        add("items", "pass", "Media count", "Text only." if n == 0 else f"{n} {'file' if n == 1 else 'files'}.")
    wrong = [f for f in files if f.kind not in s["media"]]
    if wrong:
        add("kind", "fail", "Media type", f"{name} doesn't take {wrong[0].kind}s.")

    length = len(caption)
    if s["caption"] == 0 and caption.strip():
        add("caption", "warn", "Caption", f"An {name} shows no caption; put the words on the media.")
    elif s["caption"] > 0:
        add("caption", "fail" if length > s["caption"] else "pass", "Caption length",
            f"{length:,} of {s['caption']:,} characters.")
    tags = len(HASHTAG_RE.findall(caption))
    if "hashtags" in s and tags > s["hashtags"]:
        add("hashtags", "fail", "Hashtags", f"{tags} hashtags; {LABELS[platform]} allows {s['hashtags']}.")
    elif "hashtags_soft" in s and tags > s["hashtags_soft"]:
        add("hashtags", "warn", "Hashtags", f"{tags} hashtags; more than {s['hashtags_soft']} tends to look like spam here.")
    elif tags:
        add("hashtags", "pass", "Hashtags", f"{tags} hashtags.")

    lo, hi = s["ratio"]
    for i, f in enumerate(files):
        which = f"File {i + 1}: " if n > 1 else ""
        if not (f.width and f.height):
            add(f"ratio.{i}", "warn", "Shape", f"{which}couldn't read the size of this file.")
        elif not fits(platform, placement, f.width / f.height):
            add(f"ratio.{i}", "fail" if s["ratio_strict"] else "warn", "Shape",
                f"{which}{f.width} × {f.height} is off for {name}; aim for {s['ratio_hint']}.")
        else:
            add(f"ratio.{i}", "pass", "Shape", f"{which}{f.width} × {f.height}.")
        if f.width and f.width < s["min_width"]:
            add(f"width.{i}", "warn", "Resolution",
                f"{which}{f.width} px wide; under {s['min_width']} px looks soft once the platform scales it up.")
        cap = s.get("video_mb" if f.kind == "video" else "image_mb")
        if cap and f.size > cap * 1024 * 1024:
            add(f"size.{i}", "fail", "File size", f"{which}{f.size / 1048576:.1f} MB; the limit is {cap} MB.")
        if f.kind == "video" and "duration" in s:
            dmin, dmax = s["duration"]
            if f.duration is None:
                add(f"duration.{i}", "warn", "Length", f"{which}couldn't read how long this video is.")
            elif not dmin <= f.duration <= dmax:
                add(f"duration.{i}", "fail", "Length", f"{which}{round(f.duration, 1):g} s; {name} takes {dmin:g}–{dmax:g} s.")
            else:
                add(f"duration.{i}", "pass", "Length", f"{which}{round(f.duration, 1):g} s.")

    return {"ok": all(c["status"] != "fail" for c in checks), "label": name, "checks": checks}
