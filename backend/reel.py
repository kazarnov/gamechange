"""A post made into a short video: its pictures in motion and the assistant's voice over them.

One picture or several: each moves the whole time it is on screen (a slow zoom in or out, or a pan,
so a still looks filmed), its texts come in over it, the voice reads its line in the voice the user
picked, the words show as captions, and an end card can close on the call to action ("Shop now")
on a button. Background music from FlowAI Sound, when there is some, steps aside whenever the voice
speaks. Frames are drawn with Pillow and encoded by ffmpeg as H.264 and AAC, which Instagram and X
take as they are.

Timing comes from the voice: a shot lasts as long as its words take, plus a breath before and after.
The next picture fades in over its end; the same picture again (a line of several sentences is a shot
per sentence, each with its own camera move) comes in with a straight cut, like an edit.
"""

import base64
import logging
import re
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont, ImageOps

log = logging.getLogger(__name__)

FPS = 30
VOICE_RATE = 24000
LEAD, TAIL = 0.25, 0.35  # seconds of quiet before and after a shot's words
FADE = 0.4  # one picture fading into the next
CUT = 0.0  # the same picture framed another way: a straight cut (blending them doubles everything)
MIN_SHOT, SILENT_SHOT, CARD_SHOT = 2.4, 3.0, 2.8  # seconds on screen: at least / without a line / end card
ZOOM = 1.14  # how far a picture moves in or out; it is fitted this much larger, so it stays sharp
MOTIONS = ["zoom-in", "zoom-out", "pan-left", "pan-right", "pan-up", "pan-down", "still"]
AUTO_MOTIONS = ["zoom-in", "pan-right", "zoom-out", "pan-left"]
MUSIC_VOLUME = 0.6  # between lines; under the voice the compressor takes it about 10 dB lower
# FlowAI Sound's moods (content-generator/sound/music.py), for the assistant to pick from
MOODS = {"golden-hour": "warm lo-fi", "linen": "airy ambient", "atelier": "bright acoustic",
         "pulse": "upbeat house", "night-drive": "synthwave", "bloom": "dreamy, cinematic"}


@dataclass
class Shot:
    picture: Image.Image | None  # any size, fitted to the frame; None: the plain background colour
    overlay: Image.Image | None = None  # RGBA, frame size: the slide's texts
    line: str = ""  # what the voice says over it
    voice: np.ndarray | None = None  # that line's speech, 24 kHz mono
    motion: str = "zoom-in"
    card: str = ""  # an end card: these words on a button
    enter: bool = True  # its texts come in (not when the shot before had the same picture and texts)


@dataclass
class Look:
    size: tuple[int, int]
    font: Path | None  # for captions and the end card's button
    background: str = "#1c1c1c"
    captions: bool = True
    accent: str = "#ffffff"  # the end card's button
    safe: tuple[float, float] = (0.06, 0.06)  # top and bottom shares the app covers


def ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError:
        raise RuntimeError("ffmpeg is missing: pip install imageio-ffmpeg") from None
    return imageio_ffmpeg.get_ffmpeg_exe()


def layer(size: tuple[int, int], paint: Callable[[Image.Image], Image.Image]) -> Image.Image:
    """Whatever paint draws, as a transparent layer: painted on black and on white, the difference
    gives each pixel's opacity and the true colour, soft edges and shadows included (drawing on a
    transparent picture directly darkens the edges)."""
    on_black = np.asarray(paint(Image.new("RGB", size, (0, 0, 0))).convert("RGB"), dtype=np.float32)
    on_white = np.asarray(paint(Image.new("RGB", size, (255, 255, 255))).convert("RGB"), dtype=np.float32)
    alpha = np.clip(255 - (on_white - on_black).mean(axis=2), 0, 255)
    color = np.where(alpha[..., None] > 0, on_black * 255 / np.maximum(alpha, 1)[..., None], 0)
    rgba = np.dstack([np.clip(color, 0, 255), alpha]).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def _ease(p: float) -> float:
    """Mostly steady, a little softer at the ends: a slow camera move."""
    p = min(max(p, 0.0), 1.0)
    return 0.7 * p + 0.3 * p * p * (3 - 2 * p)


def _font(path: Path | None, size: int) -> ImageFont.FreeTypeFont:
    for candidate in filter(None, (str(path) if path else None, "DejaVuSans-Bold.ttf")):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _luma(color: tuple[int, int, int]) -> float:
    return (0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]) / 255


class _Mover:
    """A picture's movement over its shot."""

    def __init__(self, picture: Image.Image | None, size: tuple[int, int], motion: str, background: str,
                 blur: float = 0):
        W, H = size
        big = (round(W * ZOOM), round(H * ZOOM))
        if picture is None:
            self.big = Image.new("RGB", big, background)
        else:
            self.big = ImageOps.fit(ImageOps.exif_transpose(picture).convert("RGB"), big, Image.LANCZOS)
        if blur:
            dark = Image.new("RGB", big, (0, 0, 0))
            self.big = Image.blend(self.big.filter(ImageFilter.GaussianBlur(blur * W)), dark, 0.45)
        self.size, self.motion = size, motion if motion in MOTIONS else "zoom-in"

    def frame(self, p: float) -> Image.Image:
        bw, bh = self.big.size
        e = _ease(p)
        m = self.motion
        if m.startswith("pan"):
            z = ZOOM  # the frame's own size out of the larger picture: room to travel, nothing lost
            w, h = bw / z, bh / z
            sx, sy = bw - w, bh - h
            x = sx * (e if m == "pan-right" else 1 - e if m == "pan-left" else 0.5)
            y = sy * (e if m == "pan-down" else 1 - e if m == "pan-up" else 0.5)
        else:
            z = 1 + (ZOOM - 1) * (e if m == "zoom-in" else 1 - e if m == "zoom-out" else 0.15 + 0.1 * e)
            w, h = bw / z, bh / z
            x, y = (bw - w) / 2, (bh - h) * 0.45  # a touch above the middle, where faces and products sit
        return self.big.resize(self.size, Image.BILINEAR, box=(x, y, x + w, y + h))


def _chunks(line: str, most: int) -> list[str]:
    """The line in caption-sized pieces, cut at the ends of sentences when it can."""
    out, cur = [], ""
    for word in line.split():
        if cur and len(cur) + 1 + len(word) > most:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
        if re.search(r"[.!?…]$", cur) and len(cur) > most * 0.45:
            out.append(cur)
            cur = ""
    if cur and out and len(cur) < most * 0.4 and len(out[-1]) + len(cur) < most * 1.4:
        out[-1] += f" {cur}"  # no word left on its own at the end
    elif cur:
        out.append(cur)
    return out


class _Caption:
    def __init__(self, text: str, look: Look, center_y: float):
        W, H = look.size
        size = max(28, round(0.066 * min(W, H)))
        font = _font(look.font, size)
        lines = _wrap(text, font, 0.86 * W)
        step = round(size * 1.18)
        stroke = max(2, round(size * 0.09))
        widths = [font.getlength(t) for t in lines]
        w, h = round(max(widths) + 4 * stroke), round(step * len(lines) + 3 * stroke)

        def paint(img: Image.Image) -> Image.Image:
            img = img.convert("RGBA")
            shade = Image.new("RGBA", img.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(shade)
            for i, (t, tw) in enumerate(zip(lines, widths)):
                d.text(((w - tw) / 2, 1.5 * stroke + i * step + size * 0.06), t, font=font, fill=(0, 0, 0, 150))
            img = Image.alpha_composite(img, shade.filter(ImageFilter.GaussianBlur(size * 0.12)))
            d = ImageDraw.Draw(img)
            for i, (t, tw) in enumerate(zip(lines, widths)):
                d.text(((w - tw) / 2, 1.5 * stroke + i * step), t, font=font, fill=(255, 255, 255),
                       stroke_width=stroke, stroke_fill=(12, 12, 12))
            return img

        self.img = layer((w, h), paint)
        self.at = (round((W - w) / 2), round(center_y * H - h / 2))

    def draw(self, frame: Image.Image, k: float):
        """k: 0..1 through its pop-in."""
        img, (x, y) = self.img, self.at
        if k < 1:
            s = 0.9 + 0.1 * _ease(k)
            w, h = max(1, round(img.width * s)), max(1, round(img.height * s))
            img = img.resize((w, h), Image.BILINEAR)
            img.putalpha(img.getchannel("A").point(lambda v: round(v * k)))
            x, y = x + (self.img.width - w) // 2, y + (self.img.height - h) // 2
        frame.paste(img, (x, y), img)


def _wrap(text: str, font, width: float) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        if cur and font.getlength(f"{cur} {word}") > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return lines + ([cur] if cur else [])


class _Button:
    """The end card's call to action: a pill in the accent colour that pops in, then breathes."""

    def __init__(self, text: str, look: Look):
        W, H = look.size
        size = max(30, round(0.085 * min(W, H)))
        font = _font(look.font, size)
        while font.getlength(text) > 0.72 * W and size > 24:
            size = int(size * 0.92)
            font = _font(look.font, size)
        accent = _rgb(look.accent)
        ink = (17, 17, 17) if _luma(accent) > 0.55 else (255, 255, 255)
        tw = font.getlength(text)
        _, top, _, bottom = font.getbbox(text, anchor="ls")
        pw, ph = round(tw + 1.8 * size), round((bottom - top) + 1.15 * size)
        glow = round(size * 0.6)
        w, h = pw + 2 * glow, ph + 2 * glow

        def paint(img: Image.Image) -> Image.Image:
            img = img.convert("RGBA")
            halo = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ImageDraw.Draw(halo).rounded_rectangle((glow, glow, glow + pw, glow + ph), radius=ph / 2,
                                                   fill=(*accent, 120))
            img = Image.alpha_composite(img, halo.filter(ImageFilter.GaussianBlur(glow * 0.55)))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((glow, glow, glow + pw, glow + ph), radius=ph / 2, fill=accent)
            d.text((glow + (pw - tw) / 2, glow + (ph - (bottom - top)) / 2 - top), text, font=font, fill=ink,
                   anchor="ls")
            return img

        self.img = layer((w, h), paint)
        self.center = (W / 2, H * 0.5)

    def draw(self, frame: Image.Image, t: float):
        """t: seconds since the card came in."""
        if t < 0.15:
            return
        k = min((t - 0.15) / 0.45, 1.0)
        back = 1 + 2.2 * (k - 1) ** 3 + 1.2 * (k - 1) ** 2  # pops slightly past its size, then settles
        s = (0.6 + 0.4 * back) if k < 1 else 1 + 0.025 * np.sin(2 * np.pi * (t - 0.6) / 1.4)
        w, h = max(1, round(self.img.width * s)), max(1, round(self.img.height * s))
        img = self.img.resize((w, h), Image.BILINEAR) if (w, h) != self.img.size else self.img
        if k < 1:
            img = img.copy()
            img.putalpha(img.getchannel("A").point(lambda v: round(v * min(1.0, k * 1.6))))
        frame.paste(img, (round(self.center[0] - w / 2), round(self.center[1] - h / 2)), img)


def _rgb(color: str) -> tuple[int, int, int]:
    try:
        return ImageColor.getrgb(color)[:3]
    except ValueError:
        return 255, 255, 255


def _blend(before: Shot, after: Shot) -> float:
    """Seconds the next shot takes to come in over this one."""
    return CUT if after.picture is not None and after.picture is before.picture and not after.card else FADE


def timeline(shots: list[Shot]) -> list[tuple[float, float]]:
    """When each shot starts and how long it lasts, in seconds."""
    out, t = [], 0.0
    for i, s in enumerate(shots):
        speech = len(s.voice) / VOICE_RATE if s.voice is not None else 0.0
        if s.card:
            d = max(CARD_SHOT, LEAD + speech + 0.9)
        else:
            d = max(MIN_SHOT if s.enter else 1.2, LEAD + speech + TAIL) if speech else SILENT_SHOT
        overlap = _blend(s, shots[i + 1]) if i < len(shots) - 1 else 0.0
        out.append((t, d + overlap))  # the next one comes in over this one's last moment
        t += d
    return out


def length(shots: list[Shot]) -> float:
    start, d = timeline(shots)[-1]
    return start + d


class _Clip:
    """One shot, ready to draw at any moment of it."""

    def __init__(self, shot: Shot, start: float, duration: float, look: Look, before: Image.Image | None,
                 blend: float = FADE):
        self.start, self.duration, self.blend = start, duration, blend
        # The cover (its texts are there from the first frame), or the same picture and texts again
        self.first = start == 0 or not shot.enter
        W, H = look.size
        if shot.card:  # the picture before it, blurred and darkened behind the button
            self.mover = _Mover(shot.picture or before, look.size, "zoom-in", look.background, blur=0.03)
            self.button = _Button(shot.card, look)
        else:
            self.mover = _Mover(shot.picture, look.size, shot.motion, look.background)
            self.button = None
        self.overlay = None
        if shot.overlay is not None and shot.overlay.getbbox():
            box = shot.overlay.getbbox()
            crop = shot.overlay.crop(box)
            self.overlay = (crop.convert("RGB"), crop.getchannel("A"), box[:2])
        self.captions: list[tuple[float, float, _Caption]] = []
        speech = len(shot.voice) / VOICE_RATE if shot.voice is not None else 0.0
        if look.captions and shot.line.strip() and speech and not shot.card:
            pieces = _chunks(shot.line, 22 if H > W else 30)
            weights = np.array([len(p) + 4 for p in pieces], dtype=float)
            ends = np.cumsum(weights) / weights.sum() * speech
            y = self._caption_y(shot.overlay, look)
            t0 = LEAD
            for piece, end in zip(pieces, ends):
                self.captions.append((t0, LEAD + end, _Caption(piece, look, y)))
                t0 = LEAD + end

    @staticmethod
    def _caption_y(overlay: Image.Image | None, look: Look) -> float:
        """Where captions go: the lower third, unless the slide's texts are there."""
        W, H = look.size
        low = min(0.7, 1 - look.safe[1] - 0.1)
        if overlay is None:
            return low
        rows = np.asarray(overlay.getchannel("A"), dtype=np.float32).mean(axis=1)

        def busy(y: float) -> float:
            a, b = int((y - 0.08) * H), int((y + 0.08) * H)
            return float(rows[max(a, 0):max(b, 1)].mean())

        return min([low, 0.36, 0.52], key=lambda y: (busy(y) > 3, busy(y)))

    def draw(self, t: float) -> Image.Image:
        """The frame t seconds into the video."""
        local = t - self.start
        frame = self.mover.frame(local / self.duration)
        if self.overlay:
            rgb, alpha, (x, y) = self.overlay
            k = 1.0 if self.first else _ease((local - 0.15) / 0.5)
            if k > 0:
                a = alpha if k >= 1 else alpha.point(lambda v: round(v * k))
                frame.paste(rgb, (x, round(y + (1 - k) * 0.02 * frame.height)), a)
        for start, end, cap in self.captions:
            if start <= local < end:
                cap.draw(frame, (local - start) / 0.12)
        if self.button:
            self.button.draw(frame, local)
        return frame


def render(shots: list[Shot], look: Look, music: bytes | None = None,
           progress: Callable[[float], None] = lambda share: None) -> tuple[bytes, dict]:
    """The video as MP4 bytes, and its width, height and duration."""
    if not shots:
        raise ValueError("there is nothing to put in the video")
    look.size = W, H = look.size[0] // 2 * 2, look.size[1] // 2 * 2  # H.264 wants even sizes
    times = timeline(shots)
    total = times[-1][0] + times[-1][1]
    clips, before = [], None
    for i, (shot, (start, d)) in enumerate(zip(shots, times)):
        clips.append(_Clip(shot, start, d, look, before, _blend(shots[i - 1], shot) if i else FADE))
        before = shot.picture or before

    voice = np.zeros(int(total * VOICE_RATE) + VOICE_RATE, dtype=np.float32)
    for shot, (start, _) in zip(shots, times):
        if shot.voice is not None and len(shot.voice):
            at = int((start + LEAD) * VOICE_RATE)
            voice[at:at + len(shot.voice)] += shot.voice[: len(voice) - at]
    voice = voice[: int(total * VOICE_RATE)]

    with tempfile.TemporaryDirectory(prefix="reel-") as tmp:
        tmp = Path(tmp)
        _write_wav(tmp / "voice.wav", voice)
        inputs = ["-i", str(tmp / "voice.wav")]
        if music:
            (tmp / "music.wav").write_bytes(music)
            inputs += ["-i", str(tmp / "music.wav")]
            fade = max(0.0, total - 1.6)
            audio = (f"[1:a]aresample=48000,asplit=2[v][key];"
                     f"[2:a]aresample=48000,volume={MUSIC_VOLUME},afade=t=in:d=0.8,afade=t=out:st={fade:.2f}:d=1.6[m];"
                     "[m][key]sidechaincompress=threshold=0.03:ratio=5:attack=15:release=450[bed];"
                     "[v][bed]amix=inputs=2:duration=first:normalize=0,"
                     "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[a]")
        else:
            audio = "[1:a]aresample=48000,loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[a]"
        out = tmp / "video.mp4"
        cmd = [ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(FPS), "-i", "pipe:0", *inputs, "-filter_complex", audio, "-map", "0:v", "-map", "[a]",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-t", f"{total:.3f}", str(out)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        frames = int(round(total * FPS))
        try:
            for f in range(frames):
                t = f / FPS
                live = [c for c in clips if c.start <= t < c.start + c.duration] or [clips[-1]]
                frame = live[0].draw(t)
                if len(live) > 1:  # the next one coming in
                    k = _ease((t - live[1].start) / max(live[1].blend, 1e-6))
                    frame = Image.blend(frame, live[1].draw(t), k)
                proc.stdin.write(frame.tobytes())
                if f % 5 == 0:
                    progress(f / frames)
            proc.stdin.close()
            err = proc.stderr.read().decode(errors="replace")
            if proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed: {err.strip().splitlines()[-1] if err.strip() else 'no output'}")
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        progress(1.0)
        return out.read_bytes(), {"width": W, "height": H, "duration": round(total, 2)}


def _write_wav(path: Path, audio: np.ndarray):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(VOICE_RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def music_wav(data: str) -> bytes:
    """FlowAI Sound's music (base64 WAV, or a data URL) as WAV bytes."""
    return base64.b64decode(data.split(",", 1)[1] if data.startswith("data:") else data)
