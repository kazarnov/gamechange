"""Changing only part of a picture: the area the user painted over on a slide (a mask).

The editing model (an image-edit workflow on the pod, e.g. FLUX.2 Klein) redraws a whole picture,
and even what it keeps comes back slightly changed. So the picture never goes to it whole: the
painted area, with some room around it so the model sees what it sits in, is cut out and sent
alone (the model then also works at a higher resolution there), and what comes back is pasted
over the original through the mask, its edge softened. Outside the painted area the picture stays
exactly as it was.

To remove or replace something, the area is painted over in flat magenta first: the model then
fills in that patch from what is around it. Told only to "remove the thing in the middle", it often
kept the thing, or took half of it away.
"""

import io

import numpy as np
from PIL import Image, ImageFilter

CONTEXT = 0.75  # room around the painted area sent with it, as a share of the area's size
MIN_REGION = 0.35  # the cut-out is at least this share of the picture's shorter side
MARKER = (255, 0, 255)  # paints over what is to be removed or replaced
FEATHER = 0.012  # of the picture's shorter side: how soft the pasted edge is
GROW = 0.008  # the painted area is grown by this much, so a stroke's edge is covered


def fit_box(size: tuple[int, int], shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """The part of a picture of `size` that a slide of `shape` shows (as ImageOps.fit cuts it)."""
    w, h = size
    ratio = shape[0] / shape[1]
    if w / h > ratio:
        cw = h * ratio
        return (w - cw) / 2, 0, (w + cw) / 2, h
    ch = w / ratio
    return 0, (h - ch) / 2, w, (h + ch) / 2


def read_mask(data: bytes, size: tuple[int, int]) -> Image.Image:
    """The page's mask (a PNG, painted pixels opaque or white) at `size`, as 0 or 255."""
    with Image.open(io.BytesIO(data)) as m:
        m.load()
        # Painted: opaque (a stroke on a transparent canvas) or light (white on black)
        mask = m.getchannel("A") if m.mode in ("RGBA", "LA") or "transparency" in m.info else m.convert("L")
    mask = mask.resize(size, Image.BILINEAR)
    return mask.point(lambda v: 255 if v >= 96 else 0)


def coverage(mask: Image.Image) -> float:
    return float(np.asarray(mask, dtype=np.uint8).mean()) / 255


def where(mask: Image.Image) -> str:
    """Where the painted area is, in words: "the upper left", "the middle"."""
    box = mask.getbbox()
    if not box:
        return "nowhere"
    w, h = mask.size
    cx, cy = (box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h
    row = "upper" if cy < 0.36 else "lower" if cy > 0.64 else ""
    col = "left" if cx < 0.36 else "right" if cx > 0.64 else ""
    return f"the {row} {col}".replace("  ", " ").strip() if row or col else "the middle"


def region(mask: Image.Image) -> tuple[int, int, int, int]:
    """The cut-out sent to the model: the painted area with room around it, inside the picture."""
    W, H = mask.size
    x0, y0, x1, y1 = mask.getbbox()
    side = max(x1 - x0, y1 - y0)
    pad = side * CONTEXT / 2
    least = MIN_REGION * min(W, H)
    w, h = max(x1 - x0 + 2 * pad, least), max(y1 - y0 + 2 * pad, least)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = min(w, W), min(h, H)
    left = min(max(cx - w / 2, 0), W - w)
    top = min(max(cy - h / 2, 0), H - h)
    return round(left), round(top), round(left + w), round(top + h)


def grow(mask: Image.Image, share: float) -> Image.Image:
    """The painted area made larger by `share` of the picture's shorter side, all round."""
    reach = share * min(mask.size)
    spread = _blur(np.asarray(mask, dtype=np.float32) / 255, reach / 2)  # two passes: about `reach` in all
    return Image.fromarray(np.where(spread > 0.01, 255, 0).astype(np.uint8))


def soft(mask: Image.Image) -> Image.Image:
    """The painted area grown a little and with a soft edge, for pasting."""
    side = min(mask.size)
    grow = max(3, round(side * GROW)) | 1  # MaxFilter takes odd sizes
    return mask.filter(ImageFilter.MaxFilter(grow)).filter(ImageFilter.GaussianBlur(max(2.0, side * FEATHER)))


def mark(img: Image.Image, mask: Image.Image) -> Image.Image:
    """The painted area in flat magenta: what to remove or replace is then plain to the model, which
    fills it in from what is around it (told only "remove it", it often keeps the thing or half of it)."""
    hole = soft(mask).point(lambda v: 255 if v > 10 else 0)
    return Image.composite(Image.new("RGB", img.size, MARKER), img.convert("RGB"), hole)


def _spread(values: np.ndarray, known: np.ndarray, radius: float) -> np.ndarray:
    """Values smoothed over the known pixels and carried into the others, from near to far."""
    out = np.zeros_like(values)
    done = np.zeros(known.shape, dtype=bool)
    while not done.all() and radius < 2 * max(known.shape):
        weight = _blur(known, radius)
        total = _blur(values * known[..., None], radius)
        ok = (weight > 1e-3) & ~done
        out[ok] = total[ok] / weight[ok, None]
        done |= ok
        radius *= 1.6
    return out


def _blur(a: np.ndarray, radius: float) -> np.ndarray:
    """Sums over a square around each pixel (twice: close to a Gaussian), along the first two axes."""
    r = max(1, int(radius))
    for _ in range(2):
        for axis in (0, 1):
            pad = [(r + 1, r) if ax == axis else (0, 0) for ax in range(a.ndim)]
            c = np.cumsum(np.pad(a, pad), axis=axis, dtype=np.float64)
            n = a.shape[axis]
            a = np.take(c, range(2 * r + 1, 2 * r + 1 + n), axis=axis) - np.take(c, range(n), axis=axis)
    return a.astype(np.float32)


def stitch(base: Image.Image, box: tuple[int, int, int, int], edited: Image.Image, mask: Image.Image) -> Image.Image:
    """The model's cut-out pasted back over `base`, only where the mask is, with a soft edge. The
    model shifts colours a little (a wall comes back a shade lighter), so the difference along the
    area's edge is measured and carried smoothly into it: the new part meets the old without a seam."""
    w, h = box[2] - box[0], box[3] - box[1]
    edited = edited.convert("RGB").resize((w, h), Image.LANCZOS)
    before = np.asarray(base.convert("RGB").crop(box), dtype=np.float32)
    after = np.asarray(edited, dtype=np.float32)
    area = np.asarray(soft(mask).crop(box), dtype=np.float32)
    x0, y0, x1, y1 = mask.getbbox()
    reach = max(4.0, 0.08 * max(x1 - x0, y1 - y0))
    edge = (area < 8) & (_blur((area > 8).astype(np.float32), reach) > 0)  # a band just outside the area
    if edge.sum() > 200:
        after = after + np.clip(_spread(before - after, edge.astype(np.float32), reach), -48, 48)
    patch = Image.new("RGB", base.size)
    patch.paste(Image.fromarray(after.clip(0, 255).astype(np.uint8)), box[:2])
    alpha = Image.new("L", base.size, 0)
    alpha.paste(soft(mask).crop(box), box[:2])
    return Image.composite(patch, base.convert("RGB"), alpha)
