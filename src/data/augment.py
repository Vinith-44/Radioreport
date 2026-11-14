"""Step 6: phone-photo / WhatsApp-style corruptions of a chest X-ray (CLAUDE.md §6 Step 6). PIL + numpy only.

photo_augment(img, rng) applies a random subset in a physically sensible order:
  geometry (perspective warp of a hand-held phone, rotation) -> screen effects (glare, monitor moire)
  -> film / sensor (printed-film texture: grain, tint, vignette) -> exposure / contrast -> blur
  -> JPEG at quality 10-40 (messaging-app recompression; applied last, like the real pipeline).
Every random choice comes from `rng` (random.Random), so a given seed always gives the same image.
"""
from __future__ import annotations

import io
import math
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _rgb(im: Image.Image) -> Image.Image:
    return im.convert("RGB")


def perspective(im: Image.Image, rng: random.Random, max_shift: float = 0.08) -> Image.Image:
    w, h = im.size
    d = lambda s: rng.uniform(-max_shift, max_shift) * s  # noqa: E731
    src = [(d(w), d(h)), (w + d(w), d(h)), (w + d(w), h + d(h)), (d(w), h + d(h))]
    dst = [(0, 0), (w, 0), (w, h), (0, h)]
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):   # coefficients map output (dst) pixels to input (src) pixels
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        B += [u, v]
    coeffs = np.linalg.solve(np.asarray(A, float), np.asarray(B, float)).tolist()
    bg = tuple(int(rng.uniform(20, 90)) for _ in range(3))     # table / wall behind the film or screen
    return _rgb(im).transform((w, h), Image.PERSPECTIVE, coeffs, Image.BILINEAR, fillcolor=bg)


def rotate(im: Image.Image, rng: random.Random, max_deg: float = 10) -> Image.Image:
    return _rgb(im).rotate(rng.uniform(-max_deg, max_deg), Image.BILINEAR, fillcolor=(40, 40, 40))


def glare(im: Image.Image, rng: random.Random) -> Image.Image:
    a = np.asarray(_rgb(im), np.float32)
    h, w = a.shape[:2]
    cy, cx = rng.uniform(0, h), rng.uniform(0, w)
    r = rng.uniform(0.1, 0.35) * max(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    spot = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))[..., None]
    a = a + rng.uniform(0.25, 0.6) * spot * (255 - a)
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def moire(im: Image.Image, rng: random.Random) -> Image.Image:
    a = np.asarray(_rgb(im), np.float32)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    f1, f2 = rng.uniform(0.15, 0.45), rng.uniform(0.15, 0.45)   # two near-equal screen / sensor gratings
    t1, t2 = rng.uniform(0, math.pi), rng.uniform(0, math.pi)
    g = np.sin(2 * math.pi * f1 * (xx * math.cos(t1) + yy * math.sin(t1))) * \
        np.sin(2 * math.pi * f2 * (xx * math.cos(t2) + yy * math.sin(t2)))
    a = a * (1 + rng.uniform(0.04, 0.12) * g[..., None])
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def film_texture(im: Image.Image, rng: random.Random) -> Image.Image:
    a = np.asarray(_rgb(im), np.float32)
    h, w = a.shape[:2]
    nrng = np.random.default_rng(rng.randrange(2**32))
    a = a + nrng.normal(0, rng.uniform(3, 10), size=(h, w, 1))                     # grain
    a = a * np.asarray([rng.uniform(0.9, 1.1) for _ in range(3)], np.float32)     # tint / white balance
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rr = ((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2
    a = a * (1 - rng.uniform(0.1, 0.35) * rr.clip(0, 1.5) / 1.5)[..., None]          # vignette
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def exposure(im: Image.Image, rng: random.Random) -> Image.Image:
    a = np.asarray(_rgb(im), np.float32) / 255.0
    a = np.power(a, rng.uniform(0.6, 1.6)) * 255.0
    out = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.5, 0.9))
    return ImageEnhance.Brightness(out).enhance(rng.uniform(0.6, 1.1))


def blur(im: Image.Image, rng: random.Random) -> Image.Image:
    return _rgb(im).filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.5)))


def jpeg(im: Image.Image, rng: random.Random, qmin: int = 10, qmax: int = 40) -> Image.Image:
    buf = io.BytesIO()
    _rgb(im).save(buf, format="JPEG", quality=rng.randint(qmin, qmax))
    buf.seek(0)
    with Image.open(buf) as j:
        j.load()
        return j.convert("RGB")


# (name, function, probability) in application order
PIPELINE = [("perspective", perspective, 0.6), ("rotate", rotate, 0.4), ("glare", glare, 0.35),
            ("moire", moire, 0.3), ("film_texture", film_texture, 0.4), ("exposure", exposure, 0.6),
            ("blur", blur, 0.4), ("jpeg", jpeg, 0.8)]


def photo_augment(im: Image.Image, rng: random.Random, return_ops: bool = False):
    """At least two corruptions are always applied (a single mild one would barely change the image)."""
    while True:
        ops = [(n, f) for n, f, p in PIPELINE if rng.random() < p]
        if len(ops) >= 2:
            break
    out = _rgb(im)
    for _, f in ops:
        out = f(out, rng)
    return (out, [n for n, _ in ops]) if return_ops else out
