"""Procedural, seamlessly-tiling PNG textures (numpy only, no image libraries needed).

Generated once into cobot_sandbox/assets/textures/ and cached. Drop your own PNGs into that folder and
reference them by file stem in a scene YAML (e.g. ``material: {texture: my_floor}``).
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from .paths import TEXTURE_DIR


def write_png(path: Path, rgb: np.ndarray) -> None:
    rgb = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    path.write_bytes(png)


def _value_noise(h, w, cy, cx, rng):
    """Smooth, periodic value noise with cy x cx lattice cells."""
    g = rng.random((cy, cx))
    ys = np.arange(h) * cy / h
    xs = np.arange(w) * cx / w
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = ys - y0, xs - x0
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    y1, x1 = (y0 + 1) % cy, (x0 + 1) % cx
    a, b = g[np.ix_(y0, x0)], g[np.ix_(y0, x1)]
    c, d = g[np.ix_(y1, x0)], g[np.ix_(y1, x1)]
    top = a + (b - a) * fx[None, :]
    bot = c + (d - c) * fx[None, :]
    return top + (bot - top) * fy[:, None]


def fbm(h, w, cells, octaves, rng, aspect=(1, 1), persistence=0.5):
    total, amp, norm = np.zeros((h, w)), 1.0, 0.0
    for o in range(octaves):
        c = cells * 2 ** o
        cy, cx = max(1, int(round(c * aspect[0]))), max(1, int(round(c * aspect[1])))
        total += amp * _value_noise(h, w, cy, cx, rng)
        norm += amp
        amp *= persistence
    return total / norm


def _lerp(c0, c1, t):
    c0, c1 = np.asarray(c0, float), np.asarray(c1, float)
    return c0 + (c1 - c0) * t[..., None]


def concrete(size=1024, tiles=4, seed=1):
    rng = np.random.default_rng(seed)
    n = fbm(size, size, 4, 7, rng)
    fine = fbm(size, size, 64, 3, rng)
    v = 0.78 + 0.22 * (n - 0.5) + 0.08 * (fine - 0.5)
    tile = size // tiles
    ty, tx = np.meshgrid(np.arange(size) // tile, np.arange(size) // tile, indexing="ij")
    tint = rng.uniform(-0.03, 0.03, (tiles, tiles))[ty, tx]
    v = v + tint
    yy, xx = np.meshgrid(np.arange(size) % tile, np.arange(size) % tile, indexing="ij")
    grout = (np.minimum(yy, tile - 1 - yy) < 2) | (np.minimum(xx, tile - 1 - xx) < 2)
    v = np.where(grout, v * 0.62, v)
    specks = rng.random((size, size)) > 0.997
    v = np.where(specks, v * 0.7, v)
    rgb = _lerp((70, 72, 76), (150, 150, 148), np.clip(v, 0, 1.2) / 1.2)
    return rgb


def wood(size=1024, planks=4, seed=2):
    rng = np.random.default_rng(seed)
    h = w = size
    warp = fbm(h, w, 2, 5, rng, aspect=(1, 3))
    streak = fbm(h, w, 16, 4, rng, aspect=(8, 1))
    y = np.arange(h)[:, None] / h * planks
    plank_id = np.floor(y).astype(int) % planks
    offset = rng.random(planks)[plank_id]
    rings = 0.5 + 0.5 * np.sin(2 * np.pi * ((y + offset) * 26 + warp * 1.4))
    t = 0.35 * rings ** 2.2 + 0.65 * streak
    rgb = _lerp((118, 74, 42), (190, 140, 92), np.clip(t, 0, 1))
    rgb *= (0.92 + 0.16 * rng.random(planks)[plank_id])[..., None]
    seam = (np.abs(((y % 1.0) - 0.5)) > 0.497)
    rgb = np.where(seam[..., None], rgb * 0.45, rgb)
    return rgb




def brushed_metal(size=512, seed=4):
    rng = np.random.default_rng(seed)
    streak = fbm(size, size, 64, 3, rng, aspect=(4, 1 / 64))
    base = fbm(size, size, 2, 3, rng)
    v = 0.75 + 0.18 * (streak - 0.5) + 0.08 * (base - 0.5)
    return _lerp((140, 144, 150), (225, 228, 232), np.clip(v, 0, 1))




GENERATORS = {
    "concrete": concrete,
    "wood": wood,
    "brushed_metal": brushed_metal,
}


def ensure_textures(force: bool = False) -> dict:
    """Generate any missing built-in textures. Returns {name: path} for every PNG available."""
    TEXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in GENERATORS.items():
        p = TEXTURE_DIR / f"{name}.png"
        if force or not p.exists():
            write_png(p, fn())
    return {p.stem: p for p in sorted(TEXTURE_DIR.glob("*.png"))}


if __name__ == "__main__":
    for k, v in ensure_textures(force=True).items():
        print(k, v)
