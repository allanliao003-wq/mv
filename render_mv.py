#!/usr/bin/env python3
"""《融化的糖》Melting Sugar — procedural character MV renderer.

Every frame is a pure function of time t (no state carried between frames),
so frames can be rendered in parallel. Visuals are driven by the song's
section map and by audio features (808/kick envelope, hi-hat envelope, RMS).

Usage:
  python3 render_mv.py --preview 5,20,50     # write PNG stills to out/preview/
  python3 render_mv.py --render              # full MV -> out/melting_sugar_mv.mp4
"""
import argparse
import math
import os
import subprocess
import sys
from functools import lru_cache
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(ROOT, "assets", "melting_sugar.mp3")
OUT_DIR = os.path.join(ROOT, "out")
FONT_CJK = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"

W, H = 1280, 720
FPS = 24
DURATION = 179.5
BAR = 60 / 80.75 * 4  # ~2.97 s per bar at ~80 BPM
LETTERBOX = 88        # 2.39:1 cinema bars

# ---------------------------------------------------------------- palette
TUNGSTEN = np.array([1.0, 0.62, 0.28], np.float32)
AMBER = np.array([0.95, 0.52, 0.12], np.float32)
ROSE = np.array([1.0, 0.45, 0.55], np.float32)
MAGENTA = np.array([1.0, 0.18, 0.62], np.float32)
CYAN = np.array([0.15, 0.75, 1.0], np.float32)
VIOLET = np.array([0.55, 0.3, 1.0], np.float32)
SKIN_DARK = np.array([0.045, 0.03, 0.03], np.float32)

# ---------------------------------------------------------------- audio features


def _load_features():
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", AUDIO, "-ac", "1", "-ar", "22050",
         "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    sr, n = 22050, 2048
    hop = sr // FPS  # one analysis frame per video frame
    nf = int(len(x) / hop)
    x = np.pad(x, (n // 2, n))
    win = np.hanning(n).astype(np.float32)
    idx = np.arange(nf) * hop
    frames = np.stack([x[i:i + n] * win for i in idx])
    S = np.abs(np.fft.rfft(frames, axis=1))
    f = np.fft.rfftfreq(n, 1 / sr)
    low = S[:, f < 120].sum(1)
    hi = S[:, f > 6000].sum(1)
    rms = np.sqrt((frames ** 2).mean(1))

    def env(v, rel):
        v = v / (np.percentile(v, 98) + 1e-9)
        out = np.zeros_like(v)
        acc = 0.0
        for i, s in enumerate(v):
            acc = max(s, acc * rel)
            out[i] = acc
        return np.clip(out, 0, 1.5)

    # onset-style kick: positive change in low band
    dlow = np.maximum(np.diff(low, prepend=low[0]), 0)
    dhi = np.maximum(np.diff(hi, prepend=hi[0]), 0)
    return {
        "kick": env(dlow, 0.80),
        "hat": env(dhi, 0.65),
        "rms": env(rms, 0.96),
        "low": env(low, 0.9),
    }


FEAT = None


def feat(name, t):
    i = int(np.clip(t * FPS, 0, len(FEAT[name]) - 1))
    return float(FEAT[name][i])


# ---------------------------------------------------------------- helpers


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def lerp(a, b, x):
    return a + (b - a) * x


def vgrad(top, bot, h=H, w=W):
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    return (np.asarray(top, np.float32) * (1 - t) + np.asarray(bot, np.float32) * t) * np.ones((1, w, 1), np.float32)


def blur(img, sigma):
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma)


def blur_fast(img, sigma, factor=4):
    """Large blur done at reduced resolution."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(sigma / factor, 0.5))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def chaikin(pts, it=3, closed=True):
    p = np.asarray(pts, np.float32)
    for _ in range(it):
        q = np.roll(p, -1, axis=0) if closed else p[1:]
        a = p if closed else p[:-1]
        new = np.empty((len(a) * 2, 2), np.float32)
        new[0::2] = 0.75 * a + 0.25 * q
        new[1::2] = 0.25 * a + 0.75 * q
        p = new
    return p


def poly_mask(pts, w, h):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(pts) * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
    return m.astype(np.float32) / 255


def paste(img, rgb, alpha, x0, y0, mode="over"):
    """Composite rgb (h,w,3) with alpha (h,w) at integer offset; clipped to frame."""
    h, w = alpha.shape
    X0, Y0 = max(x0, 0), max(y0, 0)
    X1, Y1 = min(x0 + w, img.shape[1]), min(y0 + h, img.shape[0])
    if X0 >= X1 or Y0 >= Y1:
        return
    sx, sy = X0 - x0, Y0 - y0
    a = alpha[sy:sy + Y1 - Y0, sx:sx + X1 - X0, None]
    c = rgb[sy:sy + Y1 - Y0, sx:sx + X1 - X0] if rgb.ndim == 3 else rgb
    region = img[Y0:Y1, X0:X1]
    if mode == "add":
        region += c * a
    else:
        region *= (1 - a)
        region += c * a


def fill_poly(img, pts, color, alpha=1.0, feather=0.0):
    pts = np.asarray(pts, np.float32)
    pad = int(feather * 3) + 2
    x0, y0 = np.floor(pts.min(0)).astype(int) - pad
    x1, y1 = np.ceil(pts.max(0)).astype(int) + pad
    m = poly_mask(pts - [x0, y0], x1 - x0, y1 - y0)
    if feather:
        m = blur(m, feather)
    paste(img, np.asarray(color, np.float32), m * alpha, x0, y0)


def glow_dot(layer, x, y, r, color, intensity=1.0):
    cv2.circle(layer, (int(x), int(y)), max(int(r), 1),
               tuple(float(c) * float(intensity) for c in color), -1, cv2.LINE_8)


def rng(seed):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------- text


@lru_cache(maxsize=32)
def text_mask(text, size, spacing=0):
    font = ImageFont.truetype(FONT_CJK, size)
    widths = [font.getbbox(ch)[2] for ch in text]
    tw = sum(widths) + spacing * (len(text) - 1)
    pad = size
    im = Image.new("L", (tw + 2 * pad, int(size * 1.2) + 2 * pad), 0)
    d = ImageDraw.Draw(im)
    x = pad
    for ch, cw in zip(text, widths):
        d.text((x, pad), ch, font=font, fill=255)
        x += cw + spacing
    return np.asarray(im, np.float32) / 255


def draw_text(img, text, cx, cy, size, color, alpha, spacing=0, glow=0.6):
    if alpha <= 0.002:
        return
    m = text_mask(text, size, spacing)
    h, w = m.shape
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    g = blur(m, size * 0.25)
    paste(img, np.asarray(color, np.float32), g * alpha * glow, x0, y0, "add")
    paste(img, np.asarray(color, np.float32) * 0.6 + 0.4, m * alpha, x0, y0)


# ---------------------------------------------------------------- characters
# Base sprites live in a 1000x1000 box, facing RIGHT. Head occupies the upper
# part; shoulders run off the bottom edge so busts can sit behind counters or
# run out of frame in close-ups.

MALE = [
    (440, 96), (540, 84), (618, 108), (652, 160), (654, 230),
    (652, 290), (656, 335), (672, 372), (660, 396), (672, 420), (700, 452), (722, 482),
    (700, 497), (686, 506), (697, 524), (682, 538), (692, 554), (676, 574),
    (692, 612), (678, 650), (622, 672), (572, 682),
    (576, 724), (586, 800), (660, 842), (800, 880), (905, 945), (960, 1000),
    (90, 1000), (130, 925), (270, 862), (365, 828),
    (392, 760), (400, 690), (362, 625), (332, 525), (318, 410), (332, 262), (372, 150),
]
FEMALE = [
    (430, 120), (540, 112), (618, 160), (640, 240),
    (641, 318), (652, 362), (644, 388), (656, 414), (676, 444), (692, 470),
    (668, 488), (662, 492), (676, 511), (661, 526), (672, 540), (657, 566),
    (657, 598), (641, 624), (594, 644), (560, 654),
    (560, 700), (568, 782), (642, 830), (780, 868), (882, 930), (935, 1000),
    (30, 1000), (70, 950), (130, 860), (190, 760), (235, 650), (262, 540),
    (272, 420), (290, 280), (345, 170),
]
# regions (intersected with the outline): hair / clothing
HAIR = {
    "m": [(664, 262), (600, 236), (548, 252), (505, 300), (482, 370), (452, 420), (428, 490),
          (385, 545), (300, 570), (280, 300), (360, 40), (700, 40), (700, 200)],
    "f": [(660, 300), (628, 252), (575, 238), (520, 270), (488, 330), (470, 420), (466, 520),
          (482, 610), (470, 700), (430, 790), (380, 880), (320, 960), (300, 1010),
          (-10, 1010), (-10, 0), (720, 0), (720, 220)],
}
CLOTH = {
    "m": [(566, 784), (610, 836), (700, 868), (1010, 930), (1010, 1010), (-10, 1010), (-10, 820), (380, 790)],
    "f": [(-10, 940), (1010, 958), (1010, 1010), (-10, 1010)],
}
SHEEN = {
    "m": [[(612, 128), (540, 104), (450, 118), (382, 186), (350, 280)],
          [(630, 186), (560, 150), (470, 160), (400, 230), (365, 340)],
          [(624, 234), (560, 214), (480, 240), (420, 320), (385, 440)]],
    "f": [[(606, 176), (520, 148), (420, 180), (340, 290), (310, 450), (280, 620), (220, 800)],
          [(626, 236), (540, 210), (450, 260), (380, 380), (350, 540), (310, 700), (250, 880)],
          [(600, 262), (520, 280), (460, 380), (430, 520), (410, 660), (360, 820), (300, 960)],
          [(560, 300), (500, 360), (472, 480), (460, 600), (430, 740)]],
}
DETAIL = {
    "m": [("ellipse", (470, 455, 26, 46)), ("line", [(586, 800), (624, 880), (650, 1000)]),
          ("line", [(586, 800), (540, 870), (520, 1000)])],
    "f": [("line", [(610, 846), (700, 858)]), ("line", [(575, 760), (560, 790), (520, 812)])],
}
EYE = {"m": (648, 398), "f": (632, 392)}
EARRING = (500, 560)

HAND = [
    (0, 196), (200, 192), (380, 190), (450, 172), (545, 160), (620, 176), (720, 192),
    (820, 204), (884, 212), (906, 224), (900, 240), (880, 246), (800, 244), (700, 242),
    (648, 244), (690, 262), (710, 288), (694, 310), (650, 314), (616, 304),
    (640, 326), (648, 346), (626, 360), (586, 352),
    (598, 368), (596, 388), (570, 394), (530, 378),
    (470, 352), (410, 322), (360, 304), (200, 300), (0, 306),
]
HAND_DETAIL = [[(540, 170), (600, 196), (650, 236)], [(700, 262), (660, 268)], [(632, 330), (600, 330)]]

SKIN = np.array([0.105, 0.066, 0.052], np.float32)
HAIR_COL = np.array([0.022, 0.016, 0.018], np.float32)
CLOTH_COL = {"m": np.array([0.03, 0.03, 0.04], np.float32), "f": np.array([0.05, 0.02, 0.03], np.float32)}


def _lines_mask(lines, w, h, thick):
    m = np.zeros((h, w), np.uint8)
    for ln in lines:
        if isinstance(ln, tuple) and ln[0] == "ellipse":
            x, y, a, b = ln[1]
            cv2.ellipse(m, (x, y), (a, b), 0, -110, 110, 128, thick, cv2.LINE_AA)
            continue
        if isinstance(ln, tuple):
            ln = ln[1]
        pts = chaikin(ln, 3, closed=False)
        cv2.polylines(m, [np.int32(pts)], False, 255, thick, cv2.LINE_AA)
    return cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 2.0)


@lru_cache(maxsize=4)
def base_layers(kind):
    """Stack: outline, hair, cloth, sheen, detail (1000x1000 or 1000x500 for hands)."""
    if kind.startswith("hand"):
        pts = np.array(HAND, np.float32)
        if kind == "hand_f":
            pts[:, 1] = 250 + (pts[:, 1] - 250) * 0.82
        fig = poly_mask(chaikin(pts, 3), 1000, 500)
        z = np.zeros_like(fig)
        det = _lines_mask(HAND_DETAIL, 1000, 500, 4) * fig
        return np.dstack([fig, z, z, z, det])
    fig = poly_mask(chaikin(MALE if kind == "m" else FEMALE, 3), 1000, 1000)
    hair = poly_mask(chaikin(HAIR[kind], 2), 1000, 1000) * fig
    cloth = poly_mask(chaikin(CLOTH[kind], 2), 1000, 1000) * fig
    sheen = _lines_mask(SHEEN[kind], 1000, 1000, 4) * hair
    det = _lines_mask(DETAIL[kind], 1000, 1000, 4) * fig
    return np.dstack([fig, hair, cloth, sheen, det])


@lru_cache(maxsize=96)
def sized_layers(kind, size, flip):
    L = base_layers(kind)
    s = size / 1000.0
    dsz = (max(int(L.shape[1] * s), 2), max(int(L.shape[0] * s), 2))
    L = np.dstack([cv2.resize(np.ascontiguousarray(L[..., i]), dsz, interpolation=cv2.INTER_AREA) for i in range(L.shape[2])])
    return L[:, ::-1].copy() if flip else L


def _warp(L, M, w, h):
    # cv2.warpAffine handles at most 4 channels
    a = cv2.warpAffine(np.ascontiguousarray(L[..., :4]), M, (w, h), flags=cv2.INTER_LINEAR)
    b = cv2.warpAffine(np.ascontiguousarray(L[..., 4]), M, (w, h), flags=cv2.INTER_LINEAR)
    return np.dstack([a, b])


def _shade(img, L, x0, y0, light, rim, rim_gain, size, alpha, skin, hair_col, cloth_col, wrap):
    h, w = L.shape[:2]
    m, hair, cloth, sheen, det = (L[..., i] for i in range(5))
    lx, ly = light
    norm = math.hypot(lx, ly) or 1
    lx, ly = lx / norm, ly / norm
    k = max(1.5, size * 0.010)
    shifted = cv2.warpAffine(m, np.float32([[1, 0, -lx * k], [0, 1, -ly * k]]), (w, h))
    edge = np.clip(m - shifted, 0, 1)
    k2 = size * 0.06
    shifted2 = cv2.warpAffine(m, np.float32([[1, 0, -lx * k2], [0, 1, -ly * k2]]), (w, h))
    soft = blur(np.clip(m - shifted2, 0, 1), k2 * 0.5) * m
    xs = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    ys = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    ramp = np.clip(0.5 + 0.5 * (xs * lx + ys * ly), 0, 1)[..., None]
    skin_w = np.clip(1 - hair - cloth, 0, 1)[..., None]
    base = skin[None, None] * skin_w + hair_col[None, None] * hair[..., None] + cloth_col[None, None] * cloth[..., None]
    body = base * (0.45 + 1.1 * ramp) + rim[None, None] * (soft * wrap)[..., None]
    body += rim[None, None] * (sheen * (0.04 + 0.16 * ramp[..., 0]))[..., None]
    body += rim[None, None] * (det * 0.10)[..., None]
    paste(img, body.astype(np.float32), m * alpha, x0, y0)
    rimimg = np.broadcast_to(rim * 1.6 * rim_gain, (h, w, 3)).astype(np.float32)
    paste(img, rimimg, blur(edge, max(0.6, size * 0.0015)) * alpha, x0, y0, "add")


def _sparkle(img, gx, gy, r, color, gain):
    layer = np.zeros((int(r * 12) + 4, int(r * 12) + 4, 3), np.float32)
    c = layer.shape[0] // 2
    cv2.circle(layer, (c, c), int(max(r * 0.7, 1)), (1.0, 1.0, 1.0), -1)
    layer = blur(layer, r * 0.6) * 2 + blur(layer, r * 2.5) * 2
    paste(img, layer * color * gain, np.ones(layer.shape[:2], np.float32), int(gx - c), int(gy - c), "add")


def draw_figure(img, kind, cx, by, size, flip=False, light=(1.0, -0.3), rim=TUNGSTEN,
                rim_gain=1.0, rot=0.0, glint=True, alpha=1.0, wrap=0.35, skin=None):
    """Rim-lit character bust. (cx, by) = bottom-centre of the 1000x1000 sprite box."""
    size = int(size)
    L = sized_layers(kind, size, flip)
    h, w = L.shape[:2]
    M = None
    if abs(rot) > 0.05:
        M = cv2.getRotationMatrix2D((w / 2, h * 0.9), rot, 1.0)
        L = _warp(L, M, w, h)
    x0, y0 = int(cx - w / 2), int(by - h)
    sk = SKIN if skin is None else skin
    _shade(img, L, x0, y0, light, rim, rim_gain, size, alpha, sk, HAIR_COL, CLOTH_COL[kind], wrap)

    def place(pt):
        px, py = pt
        px = (1000 - px) if flip else px
        p = np.array([px * size / 1000, py * size / 1000, 1.0])
        if M is not None:
            p[:2] = M @ p
        return x0 + p[0], y0 + p[1]

    if glint:
        gx, gy = place(EYE[kind])
        _sparkle(img, gx, gy, max(1.0, size * 0.0035), rim * 0.5 + 0.3, alpha * rim_gain * 0.5)
    if kind == "f":
        gx, gy = place(EARRING)
        _sparkle(img, gx, gy, max(1.0, size * 0.005), 0.6 + 0.4 * rim, alpha * rim_gain * 0.8)


def draw_hand(img, kind, tip_x, cy, length, flip=False, rim=TUNGSTEN, light=(0.0, -1.0), rim_gain=1.0, rot=0.0):
    """Hand reaching sideways, index finger leading; (tip_x, cy) = fingertip."""
    L = sized_layers(kind, int(length), flip)
    h, w = L.shape[:2]
    tip_off = 906 * length / 1000
    tx = tip_off if not flip else (w - tip_off)
    ty = 226 * length / 1000
    if abs(rot) > 0.05:
        M = cv2.getRotationMatrix2D((tx, ty), rot, 1.0)
        L = _warp(L, M, w, h)
    skin = np.array([0.20, 0.12, 0.085], np.float32) * (0.75 if kind == "hand_m" else 1.0)
    _shade(img, L, int(tip_x - tx), int(cy - ty), light, rim, rim_gain, length * 0.7, 1.0,
           skin, HAIR_COL, CLOTH_COL["m"], 0.5)


# ---------------------------------------------------------------- environment pieces


def bokeh(img, seed, n, t, colors, rmin, rmax, vel=(0, 0), gain=0.5, twinkle=0.3,
          area=(0, 0, W, H), streak=0, ring=True):
    r = rng(seed)
    f = 2
    lw, lh = W // f, H // f
    layers = [np.zeros((lh, lw, 3), np.float32) for _ in range(3)]
    ax0, ay0, ax1, ay1 = area
    for i in range(n):
        x0 = r.uniform(ax0, ax1)
        y0 = r.uniform(ay0, ay1)
        rad = r.uniform(rmin, rmax)
        col = colors[r.integers(len(colors))] * r.uniform(0.5, 1.0)
        ph = r.uniform(0, 6.28)
        spd = r.uniform(0.6, 1.4)
        span_x = (ax1 - ax0) + 2 * rmax
        span_y = (ay1 - ay0) + 2 * rmax
        x = ax0 - rmax + ((x0 - ax0 + vel[0] * spd * t) % span_x)
        y = ay0 - rmax + ((y0 - ay0 + vel[1] * spd * t) % span_y)
        a = gain * (1 - twinkle + twinkle * (0.5 + 0.5 * math.sin(t * 1.3 * spd + ph)))
        L = layers[i % 3]
        cv2.circle(L, (int(x / f), int(y / f)), max(int(rad / f), 1), tuple(map(float, col * a)), -1)
        if ring and rad > 10:
            cv2.circle(L, (int(x / f), int(y / f)), max(int(rad / f), 1), tuple(map(float, col * a * 1.5)), 1)
    L = layers[0] + layers[1] + layers[2]
    if streak:
        L = cv2.blur(L, (int(streak / f) * 2 + 1, 1))
    L = cv2.GaussianBlur(L, (0, 0), 1.2)
    img += cv2.resize(L, (W, H), interpolation=cv2.INTER_LINEAR)


def rain_streaks(img, seed, t, n=260, angle=0.12, speed=1500, length=(25, 60), gain=0.18, color=(0.8, 0.85, 1.0)):
    r = rng(seed)
    layer = np.zeros((H, W, 3), np.float32)
    col = np.array(color, np.float32)
    for i in range(n):
        x0, y0 = r.uniform(-100, W + 100), r.uniform(0, H + 200)
        L = r.uniform(*length)
        sp = speed * r.uniform(0.7, 1.3)
        y = (y0 + sp * t) % (H + 200) - 100
        x = x0 + angle * (y - y0)
        a = gain * r.uniform(0.3, 1.0)
        cv2.line(layer, (int(x), int(y)), (int(x + angle * L), int(y + L)), tuple(map(float, col * a)), 1, cv2.LINE_AA)
    img += blur(layer, 0.6)


def window_drops(img, seed, t, n=180, gain=1.0, tint=(1.0, 0.8, 0.6), area=(0, 0, W, H)):
    """Raindrops on a glass pane, refracting the lights behind."""
    r = rng(seed)
    hi = np.zeros((H, W, 3), np.float32)
    dark = np.zeros((H, W), np.float32)
    tint = np.array(tint, np.float32)
    ax0, ay0, ax1, ay1 = area
    for i in range(n):
        x = r.uniform(ax0, ax1)
        y0 = r.uniform(ay0, ay1)
        rad = r.uniform(2.0, 9.0) * (1 + 0.6 * (r.uniform() < 0.12))
        runner = r.uniform() < 0.10
        y = y0
        if runner:
            sp = r.uniform(25, 90)
            y = ay0 + (y0 - ay0 + sp * t) % (ay1 - ay0)
            cv2.line(hi, (int(x), int(y - 140)), (int(x), int(y)), tuple(map(float, tint * 0.06 * gain)), 2)
        cv2.circle(dark, (int(x), int(y)), int(rad), 0.35, -1)
        cv2.circle(hi, (int(x + rad * 0.15), int(y + rad * 0.45)), max(int(rad * 0.45), 1), tuple(map(float, tint * 0.55 * gain)), -1)
        cv2.circle(hi, (int(x - rad * 0.35), int(y - rad * 0.4)), max(int(rad * 0.18), 1), tuple(map(float, np.ones(3) * 0.9 * gain)), -1)
    img *= (1 - blur(dark, 1.0))[..., None]
    img += blur(hi, 0.8)


def bulb(glow, img, x, y, wire_top, t, phase, gain=1.0, size=1.0):
    sway = math.sin(t * 0.9 + phase) * 0.025
    bx = x + (y - wire_top) * sway
    cv2.line(img, (int(x), int(wire_top)), (int(bx), int(y - 20 * size)), (0.05, 0.04, 0.035), max(1, int(2 * size)), cv2.LINE_AA)
    cv2.rectangle(img, (int(bx - 6 * size), int(y - 22 * size)), (int(bx + 6 * size), int(y - 10 * size)), (0.08, 0.06, 0.04), -1)
    flick = 0.92 + 0.08 * math.sin(t * 13 + phase * 7) * math.sin(t * 3.1 + phase)
    g = gain * flick
    cv2.ellipse(glow, (int(bx), int(y)), (int(11 * size), int(15 * size)), 0, 0, 360, tuple(map(float, TUNGSTEN * 0.9 * g)), -1)
    cv2.ellipse(glow, (int(bx), int(y)), (int(4 * size), int(8 * size)), 0, 0, 360, tuple(map(float, np.array([1.0, 0.9, 0.7]) * 2.5 * g)), -1)


def add_glow(img, glow, sharp=1.5, wide=40, wide_gain=0.9):
    img += blur(glow, sharp)
    img += blur_fast(glow, wide) * wide_gain


@lru_cache(maxsize=4)
def back_bar(seed=7):
    """Out-of-focus back bar: shelves and backlit bottles (cached)."""
    img = vgrad((0.035, 0.022, 0.018), (0.012, 0.008, 0.008))
    r = rng(seed)
    for shelf_y in (250, 400):
        cv2.rectangle(img, (0, shelf_y), (W, shelf_y + 8), (0.10, 0.06, 0.03), -1)
        x = 10
        while x < W:
            bw = r.uniform(26, 46)
            bh = r.uniform(80, 130)
            col = np.array([[0.45, 0.22, 0.05], [0.12, 0.25, 0.10], [0.35, 0.30, 0.25], [0.5, 0.12, 0.08]][r.integers(4)], np.float32)
            col *= r.uniform(0.25, 0.6)
            top = shelf_y - bh
            cv2.rectangle(img, (int(x), int(top + bh * 0.3)), (int(x + bw), shelf_y), tuple(map(float, col)), -1)
            cv2.rectangle(img, (int(x + bw * 0.38), int(top)), (int(x + bw * 0.62), int(top + bh * 0.3)), tuple(map(float, col * 0.8)), -1)
            cv2.line(img, (int(x + bw * 0.2), int(top + bh * 0.35)), (int(x + bw * 0.2), shelf_y - 4), tuple(map(float, col * 2.5 + 0.05)), 2)
            x += bw + r.uniform(4, 18)
    # warm under-shelf strip light
    strip = np.zeros_like(img)
    for shelf_y in (250, 400):
        cv2.rectangle(strip, (0, shelf_y + 8), (W, shelf_y + 11), tuple(map(float, TUNGSTEN * 0.8)), -1)
    img += blur_fast(strip, 30) * 1.2 + blur(strip, 2)
    return blur(img, 6)


def counter(img, y, glow_color=TUNGSTEN, gain=1.0):
    img[y:] = vgrad((0.07, 0.035, 0.02), (0.015, 0.01, 0.008), H - y, W)
    edge = np.zeros((H, W, 3), np.float32)
    cv2.line(edge, (0, y), (W, y), tuple(map(float, glow_color * 0.9 * gain)), 2, cv2.LINE_AA)
    img += blur(edge, 1.0) + blur_fast(edge, 12) * 0.5


def rocks_glass(img, glow, cx, base_y, w, liquid=0.45, t=0.0, melt=0.0, sugar=True, refl=None):
    """Amber rocks glass with a dissolving sugar cube."""
    h = w * 0.95
    top_y = base_y - h
    tw, bw = w / 2, w * 0.43
    body = [(cx - tw, top_y), (cx + tw, top_y), (cx + bw, base_y), (cx - bw, base_y)]
    fill_poly(img, body, (0.8, 0.85, 0.9), 0.05)
    liq_y = base_y - h * 0.18 - h * 0.62 * liquid
    lt = (liq_y - top_y) / h
    lw = tw + (bw - tw) * lt
    liq = [(cx - lw + 3, liq_y), (cx + lw - 3, liq_y), (cx + bw - 3, base_y - h * 0.16), (cx - bw + 3, base_y - h * 0.16)]
    fill_poly(img, liq, AMBER * 0.55, 0.88, 1.0)
    # inner warm glow of the liquor
    fill_poly(glow, [(cx - lw * 0.5, liq_y + 10), (cx + lw * 0.3, liq_y + 10), (cx + bw * 0.3, base_y - h * 0.2), (cx - bw * 0.5, base_y - h * 0.2)], AMBER * 0.35, 1.0, 12)
    if refl is not None:
        # reflected scene inside the liquor (verse 2 "琥珀倒影")
        rh, rw = int(base_y - h * 0.16 - liq_y), int(lw * 2 - 10)
        if rh > 4 and rw > 4:
            ref = cv2.resize(refl, (rw, rh), interpolation=cv2.INTER_AREA)
            yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
            mx = xx + 4 * np.sin(yy / 9 + t * 2.0)
            my = yy + 2 * np.sin(xx / 13 + t * 1.6)
            ref = cv2.remap(ref, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            ref = ref * np.array([1.0, 0.65, 0.3], np.float32) * 1.4
            m = poly_mask(np.array(liq, np.float32) - [cx - lw + 5, liq_y], rw, rh)
            paste(img, ref, m * 0.85, int(cx - lw + 5), int(liq_y))
    # meniscus
    cv2.ellipse(img, (int(cx), int(liq_y)), (int(lw - 3), int(w * 0.035)), 0, 0, 360, tuple(map(float, AMBER * 0.9)), 1, cv2.LINE_AA)
    cv2.ellipse(glow, (int(cx - lw * 0.3), int(liq_y)), (int(lw * 0.35), 2), 0, 0, 360, tuple(map(float, TUNGSTEN * 0.6)), -1)
    if sugar:
        s = w * 0.30 * (1 - 0.7 * melt)
        sx, sy = cx + w * 0.05, liq_y - s * 0.35 * (1 - melt) + s * 0.4 * melt
        bob = math.sin(t * 1.2) * 2
        rad = s * (0.08 + 0.35 * melt)
        cube = np.array([(sx - s / 2, sy - s / 2), (sx + s / 2, sy - s / 2), (sx + s / 2, sy + s / 2), (sx - s / 2, sy + s / 2)], np.float32)
        cube[:, 1] += bob
        M = cv2.getRotationMatrix2D((sx, sy), 12 + t * 3, 1)
        cube = cube @ M[:, :2].T + M[:, 2]
        if melt > 0.05:
            cube = chaikin(cube, 1 + int(melt * 3))
        fill_poly(img, cube, (0.78, 0.70, 0.58), 0.92, max(0.5, rad * 0.15))
        # granular texture + lit top-left edge
        r2 = rng(12)
        for _ in range(40):
            gx = sx + r2.uniform(-0.4, 0.4) * s
            gy = sy + bob + r2.uniform(-0.4, 0.4) * s
            glow_dot(glow, gx, gy, 1, np.array([1.0, 0.95, 0.85]), r2.uniform(0.05, 0.25))
        cv2.polylines(glow, [np.int32(cube[: max(2, len(cube) // 2)])], False, (0.5, 0.42, 0.32), 2, cv2.LINE_AA)
        # dissolving grains drifting down & swirling
        r = rng(11)
        for i in range(70):
            ph = r.uniform(0, 1)
            life = (t * 0.12 * r.uniform(0.5, 1.5) + ph) % 1.0
            ang = r.uniform(0, 6.28) + life * 3
            px = sx + math.cos(ang) * life * lw * 0.7
            py = sy + life * (base_y - h * 0.18 - sy)
            if liq_y < py < base_y - h * 0.18:
                glow_dot(glow, px, py, 1.2, np.array([1.0, 0.85, 0.6]), 0.7 * (1 - life) * (0.3 + melt))
    # glass rims & highlights
    edge = np.zeros_like(img)
    cv2.ellipse(edge, (int(cx), int(top_y)), (int(tw), int(w * 0.05)), 0, 0, 360, (0.55, 0.5, 0.45), 1, cv2.LINE_AA)
    cv2.line(edge, (int(cx - tw), int(top_y)), (int(cx - bw), int(base_y)), (0.5, 0.45, 0.4), 2, cv2.LINE_AA)
    cv2.line(edge, (int(cx + tw), int(top_y)), (int(cx + bw), int(base_y)), (0.3, 0.25, 0.2), 1, cv2.LINE_AA)
    cv2.ellipse(edge, (int(cx), int(base_y - h * 0.08)), (int(bw), int(w * 0.04)), 0, 0, 360, (0.45, 0.35, 0.25), 2, cv2.LINE_AA)
    cv2.line(edge, (int(cx - tw * 0.78), int(top_y + h * 0.12)), (int(cx - bw * 0.78), int(base_y - h * 0.3)), (0.45, 0.42, 0.4), max(1, int(w / 140)), cv2.LINE_AA)
    img += blur(edge, 0.8)
    glow += blur(edge, 2.0) * 0.4


# ---------------------------------------------------------------- scenes
# Each scene: fn(t_global, u = local progress 0..1, p = params) -> HxWx3 float


def sc_window(t, u, p):
    """Rainy window, city bokeh behind. Used for intro (title) and outro."""
    img = vgrad((0.015, 0.02, 0.04), (0.04, 0.03, 0.05))
    bokeh(img, 1, 70, t, [TUNGSTEN, AMBER, ROSE, CYAN * 0.8, VIOLET], 14, 46, vel=(6, 0), gain=0.32, area=(0, 120, W, 640))
    window_drops(img, 2, t, n=220, gain=0.8)
    if p.get("ghosts"):
        # faint reflection of the two of them in the glass
        a = p["ghosts"]
        draw_figure(img, "m", 470, 760, 700, light=(1, -0.2), rim=TUNGSTEN, alpha=a, rim_gain=a)
        draw_figure(img, "f", 800, 760, 680, flip=True, light=(-1, -0.2), rim=ROSE, alpha=a, rim_gain=a)
    if p.get("title"):
        a = smooth((t - 2.0) / 2.5) * (1 - smooth((t - 6.0) / 1.0))
        draw_text(img, "融化的糖", W / 2, H / 2 - 20, 92, (1.0, 0.8, 0.6), a, spacing=34)
        draw_text(img, "M E L T I N G   S U G A R", W / 2, H / 2 + 60, 22, (0.9, 0.7, 0.6), a * 0.85)
    if p.get("end"):
        a = smooth((t - 171.5) / 2.0) * (1 - smooth((t - 177.0) / 2.0))
        draw_text(img, "未完待續", W / 2, H / 2, 54, (1.0, 0.75, 0.6), a, spacing=26)
    return img


def sc_glass(t, u, p):
    img = vgrad((0.03, 0.02, 0.015), (0.01, 0.008, 0.008))
    bokeh(img, 3, 30, t, [TUNGSTEN, AMBER], 25, 70, vel=(3, 0), gain=0.35, area=(0, 100, W, 420))
    glow = np.zeros_like(img)
    counter(img, 520)
    melt = lerp(p.get("melt0", 0.0), p.get("melt1", 0.4), u)
    rocks_glass(img, glow, W * 0.5 + 40, 600, 330, liquid=0.5, t=t, melt=melt)
    # bulb reflection on counter
    cv2.ellipse(glow, (int(W * 0.3), 560), (60, 6), 0, 0, 360, tuple(map(float, TUNGSTEN * 0.4)), -1)
    add_glow(img, glow, 1.5, 30, 1.0)
    return img


def bar_setting(t, dist, tilt=0.0, push=0.0, rim_m=TUNGSTEN, rim_f=ROSE):
    img = back_bar().copy()
    glow = np.zeros_like(img)
    for i, bx in enumerate((180, 470, 800, 1100)):
        bulb(glow, img, bx, 170 + 25 * (i % 2), 0, t, i * 1.7, 1.0, 1.2)
    cx = W / 2
    size = 420 + 80 * push
    sep = dist * 280
    breathe = math.sin(t * 1.4) * 2
    by = 545 + 0.02 * size
    draw_figure(img, "m", cx - sep - 90, by + breathe, size, light=(0.6, -1.0), rim=rim_m, rot=-tilt)
    draw_figure(img, "f", cx + sep + 90, by + 4 - breathe, size * 0.96, flip=True, light=(-0.6, -1.0), rim=rim_f, rot=tilt)
    counter(img, 545)
    rocks_glass(img, glow, cx - sep + 10, 600, 70, liquid=0.4, t=t, sugar=False)
    rocks_glass(img, glow, cx + sep - 10, 600, 64, liquid=0.3, t=t + 2, sugar=False)
    add_glow(img, glow, 1.5, 45, 1.0)
    return img


def sc_bar_wide(t, u, p):
    d = lerp(p["d0"], p["d1"], smooth(u))
    tilt = p.get("tilt", 0) * smooth((u - 0.3) / 0.5)
    return bar_setting(t, d, tilt)


def sc_closeup(t, u, p):
    kind = p["who"]
    img = back_bar().copy()
    img = cv2.resize(img[100:500, 200:1000], (W, H)) * 0.8
    glow = np.zeros_like(img)
    bokeh(img, 5, 18, t, [TUNGSTEN, AMBER, ROSE], 40, 90, vel=(4, 0), gain=0.25, area=(0, 0, W, H))
    rim = TUNGSTEN if kind == "m" else ROSE
    breathe = math.sin(t * 1.3) * 3
    look = p.get("look", 0) * smooth((u - 0.4) / 0.4)
    if kind == "m":
        draw_figure(img, "m", W * 0.42 - u * 30, 1250 + breathe, 1250, light=(0.8, -0.6), rim=rim, rot=-look, rim_gain=1.1)
    else:
        draw_figure(img, "f", W * 0.58 + u * 30, 1250 + breathe, 1250, flip=True, light=(-0.8, -0.6), rim=rim, rot=look, rim_gain=1.1)
    add_glow(img, glow)
    return img


def sc_two_shot(t, u, p):
    """Medium-close profiles facing each other; distance closes over the shot."""
    img = back_bar().copy()
    img = cv2.resize(img[60:560, 100:1180], (W, H)) * 0.7
    k = feat("kick", t)
    bokeh(img, 6, 26, t, [TUNGSTEN, AMBER, ROSE, VIOLET], 30, 80, vel=(3, -2), gain=0.25)
    d = lerp(p["d0"], p["d1"], smooth(u))
    sep = 140 + d * 260
    sway = math.sin(t * 2 * math.pi / BAR * 0.5) * 6
    heart = p.get("heart", 0) * k
    size = p.get("size", 900)
    tilt = p.get("tilt", 2.0)
    draw_figure(img, "m", W / 2 - sep + sway, H + size * 0.18, size, light=(0.7, -0.7), rim=TUNGSTEN, rot=-tilt, rim_gain=1 + heart * 0.5)
    draw_figure(img, "f", W / 2 + sep + sway, H + size * 0.2, size * 0.95, flip=True, light=(-0.7, -0.7), rim=ROSE, rot=tilt, rim_gain=1 + heart * 0.5)
    if p.get("dim_build"):
        # pre-chorus build: bass drops out right before the chorus -> lights dip
        x = smooth((t - p["dim_build"]) / 1.2)
        img *= 1 - 0.65 * x
    return img


def sc_hands(t, u, p):
    """Low angle along the counter: two hands sliding toward each other."""
    img = back_bar().copy()
    img = cv2.resize(img[150:450, 100:1180], (W, H)) * 0.55
    bokeh(img, 8, 20, t, [TUNGSTEN, AMBER, ROSE], 40, 90, vel=(3, 0), gain=0.22, area=(0, 60, W, 380))
    top = 400
    surf = vgrad((0.06, 0.032, 0.02), (0.11, 0.06, 0.035), 560 - top, W)
    r = rng(9)
    grain = np.zeros((560 - top, W), np.float32)
    for i in range(30):
        y = r.uniform(0, 560 - top)
        pts = np.array([(x, y + 3 * math.sin(x / r.uniform(80, 200) + i)) for x in range(0, W + 40, 40)], np.int32)
        cv2.polylines(grain, [pts], False, float(r.uniform(0.01, 0.03)), 1, cv2.LINE_AA)
    surf += blur(grain, 1.0)[..., None] * TUNGSTEN
    img[top:560] = surf
    img[560:] = vgrad((0.03, 0.016, 0.01), (0.008, 0.005, 0.004), H - 560, W)
    k = feat("kick", t)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    pool = np.exp(-(((xx - W / 2) / 380) ** 2 + ((yy - 480) / 90) ** 2)) * (yy > top)
    img += pool[..., None] * TUNGSTEN * (0.20 + 0.10 * k)
    glow = np.zeros_like(img)
    edge = np.zeros_like(img)
    cv2.line(edge, (0, 560), (W, 560), tuple(map(float, TUNGSTEN * 0.7)), 2, cv2.LINE_AA)
    img += blur(edge, 1) + blur_fast(edge, 10) * 0.4
    g = lerp(p["g0"], p["g1"], smooth(u))
    gap = 14 + g * 420
    rocks_glass(img, glow, 210, 470, 150, liquid=0.35, t=t, sugar=False)
    rocks_glass(img, glow, 1080, 462, 140, liquid=0.25, t=t + 1, sugar=False)
    draw_hand(img, "hand_m", W / 2 - gap / 2, 496, 620, rim=TUNGSTEN, light=(0.2, -1))
    draw_hand(img, "hand_f", W / 2 + gap / 2, 488, 580, flip=True, rim=ROSE, light=(-0.2, -1))
    add_glow(img, glow)
    return img


def sc_glass_reflection(t, u, p):
    """Close two-shot seen as a reflection inside the amber liquor."""
    inner = sc_two_shot(t, 0.5 + 0.5 * u, {"d0": 0.55, "d1": 0.35, "size": 820})
    img = vgrad((0.03, 0.02, 0.015), (0.01, 0.008, 0.008))
    bokeh(img, 4, 26, t, [TUNGSTEN, AMBER], 30, 80, vel=(2, 0), gain=0.3, area=(0, 60, W, 380))
    glow = np.zeros_like(img)
    counter(img, 600)
    rocks_glass(img, glow, W / 2, 700, 640 + 40 * u, liquid=0.7, t=t, sugar=False, refl=inner)
    add_glow(img, glow, 1.5, 30, 0.8)
    return img


def city_strip(t, seed, step=None):
    """Background seen through a car window: passing neon streaks."""
    tt = math.floor(t * step) / step if step else t
    img = vgrad((0.02, 0.01, 0.05), (0.06, 0.02, 0.08))
    bokeh(img, seed, 60, tt, [MAGENTA, CYAN, VIOLET, AMBER, ROSE], 12, 40, vel=(-420, 0), gain=0.55, streak=60, twinkle=0.0, ring=False, area=(0, 140, W, 600))
    bokeh(img, seed + 1, 25, tt, [MAGENTA, CYAN, TUNGSTEN], 40, 90, vel=(-180, 0), gain=0.25, twinkle=0.0, area=(0, 100, W, 560))
    return img


def passing_light(t, period, phase):
    x = ((t + phase) % period) / period
    return math.exp(-((x - 0.5) / 0.12) ** 2)


def sc_car(t, u, p):
    img = city_strip(t, 20, step=p.get("step", 8))
    window_drops(img, 21, t, n=160, gain=0.9, tint=(0.9, 0.5, 1.0))
    # interior frame: dark surround with window opening
    frame = np.ones((H, W), np.float32)
    wx0, wy0, wx1, wy1 = 60, 110, W - 60, 600
    cv2.rectangle(frame, (wx0, wy0), (wx1, wy1), 0.0, -1)
    frame = blur(frame, 6)
    interior = vgrad((0.012, 0.01, 0.02), (0.02, 0.012, 0.02))
    img = img * (1 - frame[..., None]) + interior * frame[..., None]
    k = feat("kick", t)
    pm = passing_light(t, 2.3, 0.0)
    pc = passing_light(t, 3.1, 1.2)
    close = p.get("close", 0.0) + 0.15 * smooth(u)
    sep = lerp(250, 150, close)
    size = lerp(760, 980, close)
    rim_m = CYAN * (0.6 + 0.8 * pc) + MAGENTA * 0.2
    rim_f = MAGENTA * (0.6 + 0.8 * pm) + ROSE * 0.2
    sway = math.sin(t * 2 * math.pi / BAR) * 4
    draw_figure(img, "m", W / 2 - sep + sway, H + size * 0.22, size, light=(0.5, -0.9), rim=rim_m, rim_gain=1 + 0.6 * k, rot=-2)
    draw_figure(img, "f", W / 2 + sep + sway, H + size * 0.24, size * 0.95, flip=True, light=(-0.5, -0.9), rim=rim_f, rim_gain=1 + 0.6 * k, rot=2.5)
    # neon spill sweeping across the cabin
    spill = np.zeros_like(img)
    sx = (t * 500) % (W * 2) - W / 2
    cv2.rectangle(spill, (int(sx), 0), (int(sx + 140), H), tuple(map(float, MAGENTA * 0.10)), -1)
    img += blur_fast(spill, 60)
    return img


def sc_neon_close(t, u, p):
    who = p["who"]
    img = city_strip(t, 30 + (who == "f"), step=8)
    img = blur_fast(img, 8) * 1.1
    bokeh(img, 31, 14, math.floor(t * 8) / 8, [MAGENTA, CYAN, VIOLET], 50, 110, vel=(-60, 0), gain=0.35, twinkle=0.0)
    k = feat("kick", t)
    alt = math.floor(t / (BAR / 2)) % 2
    c1, c2 = (MAGENTA, CYAN) if alt else (CYAN, MAGENTA)
    if who in ("m", "both"):
        x = W * (0.40 if who == "m" else 0.30)
        draw_figure(img, "m", x - 20 * u, H + 720, 1400, light=(0.9, -0.4), rim=c1, rim_gain=1.0 + 0.7 * k, rot=-1)
    if who in ("f", "both"):
        x = W * (0.60 if who == "f" else 0.70)
        draw_figure(img, "f", x + 20 * u, H + 720, 1360, flip=True, light=(-0.9, -0.4), rim=c2, rim_gain=1.0 + 0.7 * k, rot=1)
    return img


@lru_cache(maxsize=2)
def street_bg(seed=40):
    img = vgrad((0.02, 0.015, 0.05), (0.05, 0.02, 0.06))
    r = rng(seed)
    x = -20
    while x < W:
        bw = r.uniform(80, 200)
        top = r.uniform(60, 300)
        cv2.rectangle(img, (int(x), int(top)), (int(x + bw), 520), (0.015, 0.012, 0.025), -1)
        for wy in range(int(top) + 12, 500, 22):
            for wx in range(int(x) + 8, int(x + bw) - 10, 18):
                if r.uniform() < 0.35:
                    col = [TUNGSTEN, AMBER, CYAN * 0.6][r.integers(3)] * r.uniform(0.2, 0.6)
                    cv2.rectangle(img, (wx, wy), (wx + 8, wy + 10), tuple(map(float, col)), -1)
        x += bw + r.uniform(5, 30)
    signs = np.zeros_like(img)
    for i in range(9):
        sx, sy = r.uniform(0, W), r.uniform(200, 470)
        col = [MAGENTA, CYAN, ROSE, VIOLET][r.integers(4)]
        cv2.rectangle(signs, (int(sx), int(sy)), (int(sx + r.uniform(20, 60)), int(sy + r.uniform(50, 120))), tuple(map(float, col)), -1)
    img += signs * 0.5 + blur_fast(signs, 40) * 0.8
    # wet street reflection
    ref = img[:520][::-1][:H - 520] * 0.35
    img[520:] = ref + 0.01
    return blur(img, 5)


def sc_umbrella(t, u, p):
    img = street_bg().copy()
    tt = math.floor(t * 12) / 12
    # ripple the reflection
    k = feat("kick", t)
    bokeh(img, 41, 30, t, [MAGENTA, CYAN, TUNGSTEN, ROSE], 20, 60, vel=(2, 0), gain=0.25, area=(0, 80, W, 500))
    push = p.get("push", 0) + 0.2 * u
    size = lerp(620, 820, push)
    cx = W / 2
    sway = math.sin(t * 2 * math.pi / BAR * 0.5) * 8
    gy = H + size * 0.08
    sep = size * 0.30
    draw_figure(img, "m", cx - sep + sway, gy, size, light=(-0.3, -1), rim=CYAN, rim_gain=1 + 0.5 * k, rot=-3)
    draw_figure(img, "f", cx + sep + sway, gy + 10, size * 0.94, flip=True, light=(0.3, -1), rim=MAGENTA, rim_gain=1 + 0.5 * k, rot=4)
    # clear umbrella dome
    R = size * 0.72
    ux, uy = cx + sway + 10, max(gy - size * 0.86, LETTERBOX + R * 0.55 + 14)
    dome = [(ux + R * math.cos(a), uy - R * 0.55 * math.sin(a)) for a in np.linspace(0, math.pi, 40)]
    scal = [(ux - R + 2 * R * i / 16, uy + (6 if i % 2 else 0)) for i in range(17)]
    poly = dome + scal
    fill_poly(img, poly, (0.6, 0.7, 0.9), 0.10, 2)
    edge = np.zeros_like(img)
    cv2.polylines(edge, [np.int32(dome)], False, (0.6, 0.65, 0.8), 2, cv2.LINE_AA)
    for a in np.linspace(0.15, math.pi - 0.15, 7):
        cv2.line(edge, (int(ux), int(uy - R * 0.55)), (int(ux + R * math.cos(a)), int(uy - R * 0.55 * math.sin(a) + R * 0.55 * (1 - math.sin(a)) * 0 + 4)), (0.25, 0.28, 0.35), 1, cv2.LINE_AA)
    cv2.line(edge, (int(ux), int(uy - R * 0.55)), (int(ux + 4), int(uy + size * 0.2)), (0.3, 0.3, 0.35), 2, cv2.LINE_AA)
    img += blur(edge, 0.8) + blur_fast(edge, 10) * 0.4
    window_drops(img, 42, 0.0, n=60, gain=0.6, tint=(0.8, 0.6, 1.0), area=(int(ux - R), int(uy - R * 0.5), int(ux + R), int(uy)))
    rain_streaks(img, 43, tt, n=300, gain=0.2)
    return img


def sc_fingertips(t, u, p):
    """Bridge: drums drop out — two fingertips almost touching, slow motion."""
    img = np.zeros((H, W, 3), np.float32) + 0.006
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    spot = np.exp(-(((xx - W / 2) / 360) ** 2 + ((yy - H / 2) / 260) ** 2))
    breath = 0.5 + 0.5 * math.sin(t * 0.8)
    img += spot[..., None] * TUNGSTEN * (0.08 + 0.03 * breath)
    gap = lerp(170, 6, smooth(u * 1.05))
    draw_hand(img, "hand_m", W / 2 - gap / 2, H / 2 + 20, 720, rim=TUNGSTEN, light=(0.3, -1), rot=-4)
    draw_hand(img, "hand_f", W / 2 + gap / 2, H / 2 + 6, 680, flip=True, rim=ROSE, light=(-0.3, -1), rot=4)
    glow = np.zeros_like(img)
    # the spark between fingertips grows as the gap closes
    spark = smooth((u - 0.5) / 0.5)
    glow_dot(glow, W / 2, H / 2 + 12, 3 + 4 * spark, np.array([1.0, 0.85, 0.7]), 0.4 + 1.6 * spark)
    # floating dust
    r = rng(50)
    for i in range(120):
        x0, y0 = r.uniform(0, W), r.uniform(0, H)
        x = (x0 + 8 * t * r.uniform(-1, 1)) % W
        y = (y0 - 6 * t * r.uniform(0.2, 1)) % H
        a = spot[int(y), int(x)] * r.uniform(0.2, 0.8)
        glow_dot(glow, x, y, r.uniform(1, 2.5), TUNGSTEN, a)
    add_glow(img, glow, 1.2, 30, 1.2)
    return img


SCENES = {
    "window": sc_window, "glass": sc_glass, "bar_wide": sc_bar_wide, "closeup": sc_closeup,
    "two_shot": sc_two_shot, "hands": sc_hands, "glass_reflection": sc_glass_reflection,
    "car": sc_car, "neon_close": sc_neon_close, "umbrella": sc_umbrella, "fingertips": sc_fingertips,
}

# ---------------------------------------------------------------- edit decision list
# (start, end, scene, params, camera) — camera: (zoom0, zoom1, dx, dy)
V1, P1, C1, V2, P2, C2, BR, C3, OUT = 13.5, 34.3, 47.0, 71.5, 92.3, 107.0, 132.0, 143.0, 168.0

SHOTS = [
    # Intro — 微醺、私密的夜晚基調
    (0.0, 7.0, "window", {"title": True}, (1.0, 1.06, 0, 0)),
    (7.0, V1, "glass", {"melt0": 0.0, "melt1": 0.35}, (1.08, 1.0, -10, 0)),
    # Verse 1 — 視線交錯的試探
    (V1, 19.5, "bar_wide", {"d0": 1.0, "d1": 0.95}, (1.0, 1.05, 0, 0)),
    (19.5, 25.4, "closeup", {"who": "m", "look": 3}, (1.04, 1.0, 8, 0)),
    (25.4, 31.3, "closeup", {"who": "f", "look": 3}, (1.0, 1.04, -8, 0)),
    (31.3, P1, "bar_wide", {"d0": 0.9, "d1": 0.8, "tilt": 4}, (1.05, 1.1, 0, -5)),
    # Pre-chorus 1 — 心跳般的推進
    (P1, 40.2, "hands", {"g0": 1.0, "g1": 0.55}, (1.0, 1.08, 0, 0)),
    (40.2, C1, "two_shot", {"d0": 0.9, "d1": 0.6, "heart": 1, "dim_build": 45.4}, (1.0, 1.1, 0, 0)),
    # Chorus 1 — 霓虹雨夜車窗
    (C1, 53.0, "car", {"close": 0.0}, (1.06, 1.0, 0, 0)),
    (53.0, 59.0, "neon_close", {"who": "f"}, (1.0, 1.06, 0, 0)),
    (59.0, 65.0, "car", {"close": 0.4}, (1.0, 1.06, 0, 0)),
    (65.0, V2, "neon_close", {"who": "m"}, (1.06, 1.0, 0, 0)),
    # Verse 2 — 琥珀色酒杯倒影
    (V2, 77.5, "glass_reflection", {}, (1.0, 1.08, 0, 0)),
    (77.5, 83.4, "bar_wide", {"d0": 0.6, "d1": 0.5, "tilt": 3}, (1.0, 1.06, 0, 0)),
    (83.4, 89.3, "two_shot", {"d0": 0.55, "d1": 0.4, "size": 960}, (1.0, 1.05, 0, 0)),
    (89.3, P2, "glass", {"melt0": 0.45, "melt1": 0.6}, (1.0, 1.05, 0, 0)),
    # Pre-chorus 2
    (P2, 98.2, "hands", {"g0": 0.55, "g1": 0.12}, (1.0, 1.08, 0, 0)),
    (98.2, C2, "two_shot", {"d0": 0.4, "d1": 0.2, "heart": 1, "dim_build": 105.4, "size": 1000}, (1.0, 1.12, 0, 0)),
    # Chorus 2 — 雨中透明傘
    (C2, 113.0, "umbrella", {"push": 0.0}, (1.0, 1.05, 0, 0)),
    (113.0, 119.0, "neon_close", {"who": "both"}, (1.0, 1.05, 0, 0)),
    (119.0, 125.0, "umbrella", {"push": 0.5}, (1.0, 1.06, 0, 0)),
    (125.0, BR, "car", {"close": 0.6}, (1.0, 1.06, 0, 0)),
    # Bridge — 指尖將觸未觸
    (BR, C3, "fingertips", {}, (1.0, 1.12, 0, 0)),
    # Final chorus — montage on the bar line
    (C3, C3 + BAR, "umbrella", {"push": 0.6}, (1.08, 1.0, 0, 0)),
    (C3 + BAR, C3 + 2 * BAR, "car", {"close": 0.8}, (1.0, 1.06, 0, 0)),
    (C3 + 2 * BAR, C3 + 3 * BAR, "neon_close", {"who": "both"}, (1.06, 1.0, 0, 0)),
    (C3 + 3 * BAR, C3 + 4 * BAR, "two_shot", {"d0": 0.2, "d1": 0.1, "size": 1050, "heart": 1}, (1.0, 1.06, 0, 0)),
    (C3 + 4 * BAR, C3 + 5 * BAR, "umbrella", {"push": 0.9}, (1.0, 1.06, 0, 0)),
    (C3 + 5 * BAR, C3 + 6 * BAR, "car", {"close": 1.0}, (1.06, 1.0, 0, 0)),
    (C3 + 6 * BAR, C3 + 7 * BAR, "hands", {"g0": 0.12, "g1": 0.0}, (1.0, 1.06, 0, 0)),
    (C3 + 7 * BAR, OUT, "two_shot", {"d0": 0.1, "d1": 0.02, "size": 1100, "heart": 1, "tilt": 4}, (1.0, 1.1, 0, 0)),
    # Outro — 未完待續的餘溫
    (OUT, DURATION + 1, "window", {"ghosts": 0.25, "end": True}, (1.0, 1.08, 0, 0)),
]
XFADE = 0.35
CHORUS_STARTS = (C1, C2, C3)


def render_shot(i, t):
    s, e, name, params, _ = SHOTS[i]
    u = (t - s) / (e - s)
    return SCENES[name](t, u, params)


def camera(img, i, t):
    s, e, _, _, (z0, z1, dx, dy) = SHOTS[i]
    u = smooth((t - s) / (e - s))
    z = lerp(z0, z1, u)
    shake_x = 2.0 * math.sin(t * 1.7) + 1.2 * math.sin(t * 3.3 + 1)
    shake_y = 1.5 * math.sin(t * 1.3 + 2) + 0.8 * math.sin(t * 2.9)
    tx = dx * u + shake_x
    ty = dy * u + shake_y
    M = np.float32([[z, 0, (1 - z) * W / 2 + tx], [0, z, (1 - z) * H / 2 + ty]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------- post


@lru_cache(maxsize=1)
def _vignette():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    return np.clip(1.15 - 0.55 * d ** 1.8, 0, 1)[..., None]


@lru_cache(maxsize=1)
def _grain():
    r = rng(99)
    out = []
    for _ in range(12):
        g = r.normal(0, 1, (H // 2, W // 2)).astype(np.float32)
        g = cv2.resize(blur(g, 0.6), (W, H))
        out.append(g[..., None])
    return out


def post(img, t, fi):
    in_chorus = any(c <= t < c + 24.5 for c in CHORUS_STARTS)
    k = feat("kick", t)
    exposure = 1.0 + (0.10 * k if in_chorus else 0.03 * k)
    img = img * exposure
    # bloom
    bright = np.maximum(img - 0.55, 0)
    img = img + blur_fast(bright, 18) * 0.7 + blur_fast(bright, 60, 8) * 0.5
    # chorus entry flash
    for c in CHORUS_STARTS:
        if c - 0.05 <= t < c + 0.6:
            img = img + np.array([1.0, 0.8, 0.65], np.float32) * 0.5 * math.exp(-(t - c) * 7)
    # filmic shoulder
    img = 1 - np.exp(-img * 1.35)
    # split-tone: teal shadows, warm highlights
    lum = img.mean(2, keepdims=True)
    img = img + (1 - lum) * np.array([-0.006, 0.006, 0.018], np.float32) + lum * np.array([0.03, 0.01, -0.025], np.float32)
    img = img * _vignette()
    img = img + _grain()[fi % 12] * 0.022
    # global fades
    img = img * smooth(t / 1.5) * (1 - smooth((t - (DURATION - 2.5)) / 2.5))
    img[:LETTERBOX] = 0
    img[H - LETTERBOX:] = 0
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def render_frame(fi):
    t = fi / FPS
    idx = [i for i, (s, e, *_ ) in enumerate(SHOTS) if s - XFADE <= t < e]
    cur = max(i for i in idx if SHOTS[i][0] <= t) if any(SHOTS[i][0] <= t for i in idx) else idx[0]
    img = camera(render_shot(cur, t), cur, t)
    nxt = cur + 1
    if nxt < len(SHOTS) and SHOTS[nxt][0] - XFADE <= t and SHOTS[nxt][0] not in CHORUS_STARTS and SHOTS[nxt][0] != BR:
        a = smooth((t - (SHOTS[nxt][0] - XFADE)) / XFADE)
        img = img * (1 - a) + camera(render_shot(nxt, t), nxt, t) * a
    return post(img, t, fi)


def _init():
    global FEAT
    FEAT = _load_features()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="comma separated times (s)")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=DURATION)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "melting_sugar_mv.mp4"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if a.preview:
        _init()
        pdir = os.path.join(OUT_DIR, "preview")
        os.makedirs(pdir, exist_ok=True)
        for ts in a.preview.split(","):
            fi = int(float(ts) * FPS)
            fr = render_frame(fi)
            path = os.path.join(pdir, f"t{float(ts):06.1f}.png")
            cv2.imwrite(path, cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            print(path)
        return
    if a.render:
        f0, f1 = int(a.start * FPS), int(a.end * FPS)
        cmd = ["ffmpeg", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
               "-ss", str(a.start), "-t", str(a.end - a.start), "-i", AUDIO,
               "-map", "0:v", "-map", "1:a",
               "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-tune", "film",
               "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", a.out]
        ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        with Pool(a.workers, initializer=_init) as pool:
            for n, fr in enumerate(pool.imap(render_frame, range(f0, f1), chunksize=4)):
                ff.stdin.write(fr.tobytes())
                if n % (FPS * 5) == 0:
                    print(f"{(f0 + n) / FPS:6.1f}s / {a.end:.1f}s", flush=True)
        ff.stdin.close()
        ff.wait()
        print("wrote", a.out)


if __name__ == "__main__":
    main()
