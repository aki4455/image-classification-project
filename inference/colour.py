"""Car colour without a neural network: read the paint pixels in HSV and apply a few rules.

Why rules: the brief's 8 colours are defined by hue (which colour), saturation (how colourful)
and value (how bright). Our labelled crops have almost no rare colours (5 green, 1 yellow), so a
learned model could not learn those classes, while a hue rule handles them by definition.
The thresholds are not guessed: train/fit_colour.py tunes them on our labelled crops and saves
them to colour_params.json next to this file.

HSV in OpenCV: H = hue 0..179 (degrees / 2), S = saturation 0..255, V = brightness 0..255.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

COLOURS = ["white", "black", "grey", "red", "blue", "green", "brown", "yellow"]
RED, BLUE, GREEN, BROWN, YELLOW = (COLOURS.index(c) for c in ("red", "blue", "green", "brown", "yellow"))
PARAMS_PATH = Path(__file__).with_name("colour_params.json")

DEFAULT_PARAMS = {
    "top": 0.40, "bottom": 0.85,  # rows of the crop used as "paint": below the windows, above the road
    "s_min": 60,                  # a pixel is "coloured" if its saturation is at least this...
    "v_dark": 60,                 # ...and it is bright enough for its hue to mean anything
    "chroma_frac": 0.25,          # share of coloured pixels needed to call the car coloured
    "v_white": 160,               # otherwise: median brightness above this -> white,
    "v_black": 80,                # below this -> black, in between -> grey
}


def load_params(path: Path = PARAMS_PATH) -> dict:
    params = dict(DEFAULT_PARAMS)
    if path.exists():
        params.update(json.loads(path.read_text()))
    return params


def region_pixels(crop_bgr: np.ndarray, top: float, bottom: float, width: int = 128) -> np.ndarray:
    """Shrink the crop (averages away noise and screen moire), keep the central body area,
    return its pixels as an (N, 3) HSV array. The side margins are mostly background."""
    h, w = crop_bgr.shape[:2]
    small = cv2.resize(crop_bgr, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    body = small[int(top * sh):max(int(bottom * sh), int(top * sh) + 1), int(0.15 * sw):int(0.85 * sw)]
    return cv2.cvtColor(body, cv2.COLOR_BGR2HSV).reshape(-1, 3)


def hue_classes(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Colour index for each coloured pixel. Orange/yellow and brown/beige/gold share hues 8-34;
    bright AND saturated -> yellow (incl. orange), otherwise -> brown (incl. beige and gold)."""
    warm = (h >= 8) & (h < 35)
    return np.select([(h < 8) | (h >= 150), warm & (s >= 130) & (v >= 150), warm, h < 85, h < 150],
                     [RED, YELLOW, BROWN, GREEN, BLUE])


def pixel_summary(px: np.ndarray, s_min: int, v_dark: int) -> tuple[float, str | None, float]:
    """(share of coloured pixels, most common hue colour among them, median brightness of the rest)"""
    h, s, v = (px[:, i].astype(np.int16) for i in range(3))
    coloured = (s >= s_min) & (v >= v_dark)
    frac = float(coloured.mean()) if len(px) else 0.0
    hue_colour = None
    if coloured.any():
        counts = np.bincount(hue_classes(h[coloured], s[coloured], v[coloured]), minlength=len(COLOURS))
        hue_colour = COLOURS[int(counts.argmax())]
    rest = v[~coloured] if (~coloured).any() else v
    return frac, hue_colour, float(np.median(rest))


def decide(summary: tuple[float, str | None, float], p: dict) -> str:
    frac, hue_colour, med_v = summary
    if hue_colour is not None and frac >= p["chroma_frac"]:
        return hue_colour
    if med_v >= p["v_white"]:
        return "white"
    if med_v <= p["v_black"]:
        return "black"
    return "grey"


_PARAMS = load_params()


def predict_colour(crop_bgr: np.ndarray, p: dict | None = None) -> str:
    p = p or _PARAMS
    return decide(pixel_summary(region_pixels(crop_bgr, p["top"], p["bottom"]), p["s_min"], p["v_dark"]), p)