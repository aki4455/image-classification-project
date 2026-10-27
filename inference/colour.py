"""Car colour: HSV rules, optionally helped by a small learned model.

Rules (always available): read the paint pixels in HSV and apply a few thresholds.
Learned (if train/fit_colour_model.py chose it): ImageNet ResNet-18 features of the whole crop
+ one linear layer. It sees lighting, shadows and reflections, which a brightness rule cannot.
fit_colour_model.py compares rules, learned and a hybrid by cross-validation and saves the winner
to colour_model.pt; without that file, predict_colour() simply uses the rules.

--- the rules ---

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
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18

COLOURS = ["white", "black", "grey", "red", "blue", "green", "brown", "yellow"]
RED, BLUE, GREEN, BROWN, YELLOW = (COLOURS.index(c) for c in ("red", "blue", "green", "brown", "yellow"))
PARAMS_PATH = Path(__file__).with_name("colour_params.json")
MODEL_PATH = Path(__file__).with_name("colour_model.pt")
RULE_ONLY = {"red", "green", "yellow"}   # hybrid: rare saturated colours, where hue rules are reliable

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


def rules_colour(crop_bgr: np.ndarray, p: dict | None = None) -> str:
    p = p or _PARAMS
    return decide(pixel_summary(region_pixels(crop_bgr, p["top"], p["bottom"]), p["s_min"], p["v_dark"]), p)


# --- the learned model ---------------------------------------------------------------------

def build_trunk(pretrained: bool) -> nn.Module:
    """ResNet-18 without its ImageNet output layer: crop -> 512 numbers describing it."""
    from torchvision.models import ResNet18_Weights
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    net.fc = nn.Identity()
    return net.eval()


@torch.no_grad()
def crop_features(trunk, crops_bgr, device, batch: int = 64) -> torch.Tensor:
    """Features of each crop, averaged with its mirror image (a car's colour doesn't change when flipped)."""
    from data.dataset import eval_transform
    tf = eval_transform()
    out = []
    for i in range(0, len(crops_bgr), batch):
        x = torch.stack([tf(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops_bgr[i:i + batch]])
        x = x.to(device)
        out.append(((trunk(x) + trunk(torch.flip(x, dims=[3]))) / 2).float().cpu())
    return torch.cat(out)


_LEARNED = None   # loaded on first use: (method, trunk, linear layer, mean, std, device)


def _load_learned():
    global _LEARNED
    if _LEARNED is None:
        if not MODEL_PATH.exists():
            _LEARNED = ("rules",)
        else:
            ck = torch.load(MODEL_PATH, map_location="cpu")
            if ck["method"] == "rules":
                _LEARNED = ("rules",)
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                trunk = build_trunk(pretrained=False)        # no download: weights come from our file
                trunk.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["trunk"].items()})
                lin = nn.Linear(ck["linear"]["weight"].shape[1], len(COLOURS))
                lin.load_state_dict(ck["linear"])
                _LEARNED = (ck["method"], trunk.to(device), lin, ck["mean"], ck["std"], device)
    return _LEARNED


def predict_colour(crop_bgr: np.ndarray) -> str:
    learned = _load_learned()
    rule = rules_colour(crop_bgr)
    if learned[0] == "rules" or (learned[0] == "hybrid" and rule in RULE_ONLY):
        return rule
    _, trunk, lin, mean, std, device = learned
    f = crop_features(trunk, [crop_bgr], device)
    return COLOURS[int(lin((f - mean) / std).argmax())]