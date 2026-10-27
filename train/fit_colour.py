"""Tune the colour rules on our labelled crops, and measure them honestly with cross-validation.

The rules have 7 thresholds. We try every combination in GRID on the labelled crops and keep the
one with the highest accuracy. Choosing thresholds on the same images we score them on would be
optimistic, so the accuracy we report comes from 5-fold cross-validation: pick the best thresholds
on 4/5 of the crops, score them on the remaining 1/5, rotate, and average.

Usage (from the codebase folder)
  uv run python -m train.fit_colour labels_uzbek.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json

import cv2
import numpy as np

from inference.colour import COLOURS, DEFAULT_PARAMS, PARAMS_PATH, decide, pixel_summary, region_pixels

GRID = {
    "top": [0.30, 0.40], "bottom": [0.75, 0.85],
    "s_min": [40, 60, 80], "v_dark": [40, 60],
    "chroma_frac": [0.15, 0.25, 0.35, 0.50],
    "v_white": [140, 160, 180], "v_black": [60, 80, 100],
}


def load(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = {r["path"]: r for r in csv.DictReader(f)}.values()     # latest label per image wins
    data = []
    for r in rows:
        if r["colour"] in COLOURS:
            img = cv2.imread(r["path"])
            if img is not None:
                data.append((img, COLOURS.index(r["colour"])))
    return data


def predict_all(data):
    """Predictions of every threshold combination for every crop: array (n_combos, n_crops).
    Pixels and pixel statistics are computed once per setting that changes them, the cheap
    final decisions once per combination."""
    combos, preds = [], []
    for top, bottom in itertools.product(GRID["top"], GRID["bottom"]):
        pixels = [region_pixels(img, top, bottom) for img, _ in data]
        for s_min, v_dark in itertools.product(GRID["s_min"], GRID["v_dark"]):
            summaries = [pixel_summary(px, s_min, v_dark) for px in pixels]
            for cf, vw, vb in itertools.product(GRID["chroma_frac"], GRID["v_white"], GRID["v_black"]):
                p = {"top": top, "bottom": bottom, "s_min": s_min, "v_dark": v_dark,
                     "chroma_frac": cf, "v_white": vw, "v_black": vb}
                combos.append(p)
                preds.append([COLOURS.index(decide(s, p)) for s in summaries])
    return combos, np.array(preds)


def report(y, pred, title):
    k = len(COLOURS)
    conf = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    print(f"\n{title}: accuracy {(pred == y).mean():.3f}")
    print("true \\ pred " + " ".join(f"{c[:5]:>6}" for c in COLOURS) + "  recall")
    for i, c in enumerate(COLOURS):
        if conf[i].sum():
            print(f"{c:>11} " + " ".join(f"{n:6d}" for n in conf[i]) + f"  {conf[i, i] / conf[i].sum():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", help="labels CSV written by label.py")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    data = load(args.labels)
    y = np.array([c for _, c in data])
    print(f"{len(data)} labelled crops:", {c: int((y == i).sum()) for i, c in enumerate(COLOURS)})

    default = np.array([COLOURS.index(decide(pixel_summary(region_pixels(img, DEFAULT_PARAMS["top"],
              DEFAULT_PARAMS["bottom"]), DEFAULT_PARAMS["s_min"], DEFAULT_PARAMS["v_dark"]), DEFAULT_PARAMS))
              for img, _ in data])
    report(y, default, "hand-picked default thresholds")

    combos, preds = predict_all(data)
    correct = preds == y
    folds = np.random.default_rng(0).permutation(len(y)) % args.folds
    out_of_fold = np.empty_like(y)
    for f in range(args.folds):
        best = correct[:, folds != f].mean(1).argmax()          # chosen without looking at fold f
        out_of_fold[folds == f] = preds[best, folds == f]
    report(y, out_of_fold, f"tuned thresholds, {args.folds}-fold cross-validation (honest estimate)")

    best = int(correct.mean(1).argmax())
    PARAMS_PATH.write_text(json.dumps(combos[best], indent=2))
    print(f"\nfinal thresholds (tuned on all {len(y)} crops, {len(combos)} combinations tried): {combos[best]}")
    print(f"saved to {PARAMS_PATH}")


if __name__ == "__main__":
    main()