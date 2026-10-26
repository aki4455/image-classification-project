
import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

COLOUR_KEYS = {"w": "white", "k": "black", "g": "grey", "r": "red", "b": "blue",
               "n": "green", "o": "brown", "y": "yellow"}
BRAND_KEYS = {"1": "Chevrolet", "2": "Daewoo", "3": "Ravon"}
EXTS = {".jpg", ".jpeg", ".png"}
FIELDS = ["path", "brand", "colour", "source"]
HELP = ["colour: w white  k black  g grey  r red  b blue  n green  o brown  y yellow",
        "brand: 1 Chevrolet  2 Daewoo  3 Ravon   |   x skip   p previous   q quit"]


def load_labels(csv_path: Path) -> dict[str, dict]:
    """path -> latest label row. If an image was labelled twice, the later line wins."""
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row["path"]: row for row in csv.DictReader(f)}


def append_label(csv_path: Path, row: dict) -> None:
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def show(img: np.ndarray, lines: list[str]) -> None:
    """Draw the image enlarged (small crops are hard to read) with a text bar on top."""
    h, w = img.shape[:2]
    scale = min(900 / w, 600 / h, 3.0)
    view = cv2.resize(img, (int(w * scale), int(h * scale)))
    width = max(view.shape[1], 760)
    view = cv2.copyMakeBorder(view, 0, 0, 0, width - view.shape[1], cv2.BORDER_CONSTANT)
    bar = np.zeros((28 * len(lines) + 8, width, 3), np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(bar, text, (8, 24 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("label", np.vstack([bar, view]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("crops", type=Path, help="folder with crops (searched recursively)")
    ap.add_argument("labels", type=Path, help="CSV to append labels to")
    ap.add_argument("--source", required=True, help="where these images came from, e.g. web, own, team")
    ap.add_argument("--default-brand", default="Chevrolet")
    args = ap.parse_args()

    paths = sorted(p for p in args.crops.rglob("*") if p.suffix.lower() in EXTS)
    labels = load_labels(args.labels)
    i = next((k for k, p in enumerate(paths) if str(p) not in labels), len(paths))  # first unlabelled
    print(f"{len(paths)} crops, {len(labels)} already labelled, starting at #{i + 1}")

    brand = None
    while 0 <= i < len(paths):
        path = paths[i]
        if brand is None:   # fresh image: previous label if we went back, else the default
            brand = labels.get(str(path), {}).get("brand") or args.default_brand
        img = cv2.imread(str(path))
        if img is None:
            i += 1
            continue
        show(img, [f"[{i + 1}/{len(paths)}] {path.parent.name}/{path.name}   brand: {brand}"] + HELP)
        key = chr(cv2.waitKey(0) & 0xFF).lower()

        if key in ("q", "\x1b"):
            break
        if key == "p":
            i, brand = max(0, i - 1), None
        elif key in BRAND_KEYS:
            brand = BRAND_KEYS[key]           # redraw the same image with the new brand
        elif key in COLOUR_KEYS or key == "x":
            row = {"path": str(path), "brand": brand,
                   "colour": COLOUR_KEYS.get(key, "skip"), "source": args.source}
            append_label(args.labels, row)
            labels[str(path)] = row
            i, brand = i + 1, None
    cv2.destroyAllWindows()

    kept = [r for r in labels.values() if r["colour"] != "skip"]
    print(f"labelled: {len(labels)} | usable: {len(kept)} | skipped: {len(labels) - len(kept)}")
    print("per brand :", dict(Counter(r["brand"] for r in kept)))
    print("per colour:", dict(Counter(r["colour"] for r in kept)))


if __name__ == "__main__":
    main()