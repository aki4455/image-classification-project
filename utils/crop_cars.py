from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from detect import detector


VEHICLE_CLASSES = [2,5,7]
CONF = 0.25
MIN_SIDE = 64
PAD = 0.05
EXTS = {".jpg", ".jpeg", ".png"}


def box_area(box) -> float:
    x1,y1,x2,y2 = box
    return (x2 - x1) * (y2 - y1)


def crop_boxes(img:np.ndarray, boxes) -> list[np.ndarray]:
    h,w = img.shape[:2]
    
    crops = []
    
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        x1, y1 = max(0, int(x1 - PAD * bw)), max(0, int(y1 - PAD * bh))
        x2, y2 = min(w, int(x2 + PAD * bw)), min(h, int(y2 + PAD * bh))
        
        if min(x2 - x1, y2 - y1) >= MIN_SIDE:
            crops.append(img[y1:y2, x1:x2])
            
    return crops



def pick_main_box(boxes, img_area: float, min_frac: float, rival: float):
    if not boxes:
        return None, "no vehicle"
    
    boxes = sorted(boxes, key=box_area, reverse=True)
    main = boxes[0]
    if box_area(main) / img_area < min_frac:
        return None, "too_small"
    if len(boxes) > 1 and box_area(boxes[1]) >= rival * box_area(main):
        return None, "ambigious"
    
    return main, "ok"


def otsu_threshold(values) -> float:
    if len(values) == 0:
        return 0.0
    arr = (np.asarray(values) * 255).astype(np.uint8).reshape(-1, 1)
    t, _ = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (t + 1) / 255


def pring_histogram(values, bins: int = 10) -> None:
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    top = max(counts.max(), 1)
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        print(f"    {lo:.1f}-{hi:.1f} | {'#' * int(40 * c / top):<40} {c}")
        
        

def detect_all(model, paths):
    detections = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        r = model(img, classes=VEHICLE_CLASSES, conf=CONF, agnostic_nms=True, verbose=False)[0]
        detections.append((path, r.boxes.xyxy.tolist(), w,h))
    
    return detections


def parse_args():
    ap = argparse.ArgumentParser(description="Crop vehicles with a COCO detector")
    ap.add_argument("src", type=Path, help="folder with images (searched recursively)")
    ap.add_argument("dst", type=Path, help="output folder; subfolders are mirrored")
    ap.add_argument("mode", choices=["all", "largest"])
    ap.add_argument("--min-frac", type=float, default=None)
    ap.add_argument("--rival", type=float, default=0.5)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    model = detector.get_model()
    
    paths =sorted(p for p in args.src.rglob("*") if p.suffix.lower() in EXTS)
    detections = detect_all(model, paths)
    print(f"Images found {len(paths)} | reeadable: {len(detections)}")
    
    min_frac = 0.0
    if args.mode == "largest":
        ranked = sorted((max(box_area(b) for b in boxes) / (w * h), path.name)
                        for path, boxes, w, h in detections if boxes)
        fracs = [f for f, _ in ranked]
        print("smallest main boxes:", ", ".join(f"{n} ({f:.2f})" for f, n in ranked[:5]))
        print("largest-box share of image (photos with at least one vehicle)")
        pring_histogram(fracs)
        if args.min_frac is not None:
            min_frac, source = args.min_frac, "given"
        else:
            min_frac, source = otsu_threshold(fracs), "Otsu"
        
        print(f"min_frac = {min_frac:.3f} ({source})")
        
    counts = Counter()
    for path, boxes, w, h in detections:
        if args.mode == "largest":
            main_box, reason = pick_main_box(boxes, w * h, min_frac, args.rival)
            counts[reason] +=1
            if main_box is None:
                continue
            boxes = [main_box]
        
        img = cv2.imread(str(path))
        out_dir = args.dst / path.parent.relative_to(args.src)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, crop in enumerate(crop_boxes(img, boxes)):
            cv2.imwrite(str(out_dir / f"{path.stem}_{i}.png"), crop)
            counts["saved"] += 1
    
    print(dict(counts))
    if args.mode == "largest" and counts["too_small"] > 0.3 * len(detections):
        print("WARNING: over 30% skipped as too_small. Otsu may have cut into the good photos: || Check the histogram and pass --min_frac by hand")
        
        


if __name__ == "__main__":
    main()