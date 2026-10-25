from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np
from typing import Iterable 
from detect import detector



MIN_SIDE = 64
PAD = 0.05
EXTS = {".jpg", ".jpeg", ".png"}



def crop_boxes(img:np.ndarray, boxes: Iterable) -> list:
    h, w = img.shape[:2]
    crops = []
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        x1, y1 = max(0, int(x1 - PAD * bw)), max(0, int(y1 - PAD * bh))
        x2, y2 = min(w, int(x2 + PAD * bw)), min(h, int(y2 + PAD * bh))
        if min(x2 - x1, y2 - y1) >= MIN_SIDE:
            crops.append(img[y1:y2, x1:x2])
    
    return crops



src, dst, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
model = detector.get_model()

for img_path in sorted(p for p in src.rglob("*") if p.suffix.lower() in EXTS):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    r = model(img, classes=[2,5,7], conf=0.25, agnostic_nms=True, verbose=False)[0]
    boxes = r.boxes.xyxy.tolist()
    if mode == "largest" and boxes:
        boxes = [max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))]
    
    out_dir = dst / img_path.parent.relative_to(src)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, crop in enumerate(crop_boxes(img, boxes)):
        cv2.imwrite(str(out_dir / f"{img_path.stem}_{i}.png"), crop)