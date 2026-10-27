"""Test-day pipeline: phone photo -> brand and colour of every car -> one printed line.

    photo --YOLO--> car boxes --select--> crops --ConvNeXt--> brand (+ car model) --HSV rules--> colour

Usage (from the codebase folder)
  uv run python -m inference.predict photo1.jpg photo2.jpg
  uv run python -m inference.predict --watch "C:\\Users\\user\\Downloads\\Telegram Desktop"

Nothing here touches the internet: the detector and our weights are loaded from disk, and the
ConvNeXt is built empty (no ImageNet download) before our own weights are loaded into it.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import convnext_tiny

from data.dataset import eval_transform
from inference.colour import predict_colour
from train.network import TwoHeadConvNeXt
from utils.crop_cars import CONF, VEHICLE_CLASSES, box_area, crop_boxes
from utils.detect import detector

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CAR_MODEL_BRAND = "Chevrolet"   # the car-model head only knows our Uzbek Chevrolets


def parse_args():
    ap = argparse.ArgumentParser(description="Brand and colour of every car in a photo.")
    ap.add_argument("images", nargs="*", type=Path, help="photo files to process")
    ap.add_argument("--watch", type=Path, default=None, help="folder to watch for new photos")
    ap.add_argument("--weights", type=Path, default=Path("runs/v2_two_head/best.pt"))
    ap.add_argument("--min-conf", type=float, default=0.4,
                    help="ignore detections the detector is less sure about than this")
    ap.add_argument("--min-rel", type=float, default=0.15,
                    help="ignore cars whose box area is below this share of the biggest car's box")
    ap.add_argument("--max-cars", type=int, default=8)
    ap.add_argument("--min-model-p", type=float, default=0.5,
                    help="name the car model only if the car-model head is at least this sure")
    ap.add_argument("--out", type=Path, default=Path("outputs"), help="annotated copies go here")
    return ap.parse_args()


def load_classifier(path: Path, device: torch.device):
    """Rebuild the network exactly as in training, then load our weights. Works with the
    two-head v2 checkpoint and with the brand-only v1 baseline."""
    ckpt = torch.load(path, map_location=device)
    if "car_models" in ckpt:
        brands, car_models = ckpt["brands"], ckpt["car_models"]
        net = TwoHeadConvNeXt(len(brands), max(len(car_models), 1), pretrained=False)
    else:
        brands, car_models = ckpt["classes"], []
        net = convnext_tiny(weights=None)
        net.classifier[2] = nn.Linear(net.classifier[2].in_features, len(brands))
    net.load_state_dict(ckpt["state_dict"])
    return net.to(device).eval(), brands, car_models


def select_cars(boxes, confs, min_conf: float, min_rel: float, max_cars: int):
    """Decide which detections to report. Every extra prediction costs 1 point, and the instructor
    leaves out cars too small to identify, so we keep confident detections that are not tiny
    compared with the biggest car in the photo. Returns [(box, conf)], biggest first."""
    cands = sorted(((b, c) for b, c in zip(boxes, confs) if c >= min_conf),
                   key=lambda bc: box_area(bc[0]), reverse=True)
    if not cands:
        return []
    biggest = box_area(cands[0][0])
    return [(b, c) for b, c in cands if box_area(b) >= min_rel * biggest][:max_cars]


@torch.no_grad()
def classify(net, crops_bgr: list[np.ndarray], device, transform):
    """crops (OpenCV, BGR) -> (brand probabilities, car-model probabilities or None).
    Training images were read with PIL, which gives RGB, so we must convert BGR -> RGB here."""
    x = torch.stack([transform(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops_bgr])
    out = net(x.to(device))
    brand_logits, model_logits = out if isinstance(out, tuple) else (out, None)
    p_brand = brand_logits.float().softmax(1).cpu()
    p_model = model_logits.float().softmax(1).cpu() if model_logits is not None else None
    return p_brand, p_model


def draw(img, results, path: Path) -> None:
    """Save a copy of the photo with numbered boxes, for checking and for the report."""
    vis = img.copy()
    for i, r in enumerate(results, 1):
        x1, y1, x2, y2 = map(int, r["box"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        name = r["brand"] + (f" {r['car_model']}" if r["car_model"] else "")
        cv2.putText(vis, f"{i} {name} {r['p']:.2f} {r['colour']}", (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)


def process(path: Path, det_model, net, brands, car_models, device, transform, args) -> None:
    t0 = time.perf_counter()
    img = cv2.imread(str(path))
    if img is None:
        print(f"\n{path.name}: cannot read this file (iPhone HEIC? send it as a normal photo / JPEG)")
        return
    r = det_model(img, classes=VEHICLE_CLASSES, conf=CONF, agnostic_nms=True, verbose=False)[0]
    chosen = select_cars(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                         args.min_conf, args.min_rel, args.max_cars)

    results, crops = [], []
    for box, conf in chosen:
        crop = crop_boxes(img, [box])   # same padding and minimum size as the training crops
        if crop:
            crops.append(crop[0])
            results.append({"box": box, "det_conf": conf})
    if crops:
        p_brand, p_model = classify(net, crops, device, transform)
        for i, (res, crop) in enumerate(zip(results, crops)):
            top = p_brand[i].topk(min(2, len(brands)))
            res["brand"], res["p"] = brands[top.indices[0]], top.values[0].item()
            res["alt"] = f"{brands[top.indices[1]]} {top.values[1].item():.2f}" if len(brands) > 1 else "-"
            res["car_model"], res["p_model"] = None, 0.0
            if p_model is not None and car_models and res["brand"] == CAR_MODEL_BRAND:
                pm, im = p_model[i].max(0)
                if pm.item() >= args.min_model_p:
                    res["car_model"], res["p_model"] = car_models[im].capitalize(), pm.item()
            res["colour"] = predict_colour(crop)

    seconds = time.perf_counter() - t0
    print(f"\n=== {path.name}  ({len(results)} car(s), {seconds:.1f}s)")
    for i, res in enumerate(results, 1):
        x1, y1, x2, y2 = map(int, res["box"])
        model_txt = f" ({res['car_model']} {res['p_model']:.2f})" if res["car_model"] else ""
        print(f"  {i}. {res['brand']}{model_txt}  p={res['p']:.2f} (next: {res['alt']}) | colour {res['colour']} "
              f"| det {res['det_conf']:.2f} | box {x2 - x1}x{y2 - y1}")
    print("ANSWER:", "; ".join(f"{res['brand']} {res['colour']}" for res in results) or "(no car found)")
    if any(res["car_model"] for res in results):
        print("MODELS:", "; ".join(f"{res['brand']} {res['car_model'] or ''} {res['colour']}".replace("  ", " ")
                                   for res in results))
    draw(img, results, args.out / f"{path.stem}_pred.jpg")


def watch(folder: Path, handle) -> None:
    """Process every new image that appears in the folder (e.g. Telegram Desktop's download folder)."""
    seen = {p.resolve() for p in folder.iterdir()}
    print(f"watching {folder} - send a photo to this laptop (Ctrl+C to stop)")
    while True:
        for p in sorted(folder.iterdir(), key=lambda q: q.stat().st_mtime):
            if p.resolve() in seen or p.suffix.lower() not in EXTS:
                continue
            time.sleep(0.5)               # give the sender a moment to finish writing the file
            seen.add(p.resolve())
            handle(p)
        time.sleep(1)


def main() -> None:
    args = parse_args()
    if not args.images and not args.watch:
        raise SystemExit("give photo paths or --watch FOLDER")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    det_model = detector.get_model()
    net, brands, car_models = load_classifier(args.weights, device)
    transform = eval_transform()
    print(f"models loaded in {time.perf_counter() - t0:.1f}s on {device} | {len(brands)} brands"
          + (f", {len(car_models)} car models" if car_models else ""))

    def handle(p: Path) -> None:
        process(p, det_model, net, brands, car_models, device, transform, args)

    for p in args.images:
        handle(p)
    if args.watch:
        watch(args.watch, handle)


if __name__ == "__main__":
    main()