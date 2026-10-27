"""Web UI: upload photos or point a camera at cars, see brand, Chevrolet model and colour.

Usage (from the codebase folder)
  uv run python -m web.server                                    # opens http://127.0.0.1:8765
  uv run python -m web.server --samples "D:\\test_car_photos"     # adds those photos as one-click samples
  uv run python -m web.server --public                           # also a public HTTPS link (setup: README)

Every photo goes through inference.predict.predict_cars, the same code as the command line, with
the same defaults. The page (web/static) is plain HTML, CSS and JavaScript and loads nothing from
the internet. Live camera mode needs the page opened as localhost or HTTPS (browsers only allow cameras
there): on this computer, or anywhere through the --public link. With --host 0.0.0.0, other devices on
the network can upload photos.

API (also described at /docs)
  POST /api/predict   multipart field "file" = the photo, options as query parameters
  GET  /api/info      classes, defaults, device and training history of the loaded weights
"""
from __future__ import annotations

import argparse
import atexit
import base64
import csv
import ctypes
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from data.dataset import eval_transform
from inference.colour import COLOURS, predict_colour
from inference.predict import (CAR_MODEL_BRAND, EXTS, answer_line, classify, load_classifier, models_line,
                               parse_args as predict_defaults, predict_cars)
from utils.crop_cars import VEHICLE_CLASSES
from utils.detect import detector

STATIC = Path(__file__).with_name("static")
DEFAULTS = predict_defaults([])   # one source of truth for thresholds: the command line's defaults
CROP_THUMB = 256                  # longest side of the car crops sent back to the page
SAMPLE_THUMB = 320                # longest side of sample-photo thumbnails
PUBLIC_SETUP = "One-time setup: README.md, section 'Public link (Tailscale Funnel)'."


def parse_args():
    ap = argparse.ArgumentParser(description="Web UI for the car brand and colour classifier.")
    ap.add_argument("--weights", type=Path, default=DEFAULTS.weights)
    ap.add_argument("--samples", type=Path, default=None, help="folder of photos to offer as one-click samples")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = reachable from other devices on the network")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="don't open the page automatically")
    ap.add_argument("--public", action="store_true",
                    help="also put the page on the internet with Tailscale Funnel while the server runs: "
                         "a fixed HTTPS address that opens anywhere, camera included")
    return ap.parse_args()


def open_funnel(port: int) -> str:
    """Put this server on the internet with Tailscale Funnel and return its address,
    https://<computer name>.<tailnet name>.ts.net, which stays the same. The funnel closes with the server."""
    exe = shutil.which("tailscale") or shutil.which("tailscale", path=r"C:\Program Files\Tailscale")
    if exe is None:
        raise SystemExit(f"--public needs Tailscale, which is not installed. {PUBLIC_SETUP}")
    funnel = subprocess.Popen([exe, "funnel", f"http://127.0.0.1:{port}"], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    atexit.register(funnel.terminate)
    for line in funnel.stdout:   # until the address appears; passes on tailscale's messages, e.g. a link to approve
        if re.fullmatch(r"https://[\w.-]+\.ts\.net/?", line.strip()):
            threading.Thread(target=funnel.stdout.read, daemon=True).start()   # keep reading so it never blocks
            return line.strip().rstrip("/")
        if line.strip() and not line.startswith("Available on the internet"):
            print("tailscale:", line.rstrip(), flush=True)
    raise SystemExit(f"Tailscale Funnel did not start (exit code {funnel.wait()}). {PUBLIC_SETUP}")


def keep_awake() -> None:
    """Ask Windows not to go to sleep while the server runs, so the public link stays up (ends with the server)."""
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def read_image(data: bytes) -> np.ndarray:
    """Upload bytes -> OpenCV BGR image. imdecode is the decoder behind cv2.imread, so pixels match
    the command line exactly, including the rotation stored in a phone photo's EXIF data."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    if img is None:
        raise HTTPException(400, "Cannot read this image (iPhone HEIC? Send it as a normal photo / JPEG).")
    return img


def jpeg(img_bgr: np.ndarray, max_side: int) -> bytes:
    h, w = img_bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img_bgr = cv2.resize(img_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def top(probs: list[float], names: list[str], k: int = 3) -> list[dict]:
    order = sorted(range(len(probs)), key=lambda i: -probs[i])[:k]
    return [{"name": names[i], "p": round(probs[i], 4)} for i in order]


def load_history(run_dir: Path) -> list[dict]:
    """history.csv written by train/train.py next to best.pt, as numbers; [] if there is none."""
    path = run_dir / "history.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def create_app(args) -> FastAPI:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    det_model = detector.get_model()
    net, brands, car_models = load_classifier(args.weights, device)
    transform = eval_transform()
    model_names = [m.capitalize() for m in car_models]

    # Warm up every model once (CUDA kernels, lazily loaded colour model), so the first photo isn't slow.
    blank = np.full((480, 640, 3), 128, np.uint8)
    det_model(blank, classes=VEHICLE_CLASSES, verbose=False)
    classify(net, [blank], device, transform, flip=True)
    predict_colour(blank)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"models loaded in {time.perf_counter() - t0:.1f}s on {device_name} | {len(brands)} brands, "
          f"{len(car_models)} car models")

    gpu = threading.Lock()   # requests arrive on several threads; the models run one photo at a time
    run_dir = args.weights.parent
    figures = sorted(p.name for p in run_dir.glob("*.png"))
    samples = sorted(p.name for p in args.samples.iterdir() if p.suffix.lower() in EXTS) if args.samples else []

    app = FastAPI(title="Car brand and colour classifier",
                  description="YOLO11 finds the cars, a ConvNeXt-Tiny names brand and Chevrolet model, "
                              "a ResNet-18 colour model names the colour.")

    @app.get("/api/info")
    def info():
        return {
            "device": device_name,
            "weights": args.weights.as_posix(),
            "brands": brands,
            "car_models": model_names,
            "car_model_brand": CAR_MODEL_BRAND,
            "colours": COLOURS,
            "defaults": {k: getattr(DEFAULTS, k) for k in ("min_conf", "min_rel", "max_cars", "min_model_p", "flip")},
            "history": load_history(run_dir),
            "figures": figures,
            "samples": samples,
        }

    @app.post("/api/predict")
    def predict(file: UploadFile = File(..., description="photo (JPEG, PNG, WebP, BMP)"),
                min_conf: float = Query(DEFAULTS.min_conf, ge=0, le=1, description="detector confidence needed"),
                min_rel: float = Query(DEFAULTS.min_rel, ge=0, le=1,
                                       description="ignore cars smaller than this share of the biggest car's box"),
                max_cars: int = Query(DEFAULTS.max_cars, ge=1, le=50),
                min_model_p: float = Query(DEFAULTS.min_model_p, ge=0, le=1,
                                           description="name the Chevrolet model only if this sure"),
                flip: bool = Query(DEFAULTS.flip, description="average with the mirror image"),
                crops: bool = Query(True, description="include a JPEG of every car crop")):
        img = read_image(file.file.read())
        opts = SimpleNamespace(min_conf=min_conf, min_rel=min_rel, max_cars=max_cars,
                               min_model_p=min_model_p, flip=flip)
        with gpu:
            t = time.perf_counter()
            results, seconds = predict_cars(img, det_model, net, brands, car_models, device, transform, opts)
            seconds["total"] = time.perf_counter() - t
        cars = []
        for res in results:
            car = {
                "box": [round(v, 1) for v in res["box"]],
                "det_conf": round(res["det_conf"], 4),
                "brand": res["brand"],
                "p": round(res["p"], 4),
                "top_brands": top(res["brand_probs"], brands),
                "brand_probs": [round(p, 4) for p in res["brand_probs"]],
                "car_model": res["car_model"],
                "p_model": round(res["p_model"], 4),
                "top_models": top(res["model_probs"], model_names) if res["model_probs"] else [],
                "model_probs": [round(p, 4) for p in res["model_probs"]] if res["model_probs"] else None,
                "colour": res["colour"],
            }
            if crops:
                car["crop"] = "data:image/jpeg;base64," + base64.b64encode(jpeg(res["crop"], CROP_THUMB)).decode()
            cars.append(car)
        return {
            "width": img.shape[1],
            "height": img.shape[0],
            "cars": cars,
            "answer": answer_line(results),
            "models": models_line(results),
            "seconds": {k: round(v, 4) for k, v in seconds.items()},
            "options": vars(opts),
        }

    @app.get("/api/figures/{name}")
    def figure(name: str):
        if name not in figures:   # only the plots train.py wrote, never an arbitrary path
            raise HTTPException(404)
        return FileResponse(run_dir / name)

    @lru_cache(maxsize=256)
    def sample_thumb(name: str) -> bytes:
        img = cv2.imread(str(args.samples / name))
        return jpeg(img, SAMPLE_THUMB) if img is not None else b""

    @app.get("/api/samples/{name}")
    def sample(name: str, thumb: bool = False):
        if name not in samples:
            raise HTTPException(404)
        if thumb:
            return Response(sample_thumb(name), media_type="image/jpeg")
        return FileResponse(args.samples / name)

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def main() -> None:
    args = parse_args()
    if args.samples and not args.samples.is_dir():
        raise SystemExit(f"--samples: {args.samples} is not a folder")
    public = open_funnel(args.port) if args.public else None   # before the models load: setup problems show at once
    app = create_app(args)
    local = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{local}:{args.port}"
    print(f"web UI: {url}  (Ctrl+C to stop)")
    if local != args.host:
        print(f"other devices on this network: http://<this computer's IP, see ipconfig>:{args.port} "
              "(photo upload only: browsers allow the camera only on localhost or HTTPS)")
    if public:
        keep_awake()
        print(f"public link: {public}  (the same address every time; works while this window is open, "
              "and this computer won't go to sleep meanwhile)")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
