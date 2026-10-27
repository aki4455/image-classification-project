"""Car crops on disk + their labels: brand for every crop, car model only for our own crops.

Every item is (crop_path, brand, car_model). car_model is None for Car-1000 crops: we only name
models for the Uzbek Chevrolets we labelled ourselves (their folder name: cobalt, damas, ...).

Car-1000 crops live in  <crops_root>/<model folder>/<original stem>_<i>.jpg
brand_map.csv           says which brand each model folder is
dataset_split.json      says whether the ORIGINAL photo is in Car-1000's train, val or test split
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

# The pretrained network learned on ImageNet photos normalised with these per-channel
# statistics (R, G, B). Our inputs must be normalised the same way.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224
CAR_MODEL_BRAND = "Chevrolet"   # the only brand whose crops are labelled with a car model
CROP_EXTS = {".jpg", ".png"}


def load_brand_map(csv_path) -> dict[str, str]:
    """model folder -> brand, only for the folders we train on."""
    with open(csv_path, encoding="utf-8-sig") as f:
        return {row["folder"]: row["brand"] for row in csv.DictReader(f) if row["brand"]}


def brand_list(csv_path) -> list[str]:
    """Class names in a fixed (alphabetical) order. Position in this list = class index."""
    return sorted(set(load_brand_map(csv_path).values()))


def load_split(split_json) -> dict[str, str]:
    """original filename -> 'train' / 'val' / 'test', from Car-1000's own split file."""
    data = json.load(open(split_json, encoding="utf-8"))
    split_of = {}
    for entry in data:
        for key, split in (("train_data", "train"), ("val_data", "val"), ("test_data", "test")):
            for fname in entry.get(key, []):
                split_of[fname] = split
    return split_of


def car1000_items(crops_root, brand_map_csv, split_json) -> dict[str, list[tuple[Path, str, None]]]:
    """{'train': [(crop_path, brand, None), ...], 'val': [...], 'test': [...]}"""
    brand_of = load_brand_map(brand_map_csv)
    split_of = load_split(split_json)
    items = {"train": [], "val": [], "test": []}
    missing = 0
    for folder, brand in sorted(brand_of.items()):
        for path in sorted((Path(crops_root) / folder).glob("*")):
            if path.suffix.lower() not in CROP_EXTS:
                continue
            original = path.stem.rsplit("_", 1)[0] + ".jpg"   # 0007_0123_0.jpg -> 0007_0123.jpg
            split = split_of.get(original)
            if split is None:
                missing += 1
                continue
            items[split].append((path, brand, None))
    if missing:
        print(f"warning: {missing} crops have no entry in the split file and were left out")
    return items


def labelled_items(labels_csv, classes, val_frac=0.2, seed=0) -> dict[str, list[tuple[Path, str, str]]]:
    """Our own crops labelled with label.py -> {'train': [(path, brand, car_model), ...], 'val': [...]}.
    For Chevrolets the car model is the crop's folder name (cobalt, damas, ...), otherwise None.

    Split inside each model folder (cobalt, malibu, ...), so every model appears in both
    train and val. Skipped images and brands we don't train on are left out.
    """
    with open(labels_csv, newline="", encoding="utf-8") as f:
        rows = {r["path"]: r for r in csv.DictReader(f)}.values()   # latest label per image wins
    groups = defaultdict(list)
    unknown = 0
    for r in rows:
        if r["colour"] == "skip":
            continue
        if r["brand"] not in classes:
            unknown += 1
            continue
        folder = Path(r["path"]).parent.name.lower()
        # Chevrolet crops sit in model folders (cobalt, damas, ...); other brands' crops sit in
        # brand folders (BYD, Leapmotor, ...), which are not car models.
        car_model = folder if r["brand"] == CAR_MODEL_BRAND else None
        groups[folder].append((Path(r["path"]), r["brand"], car_model))
    if unknown:
        print(f"warning: {unknown} labelled crops have a brand we don't train on and were left out")

    rng = random.Random(seed)
    out = {"train": [], "val": []}
    for name in sorted(groups):
        items = sorted(groups[name])
        rng.shuffle(items)
        n_val = round(len(items) * val_frac)
        out["val"] += items[:n_val]
        out["train"] += items[n_val:]
    return out


def car_model_list(*item_lists) -> list[str]:
    """All car model names that appear in the given item lists, alphabetical (= class index)."""
    return sorted({m for items in item_lists for _, _, m in items if m is not None})


def train_transform():
    """Random changes applied every time an image is loaded, so the model never sees
    exactly the same picture twice and learns to ignore things that don't define a brand."""
    return v2.Compose([
        v2.ToImage(),                                     # PIL image -> uint8 tensor, shape (3, H, W)
        v2.Resize((IMG_SIZE, IMG_SIZE)),                  # squash to a square, exactly like at test time
        v2.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.85, 1.18)),  # mild zoom / shift
        v2.RandomHorizontalFlip(),                        # car facing left <-> facing right
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),  # lighting, screens
        v2.RandomApply([v2.JPEG(quality=(30, 90))], p=0.5),  # compression artefacts (needs uint8)
        v2.ToDtype(torch.float32, scale=True),            # 0..255 -> 0.0..1.0
        v2.RandomPerspective(distortion_scale=0.2, p=0.3),   # photographing a screen at an angle
        v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),  # out of focus
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform():
    """No randomness: validation and test images must be judged the same way every time."""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


NO_LABEL = -100   # car-model target for crops whose model we don't know (all Car-1000 crops)


class CarCrops(Dataset):
    """Turns (path, brand, car_model) items into (image tensor, brand index, car-model index)."""

    def __init__(self, items, brands, car_models, transform):
        self.items = items
        self.brand_index = {name: i for i, name in enumerate(brands)}
        self.model_index = {name: i for i, name in enumerate(car_models)}
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, brand, car_model = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), self.brand_index[brand], self.model_index.get(car_model, NO_LABEL)