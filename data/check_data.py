"""Sanity check before training: how many crops per brand and split, and what the
augmented training images actually look like.

Usage (from the codebase folder)
  uv run python -m data.check_data "D:\\crops\\car1000_v2" brand_map.csv "D:\\Car-1000-dataset\\dataset_split\\dataset_split.json" labels_uzbek.csv
"""
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")            # draw into a file, no window needed
import matplotlib.pyplot as plt
import torch

from data.dataset import (IMAGENET_MEAN, IMAGENET_STD, CarCrops, brand_list, car1000_items, car_model_list,
                          labelled_items, train_transform)

crops_root, brand_map_csv, split_json = sys.argv[1:4]
labels_csv = sys.argv[4] if len(sys.argv) > 4 else None
brands = brand_list(brand_map_csv)
items = car1000_items(crops_root, brand_map_csv, split_json)
if labels_csv:
    own = labelled_items(labels_csv, brands)
    items["own_train"], items["own_val"] = own["train"], own["val"]
car_models = car_model_list(items.get("own_train", []), items.get("own_val", []))

print(f"brands ({len(brands)}):", brands)
print(f"car models ({len(car_models)}):", car_models)
for split, lst in items.items():
    counts = Counter(brand for _, brand, _ in lst)
    print(f"{split:9s} {len(lst):5d} | " + " ".join(f"{c}={counts[c]}" for c in brands if counts[c]))
    model_counts = Counter(m for _, _, m in lst if m)
    if model_counts:
        print(f"{'':9s}       | car models: {dict(sorted(model_counts.items()))}")

train = items["train"] + items.get("own_train", [])
ds = CarCrops(train, brands, car_models, train_transform())
picks = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:24].tolist()
mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

fig, axes = plt.subplots(4, 6, figsize=(15, 10))
for ax, i in zip(axes.flat, picks):
    x, b, m = ds[i]
    img = (x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()   # undo Normalize; (3,H,W) -> (H,W,3)
    ax.imshow(img)
    ax.set_title(brands[b] + (f" / {car_models[m]}" if m >= 0 else ""))
    ax.axis("off")
plt.tight_layout()
plt.savefig("train_preview.png", dpi=100)
print("saved train_preview.png")