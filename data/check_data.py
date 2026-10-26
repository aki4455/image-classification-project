
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import torch

from dataset import CarCrops, IMAGENET_MEAN, IMAGENET_STD, brand_list, car1000_items, train_transform

crops_root, brand_map_csv, split_json = sys.argv[1:4]
classes = brand_list(brand_map_csv)
items = car1000_items(crops_root, brand_map_csv, split_json)

print("classes:", classes)
for split, lst in items.items():
    counts = Counter(brand for _, brand in lst)
    print(f"{split:5s} {len(lst):5d} | " + " ".join(f"{c}={counts[c]}" for c in classes))

ds = CarCrops(items["train"], classes, train_transform())
picks = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:24].tolist()
mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

fig, axes = plt.subplots(4, 6, figsize=(15, 10))
for ax, i in zip(axes.flat, picks):
    x, y = ds[i]
    img = (x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()   # undo Normalize; (3,H,W) -> (H,W,3)
    ax.imshow(img)
    ax.set_title(classes[y])
    ax.axis("off")
plt.tight_layout()
plt.savefig("train_preview.png", dpi=100)
print("saved train_preview.png")