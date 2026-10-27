"""Map Car-1000 model folders to the brand labels we train on, and pick a balanced sample.

Outputs (written next to this script's working directory, NOT into the dataset):
  brand_map.csv          one row per Car-1000 class: folder, brand (or empty), why it was excluded
  selected_images.txt    filenames to crop: at most --cap per brand, spread evenly over its models

Usage
  uv run python -m preprocessing.build_brand_map "D:\\Car-1000-dataset" --cap 1000
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

# Car-1000 automaker name (English part of name_from_old) -> the badge we predict.
# Exact match only: sub-brands such as "Geely Galaxy" or "Changan Oushang" carry their own
# badges, so they are NOT folded into the parent brand.
BRANDS = {
    "BYD": "BYD", "Chery": "Chery", "Haval": "Haval", "Geely": "Geely", "Changan": "Changan",
    "Jetour": "Jetour", "Chevrolet": "Chevrolet", "BMW": "BMW", "Mercedes-Benz": "Mercedes",
    "Zeekr": "Zeekr", "Li Auto": "Li Auto", "Exeed": "Exeed", "Hongqi": "Hongqi",
    "Leapmotor": "Leapmotor", "Kia": "Kia", "Hyundai": "Hyundai", "Toyota": "Toyota",
    "Volkswagen": "Volkswagen", "Nissan": "Nissan",
    "JAC": "JAC", "JAC Motors": "JAC", "JAC Refine": "JAC",   # Refine = JAC's MPV line, JAC badge
}
COMMERCIAL = {"Truck", "Van", "Bus"}  # vans, micro-trucks, pickups: rare on the test, odd shapes

# Single classes whose badge differs from their maker's, set by hand.
FOLDER_BRAND = {
    "0271": "Omoda",   # made by Chery, but badged OMODA - the brief: "an Omoda is Omoda"
}
# Single classes we exclude by hand, with the reason.
EXCLUDE_FOLDERS: dict[str, str] = {}


def family(maker):
    """'Geely Galaxy' -> 'Geely', 'Mercedes-Benz' -> 'Mercedes': which brand group a maker belongs to."""
    return maker.split()[0].split("-")[0]


def parse(entry):
    """'奇瑞_瑞虎7==Chery_Tiggo 7' -> ('Chery', 'Tiggo 7', 'SUV'). None if the name has no maker_model form."""
    english = entry["name_from_old"].split("==")[-1]
    if "_" not in english:
        return None
    maker, model = english.split("_", 1)
    body = entry["type_info"].split("==")[-1].split("####")[0]
    return maker, model, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Car-1000-dataset folder")
    ap.add_argument("--cap", type=int, default=1000, help="max images per brand")
    ap.add_argument("--seed", type=int, default=0, help="same seed -> same sample every run")
    args = ap.parse_args()

    info = json.load(open(args.root / "cls_info" / "class_info.json", encoding="utf-8"))
    rows, folders_by_brand = [], defaultdict(list)
    for entry in info:
        folder = f"{int(entry['id']):04d}"
        parsed = parse(entry)
        if parsed is None:
            print("could not parse, skipped:", entry)
            continue
        maker, model, body = parsed
        n = len(list((args.root / "all_images" / folder).glob("*.jpg")))
        if folder in FOLDER_BRAND:
            brand, reason = FOLDER_BRAND[folder], ""
            folders_by_brand[brand].append(folder)
        elif maker not in BRANDS:
            brand, reason = "", "not a target brand"
        elif body in COMMERCIAL:
            brand, reason = "", f"commercial ({body})"
        elif folder in EXCLUDE_FOLDERS:
            brand, reason = "", EXCLUDE_FOLDERS[folder]
        else:
            brand, reason = BRANDS[maker], ""
            folders_by_brand[brand].append(folder)
        rows.append({"folder": folder, "brand": brand, "maker": maker, "model": model,
                     "body": body, "images": n, "excluded_because": reason})

    # utf-8-sig lets Excel open the file correctly. We keep the excluded sub-brands of our target
    # makers in the table too, so every decision is visible and can be reviewed.
    with open("brand_map.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        families = {family(m) for m in BRANDS}   # 'Li Auto' -> 'Li', 'Mercedes-Benz' -> 'Mercedes'
        writer.writerows(r for r in rows if r["brand"] or family(r["maker"]) in families)

    # Round-robin over models, like dealing cards: one image from each model in turn,
    # so a brand's sample covers all its models instead of filling up from the first few.
    rng = random.Random(args.seed)
    selected = []
    print(f"{'brand':10s} {'models':>6s} {'available':>9s} {'selected':>8s}")
    for brand, folders in sorted(folders_by_brand.items()):
        pools = []
        for folder in folders:
            imgs = sorted(p.name for p in (args.root / "all_images" / folder).glob("*.jpg"))
            rng.shuffle(imgs)
            pools.append(imgs)
        available = sum(len(p) for p in pools)
        picked = []
        while len(picked) < args.cap and any(pools):
            for pool in pools:
                if pool and len(picked) < args.cap:
                    picked.append(pool.pop())
        selected += picked
        print(f"{brand:10s} {len(folders):6d} {available:9d} {len(picked):8d}")

    Path("selected_images.txt").write_text("\n".join(selected), encoding="utf-8")
    print(f"\nselected {len(selected)} images -> selected_images.txt | class table -> brand_map.csv")


if __name__ == "__main__":
    main()