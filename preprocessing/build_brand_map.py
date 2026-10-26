import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


BRANDS = {
    "BYD": "BYD", "Chery": "Chery", "Haval": "Haval", "Geely": "Geely", "Changan": "Changan",
    "Jetour": "Jetour", "Chevrolet": "Chevrolet", "BMW": "BMW", "Mercedes-Benz": "Mercedes",
    "Zeekr":"Zeekr"
}
COMMERCIAL = {"Truck", "Van", "Bus"}  

EXCLUDE_FOLDERS = {
    "0271": "Chery Omoda: sold under its own OMODA badge in Uzbekistan",
}


def family(maker):
    return maker.split()[0].split("-")[0]


def parse(entry):
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
        if maker not in BRANDS:
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

    with open("brand_map.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(r for r in rows if family(r["maker"]) in BRANDS.values())

   
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