"""Fine-tune a two-head ConvNeXt-Tiny: brand for every crop, car model for our Uzbek Chevrolets.

Usage (from the codebase folder)
  uv run python -m train.train --car1000 "D:\\crops\\car1000_v2" --brand-map brand_map.csv --split "D:\\Car-1000-dataset\\dataset_split\\dataset_split.json" --uzbek labels_uzbek.csv --out runs\\v2_two_head

Writes to --out:
  best.pt          weights + brand list + car-model list of the epoch with the best validation score
  history.csv      one line per epoch (for the report's training curves)
  confusion_*.png  confusion matrices of the best model
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.dataset import (IMG_SIZE, CarCrops, brand_list, car1000_items, car_model_list, eval_transform,
                          labelled_items, train_transform)
from train.network import TwoHeadConvNeXt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--car1000", type=Path, required=True, help="Car-1000 crops folder")
    ap.add_argument("--brand-map", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True, help="Car-1000 dataset_split.json")
    ap.add_argument("--uzbek", type=Path, default=None, help="labels CSV from label.py")
    ap.add_argument("--out", type=Path, default=Path("runs/v2_two_head"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4, help="peak learning rate of the backbone")
    ap.add_argument("--head-lr-mult", type=float, default=10.0, help="heads learn this many times faster")
    ap.add_argument("--car-model-weight", type=float, default=0.5, help="weight of the car-model loss")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def class_weights(items, brands) -> torch.Tensor:
    """weight_c = N / (K * n_c): a brand with half the average number of images counts double."""
    counts = Counter(brand for _, brand, _ in items)
    n, k = len(items), len(brands)
    return torch.tensor([n / (k * max(counts[c], 1)) for c in brands], dtype=torch.float32)


def train_one_epoch(net, loader, brand_loss_fn, model_loss_fn, model_weight, optimizer, scheduler, device):
    net.train()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y_brand, y_model in loader:
        x = x.to(device, non_blocking=True)
        y_brand, y_model = y_brand.to(device, non_blocking=True), y_model.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            brand_logits, model_logits = net(x)
            loss = brand_loss_fn(brand_logits, y_brand)
            has_model = y_model >= 0      # only our own crops have a car-model label
            if has_model.any():           # many batches contain none of them
                loss = loss + model_weight * model_loss_fn(model_logits[has_model], y_model[has_model])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(y_brand)
        correct += (brand_logits.argmax(1) == y_brand).sum().item()
        seen += len(y_brand)
    return total_loss / seen, correct / seen


@torch.no_grad()
def confusion_matrices(net, loader, device, k_brand: int, k_model: int):
    """conf[t, p] = how many images of true class t were predicted as p (brand, and car model)."""
    net.eval()
    conf_b = torch.zeros(k_brand, k_brand, dtype=torch.long)
    conf_m = torch.zeros(k_model, k_model, dtype=torch.long)
    for x, y_brand, y_model in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            brand_logits, model_logits = net(x.to(device, non_blocking=True))
        pb, pm = brand_logits.argmax(1).cpu(), model_logits.argmax(1).cpu()
        conf_b += torch.bincount(y_brand * k_brand + pb, minlength=k_brand * k_brand).view(k_brand, k_brand)
        has = y_model >= 0
        if has.any():
            conf_m += torch.bincount(y_model[has] * k_model + pm[has], minlength=k_model * k_model).view(k_model, k_model)
    return conf_b, conf_m


def scores(conf: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    """accuracy, macro accuracy (mean of per-class recall over classes present), per-class recall."""
    support = conf.sum(1)
    recall = conf.diag() / support.clamp(min=1)
    acc = conf.diag().sum().item() / max(conf.sum().item(), 1)
    macro = recall[support > 0].mean().item() if (support > 0).any() else float("nan")
    return acc, macro, recall


def plot_confusion(conf, classes, title, path) -> None:
    rec = conf.float() / conf.sum(1, keepdim=True).clamp(min=1)
    size = max(7, 0.42 * len(classes))
    fig, ax = plt.subplots(figsize=(size + 1, size))
    ax.imshow(rec.numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if conf[i, j]:
                ax.text(j, i, int(conf[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if rec[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True   # input size never changes, so let cuDNN pick its fastest kernels
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- data -------------------------------------------------------------------------
    brands = brand_list(args.brand_map)
    c1k = car1000_items(args.car1000, args.brand_map, args.split)
    uz = labelled_items(args.uzbek, brands, seed=args.seed) if args.uzbek else {"train": [], "val": []}
    car_models = car_model_list(uz["train"], uz["val"])
    kb, km = len(brands), max(len(car_models), 1)
    train_items = c1k["train"] + uz["train"]
    eval_sets = {"car1000_val": c1k["val"], "uzbek_val": uz["val"], "car1000_test": c1k["test"]}
    print(f"brands ({kb}):", brands)
    print(f"car models ({len(car_models)}):", car_models)
    print(f"train {len(train_items)} (Car-1000 {len(c1k['train'])} + own {len(uz['train'])}) | "
          + " | ".join(f"{name} {len(items)}" for name, items in eval_sets.items()))

    train_loader = DataLoader(CarCrops(train_items, brands, car_models, train_transform()), batch_size=args.batch,
                              shuffle=True, drop_last=True, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0)
    # Evaluation is short, 2 workers per loader are enough. On Windows every worker is a fresh
    # Python process that loads PyTorch's CUDA libraries again; too many of them exhaust the
    # virtual memory (WinError 1455, "paging file is too small").
    eval_workers = min(2, args.workers)
    eval_loaders = {name: DataLoader(CarCrops(items, brands, car_models, eval_transform()),
                                     batch_size=args.batch * 2, num_workers=eval_workers, pin_memory=True,
                                     persistent_workers=eval_workers > 0)
                    for name, items in eval_sets.items() if items}

    # ---- network, losses, optimiser ------------------------------------------------------
    net = TwoHeadConvNeXt(kb, km).to(device)
    print(f"parameters: {sum(p.numel() for p in net.parameters()) / 1e6:.1f} M | device: {device}")
    weights = class_weights(train_items, brands)
    print("brand weights:", {c: round(w, 2) for c, w in zip(brands, weights.tolist())})
    brand_loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=args.label_smoothing)
    model_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW([
        {"params": list(net.features.parameters()) + list(net.norm.parameters())},              # pretrained: small steps
        {"params": list(net.brand_head.parameters()) + list(net.car_model_head.parameters())},  # new: big steps
    ], lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr, args.lr * args.head_lr_mult],
        total_steps=args.epochs * len(train_loader), pct_start=0.1)

    # ---- training loop -----------------------------------------------------------------
    history, best_macro, t_start = [], -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(net, train_loader, brand_loss_fn, model_loss_fn,
                                                args.car_model_weight, optimizer, scheduler, device)
        conf_b, _ = confusion_matrices(net, eval_loaders["car1000_val"], device, kb, km)
        val_acc, val_macro, _ = scores(conf_b)
        uz_brand = uz_model = float("nan")
        if "uzbek_val" in eval_loaders:
            cb, cm = confusion_matrices(net, eval_loaders["uzbek_val"], device, kb, km)
            uz_brand, uz_model = scores(cb)[0], scores(cm)[0]
        row = {"epoch": epoch, "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
               "val_acc": round(val_acc, 4), "val_macro": round(val_macro, 4),
               "own_val_brand_acc": round(uz_brand, 4), "own_val_car_model_acc": round(uz_model, 4),
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        saved = ""
        if val_macro > best_macro:   # keep the epoch that does best on brand, on data it never trained on
            best_macro = val_macro
            torch.save({"arch": "convnext_tiny_two_head", "brands": brands, "car_models": car_models,
                        "img_size": IMG_SIZE, "state_dict": net.state_dict()}, args.out / "best.pt")
            saved = "  <- saved"
        print(f"epoch {epoch:2d} | loss {train_loss:.3f} | train {train_acc:.3f} | val {val_acc:.3f} "
              f"(macro {val_macro:.3f}) | own val brand {uz_brand:.3f}, car model {uz_model:.3f} | "
              f"{row['seconds']}s{saved}")

    with open(args.out / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"training time: {(time.time() - t_start) / 60:.1f} min")

    # ---- final evaluation of the best epoch ----------------------------------------------
    net.load_state_dict(torch.load(args.out / "best.pt", map_location=device)["state_dict"])
    for name, loader in eval_loaders.items():
        conf_b, conf_m = confusion_matrices(net, loader, device, kb, km)
        acc, macro, recall = scores(conf_b)
        print(f"\n{name}: brand accuracy {acc:.3f} | macro {macro:.3f}")
        print("  per brand:", {c: round(r, 3) for c, r, s in zip(brands, recall.tolist(), conf_b.sum(1).tolist()) if s})
        plot_confusion(conf_b, brands, f"{name}: brand (accuracy {acc:.3f})", args.out / f"confusion_{name}.png")
        if conf_m.sum():
            m_acc, _, m_recall = scores(conf_m)
            print(f"  car model accuracy {m_acc:.3f}:",
                  {c: round(r, 3) for c, r, s in zip(car_models, m_recall.tolist(), conf_m.sum(1).tolist()) if s})
            plot_confusion(conf_m, car_models, f"{name}: car model (accuracy {m_acc:.3f})",
                           args.out / f"confusion_{name}_car_model.png")
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()