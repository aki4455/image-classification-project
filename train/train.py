
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
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from data.dataset import (IMG_SIZE, CarCrops, brand_list, car1000_items, eval_transform,
                          labelled_items, train_transform)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--car1000", type=Path, required=True, help="Car-1000 crops folder")
    ap.add_argument("--brand-map", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True, help="Car-1000 dataset_split.json")
    ap.add_argument("--uzbek", type=Path, default=None, help="labels CSV from label.py")
    ap.add_argument("--out", type=Path, default=Path("runs/brand_convnext_tiny"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4, help="peak learning rate of the backbone")
    ap.add_argument("--head-lr-mult", type=float, default=10.0, help="head learns this many times faster")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def build_model(num_classes: int) -> nn.Module:
    """ImageNet-pretrained ConvNeXt-Tiny with its 1000-class output layer replaced by ours."""
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model


def class_weights(items, classes) -> torch.Tensor:
    """weight_c = N / (K * n_c): a class with half the average number of images counts double."""
    counts = Counter(brand for _, brand in items)
    n, k = len(items), len(classes)
    return torch.tensor([n / (k * max(counts[c], 1)) for c in classes], dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)                 # (batch, K) raw scores
            loss = criterion(logits, y)       # one number: how wrong this batch was
        optimizer.zero_grad(set_to_none=True)
        loss.backward()                       # gradients: how each weight should move to lower the loss
        optimizer.step()                      # move the weights a small step
        scheduler.step()                      # adjust the step size for the next batch
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        seen += len(y)
    return total_loss / seen, correct / seen


@torch.no_grad()
def confusion_matrix(model, loader, device, k: int) -> torch.Tensor:
    """conf[t, p] = how many images of true class t were predicted as class p."""
    model.eval()
    conf = torch.zeros(k, k, dtype=torch.long)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(x).argmax(1).cpu()
        conf += torch.bincount(y * k + pred, minlength=k * k).view(k, k)
    return conf


def scores(conf: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    """accuracy, macro accuracy (mean of per-class recall over classes present), per-class recall."""
    support = conf.sum(1)
    recall = conf.diag() / support.clamp(min=1)
    acc = conf.diag().sum().item() / max(conf.sum().item(), 1)
    macro = recall[support > 0].mean().item()
    return acc, macro, recall


def plot_confusion(conf, classes, title, path) -> None:
    rec = conf.float() / conf.sum(1, keepdim=True).clamp(min=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(rec.numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if conf[i, j]:
                ax.text(j, i, int(conf[i, j]), ha="center", va="center", fontsize=8,
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
    classes = brand_list(args.brand_map)
    k = len(classes)
    c1k = car1000_items(args.car1000, args.brand_map, args.split)
    uz = labelled_items(args.uzbek, classes, seed=args.seed) if args.uzbek else {"train": [], "val": []}
    train_items = c1k["train"] + uz["train"]
    eval_sets = {"car1000_val": c1k["val"], "uzbek_val": uz["val"], "car1000_test": c1k["test"]}
    print("classes:", classes)
    print(f"train {len(train_items)} (Car-1000 {len(c1k['train'])} + own {len(uz['train'])}) | "
          + " | ".join(f"{name} {len(items)}" for name, items in eval_sets.items()))

    train_loader = DataLoader(CarCrops(train_items, classes, train_transform()), batch_size=args.batch,
                              shuffle=True, drop_last=True, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0)
    # Evaluation is short, 2 workers per loader are enough. On Windows every worker is a fresh
    # Python process that loads PyTorch's CUDA libraries again; too many of them exhaust the
    # virtual memory (WinError 1455, "paging file is too small").
    eval_workers = min(2, args.workers)
    eval_loaders = {name: DataLoader(CarCrops(items, classes, eval_transform()), batch_size=args.batch * 2,
                                     num_workers=eval_workers, pin_memory=True,
                                     persistent_workers=eval_workers > 0)
                    for name, items in eval_sets.items() if items}

    # ---- model, loss, optimiser --------------------------------------------------------
    model = build_model(k).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M | device: {device}")
    weights = class_weights(train_items, classes)
    print("class weights:", {c: round(w, 2) for c, w in zip(classes, weights.tolist())})
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters()},       # pretrained backbone: small steps
        {"params": model.classifier.parameters()},     # new head: bigger steps
    ], lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr, args.lr * args.head_lr_mult],
        total_steps=args.epochs * len(train_loader), pct_start=0.1)

    # ---- training loop -----------------------------------------------------------------
    history, best_macro, t_start = [], -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_acc, val_macro, _ = scores(confusion_matrix(model, eval_loaders["car1000_val"], device, k))
        uz_acc = (scores(confusion_matrix(model, eval_loaders["uzbek_val"], device, k))[0]
                  if "uzbek_val" in eval_loaders else float("nan"))
        row = {"epoch": epoch, "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
               "val_acc": round(val_acc, 4), "val_macro": round(val_macro, 4),
               "uzbek_val_acc": round(uz_acc, 4), "seconds": round(time.time() - t0, 1)}
        history.append(row)
        saved = ""
        if val_macro > best_macro:   # keep the epoch that does best on data it never trained on
            best_macro = val_macro
            torch.save({"arch": "convnext_tiny", "classes": classes, "img_size": IMG_SIZE,
                        "state_dict": model.state_dict()}, args.out / "best.pt")
            saved = "  <- saved"
        print(f"epoch {epoch:2d} | loss {train_loss:.3f} | train {train_acc:.3f} | "
              f"val {val_acc:.3f} (macro {val_macro:.3f}) | own val {uz_acc:.3f} | {row['seconds']}s{saved}")

    with open(args.out / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"training time: {(time.time() - t_start) / 60:.1f} min")

    # ---- final evaluation of the best epoch ----------------------------------------------
    model.load_state_dict(torch.load(args.out / "best.pt", map_location=device)["state_dict"])
    for name, loader in eval_loaders.items():
        conf = confusion_matrix(model, loader, device, k)
        acc, macro, recall = scores(conf)
        print(f"\n{name}: accuracy {acc:.3f} | macro {macro:.3f}")
        print("  per brand:", {c: round(r, 3) for c, r, s in zip(classes, recall.tolist(), conf.sum(1).tolist()) if s})
        plot_confusion(conf, classes, f"{name} (accuracy {acc:.3f})", args.out / f"confusion_{name}.png")
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()