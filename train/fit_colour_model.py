"""Learned colour model vs the HSV rules: measured honestly, the better one is kept.

Learned model: frozen ImageNet ResNet-18 (never trained by us) turns each crop into 512 numbers;
one linear layer (512 -> 8 colours) is trained on our labelled crops. Few trainable weights is
deliberate: we have a few hundred labelled cars, not tens of thousands.

Three candidates, all scored with the SAME 5 folds:
  rules   - HSV thresholds, tuned on the 4 training folds (as in fit_colour.py)
  learned - the linear layer, its regularisation strength chosen on the 4 training folds only
  hybrid  - learned, except where the rules say red / green / yellow (rare, hue is decisive)
The winner is retrained on all crops and saved to inference/colour_model.pt (about 22 MB).

Usage (from the codebase folder)
  uv run python -m train.fit_colour_model labels_uzbek.csv
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch import nn

from inference.colour import COLOURS, MODEL_PATH, PARAMS_PATH, RULE_ONLY, build_trunk, crop_features
from train.fit_colour import load, predict_all, report

WEIGHT_DECAYS = [0.01, 0.1, 1.0]     # regularisation strengths to choose from


def fit_linear(X, y, wd, steps=300):
    """Multinomial logistic regression = one linear layer + cross-entropy, trained on all rows at once."""
    torch.manual_seed(0)
    mean, std = X.mean(0), X.std(0) + 1e-6      # standardise every feature to mean 0, std 1
    lin = nn.Linear(X.shape[1], len(COLOURS))
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=wd)
    Xs = (X - mean) / std
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.cross_entropy(lin(Xs), y).backward()
        opt.step()
    return lin, mean, std


@torch.no_grad()
def predict_linear(model, X):
    lin, mean, std = model
    return lin((X - mean) / std).argmax(1)


def choose_wd(X, y, folds=3):
    """Pick the weight decay by cross-validation INSIDE the data we are allowed to look at."""
    split = torch.from_numpy(np.random.default_rng(1).permutation(len(y)) % folds)
    best, best_acc = WEIGHT_DECAYS[0], -1.0
    for wd in WEIGHT_DECAYS:
        acc = np.mean([(predict_linear(fit_linear(X[split != f], y[split != f], wd), X[split == f])
                        == y[split == f]).float().mean().item() for f in range(folds)])
        if acc > best_acc:
            best, best_acc = wd, acc
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", help="labels CSV written by label.py")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    data = load(args.labels)
    y_np = np.array([c for _, c in data])
    y = torch.tensor(y_np)
    print(f"{len(data)} labelled crops:", {c: int((y_np == i).sum()) for i, c in enumerate(COLOURS)})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trunk = build_trunk(pretrained=True).to(device)
    X = crop_features(trunk, [img for img, _ in data], device)
    print("features:", tuple(X.shape))

    combos, rule_preds = predict_all(data)            # every rule threshold combination, every crop
    rule_correct = rule_preds == y_np
    folds = np.random.default_rng(0).permutation(len(y_np)) % args.folds
    oof = {name: np.empty_like(y_np) for name in ("rules", "learned", "hybrid")}
    for f in range(args.folds):
        tr, te = folds != f, folds == f                       # numpy masks (for the rules)
        tr_t, te_t = torch.from_numpy(tr), torch.from_numpy(te)  # the same masks for torch tensors
        rules_f = rule_preds[rule_correct[:, tr].mean(1).argmax(), te]
        learned_f = predict_linear(fit_linear(X[tr_t], y[tr_t], choose_wd(X[tr_t], y[tr_t])), X[te_t]).numpy()
        oof["rules"][te], oof["learned"][te] = rules_f, learned_f
        oof["hybrid"][te] = [r if COLOURS[r] in RULE_ONLY else l for r, l in zip(rules_f, learned_f)]

    for name, pred in oof.items():
        report(y_np, pred, f"{name}, {args.folds}-fold cross-validation")
    accs = {name: float((pred == y_np).mean()) for name, pred in oof.items()}
    winner = max(accs, key=accs.get)
    print("\ncross-validated accuracy:", {k: round(v, 3) for k, v in accs.items()}, "-> using", winner)

    best_rules = combos[int(rule_correct.mean(1).argmax())]
    PARAMS_PATH.write_text(json.dumps(best_rules, indent=2))
    if winner == "rules":
        torch.save({"method": "rules"}, MODEL_PATH)
    else:
        wd = choose_wd(X, y)
        lin, mean, std = fit_linear(X, y, wd)
        torch.save({"method": winner, "weight_decay": wd, "mean": mean, "std": std,
                    "linear": lin.state_dict(),
                    "trunk": {k: v.half() if v.is_floating_point() else v       # float16 halves the file
                              for k, v in trunk.state_dict().items()}}, MODEL_PATH)
    print(f"saved {MODEL_PATH.name} ({winner}) and {PARAMS_PATH.name}")


if __name__ == "__main__":
    main()