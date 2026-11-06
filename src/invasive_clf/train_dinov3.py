"""Train the final fine-tuned model (DINOv3 ViT-S/16) from a config file.

Two modes:

``python -m invasive_clf.train_dinov3 --config configs/dinov3_vits16.yaml --out-dir runs/dinov3_final``
    **Final model** (what ``19_final_evaluation.ipynb`` did, Sections 6 and 7). Trains on **train + val** for the full
    schedule and keeps the weights after the last epoch. There is no held-out set to choose an epoch on, and the 1-cycle
    schedule anneals the learning rate to zero, so the last epoch is the natural end point. The threshold is
    ``threshold`` from the config: the one the validation run tuned. Writes ``final.pt``, ``config.json``,
    ``history.csv`` and ``training_curves.png``.

``python -m invasive_clf.train_dinov3 --config configs/dinov3_vits16.yaml --out-dir runs/dinov3_val --validate``
    **Validation run** (what the experiment notebooks did, Section 6 of notebook 11). Trains on **train**, scores
    **val** after every epoch, keeps the epoch with the best macro-F1 (``best.pt``), then scores val again with the
    mirror-image average and tunes the threshold on it. This is how the final configuration was chosen and how its
    threshold (0.14) and validation macro-F1 (0.751) were obtained. Writes ``best.pt``, ``config.json``,
    ``history.csv``, ``metrics_final.json``, ``val_predictions.csv`` and the validation figures.

``test.csv`` and ``test_unseen.csv`` are never read here; they are scored by the evaluation script.

The training loop has two stages:

1. ``frozen_epochs`` with the backbone frozen: AdamW on the head only, constant learning rate ``frozen_head_lr``.
2. ``full_epochs`` of full fine-tuning: AdamW with peak learning rates ``backbone_lr`` and ``head_lr`` (optionally
   scaled per transformer block by ``layer_decay``), weight decay ``weight_decay`` except on 1-d parameters and tokens,
   and a 1-cycle cosine schedule warming up for ``pct_start`` of the stage.

Mixed precision on a GPU, gradient clipping at ``grad_clip_norm``, and a stop on a non-finite loss. The code is copied
from the notebooks; the only changes are that globals became arguments and figures are saved instead of shown.

Data: the split tables in ``--splits-dir`` (default ``data/splits``) and the photos in ``--image-dir`` (default
``data/inat_dataset_final``), as written by ``python -m invasive_clf.data.download_images``. One training run takes
about 70 minutes on a Colab T4 GPU.
"""
import argparse
import copy
import datetime
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import plots
from .data import images as img_cache
from .data.dataset import InatImages, LabelSpace, make_transforms, read_split
from .metrics import THRESHOLD_GRID, compute_metrics, scalars
from .models.classifier import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16 if (DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"


# ----------------------------------------------------------------------------------------------------------- config

def load_config(path) -> dict:
    """Read the YAML config. ``threshold_grid`` defaults to 0.01, 0.02, ..., 0.99 (the notebooks' grid)."""
    cfg = yaml.safe_load(open(path))
    cfg.setdefault("threshold_grid", THRESHOLD_GRID)
    for k in ("frozen_head_lr", "backbone_lr", "head_lr", "weight_decay", "pct_start", "grad_clip_norm", "dropout"):
        cfg[k] = float(cfg[k])                     # YAML reads e.g. "2e-5" as text; make sure every rate is a number
    return cfg


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------------------------------- one epoch, prediction

def make_loss(cfg, n_invasive: int):
    """Multi-label BCE over the invasive outputs; ``bce_pos_weight`` (if set) scales the positive term."""
    pw = None if cfg["bce_pos_weight"] is None else torch.full((n_invasive,), float(cfg["bce_pos_weight"]), device=DEVICE)
    return nn.BCEWithLogitsLoss(pos_weight=pw)


def apply_mixup_and_smoothing(x, target, cfg):
    """Label smoothing (targets -> y(1 - eps) + eps/2) and MixUp, when switched on in the config."""
    eps = cfg["label_smoothing"]
    if eps > 0:
        target = target * (1 - eps) + 0.5 * eps
    if cfg["mixup_alpha"] > 0:
        lam = float(np.random.beta(cfg["mixup_alpha"], cfg["mixup_alpha"]))
        perm = torch.randperm(x.size(0), device=x.device)
        x = lam * x + (1 - lam) * x[perm]; target = lam * target + (1 - lam) * target[perm]
    return x, target


def train_one_epoch(model, dl, optimizer, scheduler, scaler, loss_fn, cfg, frozen: bool, desc: str = ""):
    """One pass over the training loader. Returns (mean loss, mean gradient norm, max gradient norm)."""
    model.train()
    if frozen:
        model.backbone.eval()                  # frozen stage: backbone statistics untouched (matters for BN backbones)
    use_amp = cfg["amp"] and DEVICE.type == "cuda"
    losses, gnorms = [], []
    n_batches = len(dl) if not cfg["max_batches_per_epoch"] else min(len(dl), cfg["max_batches_per_epoch"])
    bar = tqdm(dl, total=n_batches, desc=desc, leave=False, unit="batch")
    for b, (x, target, _) in enumerate(bar):
        if cfg["max_batches_per_epoch"] and b >= cfg["max_batches_per_epoch"]:
            break
        x, target = x.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
        x, target = apply_mixup_and_smoothing(x, target, cfg)
        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=use_amp):
            logits = model(x)
        loss = loss_fn(logits.float(), target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at batch {b}: {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], cfg["grad_clip_norm"])
        scaler.step(optimizer); scaler.update()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item()); gnorms.append(float(gnorm))
        if b % 10 == 0 or b == n_batches - 1:
            bar.set_postfix(loss=f"{np.mean(losses[-10:]):.3f}", gnorm=f"{gnorms[-1]:.2f}",
                            lr_head=f"{optimizer.param_groups[0]['lr']:.1e}", refresh=False)
    bar.close()
    return float(np.mean(losses)), float(np.mean(gnorms)), float(np.max(gnorms))


@torch.no_grad()
def predict(model, dl, cfg, tta_hflip=False):
    """Sigmoid probabilities ``(N, 40)`` and label indices ``(N,)``. With ``tta_hflip`` the logits of each image and its
    mirror image are averaged."""
    model.eval(); use_amp = cfg["amp"] and DEVICE.type == "cuda"
    probs, ys = [], []
    for x, _, y in dl:
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=use_amp):
            logits = model(x).float()
            if tta_hflip:
                logits = 0.5 * (logits + model(torch.flip(x, dims=[3])).float())
        probs.append(torch.sigmoid(logits).cpu()); ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


# ------------------------------------------------------------------------------- optimizer parameter groups

_LAYER_RE = re.compile(r"(?:^|\.)(?:layers?|blocks|resblocks)\.(\d+)\.")   # HF: layer.N | timm: blocks.N | open_clip: resblocks.N


def backbone_layer_id(name: str, n_layers: int) -> int:
    """0 = embeddings / tokens, 1..n_layers = transformer blocks, n_layers + 1 = everything after (final norm, pooler)."""
    m = _LAYER_RE.search(name)
    if m:
        return int(m.group(1)) + 1
    if any(k in name for k in ("embed", "token", "conv1", "ln_pre")):   # patch/pos/CLS embeddings, open_clip stem
        return 0
    return n_layers + 1


def no_weight_decay(name: str, p: torch.Tensor) -> bool:
    return p.ndim <= 1 or name.endswith(".bias") or any(k in name for k in ("token", "pos_embed", "positional_embedding", "latent"))


def param_groups(model, cfg, frozen_stage=False):
    """AdamW parameter groups by (role, layer id, decay or not). Learning rate: ``head_lr`` for the head (``frozen_head_lr``
    in the frozen stage); ``backbone_lr * layer_decay^(n_layers + 1 - id)`` for the backbone when ``layer_decay`` is set,
    else ``backbone_lr``. Weight decay only on >= 2-d weights when ``weight_decay_exclude_1d``."""
    wd, excl = cfg["weight_decay"], cfg["weight_decay_exclude_1d"]
    groups = {}

    def add(key, p, lr, decay, name):
        g = groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": wd if decay else 0.0, "name": name})
        g["params"].append(p)

    head_lr = cfg["frozen_head_lr"] if frozen_stage else cfg["head_lr"]
    for n, p in model.head.named_parameters():
        decay = not (excl and no_weight_decay(n, p))
        add(("head", decay), p, head_lr, decay, "head")
    if not frozen_stage:
        names = [n for n, _ in model.backbone.named_parameters()]
        ids = [int(m.group(1)) for n in names for m in [_LAYER_RE.search(n)] if m]
        n_layers = (max(ids) + 1) if ids else 0
        for n, p in model.backbone.named_parameters():
            lid = backbone_layer_id(n, n_layers)
            scale = (cfg["layer_decay"] ** (n_layers + 1 - lid)) if cfg["layer_decay"] else 1.0
            decay = not (excl and no_weight_decay(n, p))
            add(("backbone", lid, decay), p, cfg["backbone_lr"] * scale, decay, f"backbone_l{lid}")
    return list(groups.values())


def describe_param_groups(groups) -> pd.DataFrame:
    rows = [{"group": g["name"], "lr": g["lr"], "weight_decay": g["weight_decay"], "n_params": sum(p.numel() for p in g["params"])} for g in groups]
    return pd.DataFrame(rows).groupby(["group", "lr", "weight_decay"], as_index=False)["n_params"].sum().sort_values(["group", "lr"])


# ----------------------------------------------------------------------------------------------- the two stages

def run_schedule(model, train_dl, cfg, n_invasive: int, after_epoch) -> int:
    """The two training stages. ``after_epoch(epoch, stage, train_loss, gnorm_mean, gnorm_max, optimizer, t0)`` is called
    after every epoch (logging, and in ``--validate`` mode scoring val). Returns the number of epochs run."""
    loss_fn = make_loss(cfg, n_invasive)
    use_scaler = cfg["amp"] and DEVICE.type == "cuda" and AMP_DTYPE == torch.float16
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    steps_per_epoch = len(train_dl) if not cfg["max_batches_per_epoch"] else min(len(train_dl), cfg["max_batches_per_epoch"])
    n_epochs_total = cfg["frozen_epochs"] + cfg["full_epochs"]

    # ---- stage 1: frozen backbone ----
    epoch = 0
    if cfg["frozen_epochs"] > 0:
        model.set_backbone_frozen(True)
        opt = torch.optim.AdamW(param_groups(model, cfg, frozen_stage=True))
        for _ in range(cfg["frozen_epochs"]):
            epoch += 1; t0 = time.time()
            tl, gm, gx = train_one_epoch(model, train_dl, opt, None, scaler, loss_fn, cfg, frozen=True,
                                         desc=f"epoch {epoch}/{n_epochs_total} (frozen)")
            after_epoch(epoch, "frozen", tl, gm, gx, opt, t0)

    # ---- stage 2: full fine-tuning, 1-cycle ----
    if cfg["full_epochs"] > 0:
        model.set_backbone_frozen(False)
        groups = param_groups(model, cfg)
        print(describe_param_groups(groups).to_string(index=False))
        opt = torch.optim.AdamW(groups)
        total_steps = cfg["full_epochs"] * steps_per_epoch
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=total_steps,
                                                    pct_start=max(cfg["pct_start"], 2 / total_steps),   # at least 2 warm-up steps
                                                    anneal_strategy="cos")
        for _ in range(cfg["full_epochs"]):
            epoch += 1; t0 = time.time()
            tl, gm, gx = train_one_epoch(model, train_dl, opt, sched, scaler, loss_fn, cfg, frozen=False,
                                         desc=f"epoch {epoch}/{n_epochs_total} (full)")
            after_epoch(epoch, "full", tl, gm, gx, opt, t0)
    return epoch


def _history_row(epoch, stage, train_loss, gnorm_mean, gnorm_max, optimizer, t0) -> dict:
    lr_bb = max([g["lr"] for g in optimizer.param_groups if g["name"].startswith("backbone")], default=0.0)
    lr_head = max(g["lr"] for g in optimizer.param_groups if g["name"] == "head")
    return {"epoch": epoch, "stage": stage, "train_loss": train_loss, "grad_norm_mean": gnorm_mean, "grad_norm_max": gnorm_max,
            "lr_backbone": lr_bb, "lr_head": lr_head, "seconds": time.time() - t0}


def _train_loader(images, df, tf, n_invasive, cfg) -> DataLoader:
    g = torch.Generator(); g.manual_seed(cfg["seed"])
    kw = dict(num_workers=cfg["num_workers"], pin_memory=DEVICE.type == "cuda", persistent_workers=cfg["num_workers"] > 0)
    return DataLoader(InatImages(images, df, tf, n_invasive), batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                      generator=g, **kw)


# ----------------------------------------------------------------------------------------------------- the modes

def train_final(cfg: dict, splits_dir, image_dir, out_dir, spec=None) -> dict:
    """Train on train + val for the full schedule; write final.pt (weights after the last epoch, with the config's
    threshold), config.json, history.csv and training_curves.png. Returns the run summary."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    labels = LabelSpace.from_train(read_split(splits_dir, "train"))
    train_df = read_split(splits_dir, "train", labels, image_dir)
    val_df = read_split(splits_dir, "val", labels, image_dir)
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)                 # the final training set
    cfg = {**cfg, "trained_on": "train+val", "invasive_classes": labels.invasive_classes}
    json.dump(cfg, open(out_dir / "config.json", "w"), indent=1, default=str)

    t0 = time.time()
    images = img_cache.train_images(image_dir, trainval_df["image_path"], cfg["presize"], cfg["num_workers"], "decoding train + val images")
    print(f"image cache: {len(images)} train + val images at presize {cfg['presize']} = "
          f"{images.flat.numel() / 2**30:.2f} GB ({time.time() - t0:.0f}s)")

    seed_everything(cfg["seed"])
    model, spec = build_model(cfg, labels.n_invasive, DEVICE, spec)
    train_tf, _ = make_transforms(cfg, spec.mean, spec.std)
    train_dl = _train_loader(images, trainval_df, train_tf, labels.n_invasive, cfg)
    n_epochs_total = cfg["frozen_epochs"] + cfg["full_epochs"]
    print(f"{n_epochs_total} epochs ({cfg['frozen_epochs']} frozen + {cfg['full_epochs']} full) x {len(train_dl)} batches of "
          f"{cfg['batch_size']} on {len(trainval_df)} train + val images | device {DEVICE_NAME}")

    history = []
    def log(epoch, stage, tl, gm, gx, opt, t0):
        history.append(_history_row(epoch, stage, tl, gm, gx, opt, t0))
        print(f"ep {epoch:>2}/{n_epochs_total} [{stage:<6}] loss {tl:.4f} | gnorm {gm:.2f} | {history[-1]['seconds']:.0f}s")
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    epoch = run_schedule(model, train_dl, cfg, labels.n_invasive, log)
    hist = pd.DataFrame(history); hist.attrs["frozen_epochs"] = cfg["frozen_epochs"]
    summary = {"epochs": epoch, "frozen_epochs": cfg["frozen_epochs"], "full_epochs": cfg["full_epochs"],
               "n_train_images": len(trainval_df), "train_minutes": float(hist["seconds"].sum() / 60),
               "final_train_loss": float(hist["train_loss"].iloc[-1]),
               "trained_at": datetime.datetime.now().isoformat(timespec="seconds"), "device": DEVICE_NAME}
    torch.save({"state_dict": model.state_dict(), "epoch": epoch, "threshold": cfg["threshold"], "config": cfg,
                "invasive_classes": labels.invasive_classes, "summary": summary}, out_dir / "final.pt")
    plots.plot_training_curves(hist, out_dir, f"{cfg['model_name']} on train + val")
    print(f"done: {epoch} epochs in {summary['train_minutes']:.1f} min, final training loss {summary['final_train_loss']:.4f} "
          f"-> {out_dir / 'final.pt'}")
    return summary


def train_validate(cfg: dict, splits_dir, image_dir, out_dir, spec=None) -> dict:
    """Train on train, score val after every epoch, keep the best epoch (best.pt), then score val with mirror-image TTA
    and tune the threshold on it. Writes metrics_final.json, val_predictions.csv, history.csv and the figures."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    labels = LabelSpace.from_train(read_split(splits_dir, "train"))
    train_df = read_split(splits_dir, "train", labels, image_dir)
    val_df = read_split(splits_dir, "val", labels, image_dir)
    assert not set(train_df["observer_id"]) & set(val_df["observer_id"]), "observer leakage between train and val"
    cfg = {**cfg, "trained_on": "train", "invasive_classes": labels.invasive_classes}
    json.dump(cfg, open(out_dir / "config.json", "w"), indent=1, default=str)

    t0 = time.time()
    train_images = img_cache.train_images(image_dir, train_df["image_path"], cfg["presize"], cfg["num_workers"])
    val_images = img_cache.eval_crops(image_dir, val_df["image_path"], cfg["val_resize"], cfg["image_size"], cfg["num_workers"],
                                      "decoding val images")
    print(f"image cache: {len(train_images)} train images + {len(val_images)} val crops ({time.time() - t0:.0f}s)")

    seed_everything(cfg["seed"])
    model, spec = build_model(cfg, labels.n_invasive, DEVICE, spec)
    train_tf, eval_tf = make_transforms(cfg, spec.mean, spec.std)
    train_dl = _train_loader(train_images, train_df, train_tf, labels.n_invasive, cfg)
    kw = dict(num_workers=cfg["num_workers"], pin_memory=DEVICE.type == "cuda", persistent_workers=cfg["num_workers"] > 0)
    val_dl = DataLoader(InatImages(val_images, val_df, eval_tf, labels.n_invasive), batch_size=cfg["batch_size"] * 2,
                        shuffle=False, **kw)
    n_epochs_total = cfg["frozen_epochs"] + cfg["full_epochs"]
    print(f"{n_epochs_total} epochs ({cfg['frozen_epochs']} frozen + {cfg['full_epochs']} full) x {len(train_dl)} batches of "
          f"{cfg['batch_size']} | validation: {len(val_df)} images | device {DEVICE_NAME}")

    history, best = [], {"score": -1.0, "epoch": -1}
    def evaluate_and_log(epoch, stage, tl, gm, gx, opt, t0):
        probs, y = predict(model, val_dl, cfg)
        m = compute_metrics(probs, y, None, cfg["threshold_grid"])
        row = {**_history_row(epoch, stage, tl, gm, gx, opt, t0), **scalars(m)}
        row["seconds"] = time.time() - t0
        history.append(row)
        flag = ""
        if m[cfg["select_metric"]] > best["score"]:
            best.update(score=m[cfg["select_metric"]], epoch=epoch); flag = " *best*"
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "threshold": m["threshold"], "config": cfg,
                        "invasive_classes": labels.invasive_classes}, out_dir / "best.pt")
        print(f"ep {epoch:>2}/{n_epochs_total} [{stage:<6}] loss {tl:.4f} | val macro-F1 {m['macro_f1_invasive']:.4f} "
              f"top1 {m['top1_acc_all']:.4f} mAP {m['macro_ap_invasive']:.4f} t={m['threshold']:.2f} | gnorm {gm:.2f} | "
              f"{row['seconds']:.0f}s{flag}")
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    epoch = run_schedule(model, train_dl, cfg, labels.n_invasive, evaluate_and_log)

    # ---- final evaluation of the best checkpoint (with TTA); the threshold is re-tuned on the TTA probabilities ----
    hist = pd.DataFrame(history); hist.attrs["frozen_epochs"] = cfg["frozen_epochs"]
    ckpt = torch.load(out_dir / "best.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    probs, y = predict(model, val_dl, cfg, tta_hflip=cfg["tta_hflip"])
    final = compute_metrics(probs, y, None, cfg["threshold_grid"])
    summary = {"model": cfg["model_name"], "best_epoch": best["epoch"], "epochs": epoch, "tta_hflip": cfg["tta_hflip"],
               "train_minutes": float(hist["seconds"].sum() / 60), **scalars(final)}
    all_labels = labels.all_labels
    json.dump({**summary, "per_class_f1": dict(zip(all_labels, final["per_class_f1"])),
               "per_class_ap": dict(zip(labels.invasive_classes, final["per_class_ap"])), "labels": all_labels,
               "confusion_matrix": final["confusion_matrix"]}, open(out_dir / "metrics_final.json", "w"), indent=1)
    pd.DataFrame({"observation_id": val_df["observation_id"], "label": val_df["label"], "group": val_df["group"],
                  "scientific_name": val_df["scientific_name"], "pred": [all_labels[i] for i in final["_pred"]],
                  "max_prob": probs.max(1), "argmax_class": [labels.invasive_classes[i] for i in probs.argmax(1)]}
                 ).to_csv(out_dir / "val_predictions.csv", index=False)
    print(f"best epoch {best['epoch']}: " + " | ".join(f"{k} {v:.4f}" for k, v in scalars(final).items() if k != "threshold")
          + f" | threshold {final['threshold']:.2f}")

    ttl = f"{cfg['model_name']} (validation run)"
    plots.plot_training_curves(hist, out_dir, ttl)
    plots.plot_threshold_sweep(final["_sweep"], final["threshold"], out_dir, f"{ttl}: threshold sweep on val (best epoch, TTA)")
    plots.plot_pr_roc(probs, y, final, out_dir, ttl)
    plots.plot_per_class_f1(final, labels.invasive_classes, val_df["label"].value_counts(), out_dir, f"{ttl}: per-class F1 (val)")
    plots.plot_confusion(final, all_labels, out_dir, f"{ttl}: confusion matrix (val, row-normalized)")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/dinov3_vits16.yaml")
    p.add_argument("--out-dir", required=True, help="folder for the weights, history, metrics and figures")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--image-dir", default="data/inat_dataset_final")
    p.add_argument("--validate", action="store_true", help="train on train and select the epoch and threshold on val")
    p.add_argument("--max-batches-per-epoch", type=int, default=None, help="quick check of the pipeline (e.g. 5)")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.max_batches_per_epoch:
        cfg["max_batches_per_epoch"] = args.max_batches_per_epoch
    run = train_validate if args.validate else train_final
    summary = run(cfg, args.splits_dir, args.image_dir, args.out_dir)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
