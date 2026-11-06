"""Smoke test of the DINOv3 training pipeline on the CPU, with no download and no real photos.

It uses the real split tables in data/splits, writes random JPEGs for a small subset of train and val (every other row
counts as "image missing" and is dropped, which tests that path too), replaces DINOv3 ViT-S/16 by a tiny randomly
initialized DINOv3 of the same architecture, and runs both modes of ``invasive_clf.train_dinov3`` for 1 + 1 epochs of a
few batches. It checks that every output file is written and that the saved weights load back into a fresh model.
The metrics are meaningless (random model, random photos); this only tests that the pieces fit together.

    python tests/smoke_test_dinov3.py          # about a minute
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from invasive_clf import train_dinov3 as td
from invasive_clf.models.backbones import build_dinov3_vits16
from invasive_clf.models.classifier import build_model

REPO = Path(__file__).resolve().parents[1]


def tiny_encoder():
    # Same architecture family as facebook/dinov3-vits16 (patch 16, 4 register tokens), but 2 small blocks.
    return DINOv3ViTModel(DINOv3ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                                          patch_size=16, num_register_tokens=4, image_size=224))


def fake_images(splits_dir: Path, image_dir: Path, n_per_split=dict(train=120, val=60), seed=0):
    rng = np.random.default_rng(seed)
    for split, n in n_per_split.items():
        df = pd.read_csv(splits_dir / f"{split}.csv")
        for p in df["image_path"].sample(n, random_state=seed):
            f = image_dir / p; f.parent.mkdir(parents=True, exist_ok=True)
            h, w = rng.integers(240, 400, size=2)            # different sizes and aspect ratios, like the real photos
            Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(f, quality=80)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="smoke_dinov3_"))
    try:
        splits_dir, image_dir = REPO / "data/splits", tmp / "images"
        fake_images(splits_dir, image_dir)
        cfg = td.load_config(REPO / "configs/dinov3_vits16.yaml")
        cfg.update(frozen_epochs=1, full_epochs=1, batch_size=16, num_workers=0, max_batches_per_epoch=3)

        torch.manual_seed(0); enc = tiny_encoder()
        spec = build_dinov3_vits16(cfg, encoder=enc)
        summary = td.train_final(cfg, splits_dir, image_dir, tmp / "final", spec=spec)
        for f in ("final.pt", "config.json", "history.csv", "training_curves.png"):
            assert (tmp / "final" / f).is_file(), f"final mode did not write {f}"
        ckpt = torch.load(tmp / "final/final.pt", weights_only=False)
        assert ckpt["threshold"] == 0.14 and len(ckpt["invasive_classes"]) == 40 and summary["epochs"] == 2
        model, _ = build_model(cfg, 40, "cpu", build_dinov3_vits16(cfg, encoder=tiny_encoder()))
        model.load_state_dict(ckpt["state_dict"])            # the saved weights fit a freshly built model

        torch.manual_seed(0); spec = build_dinov3_vits16(cfg, encoder=tiny_encoder())
        summary = td.train_validate(cfg, splits_dir, image_dir, tmp / "validate", spec=spec)
        for f in ("best.pt", "config.json", "history.csv", "metrics_final.json", "val_predictions.csv", "training_curves.png",
                  "threshold_sweep.png", "pr_roc.png", "per_class_f1.png", "confusion_matrix.png"):
            assert (tmp / "validate" / f).is_file(), f"validate mode did not write {f}"
        assert len(pd.read_csv(tmp / "validate/history.csv")) == 2 and 0 <= summary["macro_f1_invasive"] <= 1
        print("\nsmoke test passed: both modes ran and wrote every output")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
