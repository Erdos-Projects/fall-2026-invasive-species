"""Smoke test of the BioCLIP 2 zero-shot pipeline on the CPU, with no download and no real photos.

It builds the real 4,573-name vocabulary from data/splits, replaces BioCLIP 2 by a tiny randomly initialized CLIP model
(same open_clip code path, a few thousand times smaller), writes random JPEGs for a subset of val, and runs both modes of
``invasive_clf.build_bioclip2``. It checks the vocabulary, every output file, that the saved classifier loads back, and
that its probabilities are valid. The metrics are meaningless (random model, random photos).

    python tests/smoke_test_bioclip2.py          # about a minute
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import open_clip
import torch

from invasive_clf import build_bioclip2 as bb
from invasive_clf.models.bioclip2 import ZeroShotClassifier

sys.path.insert(0, str(Path(__file__).parent))
from smoke_test_dinov3 import fake_images  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def tiny_clip():
    vision = open_clip.CLIPVisionCfg(layers=1, width=64, head_width=32, patch_size=32, image_size=224)
    text = open_clip.CLIPTextCfg(context_length=77, vocab_size=49408, width=64, heads=2, layers=1)
    return open_clip.CLIP(embed_dim=32, vision_cfg=vision, text_cfg=text), open_clip.get_tokenizer("ViT-B-32")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="smoke_bioclip2_"))
    try:
        splits_dir, image_dir = REPO / "data/splits", tmp / "images"
        fake_images(splits_dir, image_dir, n_per_split={"val": 80})
        cfg = bb.load_config(REPO / "configs/bioclip2_zeroshot.yaml")
        cfg.update(batch_size=32, num_workers=0)

        torch.manual_seed(0); clip, tok = tiny_clip()
        classifier, spec, labels, vocab = bb.build(cfg, splits_dir, tmp / "run", clip=clip, tokenizer=tok)
        assert len(vocab) == 4573 and (vocab.y < 40).sum() == 40 and (vocab.source == "train+val").sum() == 441
        for f in (bb.ARTIFACT, "config.json", "vocabulary.csv"):
            assert (tmp / "run" / f).is_file(), f"build did not write {f}"
        loaded = ZeroShotClassifier.load(tmp / "run" / bb.ARTIFACT)
        assert loaded.threshold == 0.06 and loaded.txt.shape == (4573, 32)
        p = loaded(torch.randn(5, 32))
        assert p.shape == (5, 40) and (p >= 0).all() and (p.sum(1) <= 1 + 1e-5).all()   # the 4,533 other names hold the rest

        summary = bb.validate(cfg, splits_dir, image_dir, tmp / "run", classifier, spec, labels)
        for f in ("metrics_final.json", "val_predictions.csv", "threshold_sweep.png", "pr_roc.png", "per_class_f1.png", "confusion_matrix.png"):
            assert (tmp / "run" / f).is_file(), f"validate did not write {f}"
        assert summary["vocab_size"] == 4573 and 0 <= summary["macro_f1_invasive"] <= 1
        print("\nsmoke test passed: vocabulary rebuilt (4,573 names), both modes ran and wrote every output")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
