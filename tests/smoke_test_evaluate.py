"""Smoke test of the test-set evaluation on the CPU, with no download and no real photos.

It makes the two final-model files the evaluation reads, with the tiny stand-in models of the other smoke tests
(a 2-block DINOv3 trained for 1 + 1 epochs on fake photos, and a tiny CLIP for the BioCLIP 2 vocabulary), writes random
JPEGs for a subset of test and test_unseen, and runs ``invasive_clf.evaluate`` end to end, including the
validation-vs-test table. It checks that every output is written. The metrics are meaningless.

    python tests/smoke_test_evaluate.py          # about two minutes
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

from invasive_clf import build_bioclip2 as bb
from invasive_clf import evaluate as ev
from invasive_clf import train_dinov3 as td
from invasive_clf.models.backbones import build_dinov3_vits16

sys.path.insert(0, str(Path(__file__).parent))
from smoke_test_bioclip2 import tiny_clip  # noqa: E402
from smoke_test_dinov3 import fake_images, tiny_encoder  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main():
    tmp = Path(tempfile.mkdtemp(prefix="smoke_evaluate_"))
    try:
        splits_dir, image_dir = REPO / "data/splits", tmp / "images"
        fake_images(splits_dir, image_dir, n_per_split={"train": 60, "val": 30, "test": 100, "test_unseen": 40})

        # the two model files, as train_dinov3 and build_bioclip2 write them
        cfg = td.load_config(REPO / "configs/dinov3_vits16.yaml")
        cfg.update(frozen_epochs=1, full_epochs=1, batch_size=16, num_workers=0, max_batches_per_epoch=3)   # >= 3 steps for the 1-cycle schedule
        torch.manual_seed(0); enc = tiny_encoder()
        td.train_final(cfg, splits_dir, image_dir, tmp / "dinov3", spec=build_dinov3_vits16(cfg, encoder=enc))
        bcfg = bb.load_config(REPO / "configs/bioclip2_zeroshot.yaml")
        torch.manual_seed(0); clip, tok = tiny_clip()
        bb.build(bcfg, splits_dir, tmp / "bioclip2", clip=clip, tokenizer=tok)

        # validation metrics in the format of final_models.json (written by notebook 18)
        fake_val = {k: 0.5 for k in ("macro_f1_invasive", "top1_acc_all", "macro_ap_invasive", "bin_recall_invasive",
                                     "bin_precision_invasive", "noninv_recall")}
        json.dump({"models": {"bioclip2": {"metrics": fake_val}, "finetune": {"metrics": fake_val}}}, open(tmp / "final_models.json", "w"))

        scorers = {"bioclip2": ev.load_zeroshot(tmp / "bioclip2" / bb.ARTIFACT, clip=clip),
                   "finetune": ev.load_finetuned(tmp / "dinov3/final.pt", spec=build_dinov3_vits16(cfg, encoder=tiny_encoder()))}
        final = ev.evaluate(scorers, splits_dir, image_dir, tmp / "final_test", tmp / "final_models.json", batch_size=32, num_workers=0)

        out = tmp / "final_test"
        for f in ("test_metrics.csv", "test_vs_val.csv", "per_class_f1_test.csv", "per_class_f1_test.png", "pr_roc_test.png",
                  "errors_by_group_test.csv", "errors_by_group_test.png", "timing.csv", "accuracy_vs_runtime.csv", "unseen_results.csv",
                  "unseen_top1.png", "split_overview.csv", "split_overview.png", "class_counts.csv", "final_results.json"):
            assert (out / f).is_file(), f"evaluate did not write {f}"
        for s in ("bioclip2", "finetune"):
            for f in ("test_predictions.csv", "test_probs.npz", "metrics_test.json", "unseen_predictions.csv", "pr_roc.png",
                      "per_class_f1.png", "confusion_matrix.png"):
                assert (out / s / f).is_file(), f"evaluate did not write {s}/{f}"
            assert final["models"][s]["unseen"]["n"] == 40 and final["models"][s]["timing"]["n_images"] == 100
        assert final["models"]["finetune"]["threshold"] == 0.14 and final["models"]["bioclip2"]["threshold"] == 0.06
        print("\nsmoke test passed: both models scored on test and test_unseen, every output written")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
