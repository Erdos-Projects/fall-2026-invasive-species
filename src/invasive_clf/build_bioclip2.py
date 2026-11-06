"""Build the BioCLIP 2 zero-shot classifier from a config file (nothing is trained).

Two modes, like ``train_dinov3``:

``python -m invasive_clf.build_bioclip2 --config configs/bioclip2_zeroshot.yaml --out-dir runs/bioclip2_final``
    **Final model.** Builds the 4,573-name vocabulary from the split tables, encodes the prompts with BioCLIP 2's text
    tower once, and saves everything the scorer needs to ``bioclip2_zeroshot.pt``: the text embeddings, the label of
    each name, BioCLIP 2's logit scale, the threshold from the config (0.06, tuned on val) and the config itself. This
    is the counterpart of notebook 16's ``best.joblib``; the image tower is loaded from Hugging Face when scoring.
    A few minutes on a GPU (about 4,600 short prompts).

``python -m invasive_clf.build_bioclip2 --config configs/bioclip2_zeroshot.yaml --out-dir runs/bioclip2_val --validate``
    **Validation run.** Does the same, then scores ``val.csv`` (each photo and its mirror image) and tunes the threshold
    on it, reproducing notebook 16's numbers for this candidate (macro-F1 0.967, threshold 0.06). Writes
    ``metrics_final.json``, ``val_predictions.csv`` and the validation figures. About 20 minutes on a T4 GPU.

``test.csv`` and ``test_unseen.csv`` are never read here; they are scored by the evaluation script.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from . import plots
from .data import images as img_cache
from .data.dataset import LabelSpace, read_split
from .metrics import THRESHOLD_GRID, compute_metrics, scalars
from .models.bioclip2 import ZeroShotClassifier, build_vocabulary, encode_vocabulary, image_spec, load_bioclip2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16 if (DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
ARTIFACT = "bioclip2_zeroshot.pt"


def load_config(path) -> dict:
    cfg = yaml.safe_load(open(path))
    cfg.setdefault("threshold_grid", THRESHOLD_GRID)
    cfg["threshold"] = float(cfg["threshold"])
    return cfg


@torch.no_grad()
def zero_shot_probs(spec, classifier: ZeroShotClassifier, crops: torch.Tensor, cfg, desc="scoring") -> np.ndarray:
    """``(N, 40)`` probabilities for uint8 crops ``(N, 3, 224, 224)``: normalize, image tower, zero-shot step; with
    ``tta_hflip`` the probabilities of each photo and its mirror image are averaged (``Scorer`` of notebook 19)."""
    module = spec.module.to(DEVICE).eval(); classifier.to(DEVICE)
    mean = torch.tensor(spec.mean, device=DEVICE).view(1, 3, 1, 1); std = torch.tensor(spec.std, device=DEVICE).view(1, 3, 1, 1)
    use_amp = cfg["amp"] and DEVICE.type == "cuda"

    def probs_one(x):
        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=use_amp):
            feats = module(x)
        return classifier(feats)

    out = []
    for i in tqdm(range(0, len(crops), cfg["batch_size"]), desc=desc, unit="batch", leave=False):
        x = crops[i:i + cfg["batch_size"]].to(DEVICE, non_blocking=True).float().div_(255).sub_(mean).div_(std)
        p = probs_one(x)
        if cfg["tta_hflip"]:
            p = 0.5 * (p + probs_one(torch.flip(x, dims=[3])))
        out.append(p.float().cpu())
    return torch.cat(out).numpy()


def build(cfg: dict, splits_dir, out_dir, clip=None, tokenizer=None) -> tuple:
    """Vocabulary + text embeddings -> ``bioclip2_zeroshot.pt``. Returns (classifier, image spec, labels, vocab)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    labels = LabelSpace.from_train(read_split(splits_dir, "train"))
    vocab = build_vocabulary(splits_dir, labels)
    print(f"vocabulary: {len(vocab)} names ({(vocab.source == 'train+val').sum()} classes of train + val, of which "
          f"{(vocab.y < labels.n_invasive).sum()} invasive; {(vocab.source == 'regional').sum()} regional extras)")

    clip, tokenizer, logit_scale = load_bioclip2(cfg["model_id"], clip, tokenizer)
    t0 = time.time()
    txt = encode_vocabulary(clip, tokenizer, vocab, DEVICE, cfg["prompt"])
    print(f"text embeddings {tuple(txt.shape)} in {time.time() - t0:.0f}s | logit scale {logit_scale:.1f}")

    art = {"text_embeddings": txt.numpy().astype(np.float16),       # stored as float16, like notebook 16's best.joblib
           "vocab_to_label": vocab["y"].to_numpy().astype(np.int16), "vocab_names": vocab["name"].tolist(),
           "vocab_taxon_ids": vocab["taxon_id"].to_numpy(), "logit_scale": logit_scale, "threshold": cfg["threshold"],
           "invasive_classes": labels.invasive_classes, "labels": labels.all_labels, "config": cfg}
    torch.save(art, out_dir / ARTIFACT)
    json.dump(cfg, open(out_dir / "config.json", "w"), indent=1, default=str)
    vocab.assign(members=vocab["members"].map("; ".join)).to_csv(out_dir / "vocabulary.csv", index=False)   # for inspection
    print(f"saved {out_dir / ARTIFACT} (threshold {cfg['threshold']:.2f})")
    classifier = ZeroShotClassifier(art["text_embeddings"], art["vocab_to_label"], logit_scale, labels.n_invasive, cfg["threshold"])
    spec = image_spec(clip)
    return classifier, spec, labels, vocab


def validate(cfg: dict, splits_dir, image_dir, out_dir, classifier, spec, labels) -> dict:
    """Score val, tune the threshold, write metrics_final.json, val_predictions.csv and the figures."""
    out_dir = Path(out_dir)
    val_df = read_split(splits_dir, "val", labels, image_dir)
    t0 = time.time()
    crops = img_cache.eval_crops(image_dir, val_df["image_path"], cfg["val_resize"], cfg["image_size"], cfg["num_workers"], "decoding val images")
    probs = zero_shot_probs(spec, classifier, crops, cfg, desc="scoring val")
    y = val_df["y"].to_numpy()
    m = compute_metrics(probs, y, None, cfg["threshold_grid"])
    summary = {"model": cfg["model_name"], "method": "zero_shot", "vocab_size": len(classifier.txt), "tta_hflip": cfg["tta_hflip"],
               "minutes": (time.time() - t0) / 60, **scalars(m)}
    json.dump({**summary, "per_class_f1": dict(zip(labels.all_labels, m["per_class_f1"])),
               "per_class_ap": dict(zip(labels.invasive_classes, m["per_class_ap"])), "labels": labels.all_labels,
               "confusion_matrix": m["confusion_matrix"]}, open(out_dir / "metrics_final.json", "w"), indent=1, default=float)
    pd.DataFrame({"observation_id": val_df["observation_id"], "label": val_df["label"], "group": val_df["group"],
                  "scientific_name": val_df["scientific_name"], "pred": [labels.all_labels[i] for i in m["_pred"]],
                  "max_prob": probs.max(1), "argmax_class": [labels.invasive_classes[i] for i in probs.argmax(1)]}
                 ).to_csv(out_dir / "val_predictions.csv", index=False)
    print(" | ".join(f"{k} {v:.4f}" for k, v in scalars(m).items() if k != "threshold") + f" | threshold tuned on val {m['threshold']:.2f}"
          + ("" if abs(m["threshold"] - cfg["threshold"]) < 1e-9 else f" (config: {cfg['threshold']:.2f})"))
    ttl = "BioCLIP 2 zero-shot (validation)"
    plots.plot_threshold_sweep(m["_sweep"], m["threshold"], out_dir, f"{ttl}: threshold sweep on val (TTA)")
    plots.plot_pr_roc(probs, y, m, out_dir, ttl)
    plots.plot_per_class_f1(m, labels.invasive_classes, val_df["label"].value_counts(), out_dir, f"{ttl}: per-class F1 (val)")
    plots.plot_confusion(m, labels.all_labels, out_dir, f"{ttl}: confusion matrix (val, row-normalized)")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/bioclip2_zeroshot.yaml")
    p.add_argument("--out-dir", required=True, help="folder for bioclip2_zeroshot.pt, the vocabulary and (--validate) the metrics")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--image-dir", default="data/inat_dataset_final")
    p.add_argument("--validate", action="store_true", help="also score val and tune the threshold on it")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    classifier, spec, labels, _ = build(cfg, args.splits_dir, args.out_dir)
    if args.validate:
        print(json.dumps(validate(cfg, args.splits_dir, args.image_dir, args.out_dir, classifier, spec, labels), indent=1))


if __name__ == "__main__":
    main()
