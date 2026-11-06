"""Score the test sets once with the final models, time them, and write every result table and figure.

The script version of Sections 2b and 9 to 14 of ``19_final_evaluation.ipynb``::

    python -m invasive_clf.evaluate --dinov3 runs/dinov3_final/final.pt \\
        --bioclip2 runs/bioclip2_final/bioclip2_zeroshot.pt --out-dir runs/final_test

Either model can be left out. The inputs are the files written by ``python -m invasive_clf.train_dinov3`` (the DINOv3
ViT-S/16 trained on train + val) and ``python -m invasive_clf.build_bioclip2`` (the BioCLIP 2 zero-shot vocabulary);
BioCLIP 2's image tower is loaded from Hugging Face. About 25 minutes on a T4 GPU, almost all of it BioCLIP 2.

What it does
------------
1. Decodes ``test.csv`` and ``test_unseen.csv`` once (``Resize(256)``, ``CenterCrop(224)``, as for validation).
2. Scores ``test.csv`` with each model through the same code path (:class:`Scorer`): normalization, the forward pass of
   the photo and of its mirror image, and the probability step (sigmoid for DINOv3, the zero-shot softmax for BioCLIP 2),
   in batches of ``--batch-size``. The scoring is timed (:func:`timed_scoring`): one untimed warm-up batch, the GPU
   synchronized before and after, image decoding excluded.
3. Computes every metric of ``invasive_clf.metrics`` at the threshold each model tuned on **validation** (0.14 and 0.06).
   Nothing is tuned on the test sets.
4. Scores ``test_unseen.csv``: photos of 30 hold-out species no model saw, all ``not_invasive``. Its top-1 accuracy is
   the share each model leaves unflagged.

Outputs in ``--out-dir``
------------------------
``test_metrics.csv``, ``test_vs_val.csv`` (if ``--val-metrics`` exists), ``per_class_f1_test.csv/.png``,
``pr_roc_test.png``, ``errors_by_group_test.csv/.png``, ``timing.csv``, ``accuracy_vs_runtime.csv``,
``unseen_results.csv``, ``unseen_top1.png``, ``split_overview.csv/.png``, ``class_counts.csv``, ``final_results.json``,
and per model (``finetune/``, ``bioclip2/``): ``test_predictions.csv``, ``test_probs.npz``, ``metrics_test.json``,
``unseen_predictions.csv``, ``pr_roc.png``, ``per_class_f1.png``, ``confusion_matrix.png``.
"""
import argparse
import datetime
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from . import plots
from .data import images as img_cache
from .data.dataset import NONE_LABEL, LabelSpace, read_split
from .metrics import KEY_METRICS, compute_metrics, predict_not_invasive, scalars
from .models.bioclip2 import ZeroShotClassifier, image_spec
from .models.classifier import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16 if (DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"

# The two series of the final comparison, in the order of notebooks 18 and 19, with their names and colors in figures.
SERIES = {"bioclip2": {"label": "BioCLIP 2 ViT-L/14 (zero-shot)", "color": "#2f9e6e", "candidate": "bioclip2_zeroshot_full/00_baseline"},
          "finetune": {"label": "DINOv3 ViT-S/16 (fine-tuned)", "color": "#2a78d6", "candidate": "dinov3_vits16/04_pool_cls_patchmean"}}


# ------------------------------------------------------------------------------------------------- the models

class Scorer:
    """Maps a batch of cached uint8 crops to ``(B, 40)`` probabilities on the device: normalization with the model's
    mean/std, the forward pass, the mirror-image pass when ``tta_hflip`` is set, and the probability step: the sigmoid
    of a fine-tuned classifier's logits (``head=None``) or a :class:`ZeroShotClassifier` on the image features."""
    def __init__(self, series: str, module: nn.Module, spec, cfg: dict, threshold: float, invasive_classes: list, head=None):
        self.series, self.module, self.spec, self.cfg, self.head = series, module, spec, cfg, head
        self.tta = bool(cfg.get("tta_hflip", True)); self.amp = bool(cfg.get("amp", True)) and DEVICE.type == "cuda"
        self.threshold, self.invasive_classes = float(threshold), list(invasive_classes)
        self.n_params = sum(p.numel() for p in module.parameters())
        self.mean = torch.tensor(spec.mean, device=DEVICE).view(1, 3, 1, 1); self.std = torch.tensor(spec.std, device=DEVICE).view(1, 3, 1, 1)
        self.label, self.color = SERIES[series]["label"], SERIES[series]["color"]

    def to_device(self):
        self.module.to(DEVICE).eval()
        if self.head is not None:
            self.head.to(DEVICE)
        return self

    def to_cpu(self):
        self.module.to("cpu"); torch.cuda.empty_cache(); return self

    def _probs_one(self, x):
        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=self.amp):
            out = self.module(x)
        return torch.sigmoid(out.float()) if self.head is None else self.head(out)

    @torch.no_grad()
    def __call__(self, x_uint8: torch.Tensor) -> torch.Tensor:
        x = x_uint8.to(DEVICE, non_blocking=True).float().div_(255).sub_(self.mean).div_(self.std)
        p = self._probs_one(x)
        if self.tta:
            p = 0.5 * (p + self._probs_one(torch.flip(x, dims=[3])))
        return p


def load_finetuned(path, spec=None) -> Scorer:
    """The DINOv3 model from ``final.pt``: the architecture is rebuilt from the config saved with the weights.
    ``spec`` replaces the backbone download (used by the tests)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model, spec = build_model(cfg, len(ckpt["invasive_classes"]), "cpu", spec)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    print(f"finetune: {path} ({cfg.get('trained_on', '?')}, {ckpt['epoch']} epochs) | threshold {ckpt['threshold']:.2f}")
    return Scorer("finetune", model, spec, cfg, ckpt["threshold"], ckpt["invasive_classes"])


def load_zeroshot(path, clip=None) -> Scorer:
    """The BioCLIP 2 zero-shot model: the vocabulary file plus the image tower. ``clip`` replaces the download (tests)."""
    art = torch.load(path, map_location="cpu", weights_only=False)
    cfg = art["config"]
    if clip is None:
        import open_clip
        clip = open_clip.create_model(cfg["model_id"])                # BioCLIP 2 from Hugging Face; only the image tower is used
    spec = image_spec(clip.eval())
    head = ZeroShotClassifier.load(path)
    print(f"bioclip2: {path} (zero-shot over {len(head.txt)} names, logit scale {head.logit_scale:.0f}) | threshold {head.threshold:.2f}")
    return Scorer("bioclip2", spec.module, spec, cfg, head.threshold, art["invasive_classes"], head=head)


# ------------------------------------------------------------------------------------------------ timed scoring

def iter_batches(images: torch.Tensor, batch_size: int):
    for i in range(0, len(images), batch_size):
        yield images[i:i + batch_size]


def timed_scoring(scorer: Scorer, images: torch.Tensor, batch_size: int, repeats=1, desc="") -> tuple:
    """(probs ``(N, 40)`` float32, timing dict). The first pass produces the probabilities; every pass is timed."""
    scorer.to_device()
    n_images = len(images)
    scorer(images[:min(batch_size, n_images)])                                     # warm-up (kernels, allocator), not timed
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    seconds, probs = [], None
    for r in range(repeats):
        out = []
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for x in tqdm(iter_batches(images, batch_size), total=math.ceil(n_images / batch_size), desc=f"{desc} pass {r + 1}/{repeats}",
                      leave=False, unit="batch"):
            out.append(scorer(x))
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        seconds.append(time.perf_counter() - t0)
        if probs is None:
            probs = torch.cat(out).float().cpu().numpy()
    timing = {"n_images": n_images, "batch_size": batch_size, "tta_hflip": scorer.tta, "amp": scorer.amp, "device": DEVICE_NAME, "repeats": repeats,
              "seconds": float(min(seconds)), "seconds_mean": float(np.mean(seconds)), "seconds_all": [float(x) for x in seconds],
              "images_per_second": float(n_images / min(seconds)), "ms_per_image": float(1000 * min(seconds) / n_images),
              "peak_gpu_memory_GiB": float(torch.cuda.max_memory_allocated() / 2**30) if DEVICE.type == "cuda" else None,
              "params_M": float(scorer.n_params / 1e6)}
    scorer.to_cpu()
    return probs, timing


# --------------------------------------------------------------------------------------------------- the report

def split_tables(dfs: dict, labels: LabelSpace) -> tuple:
    """(split_overview, class_counts): photos per split and group, and photos per invasive class (Section 2b)."""
    n = labels.n_invasive
    rows = []
    for name, df in dfs.items():
        inv = df["y"] < n
        rows.append({"split": name, "photos": len(df), "invasive": int(inv.sum()), "not_invasive": int((~inv).sum()),
                     "invasive_share": float(inv.mean()), "invasive_classes": int(df.loc[inv, "label"].nunique()),
                     "species": int(df["scientific_name"].nunique()), "observers": int(df["observer_id"].nunique()),
                     **{g: int((df["group"] == g).sum()) for g in plots.GROUP_ORDER}})
    ov = pd.DataFrame(rows).set_index("split")
    trainval = pd.concat([dfs["train"], dfs["val"]], ignore_index=True)
    ov.loc["train + val"] = ov.loc[["train", "val"]].sum(numeric_only=True)
    ov.loc["train + val", "invasive_share"] = ov.loc["train + val", "invasive"] / ov.loc["train + val", "photos"]
    ov.loc["train + val", "invasive_classes"] = trainval.loc[trainval["y"] < n, "label"].nunique()
    ov.loc["train + val", "species"] = trainval["scientific_name"].nunique()
    ov.loc["train + val", "observers"] = trainval["observer_id"].nunique()
    ov = ov.loc[["train", "val", "train + val", "test", "test_unseen"]].astype({c: int for c in ov.columns if c != "invasive_share"})
    cc = pd.DataFrame({name: dfs[name].loc[dfs[name]["y"] < n, "label"].value_counts() for name in ["train", "val", "test"]}
                      ).reindex(labels.invasive_classes).fillna(0).astype(int)
    cc["train + val"] = cc["train"] + cc["val"]
    return ov, cc.sort_values("train + val", ascending=False)


def errors_by_group(scorers: dict, preds: dict, test_df: pd.DataFrame, n_invasive: int) -> pd.DataFrame:
    """Per model and group: share correct; for invasive photos also missed vs wrong class, else the false-positive rate."""
    y = test_df["y"].to_numpy(); rows = []
    for s in scorers:
        pred = preds[s]
        for g in plots.GROUP_ORDER:
            mask = (test_df["group"] == g).to_numpy()
            if not mask.any():
                continue
            if g == "invasive":
                yg, pg = y[mask], pred[mask]
                rows.append({"series": s, "group": g, "n": int(mask.sum()), "correct": float((pg == yg).mean()),
                             "left unflagged (missed)": float((pg == n_invasive).mean()),
                             "wrong invasive class": float(((pg != yg) & (pg != n_invasive)).mean())})
            else:
                rows.append({"series": s, "group": g, "n": int(mask.sum()), "correct": float((pred[mask] == n_invasive).mean()),
                             "flagged as invasive (false positive)": float((pred[mask] != n_invasive).mean())})
    return pd.DataFrame(rows).set_index(["series", "group"])


def evaluate(scorers: dict, splits_dir, image_dir, out_dir, val_metrics=None, batch_size=128, repeats=1, num_workers=2) -> dict:
    """Run the whole test evaluation for ``scorers`` (series -> :class:`Scorer`) and write every output."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    labels = LabelSpace.from_train(read_split(splits_dir, "train"))
    for s, sc in scorers.items():
        assert sc.invasive_classes == labels.invasive_classes, f"{s}: the model's class order differs from train.csv's"
    n = labels.n_invasive; all_labels = labels.all_labels

    # ---- the split (from the tables; Section 2b) ----
    dfs = {name: read_split(splits_dir, name, labels) for name in ("train", "val", "test", "test_unseen")}
    split_overview, class_counts = split_tables(dfs, labels)
    split_overview.to_csv(out_dir / "split_overview.csv"); class_counts.to_csv(out_dir / "class_counts.csv")
    plots.plot_split_overview(split_overview, class_counts, len(dfs["test_unseen"]), out_dir)
    print(split_overview.to_string())

    # ---- the photos, decoded once per evaluation size ----
    test_df = read_split(splits_dir, "test", labels, image_dir)
    unseen_df = read_split(splits_dir, "test_unseen", labels, image_dir)
    crops = {}
    def eval_images(cfg):
        key = (cfg["val_resize"], cfg["image_size"])
        if key not in crops:
            t0 = time.time()
            crops[key] = (img_cache.eval_crops(image_dir, test_df["image_path"], *key, num_workers, "decoding test images"),
                          img_cache.eval_crops(image_dir, unseen_df["image_path"], *key, num_workers, "decoding test_unseen images"))
            print(f"image cache: {len(test_df)} test + {len(unseen_df)} unseen crops of {key[1]}x{key[1]} ({time.time() - t0:.0f}s)")
        return crops[key]

    # ---- the test set, scored once and timed (Section 10) ----
    probs, metrics, timing = {}, {}, {}
    y_test = test_df["y"].to_numpy()
    for s, sc in scorers.items():
        d = out_dir / s; d.mkdir(exist_ok=True)
        p, t = timed_scoring(sc, eval_images(sc.cfg)[0], batch_size, repeats, desc=s)
        m = compute_metrics(p, y_test, sc.threshold)
        probs[s], metrics[s], timing[s] = p, m, t
        np.savez_compressed(d / "test_probs.npz", probs=p.astype(np.float32), observation_id=test_df["observation_id"].to_numpy(),
                            labels=np.array(all_labels), threshold=sc.threshold)
        pd.DataFrame({"observation_id": test_df["observation_id"], "label": test_df["label"], "group": test_df["group"],
                      "scientific_name": test_df["scientific_name"], "pred": [all_labels[i] for i in m["_pred"]],
                      "max_prob": p.max(1), "argmax_class": [labels.invasive_classes[i] for i in p.argmax(1)]}
                     ).to_csv(d / "test_predictions.csv", index=False)
        json.dump({"series": s, "candidate": SERIES[s]["candidate"], "set": "test", "n": int(len(y_test)), "tta_hflip": sc.tta,
                   **scalars(m), "per_class_f1": dict(zip(all_labels, m["per_class_f1"])),
                   "per_class_ap": dict(zip(labels.invasive_classes, m["per_class_ap"])), "labels": all_labels,
                   "confusion_matrix": m["confusion_matrix"], "timing": t}, open(d / "metrics_test.json", "w"), indent=1, default=float)
        print(f"{s:<9} test macro-F1 {m['macro_f1_invasive']:.4f} | top-1 {m['top1_acc_all']:.4f} | mAP {m['macro_ap_invasive']:.4f} | "
              f"t={sc.threshold:.2f} | {t['seconds']:.1f}s for {t['n_images']} images = {t['ms_per_image']:.2f} ms/image")

    # ---- results on the test set (Section 11) ----
    models = {s: {"label": sc.label, "color": sc.color} for s, sc in scorers.items()}
    test_table = pd.DataFrame({s: scalars(metrics[s]) for s in scorers}).T
    test_table.insert(0, "model", [sc.label for sc in scorers.values()])
    test_table.to_csv(out_dir / "test_metrics.csv")
    print("\ntest set metrics at the validation threshold\n" + test_table[["model"] + KEY_METRICS].round(4).to_string())

    val = {}
    if val_metrics and Path(val_metrics).is_file():
        sel = json.load(open(val_metrics))["models"]
        val = {s: sel[s]["metrics"] for s in scorers if s in sel}
        order = ["macro_f1_invasive", "top1_acc_all", "macro_ap_invasive", "bin_recall_invasive", "bin_precision_invasive", "noninv_recall"]
        rows = [{"series": s, "metric": k, "val": val[s].get(k), "test": metrics[s][k],
                 "test - val": metrics[s][k] - val[s][k] if val[s].get(k) is not None else np.nan} for s in val for k in order]
        tv = (pd.DataFrame(rows).pivot(index="series", columns="metric", values=["val", "test", "test - val"]).loc[list(val)]
              .swaplevel(axis=1).reindex(columns=pd.MultiIndex.from_product([order, ["val", "test", "test - val"]])))
        tv.set_axis([f"{m} | {w}" for m, w in tv.columns], axis=1).to_csv(out_dir / "test_vs_val.csv")
        print("\nvalidation (the numbers the models were chosen on) vs test\n" + tv.round(4).to_string())
    else:
        print(f"\n{val_metrics} not found: test_vs_val.csv skipped")

    support = test_df["label"].value_counts()
    f1 = pd.DataFrame({s: metrics[s]["per_class_f1"][:n] for s in scorers}, index=labels.invasive_classes)
    f1 = f1.loc[sorted(labels.invasive_classes, key=lambda c: -support.get(c, 0))]
    f1.to_csv(out_dir / "per_class_f1_test.csv")
    plots.plot_per_class_f1_compare(f1, support, models, {s: metrics[s]["macro_f1_invasive"] for s in scorers}, out_dir)
    plots.plot_pr_roc_compare(probs, metrics, y_test, models, out_dir)
    for s, sc in scorers.items():
        ttl = f"{sc.label} on the test set"
        plots.plot_pr_roc(probs[s], y_test, metrics[s], out_dir / s, ttl)
        plots.plot_per_class_f1(metrics[s], labels.invasive_classes, support, out_dir / s, f"{ttl}: per-class F1", split="test")
        plots.plot_confusion(metrics[s], all_labels, out_dir / s, f"{ttl}: confusion matrix")

    # ---- where the errors are (Section 11b) ----
    errors = errors_by_group(scorers, {s: metrics[s]["_pred"] for s in scorers}, test_df, n)
    errors.to_csv(out_dir / "errors_by_group_test.csv")
    plots.plot_errors_by_group(errors, models, out_dir)
    print("\n" + errors.round(4).to_string())

    # ---- accuracy against inference runtime (Section 12) ----
    timing_table = pd.DataFrame({s: {"label": sc.label, "params_M": timing[s]["params_M"], "top1_acc_all": metrics[s]["top1_acc_all"],
                                     "macro_f1_invasive": metrics[s]["macro_f1_invasive"], "ms_per_image": timing[s]["ms_per_image"],
                                     "images_per_second": timing[s]["images_per_second"], "seconds_test_set": timing[s]["seconds"],
                                     "n_images": timing[s]["n_images"], "batch_size": timing[s]["batch_size"], "tta_hflip": timing[s]["tta_hflip"],
                                     "amp": timing[s]["amp"], "peak_gpu_memory_GiB": timing[s]["peak_gpu_memory_GiB"], "device": timing[s]["device"]}
                                 for s, sc in scorers.items()}).T
    timing_table.to_csv(out_dir / "timing.csv")
    acc_rt = pd.DataFrame({sc.label: {"parameters (M)": timing[s]["params_M"], "top-1 accuracy": metrics[s]["top1_acc_all"],
                                      "macro-F1 (invasive)": metrics[s]["macro_f1_invasive"], "ms per image": timing[s]["ms_per_image"],
                                      "images per second": timing[s]["images_per_second"], "seconds elapsed": timing[s]["seconds"],
                                      "peak GPU memory (GiB)": timing[s]["peak_gpu_memory_GiB"]} for s, sc in scorers.items()}).T.astype(float)
    acc_rt.to_csv(out_dir / "accuracy_vs_runtime.csv")
    print(f"\ntest set: {len(test_df)} images, batches of {batch_size}, mirror TTA, {DEVICE_NAME}\n" + acc_rt.round(4).to_string())

    # ---- unseen species (Section 13) ----
    unseen, rows = {}, []
    for s, sc in scorers.items():
        p, _ = timed_scoring(sc, eval_images(sc.cfg)[1], batch_size, 1, desc=f"{s} unseen")
        pred = predict_not_invasive(p, sc.threshold)
        out = pd.DataFrame({"observation_id": unseen_df["observation_id"], "scientific_name": unseen_df["scientific_name"],
                            "pred": [all_labels[i] for i in pred], "max_prob": p.max(1),
                            "argmax_class": [labels.invasive_classes[i] for i in p.argmax(1)]})
        out.to_csv(out_dir / s / "unseen_predictions.csv", index=False)
        flagged = out[out["pred"] != NONE_LABEL]
        unseen[s] = {"n": int(len(out)), "top1_acc": float((pred == n).mean()), "n_flagged": int(len(flagged)),
                     "n_species": int(unseen_df["scientific_name"].nunique()), "species_with_a_flag": int(flagged["scientific_name"].nunique()),
                     "flags_by_predicted_class": flagged["pred"].value_counts().to_dict()}
        rows.append({"series": s, "model": sc.label, "photos": len(out), "hold-out species": unseen[s]["n_species"],
                     "top-1 accuracy": unseen[s]["top1_acc"], "flagged as invasive": unseen[s]["n_flagged"],
                     "species with at least one flag": unseen[s]["species_with_a_flag"]})
        top = sorted(unseen[s]["flags_by_predicted_class"].items(), key=lambda kv: -kv[1])[:3]
        print(f"{s}: {unseen[s]['n_flagged']} of {unseen[s]['n']} unseen-species photos flagged"
              + (f"; most often as {', '.join(f'{c} ({k})' for c, k in top)}" if top else ""))
    pd.DataFrame(rows).set_index("series").to_csv(out_dir / "unseen_results.csv")
    plots.plot_unseen(unseen, models, out_dir)

    # ---- record everything (Section 14) ----
    results = {s: {"label": sc.label, "candidate": SERIES[s]["candidate"], "threshold": sc.threshold,
                   "threshold_source": "tuned on validation by the chosen run", "val_metrics": val.get(s),
                   "test_metrics": scalars(metrics[s]), "test_per_class_f1": dict(zip(all_labels, metrics[s]["per_class_f1"])),
                   "timing": timing[s], "unseen": unseen[s]} for s, sc in scorers.items()}
    final = {"evaluated_at": datetime.datetime.now().isoformat(timespec="seconds"), "device": DEVICE_NAME, "eval_batch_size": batch_size,
             "timing_repeats": repeats,
             "splits": {k: {c: (float(v) if c == "invasive_share" else int(v)) for c, v in r.items()} for k, r in split_overview.to_dict("index").items()},
             "models": results,
             "notes": ["The fine-tuned model was trained on train + val with the hyperparameters of the chosen iteration for the full "
                       "schedule; the weights after the last epoch are used. BioCLIP 2 is used frozen (zero-shot); nothing is trained.",
                       "Thresholds are the ones tuned on validation by the chosen runs; nothing was tuned on test or test_unseen.",
                       "Timing: one untimed warm-up batch, then the whole test set in batches of eval_batch_size with mirror TTA, "
                       "GPU synchronized; image decoding excluded.",
                       "unseen.top1_acc is the share of test_unseen photos (all not_invasive) left unflagged at the threshold."]}
    json.dump(final, open(out_dir / "final_results.json", "w"), indent=1, default=float)
    print(f"\nwritten to {out_dir.resolve()}")
    return final


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dinov3", help="final.pt written by invasive_clf.train_dinov3 (train + val model)")
    p.add_argument("--bioclip2", help="bioclip2_zeroshot.pt written by invasive_clf.build_bioclip2")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--image-dir", default="data/inat_dataset_final")
    p.add_argument("--val-metrics", default="results/final_selection/final_models.json",
                   help="validation metrics of the chosen runs (written by notebook 18), for test_vs_val.csv")
    p.add_argument("--batch-size", type=int, default=128, help="the same for every model, so the timings are comparable")
    p.add_argument("--timing-repeats", type=int, default=1, help="timed passes over the test set per model; > 1 reports the minimum")
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args(argv)
    assert args.dinov3 or args.bioclip2, "give --dinov3 and/or --bioclip2"
    scorers = {}
    if args.bioclip2:
        scorers["bioclip2"] = load_zeroshot(args.bioclip2)
    if args.dinov3:
        scorers["finetune"] = load_finetuned(args.dinov3)
    evaluate(scorers, args.splits_dir, args.image_dir, args.out_dir, args.val_metrics, args.batch_size, args.timing_repeats, args.num_workers)


if __name__ == "__main__":
    main()
