"""Figures written by the training and evaluation scripts.

Copied from Section 5b of the team's experiment notebooks (05 to 13) and Section 5 of ``19_final_evaluation.ipynb``,
with the same style so the figures match the notebooks. Every function saves a PNG in ``out_dir`` and closes the figure
(the notebooks showed it inline instead).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # write files; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})
C_BLUE, C_ORANGE, C_GRAY = "#2a78d6", "#eb6834", "#9a9995"


def _save(fig, out_dir: Path, name: str):
    fig.savefig(Path(out_dir) / name); plt.close(fig)


def plot_training_curves(hist: pd.DataFrame, out_dir: Path, title=""):
    """Training loss, validation metrics (when the run had a validation set) or gradient norms (when it did not), and
    learning rates per epoch. A dashed line marks the end of the frozen stage."""
    has_val = "macro_f1_invasive" in hist
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    ax = axes[0]; ax.plot(hist["epoch"], hist["train_loss"], color=C_BLUE, lw=2)
    ax.set_xlabel("epoch"); ax.set_ylabel("BCE loss"); ax.set_title("training loss")
    ax = axes[1]
    if has_val:
        ax.plot(hist["epoch"], hist["macro_f1_invasive"], color=C_BLUE, lw=2, label="macro-F1 (invasive, primary)")
        ax.plot(hist["epoch"], hist["top1_acc_all"], color=C_ORANGE, lw=2, label="top-1 accuracy (41 classes, KPI)")
        ax.plot(hist["epoch"], hist["macro_ap_invasive"], color=C_GRAY, lw=1.5, label="macro-AP (invasive)")
        ax.set_ylim(0, 1.15); ax.set_xlabel("epoch"); ax.set_title("validation metrics")
        ax.legend(fontsize=7, loc="upper left", ncol=3, frameon=False)
    else:
        ax.plot(hist["epoch"], hist["grad_norm_mean"], color=C_BLUE, lw=2, label="mean over the epoch")
        ax.plot(hist["epoch"], hist["grad_norm_max"], color=C_GRAY, lw=1.5, label="max over the epoch")
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_title("gradient norm (before clipping)"); ax.legend(fontsize=7)
    ax = axes[2]; ax.plot(hist["epoch"], hist["lr_head"], color=C_ORANGE, lw=2, label="head LR")
    ax.plot(hist["epoch"], hist["lr_backbone"], color=C_BLUE, lw=2, label="backbone LR")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_title("learning rate (end of epoch)"); ax.legend(fontsize=7)
    for a in axes:
        if hist.attrs.get("frozen_epochs"):
            a.axvline(hist.attrs["frozen_epochs"] + 0.5, color=C_GRAY, ls="--", lw=1)
    fig.suptitle(title); fig.tight_layout(); _save(fig, out_dir, "training_curves.png")


def plot_threshold_sweep(sweep: pd.DataFrame, t: float, out_dir: Path, title=""):
    """Macro-F1 and top-1 accuracy against the threshold, with the chosen threshold marked."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(sweep["threshold"], sweep["macro_f1_invasive"], color=C_BLUE, lw=2, label="macro-F1 (invasive)")
    ax.plot(sweep["threshold"], sweep["top1_acc_all"], color=C_ORANGE, lw=2, label="top-1 accuracy (41)")
    ax.axvline(t, color=C_GRAY, ls="--", lw=1); ax.text(t, 0.02, f" t = {t:.2f}", fontsize=8)
    ax.set_xlabel("invasive threshold on max class probability"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.set_title(title or "threshold sweep (validation)", fontsize=8); fig.tight_layout(); _save(fig, out_dir, "threshold_sweep.png")


def plot_pr_roc(probs, y, m: dict, out_dir: Path, title=""):
    """Binary invasive-vs-not PR curve, the per-class PR curves, and the ROC curve, with the operating point marked."""
    n_invasive = probs.shape[1]
    is_inv = y < n_invasive; score = probs.max(1); t = m["threshold"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    p, r, _ = precision_recall_curve(is_inv, score)
    ax = axes[0]; ax.plot(r, p, color=C_BLUE, lw=2)
    ax.scatter([m["bin_recall_invasive"]], [m["bin_precision_invasive"]], color=C_ORANGE, s=40, zorder=3, label=f"operating point, t={t:.2f}")
    ax.axhline(is_inv.mean(), color=C_GRAY, ls=":", lw=1, label=f"prevalence {is_inv.mean():.2f}")
    ax.set_xlabel("recall (invasive)"); ax.set_ylabel("precision (invasive)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title(f"binary invasive detection: AP = {m['bin_ap_invasive']:.3f}"); ax.legend(fontsize=7, loc="lower left")
    ax = axes[1]
    for c in range(n_invasive):
        yc = (y == c)
        if yc.any():
            pc, rc, _ = precision_recall_curve(yc, probs[:, c]); ax.plot(rc, pc, color=C_BLUE, alpha=0.25, lw=1)
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title(f"per-class PR, {n_invasive} invasive classes: macro-AP = {m['macro_ap_invasive']:.3f}")
    ax = axes[2]
    if 0 < is_inv.sum() < len(y):
        fpr, tpr, _ = roc_curve(is_inv, score); ax.plot(fpr, tpr, color=C_BLUE, lw=2)
        fpr_t = float((~is_inv & (score >= t)).sum() / max((~is_inv).sum(), 1))
        ax.scatter([fpr_t], [m["bin_recall_invasive"]], color=C_ORANGE, s=40, zorder=3)
    ax.plot([0, 1], [0, 1], color=C_GRAY, ls=":", lw=1)
    ax.set_xlabel("false-positive rate (not_invasive flagged)"); ax.set_ylabel("true-positive rate (invasive found)")
    ax.set_title(f"ROC: AUROC = {m['bin_auroc_invasive']:.3f}")
    fig.suptitle(title); fig.tight_layout(); _save(fig, out_dir, "pr_roc.png")


def plot_confusion(m: dict, labels: list, out_dir: Path, title="", normalize=True):
    """41 x 41 confusion matrix (rows = true label), row-normalized by default so the diagonal is the recall."""
    cm = np.array(m["confusion_matrix"], dtype=float)
    if normalize:
        cm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6); ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.grid(False)
    ax.set_title(title or ("confusion matrix, row-normalized (recall on the diagonal)" if normalize else "confusion matrix (counts)"))
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); fig.tight_layout(); _save(fig, out_dir, "confusion_matrix.png")


def plot_per_class_f1(m: dict, invasive_classes: list, support: pd.Series, out_dir: Path, title="", split="validation"):
    """F1 of each invasive class at the chosen threshold, sorted, with the number of photos of that class."""
    n_invasive = len(invasive_classes)
    f1 = pd.Series(m["per_class_f1"][:n_invasive], index=invasive_classes).sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.22 * n_invasive + 1.5))
    ax.barh(f1.index, f1.values, color=C_BLUE, height=0.7)
    for i, (name, v) in enumerate(f1.items()):
        ax.text(min(v + 0.01, 0.9), i, f"n={int(support.get(name, 0))}", va="center", fontsize=6, color="#52514e")
    ax.set_xlim(0, 1); ax.set_xlabel(f"F1 on {split} (at the threshold)"); ax.tick_params(axis="y", labelsize=7)
    ax.set_title(title or f"per-class F1; macro-F1 (invasive) = {m['macro_f1_invasive']:.3f}")
    fig.tight_layout(); _save(fig, out_dir, "per_class_f1.png")


# ------------------------------------------------------------------------------------------------------------------
# Test-set figures comparing the final models (Sections 2b, 11, 11b and 13 of 19_final_evaluation.ipynb). ``models`` is
# a dict  key -> {"label": name shown in the legend, "color": hex}; the other dicts are keyed the same way.

GROUP_ORDER = ["invasive", "native_background", "introduced_unlisted", "listed_not_invasive", "native_listed"]
GROUP_LABELS = {"invasive": "invasive (one of the 40 classes)", "native_background": "not invasive: native background species",
                "introduced_unlisted": "not invasive: introduced, not on the list", "listed_not_invasive": "not invasive: listed but not invasive here",
                "native_listed": "not invasive: native, on a watch list", "holdout_unseen": "not invasive: hold-out species (unseen)"}
GROUP_COLORS = {"invasive": "#eb6834", "native_background": "#2a78d6", "introduced_unlisted": "#9452d6",
                "listed_not_invasive": "#2f9e6e", "native_listed": "#0e9bb8", "holdout_unseen": "#9a9995"}


def plot_split_overview(split_overview: pd.DataFrame, class_counts: pd.DataFrame, n_unseen: int, out_dir: Path):
    """Left: composition of each split by group. Right: photos per invasive class, train + val next to test."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    comp = split_overview.loc[["train", "val", "test"], GROUP_ORDER].T
    comp["test_unseen"] = 0
    comp.loc["holdout_unseen"] = [0, 0, 0, n_unseen]
    shares = comp / comp.sum(axis=0)
    bottom = np.zeros(shares.shape[1])
    for g in shares.index:
        ax.bar(shares.columns, shares.loc[g].values, bottom=bottom, color=GROUP_COLORS[g], label=GROUP_LABELS[g], width=0.65)
        bottom += shares.loc[g].values
    for i, n in enumerate(comp.sum(axis=0).values):
        ax.text(i, 1.01, f"n = {int(n)}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.08); ax.set_ylabel("share of the split"); ax.set_title("composition of each split")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax = axes[1]
    n = len(class_counts); x = np.arange(n); w = 0.4
    ax.bar(x - w / 2, class_counts["train + val"].values, width=w, color=C_BLUE, label="train + val (the final training set)")
    ax.bar(x + w / 2, class_counts["test"].values, width=w, color=C_ORANGE, label="test")
    ax.set_xticks(x); ax.set_xticklabels(class_counts.index, rotation=90, fontsize=6)
    ax.set_ylabel("photos"); ax.set_title(f"photos per invasive class ({n} classes, sorted by training count)"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(Path(out_dir) / "split_overview.png", bbox_inches="tight"); plt.close(fig)


def plot_per_class_f1_compare(f1: pd.DataFrame, support: pd.Series, models: dict, macro: dict, out_dir: Path):
    """Per-class F1 on test of every model side by side; ``f1`` has one column per model, rows sorted by test support."""
    fig, ax = plt.subplots(figsize=(13, 4.2))
    x = np.arange(len(f1)); w = 0.8 / len(models)
    for j, s in enumerate(models):
        ax.bar(x + (j - (len(models) - 1) / 2) * w, f1[s].values, width=w, color=models[s]["color"],
               label=f"{models[s]['label']}: macro-F1 {macro[s]:.3f}")
    ax.set_xticks(x); ax.set_xticklabels([f"{c} (n={support.get(c, 0)})" for c in f1.index], rotation=90, fontsize=6)
    ax.set_ylim(0, 1.02); ax.set_ylabel("F1-score"); ax.set_title("per-class F1-score on the test set, classes sorted by frequency")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout(); _save(fig, out_dir, "per_class_f1_test.png")


def plot_pr_roc_compare(probs: dict, metrics: dict, y: np.ndarray, models: dict, out_dir: Path):
    """Binary invasive detection on test: PR and ROC curves of every model, dots at the validation threshold."""
    n_invasive = next(iter(probs.values())).shape[1]
    is_inv = y < n_invasive
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for s in models:
        score = probs[s].max(1); m = metrics[s]; c = models[s]["color"]
        pr_p, pr_r, _ = precision_recall_curve(is_inv, score); axes[0].plot(pr_r, pr_p, color=c, lw=2, label=f"{models[s]['label']} (AP {m['bin_ap_invasive']:.3f})")
        axes[0].scatter([m["bin_recall_invasive"]], [m["bin_precision_invasive"]], color=c, edgecolors="black", s=45, zorder=4)
        fpr, tpr, _ = roc_curve(is_inv, score); axes[1].plot(fpr, tpr, color=c, lw=2, label=f"{models[s]['label']} (AUROC {m['bin_auroc_invasive']:.3f})")
        fpr_t = float((~is_inv & (score >= m["threshold"])).sum() / max((~is_inv).sum(), 1))
        axes[1].scatter([fpr_t], [m["bin_recall_invasive"]], color=c, edgecolors="black", s=45, zorder=4)
    axes[0].axhline(is_inv.mean(), color=C_GRAY, ls=":", lw=1); axes[0].set_xlabel("recall (invasive)"); axes[0].set_ylabel("precision (invasive)")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.02); axes[0].set_title("binary invasiveness: precision-recall (dots = validation threshold)")
    axes[0].legend(fontsize=7, loc="lower left")
    axes[1].plot([0, 1], [0, 1], color=C_GRAY, ls=":", lw=1); axes[1].set_xlabel("false positive rate"); axes[1].set_ylabel("true-positive rate")
    axes[1].set_title("binary invasiveness: ROC"); axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); _save(fig, out_dir, "pr_roc_test.png")


def plot_errors_by_group(errors: pd.DataFrame, models: dict, out_dir: Path):
    """Test error rate per group: invasive photos missed or misclassified, false positives in each not_invasive group."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    groups = [g for g in GROUP_ORDER if g in errors.index.get_level_values("group")]
    x = np.arange(len(groups)); w = 0.8 / len(models)
    for j, s in enumerate(models):
        vals = [1 - errors.loc[(s, g), "correct"] for g in groups]
        bars = ax.bar(x + (j - (len(models) - 1) / 2) * w, vals, width=w, color=models[s]["color"], label=models[s]["label"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.1%}", ha="center", fontsize=6)
    first = next(iter(models))
    ax.set_xticks(x); ax.set_xticklabels([f"{g}\n(n={int(errors.loc[(first, g), 'n'])})" for g in groups], fontsize=7)
    ax.set_ylabel("error rate"); ax.set_title("test set error rate by group: missed or misclassified (invasive), false positives (the not_invasive groups)")
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=len(models))
    ax.set_ylim(0, max(0.05, (1 - errors["correct"]).max() * 1.2 + 0.02))
    fig.tight_layout(); fig.savefig(Path(out_dir) / "errors_by_group_test.png", bbox_inches="tight"); plt.close(fig)


def plot_unseen(unseen: dict, models: dict, out_dir: Path):
    """Share of test_unseen photos (all not_invasive) each model leaves unflagged."""
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    vals = [unseen[s]["top1_acc"] for s in models]
    bars = ax.bar([models[s]["label"] for s in models], vals, color=[models[s]["color"] for s in models], width=0.6)
    for b, v, s in zip(bars, vals, models):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.1%}\n({unseen[s]['n_flagged']} flagged)", ha="center", fontsize=8)
    first = unseen[next(iter(models))]
    ax.set_ylim(max(0, min(vals) - 0.1), 1.06); ax.set_ylabel("top-1 accuracy"); ax.tick_params(axis="x", labelsize=7)
    ax.set_title(f"unseen species ({first['n']} photos of {first['n_species']} hold-out species, all not_invasive)", fontsize=9)
    fig.tight_layout(); _save(fig, out_dir, "unseen_top1.png")
