"""Evaluation metrics shared by every model in the project.

Copied from Section 5 of the team's experiment notebooks (05 to 13), so a number computed here means exactly what it
meant on validation. The only change from the notebooks is that the number of invasive classes is read from the shape of
``probs`` instead of a global variable.

Task encoding
-------------
The models output one probability per invasive species (40 classes), not one per label: ``probs`` has shape
``(N, n_invasive)``. The 41st label, ``not_invasive``, has no output of its own. A photo is predicted as the most
probable invasive species when that probability reaches a threshold ``t``, and as ``not_invasive`` otherwise. True labels
``y`` are integers in ``0 .. n_invasive``, where ``n_invasive`` means ``not_invasive``.

Metrics returned by :func:`compute_metrics`
-------------------------------------------
=========================  ==========================================================================================
``macro_f1_invasive``      **primary metric**: per-class F1 averaged over the invasive classes only
``top1_acc_all``           **KPI**: accuracy of the 41-way prediction (40 species + ``not_invasive``) over all photos
``macro_ap_invasive``      mean over the invasive classes of average precision (one-vs-rest, threshold-free)
``top1/top3_recall_inv.``  among invasive photos: is the true species ranked first / in the top three (threshold-free)
``bin_*_invasive``         binary "invasive or not", with score = largest invasive probability and threshold ``t``
``noninv_*``               the same binary task seen from the ``not_invasive`` side
``macro_f1_all41``         per-class F1 averaged over all 41 labels
=========================  ==========================================================================================
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix, MulticlassF1Score

SCALAR_METRICS = ["macro_f1_invasive", "top1_acc_all", "macro_ap_invasive", "top1_recall_invasive", "top3_recall_invasive",
                  "threshold", "bin_precision_invasive", "bin_recall_invasive", "bin_f1_invasive", "bin_ap_invasive",
                  "bin_auroc_invasive", "noninv_precision", "noninv_recall", "noninv_auroc", "macro_f1_all41"]
KEY_METRICS = ["macro_f1_invasive", "top1_acc_all", "macro_ap_invasive", "top1_recall_invasive", "top3_recall_invasive",
               "bin_precision_invasive", "bin_recall_invasive", "bin_auroc_invasive", "noninv_auroc", "threshold"]
THRESHOLD_GRID = [round(t, 3) for t in np.linspace(0.01, 0.99, 99)]


def predict_not_invasive(probs: np.ndarray, t: float) -> np.ndarray:
    """Predicted label index per photo: the most probable species if its probability reaches ``t``, else ``not_invasive``."""
    n_invasive = probs.shape[1]
    pred = probs.argmax(1)
    pred[probs.max(1) < t] = n_invasive
    return pred


def macro_f1_invasive(y: np.ndarray, pred: np.ndarray, n_invasive: int) -> float:
    """Per-class F1 over the 41 labels, averaged over the invasive classes only."""
    f1 = MulticlassF1Score(num_classes=n_invasive + 1, average=None)(torch.as_tensor(pred), torch.as_tensor(y))
    return float(f1[:n_invasive].mean())


def tune_threshold(probs: np.ndarray, y: np.ndarray, grid) -> tuple[float, pd.DataFrame]:
    """Sweep ``grid`` and return the threshold with the highest macro-F1 (invasive), plus the sweep table for plotting."""
    n_invasive = probs.shape[1]
    sweep = pd.DataFrame({"threshold": grid})
    sweep["macro_f1_invasive"] = [macro_f1_invasive(y, predict_not_invasive(probs, t), n_invasive) for t in grid]
    sweep["top1_acc_all"] = [float((predict_not_invasive(probs, t) == y).mean()) for t in grid]
    best = sweep.loc[sweep["macro_f1_invasive"].idxmax()]
    return float(best["threshold"]), sweep


def compute_metrics(probs: np.ndarray, y: np.ndarray, threshold: float | None, grid=THRESHOLD_GRID) -> dict:
    """Every metric of the project for one set of predictions.

    Args:
        probs: ``(N, n_invasive)`` probabilities (sigmoid of the logits for a fine-tuned model).
        y: ``(N,)`` true label indices in ``0 .. n_invasive`` (``n_invasive`` = ``not_invasive``).
        threshold: the threshold to use. ``None`` tunes it on these predictions (validation only); the test sets are
            always scored at the threshold tuned on validation.
        grid: thresholds tried when ``threshold`` is ``None``.

    Returns:
        A dict with the scalar metrics listed in the module docstring, ``per_class_f1`` (41 values), ``per_class_ap``
        (one per invasive class), ``confusion_matrix`` (41 x 41, rows = true label), and two private entries:
        ``_sweep`` (the threshold sweep, or ``None``) and ``_pred`` (the predicted label indices).
    """
    probs = np.asarray(probs, dtype=np.float64); y = np.asarray(y)
    n_invasive = probs.shape[1]
    is_inv = y < n_invasive
    onehot = np.eye(n_invasive + 1)[y][:, :n_invasive]              # (N, n_invasive) binary targets

    m = {}
    # threshold-free, ranking metrics over the invasive classes
    per_class_ap = np.array([average_precision_score(onehot[:, c], probs[:, c]) if onehot[:, c].any() else np.nan
                             for c in range(n_invasive)])
    m["macro_ap_invasive"] = float(np.nanmean(per_class_ap))
    if is_inv.any():
        rank = (-probs[is_inv]).argsort(1)
        m["top1_recall_invasive"] = float((rank[:, 0] == y[is_inv]).mean())
        m["top3_recall_invasive"] = float((rank[:, :3] == y[is_inv, None]).any(1).mean())
    # threshold
    sweep = None
    if threshold is None:
        threshold, sweep = tune_threshold(probs, y, grid)
    m["threshold"] = float(threshold)
    pred = predict_not_invasive(probs, threshold)
    # 41-way metrics at the fixed threshold
    f1_per_class = MulticlassF1Score(num_classes=n_invasive + 1, average=None)(torch.as_tensor(pred), torch.as_tensor(y)).numpy()
    m["macro_f1_invasive"] = float(f1_per_class[:n_invasive].mean())
    m["macro_f1_all41"] = float(f1_per_class.mean())
    m["top1_acc_all"] = float(MulticlassAccuracy(num_classes=n_invasive + 1, average="micro")(torch.as_tensor(pred), torch.as_tensor(y)))
    # binary invasive detection
    score_inv = probs.max(1); pred_inv = pred < n_invasive
    m["bin_precision_invasive"] = precision_score(is_inv, pred_inv, zero_division=0)
    m["bin_recall_invasive"] = recall_score(is_inv, pred_inv, zero_division=0)
    m["bin_f1_invasive"] = f1_score(is_inv, pred_inv, zero_division=0)
    m["bin_ap_invasive"] = average_precision_score(is_inv, score_inv)
    m["bin_auroc_invasive"] = roc_auc_score(is_inv, score_inv) if 0 < is_inv.sum() < len(y) else np.nan
    # the not_invasive side
    m["noninv_precision"] = precision_score(~is_inv, ~pred_inv, zero_division=0)
    m["noninv_recall"] = recall_score(~is_inv, ~pred_inv, zero_division=0)
    m["noninv_auroc"] = roc_auc_score(~is_inv, 1 - score_inv) if 0 < is_inv.sum() < len(y) else np.nan
    # arrays
    m["per_class_f1"] = f1_per_class.tolist()
    m["per_class_ap"] = per_class_ap.tolist()
    m["confusion_matrix"] = MulticlassConfusionMatrix(num_classes=n_invasive + 1)(torch.as_tensor(pred), torch.as_tensor(y)).numpy().tolist()
    m["_sweep"] = sweep; m["_pred"] = pred
    return m


def scalars(m: dict) -> dict:
    """The scalar entries of a :func:`compute_metrics` result, in the standard order."""
    return {k: float(m[k]) for k in SCALAR_METRICS if k in m}
