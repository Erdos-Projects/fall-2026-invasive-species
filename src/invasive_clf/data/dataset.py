"""Split tables, label encoding, augmentation and the PyTorch dataset.

Copied from Sections 2 and 4 of the team's experiment notebooks (05 to 13) and ``19_final_evaluation.ipynb``.

Label encoding
--------------
The 40 invasive classes are the labels of ``train.csv`` other than ``not_invasive``, sorted alphabetically; they get
indices ``0..39``. ``not_invasive`` gets index 40 and has no model output: its training target is all zeros.

Augmentation (training only)
----------------------------
From the cached image at ``presize``: ``RandomResizedCrop(image_size, scale=rrc_scale)``, horizontal flip, rotation by up
to ``rotation_deg`` degrees, color jitter, then normalization with the backbone's mean and std. Scored photos are already
resized and center-cropped in the cache (see ``images.py``), so they are only normalized.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

NONE_LABEL = "not_invasive"
SPLITS = ("train", "val", "test", "test_unseen")


@dataclass
class LabelSpace:
    """The 40 invasive classes in index order, plus ``not_invasive`` as the last label."""
    invasive_classes: list

    @classmethod
    def from_train(cls, train_df: pd.DataFrame) -> "LabelSpace":
        return cls(sorted(c for c in train_df["label"].unique() if c != NONE_LABEL))

    @property
    def n_invasive(self) -> int:
        return len(self.invasive_classes)

    @property
    def all_labels(self) -> list:
        return self.invasive_classes + [NONE_LABEL]          # 41 names in index order

    @property
    def class_to_idx(self) -> dict:
        return {c: i for i, c in enumerate(self.all_labels)}


def read_split(splits_dir, name: str, labels: LabelSpace | None = None, image_dir=None) -> pd.DataFrame:
    """Read ``<splits_dir>/<name>.csv``; add the label index ``y`` if ``labels`` is given; if ``image_dir`` is given,
    drop the rows whose image file is missing (the download may have gaps) and say how many were dropped."""
    df = pd.read_csv(Path(splits_dir) / f"{name}.csv")
    if labels is not None:
        unknown = set(df["label"]) - set(labels.all_labels)
        assert not unknown, f"{name}.csv has labels that train.csv does not: {sorted(unknown)[:5]}"
        df["y"] = df["label"].map(labels.class_to_idx)
    if image_dir is not None:
        exists = df["image_path"].map(lambda p: (Path(image_dir) / p).is_file())
        if (~exists).any():
            print(f"{name}: {(~exists).sum()} of {len(df)} rows have no image file under {image_dir} and are dropped")
        df = df[exists].reset_index(drop=True)
        assert len(df), f"no images found for {name} under {image_dir}; run `python -m invasive_clf.data.download_images` first"
    return df


def make_transforms(cfg: dict, mean, std):
    """(train transform, eval transform), applied per sample to the cached uint8 tensors inside the loader workers."""
    norm = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    cj = cfg["color_jitter"]
    train_tf = v2.Compose([
        v2.RandomResizedCrop(cfg["image_size"], scale=tuple(cfg["rrc_scale"]), antialias=True),   # from the presize image
        v2.RandomHorizontalFlip(p=cfg["hflip_p"]),
        v2.RandomRotation(cfg["rotation_deg"]),
        v2.ColorJitter(**cj),
        *norm])
    eval_tf = v2.Compose(norm)                   # the cached eval images are already resized and center-cropped
    return train_tf, eval_tf


class InatImages(Dataset):
    """Cached images + labels. Returns ``(image, target, y)``: ``target`` is the 40-long multi-label vector (all zeros for
    ``not_invasive``), ``y`` the label index used by the metrics."""
    def __init__(self, images, df: pd.DataFrame, transform, n_invasive: int):
        self.images = images                     # RaggedUint8 (train) or a stacked uint8 tensor (eval), indexed like df
        self.y = df["y"].to_numpy(); self.transform = transform; self.n = n_invasive
        assert len(images) == len(df), "image cache does not match the dataframe"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.transform(self.images[i])
        target = torch.zeros(self.n, dtype=torch.float32)
        if self.y[i] < self.n:
            target[self.y[i]] = 1.0                     # not_invasive stays all-zero
        return x, target, int(self.y[i])
