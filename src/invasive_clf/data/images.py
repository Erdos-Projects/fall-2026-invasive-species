"""Decode every photo once and keep it in memory as uint8 tensors.

Copied from Section 2 of the team's experiment notebooks (05 to 13) and Section 2c of ``19_final_evaluation.ipynb``.
Decoding about 10,000 JPEGs is the slowest step of an epoch, so the notebooks decode each image once, resize it, and
keep the result in RAM; each epoch then only applies the random augmentation to small tensors.

* Training photos are resized so their **shorter side** is ``presize`` (aspect ratio kept). They have different sizes,
  so they are stored "ragged" in one flat tensor (:class:`RaggedUint8`).
* Photos that are only scored (val, test, test_unseen) get the deterministic evaluation transform right away:
  ``Resize(val_resize)`` then ``CenterCrop(image_size)``. They all have the same size and are stacked in one tensor.

At the design sizes (presize 256, crops of 224) train + val take about 2.4 GB of RAM.
"""
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


class _Decode(Dataset):
    # One-off decode + deterministic resize; wrapped in a DataLoader only to use several worker processes.
    def __init__(self, root, paths, tf):
        self.root, self.paths, self.tf = Path(root), list(paths), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = ImageOps.exif_transpose(Image.open(self.root / self.paths[i])).convert("RGB")
        return T.functional.pil_to_tensor(self.tf(img))                  # uint8, C x H x W


def decode_all(root, paths, tf, desc: str = "decoding images", num_workers: int = 2) -> list[torch.Tensor]:
    """Decode ``root / path`` for every path, apply ``tf``, return a list of uint8 ``C x H x W`` tensors in order."""
    dl = DataLoader(_Decode(root, paths, tf), batch_size=None, num_workers=num_workers,   # batch_size=None: items stay ragged
                    generator=torch.Generator().manual_seed(0))                         # do not touch the global RNG
    return [t for t in tqdm(dl, total=len(paths), desc=desc, unit="img", leave=False)]


class RaggedUint8:
    """Images of different sizes in ONE flat uint8 tensor plus shapes and offsets. A single allocation is shared copy-on-write
    by the forked DataLoader workers, whereas a Python list of thousands of tensors slowly gets duplicated into every worker."""
    def __init__(self, flat, shapes, offsets):
        self.flat, self.shapes, self.offsets = flat, shapes, offsets

    @classmethod
    def from_list(cls, tensors):
        shapes = torch.tensor([list(t.shape) for t in tensors])
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), shapes.prod(1).cumsum(0)])
        return cls(torch.cat([t.flatten() for t in tensors]), shapes, offsets)

    def __len__(self):
        return len(self.shapes)

    def __getitem__(self, i):
        return self.flat[self.offsets[i]:self.offsets[i + 1]].view(*self.shapes[i].tolist())


def train_images(root, paths, presize: int, num_workers: int = 2, desc="decoding training images") -> RaggedUint8:
    """Training photos resized to shorter side ``presize``; the random crop to ``image_size`` happens per epoch."""
    return RaggedUint8.from_list(decode_all(root, paths, T.Resize(presize), desc, num_workers))


def eval_crops(root, paths, val_resize: int, image_size: int, num_workers: int = 2, desc="decoding images") -> torch.Tensor:
    """Photos to score, as one ``(N, 3, image_size, image_size)`` uint8 tensor: Resize(val_resize), CenterCrop(image_size)."""
    tf = T.Compose([T.Resize(val_resize), T.CenterCrop(image_size)])
    return torch.stack(decode_all(root, paths, tf, desc, num_workers))
