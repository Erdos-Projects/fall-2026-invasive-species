"""BioCLIP 2 zero-shot classifier: the final frozen model.

BioCLIP 2 (``imageomics/bioclip-2`` on Hugging Face, an ``open_clip`` ViT-L/14 with 304M parameters in the image tower)
was trained to match photos of organisms with text naming them. Used zero-shot, nothing is trained:

1. **Vocabulary.** 4,573 names: the 441 classes of ``train.csv`` and ``val.csv`` (40 invasive + 401 background) and
   4,132 more regional species from ``classes.csv`` (:func:`build_vocabulary`). Every name except the 40 invasive ones
   counts as ``not_invasive``. The 30 hold-out species of ``test_unseen.csv`` are deliberately not in it.
2. **Text embeddings.** Each name becomes the prompt ``"a photo of <scientific name>"``; the text tower turns it into a
   768-d unit vector. A class that merges several taxa (a complex) gets the normalized mean of its members' vectors
   (:func:`encode_vocabulary`). This is done once.
3. **Scoring a photo** (:class:`ZeroShotClassifier`). The image tower turns the photo into a 768-d vector; its cosine
   similarity with every name, times BioCLIP 2's learned logit scale, goes through a softmax over the 4,573 names. The
   probability of invasive class *c* is the probability of its name. The decision rule is the one of every model in
   the project (``invasive_clf.metrics``): the most likely invasive class if its probability reaches the threshold
   (0.06, tuned on val), otherwise ``not_invasive``. The photo and its mirror image are scored and their
   probabilities averaged.

Copied from notebooks 14 (vocabulary of 441 classes and prompts), 16 (the 4,132 extra names, the saved candidate) and
19 (``build_open_clip``, ``FrozenMethod``, ``Scorer``).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.dataset import NONE_LABEL, LabelSpace
from .backbones import BackboneSpec

BIOCLIP2 = "hf-hub:imageomics/bioclip-2"
PROMPT = "a photo of {}"            # the "scientific" prompt, chosen on val in notebook 14


def _split(s) -> list[str]:
    return [x.strip() for x in re.split(r"[;|]", str(s)) if x.strip()]


def _split_ids(s) -> list[int]:
    return [int(x) for x in _split(s) if x.isdigit()]


# ------------------------------------------------------------------------------------------------- vocabulary

def build_vocabulary(splits_dir, labels: LabelSpace) -> pd.DataFrame:
    """The 4,573-name zero-shot vocabulary, one row per class, with columns ``taxon_id``, ``name``, ``members`` (the
    scientific names whose prompts are averaged), ``label`` and ``y`` (label index; ``n_invasive`` = not_invasive) and
    ``source`` (``"train+val"`` or ``"regional"``).

    * The first part is every class (taxon_id) in train or val, as in notebook 14. Members of merged complexes come from
      ``classes.csv``.
    * The second part is notebook 16's extras from ``classes.csv``: species or complexes not already in the vocabulary,
      sharing no member taxon with it, and not one of the ``test_unseen`` hold-out classes; sorted by number of
      observations. They all count as ``not_invasive``.
    """
    splits_dir = Path(splits_dir)
    train_df, val_df = pd.read_csv(splits_dir / "train.csv"), pd.read_csv(splits_dir / "val.csv")
    classes = pd.read_csv(splits_dir / "classes.csv")

    # --- part 1: the classes of train + val (notebook 14) ---
    rows = pd.concat([train_df, val_df], ignore_index=True)
    vocab = (rows.groupby("taxon_id").agg(name=("scientific_name", "first"), label=("label", "first"), n_labels=("label", "nunique"))
             .reset_index())
    assert (vocab.n_labels == 1).all(), "a taxon_id carries more than one label"
    cl = classes.drop_duplicates("class_key").set_index("class_key")
    vocab["members"] = [(_split(cl.at[t, "member_names"]) if t in cl.index and isinstance(cl.at[t, "member_names"], str) else [])
                        or [n] for t, n in zip(vocab.taxon_id, vocab.name)]
    vocab = vocab.sort_values(["label", "name"]).reset_index(drop=True)
    vocab["source"] = "train+val"

    # --- part 2: the regional extras (notebook 16, Section 9) ---
    ids = set(vocab.taxon_id)
    in_vocab = set(ids)
    for r in classes[classes.class_key.isin(ids)].itertuples():
        in_vocab |= set(_split_ids(r.member_taxon_ids))
    cand = classes[~classes.class_key.isin(ids) & ~classes.holdout_unseen.astype(bool) & classes.class_rank.isin(["species", "complex"])]
    cand = cand[~cand.member_taxon_ids.map(lambda s: bool(set(_split_ids(s)) & in_vocab)) & ~cand.class_key.isin(in_vocab)]
    cand = cand.drop_duplicates("class_key").sort_values("n_window", ascending=False).reset_index(drop=True)
    extras = pd.DataFrame({"taxon_id": cand.class_key, "name": cand.class_name, "label": NONE_LABEL,
                           "members": [_split(m) or [n] for m, n in zip(cand.member_names, cand.class_name)], "source": "regional"})

    vocab = pd.concat([vocab.drop(columns="n_labels"), extras], ignore_index=True)
    vocab["y"] = vocab["label"].map(labels.class_to_idx)
    assert vocab["y"].notna().all(), "the vocabulary has labels that train.csv does not"
    vocab["y"] = vocab["y"].astype(int)
    assert (vocab.y < labels.n_invasive).sum() == labels.n_invasive, "every invasive class must have exactly one vocabulary entry"
    return vocab


# ------------------------------------------------------------------------------------------- BioCLIP 2 itself

class OpenClipVisual(nn.Module):
    """The image tower of an open_clip model; ``forward`` returns the projected CLIP image embedding (768-d for BioCLIP 2),
    the space the text embeddings live in."""
    def __init__(self, visual):
        super().__init__(); self.m = visual

    def forward(self, x):
        return self.m(x)


def load_bioclip2(model_id: str = BIOCLIP2, clip=None, tokenizer=None):
    """(open_clip model, tokenizer, logit scale). ``clip`` and ``tokenizer`` replace the download (used by the tests)."""
    import open_clip
    clip = clip if clip is not None else open_clip.create_model(model_id)
    tokenizer = tokenizer if tokenizer is not None else open_clip.get_tokenizer(model_id)
    return clip.eval(), tokenizer, float(clip.logit_scale.exp().item())


def image_spec(clip) -> BackboneSpec:
    """The image tower as a :class:`BackboneSpec`, with the normalization BioCLIP 2 was trained with."""
    import open_clip
    visual = clip.visual
    mean = tuple(getattr(visual, "image_mean", None) or open_clip.OPENAI_DATASET_MEAN)
    std = tuple(getattr(visual, "image_std", None) or open_clip.OPENAI_DATASET_STD)
    return BackboneSpec(OpenClipVisual(visual), visual.output_dim, mean, std, notes="BioCLIP 2 image tower (open_clip), projected features")


@torch.no_grad()
def encode_vocabulary(clip, tokenizer, vocab: pd.DataFrame, device, prompt: str = PROMPT, batch_size: int = 256) -> torch.Tensor:
    """``(V, d)`` unit text embeddings, one per vocabulary row: the normalized mean of its members' prompt embeddings."""
    clip = clip.to(device).eval()
    per_class = [[prompt.format(n) for n in members] for members in vocab["members"]]
    texts = [p for ps in per_class for p in ps]
    out = []
    for i in range(0, len(texts), batch_size):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            e = clip.encode_text(tokenizer(texts[i:i + batch_size]).to(device)).float()
        out.append(F.normalize(e, dim=-1).cpu())
    flat = torch.cat(out)
    return F.normalize(torch.stack([c.mean(0) for c in torch.split(flat, [len(ps) for ps in per_class])]), dim=-1)


class ZeroShotClassifier:
    """Probabilities over the invasive classes from image features (``FrozenMethod`` of notebook 19, zero-shot case).

    ``text_embeddings`` (V, d) unit vectors; ``vocab_to_label`` (V,) label index of each name (``n_invasive`` =
    not_invasive); ``logit_scale`` BioCLIP 2's learned temperature (about 100)."""
    def __init__(self, text_embeddings, vocab_to_label, logit_scale: float, n_invasive: int, threshold: float, device="cpu"):
        self.txt = torch.as_tensor(np.asarray(text_embeddings, dtype=np.float32), device=device)
        v2l = np.asarray(vocab_to_label).astype(int)
        M = np.zeros((len(v2l), n_invasive), np.float32); inv = np.flatnonzero(v2l < n_invasive); M[inv, v2l[inv]] = 1
        self.M = torch.as_tensor(M, device=device)            # name -> invasive class (zero row = not_invasive)
        self.logit_scale, self.threshold = float(logit_scale), float(threshold)

    def to(self, device):
        self.txt, self.M = self.txt.to(device), self.M.to(device); return self

    def __call__(self, feats: torch.Tensor) -> torch.Tensor:
        Fq = F.normalize(feats.float(), dim=-1)
        return torch.softmax(self.logit_scale * Fq @ self.txt.T, dim=1) @ self.M

    @classmethod
    def load(cls, path, device="cpu") -> "ZeroShotClassifier":
        """From the file written by ``python -m invasive_clf.build_bioclip2``."""
        art = torch.load(path, map_location="cpu", weights_only=False)
        return cls(art["text_embeddings"], art["vocab_to_label"], art["logit_scale"], len(art["invasive_classes"]),
                   art["threshold"], device)
