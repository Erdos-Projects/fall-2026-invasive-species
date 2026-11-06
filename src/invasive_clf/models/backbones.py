"""Pretrained backbones.

Copied from Section 3 of the team's experiment notebooks (05 to 13). A backbone builder returns a :class:`BackboneSpec`:
a module whose ``forward(x)`` maps a normalized image batch ``(B, 3, H, W)`` to one pooled feature vector per image
``(B, feat_dim)``, plus the normalization constants the checkpoint was trained with.

Only the backbone of the final fine-tuned model is kept here. The notebooks also have builders for PlantNet DINOv3,
ResNets, ConvNeXt and SigLIP 2 (via ``timm``).
BioCLIP 2's image tower, used frozen by the final zero-shot model, is in ``bioclip2.py``.

DINOv3 ViT-S/16
---------------
``facebook/dinov3-vits16-pretrain-lvd1689m`` on Hugging Face: a ViT-S/16 (21.6M parameters, 384-d tokens, 4 register
tokens) trained self-supervised on 1.7 billion images. The weights are **gated**: accept the license on the model page,
then log in with ``hf auth login`` or set the ``HF_TOKEN`` environment variable. Needs ``transformers >= 4.57.3``.

Pooling (``backbone_kwargs.pooling``):

* ``"cls"``: the final-normed CLS token (384-d).
* ``"cls+patchmean"``: the CLS token concatenated with the mean of the patch tokens (768-d). The register tokens are
  skipped. This is the setting of the final model (run ``04_pool_cls_patchmean`` of notebook 11).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DINOV3_VITS16 = "facebook/dinov3-vits16-pretrain-lvd1689m"


@dataclass
class BackboneSpec:
    module: nn.Module          # forward(x) -> (B, feat_dim) pooled features
    feat_dim: int
    mean: tuple                # normalization of the checkpoint
    std: tuple
    notes: str = ""


class HFPooledBackbone(nn.Module):
    """Wraps a Hugging Face vision model. pooling="cls": ``.pooler_output`` (DINOv3: the final-normed CLS token).
    pooling="cls+patchmean": CLS concatenated with the mean of the patch tokens (skips the CLS and register tokens)."""
    def __init__(self, hf_model, pooling="cls", n_prefix_tokens=1):
        super().__init__(); self.m = hf_model; self.pooling = pooling; self.n_prefix = n_prefix_tokens
        assert pooling in ("cls", "cls+patchmean"), pooling

    def forward(self, x):
        out = self.m(pixel_values=x)
        if self.pooling == "cls":
            return out.pooler_output
        return torch.cat([out.pooler_output, out.last_hidden_state[:, self.n_prefix:].mean(1)], dim=1)


def build_dinov3_vits16(cfg: dict, encoder=None) -> BackboneSpec:
    """Plain DINOv3 ViT-S/16 with ImageNet mean/std. ``encoder`` replaces the download (used by the tests)."""
    if encoder is None:
        from transformers import AutoModel
        encoder = AutoModel.from_pretrained(DINOV3_VITS16)
    pooling = cfg.get("backbone_kwargs", {}).get("pooling", "cls")
    feat_dim = encoder.config.hidden_size * (2 if pooling == "cls+patchmean" else 1)
    return BackboneSpec(HFPooledBackbone(encoder, pooling, 1 + encoder.config.num_register_tokens), feat_dim,
                        IMAGENET_MEAN, IMAGENET_STD, notes=f"DINOv3 ViT-S/16, ImageNet mean/std, pooling={pooling}")


BACKBONE_REGISTRY = {
    "dinov3_vits16": build_dinov3_vits16,
}
