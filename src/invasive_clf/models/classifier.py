"""The fine-tuned classifier: backbone -> pooled features -> Dropout -> Linear (one output per invasive class).

Copied from Section 3b of the team's experiment notebooks (05 to 13). The head is initialized with Xavier-uniform
weights and zero biases; no new BatchNorm layers are added. The sigmoid of each output is the probability of that
invasive species; ``not_invasive`` has no output (see ``invasive_clf.metrics``).
"""
import torch.nn as nn

from .backbones import BACKBONE_REGISTRY, BackboneSpec


class InvasiveClassifier(nn.Module):
    def __init__(self, spec: BackboneSpec, n_classes: int, dropout: float):
        super().__init__()
        self.backbone = spec.module
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(spec.feat_dim, n_classes)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.dropout(self.backbone(x)))

    def set_backbone_frozen(self, frozen: bool):
        for p in self.backbone.parameters():
            p.requires_grad_(not frozen)


def build_model(cfg: dict, n_classes: int, device, spec: BackboneSpec | None = None):
    """Build the backbone named by ``cfg["model_name"]`` (unless ``spec`` is given) and put the classifier on ``device``."""
    spec = spec or BACKBONE_REGISTRY[cfg["model_name"]](cfg)
    model = InvasiveClassifier(spec, n_classes, cfg["dropout"]).to(device)
    n_bb = sum(p.numel() for p in model.backbone.parameters()); n_head = sum(p.numel() for p in model.head.parameters())
    print(f"{cfg['model_name']}: backbone {n_bb/1e6:.1f}M params, head {n_head/1e3:.1f}K params, feat_dim {spec.feat_dim} | {spec.notes}")
    return model, spec
