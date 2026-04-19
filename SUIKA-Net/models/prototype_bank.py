import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeBank(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int, momentum: float = 0.9) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.momentum = momentum
        init_proto = F.normalize(torch.randn(num_classes, emb_dim), dim=-1)
        self.register_buffer("prototypes", init_proto)

    @torch.no_grad()
    def update(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        feats = F.normalize(feats, dim=-1)
        for c in labels.unique().tolist():
            mask = labels == c
            if mask.sum().item() == 0:
                continue
            feat_c = feats[mask].mean(dim=0)
            old = self.prototypes[c]
            new = self.momentum * old + (1.0 - self.momentum) * feat_c
            self.prototypes[c] = F.normalize(new.unsqueeze(0), dim=-1).squeeze(0)

    def similarity(self, feats: torch.Tensor, normalize_feats: bool = True) -> torch.Tensor:
        if normalize_feats:
            feats = F.normalize(feats, dim=-1)
        return feats @ self.prototypes.t()

