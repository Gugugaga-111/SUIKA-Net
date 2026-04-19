from typing import Dict, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mixstyle import MixStyle


class AnimeNet(nn.Module):
    """
    Shared-backbone multi-view network.

    Inputs:
      - global view
      - head view
      - mask-prompt view
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        emb_dim: int = 512,
        pretrained: bool = True,
        proj_hidden_dim: Optional[int] = None,
        drop_rate: float = 0.0,
        use_mixstyle: bool = False,
        mixstyle_p: float = 0.5,
        mixstyle_alpha: float = 0.3,
        view_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.use_mixstyle = use_mixstyle

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_rate=drop_rate,
        )
        feat_dim = getattr(self.backbone, "num_features")
        hidden = proj_hidden_dim or feat_dim

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, emb_dim),
        )
        self.classifier = nn.Linear(emb_dim, num_classes)

        self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha) if use_mixstyle else nn.Identity()
        self.view_weights = view_weights or {"global": 1.0, "head": 1.0, "mask": 1.0}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # For portability across backbones, MixStyle is applied on the input tensor.
        x = self.mixstyle(x)
        feat = self.backbone(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[-1]
        if feat.dim() > 2:
            feat = feat.mean(dim=tuple(range(2, feat.dim())))
        emb = self.proj(feat)
        return F.normalize(emb, dim=-1)

    def forward(
        self,
        x_global: torch.Tensor,
        x_head: Optional[torch.Tensor] = None,
        x_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}

        z_global = self.encode(x_global)
        out["z_global"] = z_global

        z_list = [z_global]
        w_list = [self.view_weights.get("global", 1.0)]

        if x_head is not None:
            z_head = self.encode(x_head)
            out["z_head"] = z_head
            z_list.append(z_head)
            w_list.append(self.view_weights.get("head", 1.0))

        if x_mask is not None:
            z_mask = self.encode(x_mask)
            out["z_mask"] = z_mask
            z_list.append(z_mask)
            w_list.append(self.view_weights.get("mask", 1.0))

        z = torch.stack(z_list, dim=0)  # [V, B, D]
        w = torch.tensor(w_list, dtype=z.dtype, device=z.device).view(-1, 1, 1)
        z_fuse = (z * w).sum(dim=0) / torch.clamp(w.sum(dim=0), min=1e-12)
        z_fuse = F.normalize(z_fuse, dim=-1)

        out["z_fuse"] = z_fuse
        out["logits"] = self.classifier(z_fuse)
        return out
