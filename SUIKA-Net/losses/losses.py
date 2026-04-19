from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def build_class_weights(cls_counts: Dict[int, int], beta: float = 0.9999) -> torch.Tensor:
    if not cls_counts:
        raise ValueError("cls_counts is empty")
    max_label = max(cls_counts.keys())
    counts = torch.ones(max_label + 1, dtype=torch.float32)
    for c, n in cls_counts.items():
        counts[c] = max(float(n), 1.0)

    effective_num = 1.0 - torch.pow(beta, counts)
    weights = (1.0 - beta) / torch.clamp(effective_num, min=1e-12)
    weights = weights / weights.sum() * len(weights)
    return weights


def view_consistency_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(z1, z2, dim=-1).mean()


def prototype_loss(
    proto_bank,
    feats: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    sim = proto_bank.similarity(feats) / temperature
    return F.cross_entropy(sim, labels, label_smoothing=label_smoothing)


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    proto_bank=None,
    ce_weight: Optional[torch.Tensor] = None,
    lambda_view: float = 0.2,
    lambda_proto: float = 0.5,
    temperature: float = 0.07,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits = outputs["logits"]
    z_fuse = outputs["z_fuse"]

    cls_loss = F.cross_entropy(logits, labels, weight=ce_weight, label_smoothing=label_smoothing)

    pair_losses = []
    if "z_head" in outputs:
        pair_losses.append(view_consistency_loss(outputs["z_global"], outputs["z_head"]))
    if "z_mask" in outputs:
        pair_losses.append(view_consistency_loss(outputs["z_global"], outputs["z_mask"]))
    if "z_head" in outputs and "z_mask" in outputs:
        pair_losses.append(view_consistency_loss(outputs["z_head"], outputs["z_mask"]))
    if pair_losses:
        v_loss = torch.stack(pair_losses).mean()
    else:
        v_loss = torch.zeros(1, device=labels.device, dtype=logits.dtype).squeeze(0)

    if proto_bank is not None and lambda_proto > 0:
        p_loss = prototype_loss(
            proto_bank=proto_bank,
            feats=z_fuse,
            labels=labels,
            temperature=temperature,
            label_smoothing=label_smoothing,
        )
    else:
        p_loss = torch.zeros(1, device=labels.device, dtype=logits.dtype).squeeze(0)

    total = cls_loss + lambda_view * v_loss + lambda_proto * p_loss
    loss_dict = {
        "cls_loss": float(cls_loss.detach().item()),
        "view_loss": float(v_loss.detach().item()),
        "proto_loss": float(p_loss.detach().item()),
        "loss": float(total.detach().item()),
    }
    return total, loss_dict

