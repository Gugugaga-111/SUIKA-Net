from dataclasses import dataclass
from typing import Sequence, Tuple

import torch


def accuracy_topk(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Sequence[int] = (1,),
) -> Tuple[float, ...]:
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B, C], got shape={tuple(logits.shape)}")
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    out = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        out.append(float(correct_k.mul_(100.0 / batch_size).item()))
    return tuple(out)


@dataclass
class AccuracyMeter:
    total: int = 0
    correct_top1: int = 0
    correct_top5: int = 0
    loss_sum: float = 0.0

    def update(self, loss: float, logits: torch.Tensor, targets: torch.Tensor) -> None:
        bsz = int(targets.size(0))
        self.total += bsz
        self.loss_sum += loss * bsz

        topk = (1, 5 if logits.size(1) >= 5 else logits.size(1))
        top_res = accuracy_topk(logits, targets, topk=topk)
        top1 = top_res[0]
        top5 = top_res[-1]

        self.correct_top1 += int(round(top1 / 100.0 * bsz))
        self.correct_top5 += int(round(top5 / 100.0 * bsz))

    @property
    def avg_loss(self) -> float:
        return self.loss_sum / max(self.total, 1)

    @property
    def top1(self) -> float:
        return 100.0 * self.correct_top1 / max(self.total, 1)

    @property
    def top5(self) -> float:
        return 100.0 * self.correct_top5 / max(self.total, 1)

    def as_dict(self):
        return {
            "loss": float(self.avg_loss),
            "top1": float(self.top1),
            "top5": float(self.top5),
            "num_samples": int(self.total),
        }

