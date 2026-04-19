from typing import Dict, List, Optional

import torch
from torch.utils.data import WeightedRandomSampler


def build_class_balanced_sampler(
    labels: List[int],
    power: float = 1.0,
    replacement: bool = True,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    counts: Dict[int, int] = {}
    for y in labels:
        counts[y] = counts.get(y, 0) + 1

    weights = []
    for y in labels:
        freq = max(counts[y], 1)
        w = (1.0 / float(freq)) ** power
        weights.append(w)

    weights_t = torch.as_tensor(weights, dtype=torch.double)
    if num_samples is None:
        num_samples = len(labels)
    return WeightedRandomSampler(
        weights=weights_t,
        num_samples=num_samples,
        replacement=replacement,
    )

