from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch
from torch import Tensor


class AttnResStats:
    def __init__(self) -> None:
        self.depth_means: List[Tensor] = []
        self.depth_entropies: List[Tensor] = []
        self.depth_weights: List[Tensor] = []

    def add(self, weights: Tensor) -> None:
        if weights.numel() == 0:
            return
        w = weights.detach()
        mean = w.mean()
        entropy = -(w * torch.log(w.clamp_min(1e-12))).sum(dim=-1).mean()
        self.depth_means.append(mean)
        self.depth_entropies.append(entropy)
        self.depth_weights.append(w)

    def summary(self) -> Dict[str, float]:
        if not self.depth_means:
            return {"depth_mean": 0.0, "depth_entropy": 0.0, "attn_depth_weights": []}
        mean_val = torch.stack(self.depth_means).mean().item()
        entropy_val = torch.stack(self.depth_entropies).mean().item()
        return {
            "depth_mean": mean_val,
            "depth_entropy": entropy_val,
            "attn_depth_weights": list(self.depth_weights),
        }


@dataclass
class FullAttnResState:
    history: List[Tensor]
    stats: AttnResStats = field(default_factory=AttnResStats)


@dataclass
class BlockAttnResState:
    blocks: List[Tensor]
    partial_block: Tensor
    block_size: int
    stats: AttnResStats = field(default_factory=AttnResStats)
    layer_counter: int = 0

    def increment_layer(self) -> None:
        self.layer_counter += 1

    def should_close_block(self) -> bool:
        return self.block_size > 0 and self.layer_counter % self.block_size == 0
