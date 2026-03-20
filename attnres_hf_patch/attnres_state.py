from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor


class AttnResStats:
    def __init__(self) -> None:
        self.depth_means: List[Tensor] = []
        self.depth_entropies: List[Tensor] = []
        self.depth_weights: List[Tensor] = []
        self.layer_data: Dict[int, Dict[str, List[Tensor]]] = {}

    def add(self, weights: Tensor, layer_idx: Optional[int] = None) -> None:
        if weights.numel() == 0:
            return
        w = weights.detach().cpu()
        mean = w.mean()
        entropy = -(w * torch.log(w.clamp_min(1e-12))).sum(dim=-1).mean()
        self.depth_means.append(mean)
        self.depth_entropies.append(entropy)
        self.depth_weights.append(w)
        if layer_idx is not None:
            entry = self.layer_data.setdefault(layer_idx, {"means": [], "entropies": []})
            entry["means"].append(mean)
            entry["entropies"].append(entropy)

    def summary(self) -> Dict[str, object]:
        if not self.depth_means:
            return {
                "depth_mean": 0.0,
                "depth_entropy": 0.0,
                "attn_depth_weights": [],
                "per_layer": {"means": {}, "entropies": {}},
                "depth_entropy_values": [],
            }
        mean_val = torch.stack(self.depth_means).mean().item()
        entropy_val = torch.stack(self.depth_entropies).mean().item()
        per_layer = {"means": {}, "entropies": {}}
        for layer_idx, data in self.layer_data.items():
            if data["means"]:
                per_layer["means"][layer_idx] = torch.stack(data["means"]).mean().item()
            if data["entropies"]:
                per_layer["entropies"][layer_idx] = torch.stack(data["entropies"]).mean().item()
        entropy_values = [float(t.item()) for t in self.depth_entropies]
        return {
            "depth_mean": mean_val,
            "depth_entropy": entropy_val,
            "attn_depth_weights": list(self.depth_weights),
            "per_layer": per_layer,
            "depth_entropy_values": entropy_values,
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
