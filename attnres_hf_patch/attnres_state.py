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
        self.weight_max_values: List[Tensor] = []
        self.weight_top1_mean_values: List[Tensor] = []
        self.layer_data: Dict[int, Dict[str, List[Tensor]]] = {}

    def add(self, weights: Tensor, layer_idx: Optional[int] = None) -> None:
        if weights.numel() == 0:
            return
        w = weights.detach().cpu()
        mean = w.mean()
        entropy = -(w * torch.log(w.clamp_min(1e-12))).sum(dim=-1).mean()
        weight_max = w.max()
        weight_top1_mean = w.max(dim=2).values.mean()
        self.depth_means.append(mean)
        self.depth_entropies.append(entropy)
        self.depth_weights.append(w)
        self.weight_max_values.append(weight_max)
        self.weight_top1_mean_values.append(weight_top1_mean)
        if layer_idx is not None:
            entry = self.layer_data.setdefault(
                layer_idx,
                {"means": [], "entropies": [], "weight_max": [], "weight_top1_mean": []},
            )
            entry["means"].append(mean)
            entry["entropies"].append(entropy)
            entry["weight_max"].append(weight_max)
            entry["weight_top1_mean"].append(weight_top1_mean)

    def summary(self) -> Dict[str, object]:
        if not self.depth_means:
            return {
                "depth_mean": 0.0,
                "depth_entropy": 0.0,
                "attn_depth_weights": [],
                "attnres_weight_max": 0.0,
                "attnres_weight_top1_mean": 0.0,
                "attnres_weight_max_values": [],
                "attnres_weight_top1_mean_values": [],
                "per_layer": {"means": {}, "entropies": {}, "weight_max": {}, "weight_top1_mean": {}},
                "depth_entropy_values": [],
            }
        mean_val = torch.stack(self.depth_means).mean().item()
        entropy_val = torch.stack(self.depth_entropies).mean().item()
        weight_max_val = torch.stack(self.weight_max_values).mean().item()
        weight_top1_mean_val = torch.stack(self.weight_top1_mean_values).mean().item()
        per_layer = {"means": {}, "entropies": {}, "weight_max": {}, "weight_top1_mean": {}}
        for layer_idx, data in self.layer_data.items():
            if data["means"]:
                per_layer["means"][layer_idx] = torch.stack(data["means"]).mean().item()
            if data["entropies"]:
                per_layer["entropies"][layer_idx] = torch.stack(data["entropies"]).mean().item()
            if data["weight_max"]:
                per_layer["weight_max"][layer_idx] = torch.stack(data["weight_max"]).mean().item()
            if data["weight_top1_mean"]:
                per_layer["weight_top1_mean"][layer_idx] = torch.stack(data["weight_top1_mean"]).mean().item()
        entropy_values = [float(t.item()) for t in self.depth_entropies]
        return {
            "depth_mean": mean_val,
            "depth_entropy": entropy_val,
            "attn_depth_weights": list(self.depth_weights),
            "attnres_weight_max": weight_max_val,
            "attnres_weight_top1_mean": weight_top1_mean_val,
            "attnres_weight_max_values": [float(t.item()) for t in self.weight_max_values],
            "attnres_weight_top1_mean_values": [float(t.item()) for t in self.weight_top1_mean_values],
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
