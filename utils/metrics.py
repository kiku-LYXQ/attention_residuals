from __future__ import annotations

from typing import Dict, Iterable, List

import torch
from torch import Tensor, nn


def hidden_norms(hidden_states: Dict[str, Tensor]) -> Dict[str, float]:
    """Compute L2 norm for each hidden layer snapshot."""
    return {name: tensor.norm(p=2).item() for name, tensor in hidden_states.items()}


def grad_norms(model: nn.Module) -> Dict[str, float]:
    """Compute gradient norms for every parameter group."""
    norms: Dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        norms[name] = param.grad.norm(p=2).item()
    return norms


def depth_attention_mean(weights: Iterable[Tensor]) -> Tensor:
    """Return mean attention weights aggregated over layers."""
    means = [w.mean(dim=2) for w in weights]
    stacked = torch.stack(means)
    return stacked.mean(dim=0)
