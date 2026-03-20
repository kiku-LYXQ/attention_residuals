from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """RMS layer normalization as described in the paper.

    RMSNorm scales the input by rms norm rather than computing a full variance.
    This implementation follows the formula: x * (g / (sqrt(mean(x^2)) + eps)).
    """

    def __init__(self, dim: int, eps: float = 1e-8, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.dim, "RMSNorm dimension mismatch"
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        denom = rms + self.eps
        scale = self.weight if self.elementwise_affine else 1.0
        return x / denom * scale
