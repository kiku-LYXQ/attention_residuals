from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor, nn

from .norms import RMSNorm


class DepthAttnResidual(nn.Module):
    """Implements depth-wise attention residual as described for Full AttnRes."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scoring_proj = nn.Linear(dim, 1, bias=False)
        self.norm = RMSNorm(dim)

    def forward(self, inputs: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Aggregate depth sources with softmax along depth dimension."""
        assert inputs, "At least one depth source must be provided"
        stacked = torch.stack(inputs, dim=2)  # shape: (B, T, depth, D)
        b, t, depth, d = stacked.shape
        flattened = stacked.reshape(-1, d)
        normalized = self.norm(flattened).reshape(b, t, depth, d)
        scores = self.scoring_proj(normalized).squeeze(-1)
        weights = torch.softmax(scores, dim=2)
        weighted = (weights.unsqueeze(-1) * stacked).sum(dim=2)
        return weighted, weights


@dataclass
class BlockState:
    blocks: List[Tensor]
    partial_block: Tensor
    layer_counter: int = 0


class BlockDepthAttnResidual(nn.Module):
    """Implements Block Attention Residuals, operating over completed blocks + partial block."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scoring_proj = nn.Linear(dim, 1, bias=False)
        self.norm = RMSNorm(dim)

    def _compose(self, sources: List[Tensor]) -> Tuple[Tensor, Tensor]:
        stacked = torch.stack(sources, dim=2)  # (B, T, depth, D)
        b, t, depth, d = stacked.shape
        flattened = stacked.reshape(-1, d)
        normalized = self.norm(flattened).reshape(b, t, depth, d)
        scores = self.scoring_proj(normalized).squeeze(-1)
        attn = torch.softmax(scores, dim=2)
        output = (attn.unsqueeze(-1) * stacked).sum(dim=2)
        return output, attn

    def forward(
        self,
        state: BlockState,
    ) -> Tuple[Tensor, Tensor]:
        """Attend over blocks + partial_block in source dimension."""
        sources = [*state.blocks, state.partial_block]
        return self._compose(sources)
