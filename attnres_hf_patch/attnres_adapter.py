from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from models.attnres import BlockDepthAttnResidual, BlockState, DepthAttnResidual
from .attnres_state import BlockAttnResState, FullAttnResState


class FullAttnResAdapter(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn_res = DepthAttnResidual(dim)
        self.mlp_res = DepthAttnResidual(dim)

    def before_attention(self, state: FullAttnResState) -> Tuple[Tensor, Tensor]:
        out, weights = self.attn_res(state.history)
        state.stats.add(weights)
        return out, weights

    def before_mlp(self, state: FullAttnResState) -> Tuple[Tensor, Tensor]:
        out, weights = self.mlp_res(state.history)
        state.stats.add(weights)
        return out, weights

    @staticmethod
    def append_history(state: FullAttnResState, value: Tensor) -> None:
        state.history.append(value)


class BlockAttnResAdapter(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block_res = BlockDepthAttnResidual(dim)

    def _current_state(self, state: BlockAttnResState) -> BlockState:
        return BlockState(blocks=list(state.blocks), partial_block=state.partial_block)

    def before_attention(self, state: BlockAttnResState) -> Tuple[Tensor, Tensor]:
        out, weights = self.block_res(self._current_state(state))
        state.stats.add(weights)
        return out, weights

    def before_mlp(self, state: BlockAttnResState) -> Tuple[Tensor, Tensor]:
        out, weights = self.block_res(self._current_state(state))
        state.stats.add(weights)
        return out, weights

    @staticmethod
    def update_partial(state: BlockAttnResState, addendum: Tensor) -> None:
        state.partial_block = state.partial_block + addendum
