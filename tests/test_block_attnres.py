from __future__ import annotations

import torch

from models.attnres import BlockDepthAttnResidual, BlockState


def _make_seq(batch: int = 1, seq_len: int = 4, dim: int = 16) -> torch.Tensor:
    return torch.randn(batch, seq_len, dim)


def test_block_state_advances():
    block_size = 2
    dim = 16
    residual = BlockDepthAttnResidual(dim=dim)
    embed = _make_seq(dim=dim)
    state = BlockState(blocks=[embed], partial_block=embed)
    for _ in range(1, 5):
        _, attn = residual(state)
        assert torch.allclose(attn.sum(dim=2), torch.ones_like(attn.sum(dim=2)))
        state.partial_block = state.partial_block + torch.randn_like(state.partial_block) * 0.01
        state.layer_counter += 1
        if state.layer_counter % block_size == 0:
            state.blocks.append(state.partial_block.clone())
            state.partial_block = state.partial_block.clone()
    assert len(state.blocks) >= 3, "Blocks should grow as layers end"
    assert state.partial_block.shape == embed.shape
