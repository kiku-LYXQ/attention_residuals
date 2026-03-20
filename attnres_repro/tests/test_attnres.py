from __future__ import annotations

import torch

from attnres_repro.models.attnres import BlockDepthAttnResidual, BlockState, DepthAttnResidual


def _make_seq(batch: int = 2, seq_len: int = 4, dim: int = 16):
    return torch.randn(batch, seq_len, dim)


def test_depth_attn_weights_sum():
    module = DepthAttnResidual(dim=16)
    history = [_make_seq(), _make_seq()]
    _, weights = module(history)
    assert torch.allclose(weights.sum(dim=2), torch.ones_like(weights.sum(dim=2)))
    assert (weights[:, :, 0] > 0).any(), "Embedding source should contribute"


def test_block_depth_attn_residual_outputs():
    module = BlockDepthAttnResidual(dim=16)
    embed = _make_seq()
    blocks = [embed]
    partial = _make_seq()
    state = BlockState(blocks=blocks, partial_block=partial)
    output, attn = module(state)
    assert output.shape == partial.shape
    assert torch.allclose(attn.sum(dim=2), torch.ones_like(attn.sum(dim=2)))
