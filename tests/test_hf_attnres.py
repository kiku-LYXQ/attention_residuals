from __future__ import annotations

import torch

from attnres_hf_patch.attnres_adapter import BlockAttnResAdapter, FullAttnResAdapter
from attnres_hf_patch.attnres_state import BlockAttnResState, FullAttnResState
from attnres_hf_patch.config import HfAttnResMode
from attnres_hf_patch.modeling_llama_attnres import AttnResLlamaModelBase
from transformers import LlamaConfig


def _hf_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=32,
    )


def test_full_attnres_forward_shape():
    config = _hf_config()
    model = AttnResLlamaModelBase(config, mode=HfAttnResMode.FULL)
    tokens = torch.randint(0, config.vocab_size, (2, 8))
    outputs = model(input_ids=tokens)
    assert outputs.last_hidden_state.shape == (2, 8, config.hidden_size)


def test_full_attnres_stats():
    config = _hf_config()
    model = AttnResLlamaModelBase(config, mode=HfAttnResMode.FULL)
    tokens = torch.randint(0, config.vocab_size, (2, 8))
    outputs, stats = model(input_ids=tokens, return_depth_weights=True)
    assert outputs.last_hidden_state.shape == (2, 8, config.hidden_size)
    assert isinstance(stats, dict)
    assert stats["depth_mean"] > 0
    assert stats["depth_entropy"] > 0
    assert "attnres_weight_max" in stats
    assert "attnres_weight_top1_mean" in stats
    assert len(stats["attnres_weight_max_values"]) == len(stats["attnres_weight_top1_mean_values"])


def test_full_adapter_softmax():
    config = _hf_config()
    adapter = FullAttnResAdapter(config.hidden_size)
    state = FullAttnResState(history=[torch.randn(2, 8, config.hidden_size)])
    _, weights = adapter.before_attention(state)
    sums = weights.sum(dim=2)
    assert torch.allclose(sums, torch.ones_like(sums))


def test_embedding_in_history():
    config = _hf_config()
    adapter = FullAttnResAdapter(config.hidden_size)
    embed = torch.randn(2, 8, config.hidden_size)
    state = FullAttnResState(history=[embed])
    adapter.append_history(state, embed * 0.5)
    assert state.history[0] is embed


def test_block_state_progression():
    config = _hf_config()
    adapter = BlockAttnResAdapter(config.hidden_size)
    embed = torch.randn(2, 8, config.hidden_size)
    state = BlockAttnResState(blocks=[embed], partial_block=embed.clone(), block_size=1)
    before = state.partial_block.clone()
    attn_out, _ = adapter.before_attention(state)
    adapter.update_partial(state, attn_out)
    assert not torch.allclose(state.partial_block, before)


def test_block_backward():
    config = _hf_config()
    model = AttnResLlamaModelBase(config, mode=HfAttnResMode.BLOCK, block_size=1)
    tokens = torch.randint(0, config.vocab_size, (1, 8))
    outputs = model(input_ids=tokens)
    loss = outputs.last_hidden_state.mean()
    loss.backward()
    assert all(param.grad is None or torch.isfinite(param.grad).all() for param in model.parameters())
