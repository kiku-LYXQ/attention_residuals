from __future__ import annotations

import torch

from models.transformer import (
    BaselineTransformer,
    BlockAttnResTransformer,
    FullAttnResTransformer,
)


def _make_model(model_cls, **kwargs):
    return model_cls(**kwargs)


def _run_shape_test(model, batch_size: int = 2, seq_len: int = 8):
    tokens = torch.randint(0, model.output_proj.out_features, (batch_size, seq_len))
    logits, stats = model(tokens)
    assert logits.shape == (batch_size, seq_len, model.output_proj.out_features)


def test_baseline_shape():
    model = _make_model(
        BaselineTransformer,
        vocab_size=32,
        max_len=16,
        num_layers=2,
        embed_dim=32,
        num_heads=4,
        mlp_dim=64,
        dropout=0.0,
    )
    _run_shape_test(model)


def test_full_attnres_shape():
    model = _make_model(
        FullAttnResTransformer,
        vocab_size=32,
        max_len=16,
        num_layers=2,
        embed_dim=32,
        num_heads=4,
        mlp_dim=64,
        dropout=0.0,
    )
    _run_shape_test(model)


def test_block_attnres_shape():
    model = _make_model(
        BlockAttnResTransformer,
        vocab_size=32,
        max_len=16,
        num_layers=2,
        embed_dim=32,
        num_heads=4,
        mlp_dim=64,
        block_size=1,
        dropout=0.0,
    )
    _run_shape_test(model)
