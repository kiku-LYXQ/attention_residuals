from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from .attnres import BlockDepthAttnResidual, BlockState, DepthAttnResidual
from .layers import BasePreNormBlock, MultiHeadSelfAttention


class BaseTransformer(nn.Module):
    """Shared utilities for all attention residual variants."""

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(max_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim

    def _get_positional(self, seq_len: int) -> Tensor:
        return self.pos_embed[:seq_len]

    def _forward(self, x: Tensor, **kwargs) -> Tuple[Tensor, Dict[str, Tensor]]:
        raise NotImplementedError

    def forward(
        self,
        input_ids: Tensor,
        return_depth_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        bsz, seq_len = input_ids.shape
        assert seq_len <= self.max_len, "Sequence length exceeds max_len"
        token_embeddings = self.token_embed(input_ids)
        pos_embeddings = self._get_positional(seq_len)
        x = token_embeddings + pos_embeddings
        x = self.dropout(x)
        final, stats = self._forward(x)
        logits = self.output_proj(final)
        if return_depth_weights:
            return logits, stats
        return logits, None


class BaselineTransformer(BaseTransformer):
    """PreNorm transformer with additive residuals (baseline)."""

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(vocab_size, max_len, num_layers, embed_dim, num_heads, mlp_dim, dropout)
        self.layers = nn.ModuleList([
            BasePreNormBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

    def _forward(self, x: Tensor, **kwargs) -> Tuple[Tensor, Dict[str, Tensor]]:
        attn_weights = []
        hidden_states: Dict[str, Tensor] = {}
        for idx, layer in enumerate(self.layers, start=1):
            x, attn = layer(x)
            attn_weights.append(attn)
            hidden_states[f"layer_{idx}"] = x
        stats = {
            "baseline_attn": torch.stack(attn_weights) if attn_weights else torch.empty(0),
            "hidden_states": hidden_states,
        }
        return x, stats


class FullAttnResTransformer(BaseTransformer):
    """Transformer that applies full depth attention residuals before attention and MLP."""

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(vocab_size, max_len, num_layers, embed_dim, num_heads, mlp_dim, dropout)
        self.layers = nn.ModuleList([
            BasePreNormBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.depth_res_attn = DepthAttnResidual(embed_dim)
        self.depth_res_mlp = DepthAttnResidual(embed_dim)

    def _forward(self, hidden: Tensor, **kwargs) -> Tuple[Tensor, Dict[str, Tensor]]:
        history: List[Tensor] = [hidden]
        attn_depth_weights: List[Tensor] = []
        mlp_depth_weights: List[Tensor] = []
        hidden_states: Dict[str, Tensor] = {}
        x = hidden
        for idx, layer in enumerate(self.layers, start=1):
            attn_input, attn_weights = self.depth_res_attn(history)
            attn_input = layer.norm1(attn_input)
            attn_out, _ = layer.attn(attn_input)
            x = attn_input + attn_out
            history.append(x)
            mlp_input, mlp_weights = self.depth_res_mlp(history)
            mlp_input = layer.norm2(mlp_input)
            mlp_out = layer.mlp(mlp_input)
            x = x + mlp_out
            history.append(x)
            attn_depth_weights.append(attn_weights)
            mlp_depth_weights.append(mlp_weights)
            hidden_states[f"layer_{idx}"] = x
        stats = {
            "attn_depth_weights": attn_depth_weights,
            "mlp_depth_weights": mlp_depth_weights,
            "hidden_states": hidden_states,
        }
        return x, stats


class BlockAttnResTransformer(BaseTransformer):
    """Transformer that applies block attention residual machinery."""

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(vocab_size, max_len, num_layers, embed_dim, num_heads, mlp_dim, dropout)
        self.layers = nn.ModuleList([
            BasePreNormBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.block_res = BlockDepthAttnResidual(embed_dim)
        self.block_size = block_size

    def _forward(self, hidden: Tensor, **kwargs) -> Tuple[Tensor, Dict[str, Tensor]]:
        state = BlockState(blocks=[hidden], partial_block=hidden)
        attn_weights: List[Tensor] = []
        mlp_weights: List[Tensor] = []
        hidden_states: Dict[str, Tensor] = {}
        x = hidden
        for idx, layer in enumerate(self.layers, start=1):
            attn_input, attn_alpha = self.block_res(state)
            attn_input = layer.norm1(attn_input)
            attn_out, _ = layer.attn(attn_input)
            x = attn_input + attn_out
            state.partial_block = state.partial_block + attn_out
            attn_weights.append(attn_alpha)
            mlp_input, mlp_alpha = self.block_res(state)
            mlp_input = layer.norm2(mlp_input)
            mlp_out = layer.mlp(mlp_input)
            x = x + mlp_out
            state.partial_block = state.partial_block + mlp_out
            mlp_weights.append(mlp_alpha)
            state.layer_counter += 1
            if state.layer_counter % self.block_size == 0:
                state.blocks.append(state.partial_block.clone())
                state.partial_block = state.partial_block.clone()
            hidden_states[f"layer_{idx}"] = state.partial_block
        stats = {
            "attn_depth_weights": attn_weights,
            "mlp_depth_weights": mlp_weights,
            "hidden_states": hidden_states,
        }
        return state.partial_block, stats
