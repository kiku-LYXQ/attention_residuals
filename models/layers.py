from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .norms import RMSNorm


class MultiHeadSelfAttention(nn.Module):
    """Simple MHA with qkv projections and dropout control."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % num_heads == 0, "Embedding dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        # [B, T, heads, head_dim] -> [B, heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.dim)
        output = self.out(attn_output)
        return output, attn_probs.mean(dim=1)


class MLP(nn.Module):
    """Simple two-layer feed-forward with GeLU activation."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class BasePreNormBlock(nn.Module):
    """Baseline PreNorm transformer block used by the baseline mode."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        attn_input = self.norm1(x)
        attn_out, attn_weights = self.attn(attn_input, attn_mask)
        x = x + attn_out
        mlp_input = self.norm2(x)
        x = x + self.mlp(mlp_input)
        return x, attn_weights
