from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset


class TokenSequenceDataset(Dataset):
    """Sequence dataset for pretrained tokenizer output.

    Each example is Vector[seq_len+1], where
    - input_ids = example[:-1]
    - targets   = example[1:]
    """

    def __init__(self, tokens: torch.Tensor, seq_len: int) -> None:
        assert tokens.dim() == 1, "Tokens must be 1D"
        self.seq_len = seq_len
        self.tokens = tokens

    def __len__(self) -> int:
        usable_tokens = self.tokens.numel() - self.seq_len
        return max(1, usable_tokens)

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        if chunk.numel() < self.seq_len + 1:
            pad = torch.full((self.seq_len + 1 - chunk.numel(),), self.tokens[-1].item(), dtype=self.tokens.dtype)
            chunk = torch.cat([chunk, pad], dim=0)
        return chunk
