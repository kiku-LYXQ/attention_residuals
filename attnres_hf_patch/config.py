from __future__ import annotations

from enum import Enum


class HfAttnResMode(str, Enum):
    BASELINE = "baseline_llama"
    FULL = "full_attnres_llama"
    BLOCK = "block_attnres_llama"
