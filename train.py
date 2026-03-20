from __future__ import annotations

import argparse
import logging
import pathlib
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel

from models.transformer import (
    BaselineTransformer,
    BlockAttnResTransformer,
    FullAttnResTransformer,
)
from utils.logging_utils import configure_logging
from utils.metrics import depth_attention_mean
from utils.seed import set_seed

from attnres_hf_patch.config import HfAttnResMode
from attnres_hf_patch.modeling_llama_attnres import AttnResLlamaModelBase, HfAttnResCausalLM


class ToyTokenDataset(Dataset):
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        size: int,
        text_file: pathlib.Path | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.size = size
        if text_file and text_file.exists():
            text = text_file.read_text().strip()
            tokens = [ord(c) % vocab_size for c in text]
            self.data = torch.tensor(tokens, dtype=torch.long)
        else:
            self.data = torch.randint(0, vocab_size, (size * seq_len,), dtype=torch.long)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.seq_len
        end = start + self.seq_len
        return self.data[start:end]


def load_config(path: pathlib.Path) -> Dict:
    with path.open() as fp:
        return yaml.safe_load(fp)


def _build_llama_config(model_config: Dict) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=model_config["vocab_size"],
        hidden_size=model_config["embed_dim"],
        intermediate_size=model_config["mlp_dim"],
        num_attention_heads=model_config["num_heads"],
        num_hidden_layers=model_config["num_layers"],
        max_position_embeddings=model_config["max_len"],
        rms_norm_eps=1e-6,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=0,
    )


def build_model(model_config: Dict, model_type: str | None = None) -> torch.nn.Module:
    mode = model_type or model_config["mode"]
    common = dict(
        vocab_size=model_config["vocab_size"],
        max_len=model_config["max_len"],
        num_layers=model_config["num_layers"],
        embed_dim=model_config["embed_dim"],
        num_heads=model_config["num_heads"],
        mlp_dim=model_config["mlp_dim"],
        dropout=model_config.get("dropout", 0.1),
    )
    match mode:
        case "baseline":
            return BaselineTransformer(**common)
        case "full_attnres":
            return FullAttnResTransformer(**common)
        case "block_attnres":
            return BlockAttnResTransformer(block_size=model_config.get("block_size", 2), **common)
        case "baseline_llama":
            return LlamaForCausalLM(_build_llama_config(model_config))
        case "full_attnres_llama":
            base = AttnResLlamaModelBase(
                _build_llama_config(model_config),
                mode=HfAttnResMode.FULL,
                block_size=model_config.get("block_size", 2),
            )
            return HfAttnResCausalLM(base, vocab_size=model_config["vocab_size"])
        case "block_attnres_llama":
            base = AttnResLlamaModelBase(
                _build_llama_config(model_config),
                mode=HfAttnResMode.BLOCK,
                block_size=model_config.get("block_size", 2),
            )
            return HfAttnResCausalLM(base, vocab_size=model_config["vocab_size"])
        case _:
            raise ValueError(f"Unknown model mode {mode}")


def train(config_path: pathlib.Path, model_type: str | None = None) -> None:
    config = load_config(config_path)
    set_seed(config.get("seed", 42))
    configure_logging(config.get("log_level", logging.INFO))
    device = torch.device(config.get("device", "cpu"))
    model = build_model(config["model"], model_type).to(device)
    lr = float(config.get("lr", 5e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataset_file = config.get("dataset", {}).get("text_file")
    dataset = ToyTokenDataset(
        seq_len=config["trainer"]["seq_len"],
        vocab_size=config["model"]["vocab_size"],
        size=config["trainer"]["dataset_size"],
        text_file=pathlib.Path(dataset_file) if dataset_file else None,
    )
    loader = DataLoader(dataset, batch_size=config["trainer"]["batch_size"], shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    logging.info("Starting training for %d steps", config["trainer"]["steps"])
    total_steps = config["trainer"]["steps"]
    loader_iter = iter(loader)
    for step in range(1, total_steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        batch = batch.to(device)
        optimizer.zero_grad()
        try:
            outputs = model(batch, return_depth_weights=True)
        except TypeError:
            outputs = model(batch)
        if isinstance(outputs, tuple):
            logits, stats = outputs
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
            stats = {}
        elif hasattr(outputs, "last_hidden_state"):
            logits = outputs.last_hidden_state
            stats = {}
        else:
            logits = outputs
            stats = {}
        shift_logits = logits[:, :-1, :].contiguous()
        targets = batch[:, 1:].contiguous()
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        depth_mean = None
        if stats and "attn_depth_weights" in stats:
            depth_mean = depth_attention_mean(stats["attn_depth_weights"]).mean().item()
        logging.info(
            "step=%d loss=%.4f depth_mean=%.4f",
            step,
            loss.item(),
            depth_mean if depth_mean is not None else 0.0,
        )


def main() -> None:
    parser = argparse.ArgumentParser("attnres-train")
    parser.add_argument("config", type=pathlib.Path, help="Path to YAML configuration")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=[
            "baseline",
            "full_attnres",
            "block_attnres",
            "baseline_llama",
            "full_attnres_llama",
            "block_attnres_llama",
        ],
        default=None,
        help="Model type override",
    )
    args = parser.parse_args()
    train(args.config, args.model_type)


if __name__ == "__main__":
    main()
