from __future__ import annotations

import argparse
import pathlib
import logging
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from attnres_repro.train import ToyTokenDataset, build_model, load_config
from attnres_repro.utils.logging_utils import configure_logging
from attnres_repro.utils.metrics import depth_attention_mean, grad_norms, hidden_norms
from attnres_repro.utils.seed import set_seed


def evaluate(config_path: pathlib.Path) -> None:
    config = load_config(config_path)
    set_seed(config.get("seed", 42))
    configure_logging()
    device = torch.device(config.get("device", "cpu"))
    model = build_model(config["model"]).to(device)
    dataset = ToyTokenDataset(
        seq_len=config["trainer"]["seq_len"],
        vocab_size=config["model"]["vocab_size"],
        size=4,
    )
    loader = DataLoader(dataset, batch_size=2)
    batch = next(iter(loader)).to(device)
    logits, stats = model(batch, return_depth_weights=True)
    shift_logits = logits[:, :-1, :].contiguous()
    targets = batch[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), targets.view(-1)
    )
    loss.backward()
    metrics: Dict[str, Optional[float]] = {}
    if stats and "hidden_states" in stats:
        metrics["hidden_norm_mean"] = hidden_norms(stats["hidden_states"]).get("layer_1")
    metrics["grad_norm_mean"] = sum(grad_norms(model).values())
    if stats and "attn_depth_weights" in stats:
        metrics["depth_mean"] = depth_attention_mean(stats["attn_depth_weights"]).mean().item()
    logging.info("Evaluation loss=%.4f metrics=%s", loss.item(), metrics)


def main() -> None:
    parser = argparse.ArgumentParser("attnres-eval")
    parser.add_argument("config", type=pathlib.Path, help="Path to YAML configuration")
    args = parser.parse_args()
    evaluate(args.config)


if __name__ == "__main__":
    main()
