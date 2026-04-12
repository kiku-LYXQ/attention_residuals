from __future__ import annotations

import sys

import yaml
import torch
from torch import nn

from attnres_hf_patch.attnres_state import AttnResStats
from train import build_attnres_log_metrics, train


class _FakeModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.vocab_size = vocab_size

    def forward(self, batch, return_depth_weights: bool = False):
        batch_size, seq_len = batch.shape
        logits = torch.zeros(batch_size, seq_len, self.vocab_size, device=batch.device)
        logits[..., 0] = 1.0 + self.bias
        stats = {
            "depth_entropy": 0.25,
            "depth_entropy_values": [0.25, 0.5],
            "attn_depth_weights": [torch.tensor([[[0.2, 0.8]]], dtype=torch.float32)],
            "per_layer": {
                "means": {},
                "entropies": {},
                "weight_max": {0: 0.8},
                "weight_top1_mean": {0: 0.7},
            },
            "attnres_weight_max": 0.8,
            "attnres_weight_top1_mean": 0.7,
            "attnres_weight_max_values": [0.8, 0.8],
            "attnres_weight_top1_mean_values": [0.7, 0.7],
        }
        return logits, stats


class _FakeWandbRun:
    def __init__(self) -> None:
        self.logged: list[tuple[int, dict]] = []
        self.finished = False

    def log(self, metrics, step=None):
        self.logged.append((step, dict(metrics)))

    def finish(self):
        self.finished = True


class _FakeWandb:
    def __init__(self) -> None:
        self.run = _FakeWandbRun()

    def init(self, **kwargs):
        return self.run

    @staticmethod
    def Histogram(values):
        return {"hist": list(values)}


class _FakeSummaryWriter:
    instances: list["_FakeSummaryWriter"] = []

    def __init__(self, log_dir=None):
        self.log_dir = log_dir
        self.scalars: list[tuple[str, float, int]] = []
        self.closed = False
        self.__class__.instances.append(self)

    def add_scalar(self, key, value, step):
        self.scalars.append((key, float(value), int(step)))

    def close(self):
        self.closed = True


def test_attnres_stats_summary_includes_top1_mean():
    stats = AttnResStats()
    weights = torch.tensor(
        [
            [[0.1, 0.9], [0.8, 0.2]],
            [[0.3, 0.7], [0.4, 0.6]],
        ],
        dtype=torch.float32,
    )
    stats.add(weights, layer_idx=3)
    summary = stats.summary()

    expected_top1_mean = weights.max(dim=2).values.mean().item()
    expected_weight_max = weights.max().item()

    assert summary["attnres_weight_top1_mean"] == expected_top1_mean
    assert summary["attnres_weight_max"] == expected_weight_max
    assert summary["attnres_weight_top1_mean_values"] == [expected_top1_mean]
    assert summary["attnres_weight_max_values"] == [expected_weight_max]
    assert summary["per_layer"]["weight_top1_mean"][3] == expected_top1_mean
    assert summary["per_layer"]["weight_max"][3] == expected_weight_max


def test_build_attnres_log_metrics_includes_overall_and_per_layer():
    metrics = build_attnres_log_metrics(
        {
            "attnres_weight_max": 0.95,
            "attnres_weight_top1_mean": 0.72,
            "per_layer": {
                "weight_max": {0: 0.9, 1: 0.8},
                "weight_top1_mean": {0: 0.7, 1: 0.6},
            },
        }
    )

    assert metrics["attnres/weight_max"] == 0.95
    assert metrics["attnres/weight_top1_mean"] == 0.72
    assert metrics["attnres/weight_max_layer_0"] == 0.9
    assert metrics["attnres/weight_top1_mean_layer_1"] == 0.6


def test_train_logs_new_attnres_metrics_to_wandb_and_tensorboard(tmp_path, monkeypatch):
    config = {
        "seed": 0,
        "model": {
            "mode": "baseline",
            "vocab_size": 8,
            "max_len": 4,
            "num_layers": 1,
            "embed_dim": 8,
            "num_heads": 1,
            "mlp_dim": 8,
            "dropout": 0.0,
        },
        "trainer": {
            "steps": 1,
            "seq_len": 4,
            "batch_size": 1,
            "dataset_size": 1,
            "device": "cpu",
            "lr": 1e-3,
            "warmup_ratio": 0.0,
            "max_grad_norm": 1.0,
            "tensorboard_log_dir": str(tmp_path / "tb"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    fake_wandb = _FakeWandb()
    fake_wandb_module = type(sys)("wandb")
    fake_wandb_module.init = fake_wandb.init
    fake_wandb_module.Histogram = fake_wandb.Histogram
    monkeypatch.setattr("train.build_model", lambda *args, **kwargs: _FakeModel(vocab_size=8))
    monkeypatch.setattr("train.SummaryWriter", _FakeSummaryWriter)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb_module)

    train(config_path, wandb_mode="offline", wandb_project="test-project", wandb_run_name="test-run")

    assert fake_wandb.run.logged, "expected at least one wandb log call"
    _, logged_metrics = fake_wandb.run.logged[0]
    assert logged_metrics["attnres/weight_top1_mean"] == 0.7
    assert logged_metrics["attnres/weight_max"] == 0.8
    assert logged_metrics["attnres/weight_top1_mean_layer_0"] == 0.7
    assert logged_metrics["attnres/weight_max_layer_0"] == 0.8

    writer = _FakeSummaryWriter.instances[-1]
    scalar_keys = {key for key, _, _ in writer.scalars}
    assert "attnres/weight_top1_mean" in scalar_keys
    assert "attnres/weight_max" in scalar_keys
    assert "attnres/weight_top1_mean_layer_0" in scalar_keys
    assert "attnres/weight_max_layer_0" in scalar_keys
    assert writer.closed
