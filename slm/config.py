"""YAML-backed configuration for pretraining and fine-tuning runs."""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import yaml

from .model import ModelConfig


@dataclass
class TrainConfig:
    # data / io
    data_dir: str = "data/fineweb-edu"
    out_dir: str = "checkpoints/run"
    tokenizer_path: str = "tokenizer/tokenizer.json"
    # batch: tokens per optimizer step = micro_batch_size * grad_accum_steps * seq_len (* world_size)
    micro_batch_size: int = 8
    grad_accum_steps: int = 32
    max_steps: int = 38000
    # optimizer
    optimizer: str = "adamw"  # "adamw" | "adamw8bit" (needs bitsandbytes)
    lr: float = 4.0e-4
    min_lr: float = 4.0e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # runtime
    dtype: str = "bfloat16"  # "bfloat16" | "float16" | "float32"
    compile: bool = True
    seed: int = 1337
    # logging / eval / checkpointing
    log_interval: int = 10
    eval_interval: int = 1000
    eval_steps: int = 100
    save_interval: int = 1000
    keep_checkpoints: int = 3
    wandb_project: str = ""  # empty = disabled
    wandb_run_name: str = ""


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _build(cls, data: dict):
    valid = {f.name for f in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return Config(
        model=_build(ModelConfig, raw.get("model", {})),
        train=_build(TrainConfig, raw.get("train", {})),
    )
