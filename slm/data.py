"""Memory-mapped token-shard dataset (nanoGPT-style).

`scripts/prepare_data.py` writes uint16 token streams as `*.bin` shards under
<data_dir>/train and <data_dir>/val. Training samples random windows from the
concatenated stream; documents are separated by the tokenizer's EOS token.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch


class TokenShards:
    def __init__(self, split_dir: str):
        paths = sorted(glob.glob(os.path.join(split_dir, "*.bin")))
        if not paths:
            raise FileNotFoundError(
                f"No .bin shards in {split_dir} - run scripts/prepare_data.py first"
            )
        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        self.sizes = np.array([len(s) for s in self.shards], dtype=np.int64)
        self.total_tokens = int(self.sizes.sum())

    def sample_batch(self, batch_size: int, seq_len: int, rng: np.random.Generator):
        """Random (x, y) windows for pretraining. Shards are weighted by size."""
        xs = np.empty((batch_size, seq_len), dtype=np.int64)
        ys = np.empty((batch_size, seq_len), dtype=np.int64)
        probs = self.sizes / self.sizes.sum()
        for i in range(batch_size):
            s = self.shards[rng.choice(len(self.shards), p=probs)]
            off = int(rng.integers(0, len(s) - seq_len - 1))
            chunk = s[off : off + seq_len + 1].astype(np.int64)
            xs[i] = chunk[:-1]
            ys[i] = chunk[1:]
        return torch.from_numpy(xs), torch.from_numpy(ys)


class PretrainLoader:
    """Infinite loader with async H2D copies when CUDA pinned memory is available."""

    def __init__(self, split_dir: str, batch_size: int, seq_len: int, device: str, seed: int = 0):
        self.shards = TokenShards(split_dir)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.rng = np.random.default_rng(seed)

    def next_batch(self):
        x, y = self.shards.sample_batch(self.batch_size, self.seq_len, self.rng)
        if "cuda" in self.device:
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
