"""Stream a HF dataset, tokenize it, and write uint16 .bin shards for training.

Documents are concatenated with an <|endoftext|> separator. Output layout:
    <out>/train/shard_0000.bin, ...
    <out>/val/shard_0000.bin

Example (~10B tokens of FineWeb-Edu; use --max-train-tokens to cap):
    python scripts/prepare_data.py \
        --dataset HuggingFaceFW/fineweb-edu --name sample-10BT \
        --tokenizer tokenizer/tokenizer.json --out data/fineweb-edu \
        --max-train-tokens 10000000000
"""

import argparse
import os

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


class ShardWriter:
    def __init__(self, out_dir: str, shard_tokens: int):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.shard_tokens = shard_tokens
        self.buf = np.empty(shard_tokens, dtype=np.uint16)
        self.fill = 0
        self.shard_idx = 0
        self.total = 0

    def add(self, tokens: list[int]):
        arr = np.asarray(tokens, dtype=np.uint16)
        self.total += len(arr)
        while len(arr) > 0:
            space = self.shard_tokens - self.fill
            take = min(space, len(arr))
            self.buf[self.fill : self.fill + take] = arr[:take]
            self.fill += take
            arr = arr[take:]
            if self.fill == self.shard_tokens:
                self._flush()

    def _flush(self):
        if self.fill == 0:
            return
        path = os.path.join(self.out_dir, f"shard_{self.shard_idx:04d}.bin")
        self.buf[: self.fill].tofile(path)
        self.shard_idx += 1
        self.fill = 0

    def close(self):
        self._flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--name", default="sample-10BT", help="dataset config name ('' for none)")
    p.add_argument("--text-key", default="text")
    p.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    p.add_argument("--out", default="data/fineweb-edu")
    p.add_argument("--max-train-tokens", type=int, default=10_000_000_000)
    p.add_argument("--val-tokens", type=int, default=5_000_000)
    p.add_argument("--shard-tokens", type=int, default=100_000_000, help="tokens per shard (100M = 200MB)")
    p.add_argument("--batch-docs", type=int, default=512, help="docs per encode_batch call")
    args = p.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    eot = tokenizer.token_to_id("<|endoftext|>")
    assert eot is not None, "tokenizer must define <|endoftext|>"
    assert tokenizer.get_vocab_size() <= 65536, "uint16 storage requires vocab <= 65536"

    ds = load_dataset(args.dataset, name=args.name or None, split="train", streaming=True)

    val_writer = ShardWriter(os.path.join(args.out, "val"), args.shard_tokens)
    train_writer = ShardWriter(os.path.join(args.out, "train"), args.shard_tokens)

    pbar = tqdm(total=args.max_train_tokens + args.val_tokens, unit="tok", unit_scale=True)
    batch: list[str] = []

    def process(texts: list[str]) -> bool:
        """Returns False when the train budget is exhausted."""
        encs = tokenizer.encode_batch(texts)
        for enc in encs:
            tokens = enc.ids + [eot]
            # Fill val first so it comes from held-out leading documents.
            writer = val_writer if val_writer.total < args.val_tokens else train_writer
            writer.add(tokens)
            pbar.update(len(tokens))
            if train_writer.total >= args.max_train_tokens:
                return False
        return True

    done = False
    for row in ds:
        text = row[args.text_key]
        if text:
            batch.append(text)
        if len(batch) >= args.batch_docs:
            if not process(batch):
                done = True
                break
            batch = []
    if not done and batch:
        process(batch)

    val_writer.close()
    train_writer.close()
    pbar.close()
    print(f"train tokens: {train_writer.total:,}  ({train_writer.shard_idx} shards)")
    print(f"val tokens:   {val_writer.total:,}")


if __name__ == "__main__":
    main()
