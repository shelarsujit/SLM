"""Supervised fine-tuning (instruction tuning) on a chat dataset.

Loss is masked to assistant responses using the ChatML template in slm/chat.py.
Works with alpaca-style datasets (instruction/input/output columns) or
messages-style datasets (a "messages" list of {role, content} dicts).

Example:
    python scripts/sft.py --ckpt checkpoints/slm-1b/best.pt \
        --tokenizer tokenizer/tokenizer.json \
        --dataset yahma/alpaca-cleaned \
        --out-dir checkpoints/slm-1b-sft
"""

import argparse
import math
import os
import sys
import time

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slm.chat import DEFAULT_SYSTEM, encode_for_sft
from slm.model import SLM, ModelConfig

PTDTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def to_messages(row: dict) -> list[dict] | None:
    if "messages" in row:
        msgs = [m for m in row["messages"] if m.get("role") in ("system", "user", "assistant")]
        return msgs if any(m["role"] == "assistant" for m in msgs) else None
    if "instruction" in row and "output" in row:
        user = row["instruction"]
        if row.get("input"):
            user += "\n\n" + row["input"]
        return [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["output"]},
        ]
    return None


def collate(batch: list[tuple[list[int], list[int]]], pad_id: int, device: str):
    max_len = max(len(ids) for ids, _ in batch)
    x = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    y = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(batch):
        # next-token shift: predict ids[t+1] at position t
        x[i, : len(ids) - 1] = torch.tensor(ids[:-1])
        y[i, : len(labels) - 1] = torch.tensor(labels[1:])
    return x.to(device), y.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="pretrained checkpoint (.pt)")
    p.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    p.add_argument("--dataset", default="yahma/alpaca-cleaned")
    p.add_argument("--name", default="", help="dataset config name")
    p.add_argument("--split", default="train")
    p.add_argument("--out-dir", default="checkpoints/sft")
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--min-lr", type=float, default=2e-6)
    p.add_argument("--warmup-frac", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mc = ModelConfig(**ckpt["model_config"])
    model = SLM(mc).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {args.ckpt} ({model.num_params():,} params)")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    pad_id = tokenizer.token_to_id("<|pad|>")

    print(f"tokenizing {args.dataset} ...")
    ds = load_dataset(args.dataset, name=args.name or None, split=args.split)
    examples = []
    for row in ds:
        msgs = to_messages(row)
        if not msgs:
            continue
        enc = encode_for_sft(tokenizer, msgs, min(args.max_len, mc.seq_len))
        if enc:
            examples.append(enc)
    print(f"{len(examples):,} usable examples")

    steps_per_epoch = math.ceil(len(examples) / (args.batch_size * args.grad_accum))
    max_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(max_steps * args.warmup_frac))

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        frac = (step - warmup) / max(1, max_steps - warmup)
        return args.min_lr + 0.5 * (1 + math.cos(math.pi * frac)) * (args.lr - args.min_lr)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=PTDTYPE[args.dtype]) \
        if device == "cuda" else torch.autocast(device_type="cpu", enabled=False)
    scaler = torch.amp.GradScaler(enabled=(args.dtype == "float16" and device == "cuda"))

    os.makedirs(args.out_dir, exist_ok=True)
    g = torch.Generator().manual_seed(args.seed)
    model.train()
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        order = torch.randperm(len(examples), generator=g).tolist()
        micro_batches = [
            [examples[j] for j in order[i : i + args.batch_size]]
            for i in range(0, len(order), args.batch_size)
        ]
        for i in range(0, len(micro_batches), args.grad_accum):
            group = micro_batches[i : i + args.grad_accum]
            lr = lr_at(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            loss_accum = 0.0
            for mb in group:
                x, y = collate(mb, pad_id, device)
                with autocast_ctx:
                    _, loss, _ = model(x, targets=y)
                    loss = loss / len(group)
                loss_accum += loss.item()
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_interval == 0:
                dt = time.time() - t0
                t0 = time.time()
                print(f"epoch {epoch+1} | step {step}/{max_steps} | "
                      f"loss {loss_accum:.4f} | lr {lr:.2e} | "
                      f"{args.log_interval/dt:.2f} it/s")

        path = os.path.join(args.out_dir, f"sft_epoch{epoch+1}.pt")
        torch.save({
            "model": model.state_dict(),
            "step": step,
            "model_config": vars(mc),
            "sft_args": vars(args),
        }, path)
        print(f"saved {path}")

    final = os.path.join(args.out_dir, "final.pt")
    torch.save({"model": model.state_dict(), "step": step,
                "model_config": vars(mc), "sft_args": vars(args)}, final)
    print(f"done. final model at {final}")


if __name__ == "__main__":
    main()
