"""Pretrain an SLM on token shards produced by prepare_data.py.

Single GPU:
    python scripts/train.py --config configs/slm-1b.yaml

Multi-GPU (single node):
    torchrun --standalone --nproc_per_node=4 scripts/train.py --config configs/slm-1b.yaml

Resume:
    python scripts/train.py --config configs/slm-1b.yaml --resume checkpoints/slm-1b/latest.pt
"""

import argparse
import glob
import math
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slm.config import load_config
from slm.data import PretrainLoader
from slm.model import SLM

PTDTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def lr_at(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    frac = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * frac)) * (cfg.lr - cfg.min_lr)


def build_optimizer(model, cfg):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if cfg.optimizer == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    return torch.optim.AdamW(
        groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=torch.cuda.is_available()
    )


def prune_checkpoints(out_dir: str, keep: int):
    ckpts = sorted(glob.glob(os.path.join(out_dir, "step_*.pt")))
    for path in ckpts[:-keep]:
        os.remove(path)


@torch.no_grad()
def evaluate(model, loader, steps: int, autocast_ctx) -> float:
    model.eval()
    total = 0.0
    for _ in range(steps):
        x, y = loader.next_batch()
        with autocast_ctx:
            _, loss, _ = model(x, targets=y)
        total += loss.item()
    model.train()
    return total / steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default="", help="path to a checkpoint to resume from")
    p.add_argument("--data-dir", default="", help="override train.data_dir")
    p.add_argument("--out-dir", default="", help="override train.out_dir")
    args = p.parse_args()

    cfg = load_config(args.config)
    tc, mc = cfg.train, cfg.model
    if args.data_dir:
        tc.data_dir = args.data_dir
    if args.out_dir:
        tc.out_dir = args.out_dir

    # --- distributed setup (no-op for a single process) ---
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    master = rank == 0

    if device == "cpu" and tc.dtype != "float32":
        tc.dtype = "float32"
    torch.manual_seed(tc.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokens_per_step = tc.micro_batch_size * tc.grad_accum_steps * mc.seq_len * world_size
    if master:
        print(f"world_size={world_size}  tokens/optimizer-step={tokens_per_step:,}")

    model = SLM(mc).to(device)
    if master:
        print(f"model parameters: {model.num_params():,} "
              f"({model.num_params(non_embedding=True):,} non-embedding)")

    optimizer = build_optimizer(model, tc)
    scaler = torch.amp.GradScaler(enabled=(tc.dtype == "float16"))
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=PTDTYPE[tc.dtype])
        if "cuda" in device and tc.dtype != "float32"
        else torch.autocast(device_type="cpu", enabled=False)
    )

    start_step, tokens_seen = 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        tokens_seen = ckpt.get("tokens_seen", 0)
        if master:
            print(f"resumed from {args.resume} at step {start_step}")

    if tc.compile:
        model = torch.compile(model)
    raw_model = model
    if ddp:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])
        raw_model = model.module

    train_loader = PretrainLoader(
        os.path.join(tc.data_dir, "train"), tc.micro_batch_size, mc.seq_len, device,
        seed=tc.seed + rank,
    )
    val_loader = PretrainLoader(
        os.path.join(tc.data_dir, "val"), tc.micro_batch_size, mc.seq_len, device,
        seed=tc.seed,
    )

    use_wandb = bool(tc.wandb_project) and master
    if use_wandb:
        import wandb

        wandb.init(project=tc.wandb_project, name=tc.wandb_run_name or None,
                   config={"model": vars(mc), "train": vars(tc)}, resume="allow")

    if master:
        os.makedirs(tc.out_dir, exist_ok=True)

    def save_checkpoint(step: int, tag: str):
        if not master:
            return
        state = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "model_config": vars(mc),
            "train_config": vars(tc),
        }
        path = os.path.join(tc.out_dir, f"{tag}.pt")
        torch.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)

    model.train()
    best_val = float("inf")
    t0 = time.time()
    for step in range(start_step, tc.max_steps):
        lr = lr_at(step, tc)
        for g in optimizer.param_groups:
            g["lr"] = lr

        loss_accum = 0.0
        for micro in range(tc.grad_accum_steps):
            x, y = train_loader.next_batch()
            if ddp:
                model.require_backward_grad_sync = micro == tc.grad_accum_steps - 1
            with autocast_ctx:
                _, loss, _ = model(x, targets=y)
                loss = loss / tc.grad_accum_steps
            loss_accum += loss.item()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        tokens_seen += tokens_per_step

        if master and (step + 1) % tc.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            tok_s = tokens_per_step * tc.log_interval / dt
            print(f"step {step+1:>7}/{tc.max_steps} | loss {loss_accum:.4f} | "
                  f"lr {lr:.2e} | gnorm {grad_norm:.2f} | {tok_s:,.0f} tok/s | "
                  f"tokens {tokens_seen/1e9:.2f}B")
            if use_wandb:
                wandb.log({"train/loss": loss_accum, "train/lr": lr,
                           "train/grad_norm": grad_norm.item(), "train/tok_per_s": tok_s,
                           "train/tokens_B": tokens_seen / 1e9}, step=step + 1)

        if (step + 1) % tc.eval_interval == 0 or step + 1 == tc.max_steps:
            val_loss = evaluate(model, val_loader, tc.eval_steps, autocast_ctx)
            if ddp:
                t = torch.tensor(val_loss, device=device)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                val_loss = t.item()
            if master:
                print(f"step {step+1}: val loss {val_loss:.4f}")
                if use_wandb:
                    wandb.log({"val/loss": val_loss}, step=step + 1)
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(step + 1, "best")
            t0 = time.time()

        if (step + 1) % tc.save_interval == 0 or step + 1 == tc.max_steps:
            save_checkpoint(step + 1, f"step_{step+1:07d}")
            save_checkpoint(step + 1, "latest")
            if master:
                prune_checkpoints(tc.out_dir, tc.keep_checkpoints)
            t0 = time.time()

    if master:
        print(f"done. best val loss {best_val:.4f}; checkpoints in {tc.out_dir}")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
