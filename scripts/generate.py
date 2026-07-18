"""Sample from a trained checkpoint.

Raw completion:
    python scripts/generate.py --ckpt checkpoints/slm-1b/best.pt \
        --prompt "The history of aviation began"

Chat mode (after SFT):
    python scripts/generate.py --ckpt checkpoints/slm-1b-sft/final.pt --chat
"""

import argparse
import os
import sys

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slm.chat import EOT, IM_END, render_messages
from slm.model import SLM, ModelConfig


def load_model(ckpt_path: str, device: str) -> SLM:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SLM(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def sample(model, tokenizer, prompt: str, args, device: str, stop_ids: set[int]) -> str:
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        y = model.generate(
            x, args.max_new_tokens, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, stop_ids=stop_ids,
        )
    out_ids = y[0, len(ids):].tolist()
    out_ids = [t for t in out_ids if t not in stop_ids]
    return tokenizer.decode(out_ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--chat", action="store_true", help="interactive ChatML chat loop")
    p.add_argument("--system", default="You are a helpful assistant.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=-1, help="-1 = random")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.seed >= 0:
        torch.manual_seed(args.seed)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    model = load_model(args.ckpt, device)
    stop_ids = {tokenizer.token_to_id(EOT), tokenizer.token_to_id(IM_END)}
    stop_ids.discard(None)

    if not args.chat:
        print(args.prompt + sample(model, tokenizer, args.prompt, args, device, stop_ids))
        return

    messages = [{"role": "system", "content": args.system}]
    print("chat mode - type 'exit' to quit")
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user.lower() in ("exit", "quit"):
            break
        messages.append({"role": "user", "content": user})
        prompt = render_messages(messages, add_generation_prompt=True)
        reply = sample(model, tokenizer, prompt, args, device, stop_ids).strip()
        print(f"slm> {reply}")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
