# SLM — train your own 1–2B language model on a single GPU

An end-to-end, dependency-light pipeline for pretraining and fine-tuning a
small language model (SLM) on solo-person hardware: one rented GPU from
RunPod, Vast.ai, Lambda, etc. Pure PyTorch, ~1,500 lines, no Megatron/DeepSpeed.

**Pipeline:** train tokenizer → prepare data → pretrain → fine-tune (SFT) →
chat / export to Hugging Face format.

The model is a Llama-style decoder (RMSNorm, SwiGLU, RoPE, grouped-query
attention), so the final checkpoint exports 1:1 to `LlamaForCausalLM` and
works with `transformers`, vLLM, and llama.cpp conversion.

## Hardware & cost (rough, honest numbers)

Training compute ≈ `6 × params × tokens` FLOPs. Throughput below assumes
bf16 + `torch.compile` at ~35–45% hardware utilization for the 1B model.

| GPU | VRAM | Typical rent | 1B tok/s | 1B @ 20B tokens |
|---|---|---|---|---|
| RTX 3090 | 24 GB | $0.15–0.30/hr | ~5–6k | ~40 days (~$150–250) |
| RTX 4090 | 24 GB | $0.30–0.60/hr | ~9–11k | ~22 days (~$180–300) |
| A100 80GB | 80 GB | $1.00–1.80/hr | ~20–25k | ~10 days (~$280–450) |
| H100 80GB | 80 GB | $2.00–3.00/hr | ~40–50k | ~5 days (~$260–360) |

Guidance:
- **Start with `configs/slm-100m.yaml`** and a small data cap to validate the
  whole pipeline in a few hours before spending real money.
- A 1B model wants ~20B tokens (Chinchilla-optimal); 2B wants ~40B. Training
  the 2B config solo is realistic on one H100 (~2 weeks) or a multi-GPU node
  via `torchrun` — the trainer supports DDP out of the box.
- Rented instances die. Checkpoints are written every 1,000 steps and
  `--resume checkpoints/<run>/latest.pt` picks up exactly where you left off.
  Use spot/interruptible instances to halve costs; keep checkpoints on the
  provider's persistent volume.

## Setup

```bash
pip install -r requirements.txt
# on a 24GB card also: pip install bitsandbytes
```

## 1. Train a tokenizer (32k byte-level BPE)

```bash
python scripts/train_tokenizer.py \
  --dataset HuggingFaceFW/fineweb-edu --name sample-10BT \
  --out tokenizer/tokenizer.json
```

Streams ~2 GB of text and takes minutes. Special tokens: `<|endoftext|>`,
`<|im_start|>`, `<|im_end|>` (ChatML), `<|pad|>`.

## 2. Prepare pretraining data

Streams the dataset, tokenizes, and writes memory-mapped `uint16` shards
(~2 bytes/token on disk — 10B tokens ≈ 20 GB):

```bash
python scripts/prepare_data.py \
  --dataset HuggingFaceFW/fineweb-edu --name sample-10BT \
  --tokenizer tokenizer/tokenizer.json \
  --out data/fineweb-edu \
  --max-train-tokens 10000000000
```

FineWeb-Edu's `sample-10BT` gives ~10B high-quality tokens; for a full 20B-token
run use `--name sample-100BT --max-train-tokens 20000000000`. (Repeating 10B
tokens for 2 epochs also works nearly as well at this scale.) Any HF text
dataset works via `--dataset/--text-key`.

For a pilot run: `--max-train-tokens 300000000` (300M tokens, ~600 MB).

## 3. Pretrain

```bash
# pilot
python scripts/train.py --config configs/slm-100m.yaml

# the real thing (~1.0B params)
python scripts/train.py --config configs/slm-1b.yaml

# resume after an interruption / new spot instance
python scripts/train.py --config configs/slm-1b.yaml \
  --resume checkpoints/slm-1b/latest.pt

# multi-GPU (same config; tokens/step scales with GPU count)
torchrun --standalone --nproc_per_node=4 scripts/train.py --config configs/slm-1b.yaml
```

Configs: `slm-100m.yaml` (pilot), `slm-1b.yaml` (~1.0B), `slm-2b.yaml` (~1.9B).
Each file documents the settings for 24 GB cards (8-bit AdamW + smaller micro
batch). Set `wandb_project` in the config for live loss curves; otherwise
progress (loss, lr, tok/s, tokens seen) prints every 10 steps and validation
loss every 1,000.

Run it under `tmux`/`screen` (or `nohup ... &`) so SSH disconnects don't kill
training.

## 4. Supervised fine-tuning (make it a chatbot)

Fine-tunes on instruction data with loss masked to assistant responses
(ChatML format). Takes ~1–3 hours for a 1B model on alpaca-cleaned:

```bash
python scripts/sft.py \
  --ckpt checkpoints/slm-1b/best.pt \
  --tokenizer tokenizer/tokenizer.json \
  --dataset yahma/alpaca-cleaned \
  --out-dir checkpoints/slm-1b-sft
```

Also accepts messages-style datasets (a `messages` column of
`{role, content}` dicts), e.g. `--dataset HuggingFaceH4/ultrachat_200k --split train_sft`.

## 5. Generate / chat

```bash
# raw completion (base model)
python scripts/generate.py --ckpt checkpoints/slm-1b/best.pt \
  --prompt "The three laws of thermodynamics state"

# interactive chat (after SFT)
python scripts/generate.py --ckpt checkpoints/slm-1b-sft/final.pt --chat
```

## 6. Export to Hugging Face format

```bash
python scripts/export_hf.py --ckpt checkpoints/slm-1b-sft/final.pt \
  --tokenizer tokenizer/tokenizer.json --out export/my-slm-1b
```

The exported folder is a standard Llama repo (safetensors + config +
tokenizer + chat template):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("export/my-slm-1b")
tok = AutoTokenizer.from_pretrained("export/my-slm-1b")
```

Push it to the Hub with `huggingface-cli upload export/my-slm-1b`.

## Repository layout

```
slm/
  model.py       # Llama-style transformer (RoPE, GQA, SwiGLU, KV-cache generate)
  data.py        # memory-mapped token-shard loader
  chat.py        # ChatML formatting + SFT loss masking
  config.py      # YAML config loading
scripts/
  train_tokenizer.py   # 32k BPE tokenizer
  prepare_data.py      # stream + tokenize + shard datasets
  train.py             # pretraining (bf16, grad accum, cosine LR, DDP, resume)
  sft.py               # instruction fine-tuning
  generate.py          # sampling / interactive chat
  export_hf.py         # convert to HF LlamaForCausalLM
configs/
  slm-100m.yaml  slm-1b.yaml  slm-2b.yaml
```

## What to expect

A 1B model trained on 20B tokens of FineWeb-Edu lands around GPT-2-XL /
TinyLlama-early-checkpoint quality: fluent English, decent common-sense
completion, basic instruction following after SFT — not GPT-4. It's an
excellent base for domain fine-tuning and a complete education in how modern
LLMs are actually built.

Tips that matter at this scale:
- Data quality dominates. FineWeb-Edu >> raw CommonCrawl at equal token count.
- Don't undertrain: a 1B model on 2B tokens will be visibly worse than a
  300M model on the same budget's worth of compute.
- Watch validation loss; a healthy 1B/FineWeb-Edu run ends around ~2.3–2.6.
- If loss spikes and doesn't recover, resume from the last good checkpoint
  with a lower `lr`.
