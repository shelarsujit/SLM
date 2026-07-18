"""Export a trained checkpoint to Hugging Face LlamaForCausalLM format.

The architecture (RMSNorm + SwiGLU + rotate-half RoPE + GQA) matches HF Llama
exactly, so this is a pure renaming of weights. The exported folder loads with
transformers, vLLM, llama.cpp converters, etc.

    python scripts/export_hf.py --ckpt checkpoints/slm-1b-sft/final.pt \
        --tokenizer tokenizer/tokenizer.json --out export/my-slm-1b
"""

import argparse
import json
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slm.model import ModelConfig


def convert_state_dict(sd: dict, cfg: ModelConfig) -> dict:
    out = {"model.embed_tokens.weight": sd["tok_embeddings.weight"]}
    for i in range(cfg.n_layers):
        s, d = f"layers.{i}", f"model.layers.{i}"
        out[f"{d}.self_attn.q_proj.weight"] = sd[f"{s}.attn.wq.weight"]
        out[f"{d}.self_attn.k_proj.weight"] = sd[f"{s}.attn.wk.weight"]
        out[f"{d}.self_attn.v_proj.weight"] = sd[f"{s}.attn.wv.weight"]
        out[f"{d}.self_attn.o_proj.weight"] = sd[f"{s}.attn.wo.weight"]
        out[f"{d}.mlp.gate_proj.weight"] = sd[f"{s}.mlp.w_gate.weight"]
        out[f"{d}.mlp.up_proj.weight"] = sd[f"{s}.mlp.w_up.weight"]
        out[f"{d}.mlp.down_proj.weight"] = sd[f"{s}.mlp.w_down.weight"]
        out[f"{d}.input_layernorm.weight"] = sd[f"{s}.attn_norm.weight"]
        out[f"{d}.post_attention_layernorm.weight"] = sd[f"{s}.mlp_norm.weight"]
    out["model.norm.weight"] = sd["norm_f.weight"]
    if not cfg.tie_embeddings:
        out["lm_head.weight"] = sd["lm_head.weight"]
    return out


CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}

    dtype = getattr(torch, args.dtype)
    hf_sd = {k: v.to(dtype).contiguous() for k, v in convert_state_dict(sd, cfg).items()}

    os.makedirs(args.out, exist_ok=True)

    from safetensors.torch import save_file

    save_file(hf_sd, os.path.join(args.out, "model.safetensors"),
              metadata={"format": "pt"})

    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.d_model,
        "intermediate_size": cfg.ffn_hidden,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "max_position_embeddings": cfg.seq_len,
        "rope_theta": cfg.rope_theta,
        "rms_norm_eps": cfg.norm_eps,
        "tie_word_embeddings": cfg.tie_embeddings,
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "torch_dtype": args.dtype,
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    shutil.copy(args.tokenizer, os.path.join(args.out, "tokenizer.json"))
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>",
        "model_max_length": cfg.seq_len,
        "chat_template": CHAT_TEMPLATE,
    }
    with open(os.path.join(args.out, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    with open(os.path.join(args.out, "generation_config.json"), "w") as f:
        json.dump({"bos_token_id": 0, "eos_token_id": 0, "do_sample": True,
                   "temperature": 0.8, "top_k": 50}, f, indent=2)

    print(f"exported to {args.out}/")
    print("load with: AutoModelForCausalLM.from_pretrained("
          f"'{args.out}'), AutoTokenizer.from_pretrained('{args.out}')")


if __name__ == "__main__":
    main()
