"""Llama-style decoder-only transformer, sized for 0.1B-2B params.

Architecture: RMSNorm (pre-norm), SwiGLU MLP, rotary embeddings (NeoX
rotate-half convention, so weights map 1:1 onto HF LlamaForCausalLM),
grouped-query attention, optional tied embeddings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    d_model: int = 2048
    n_layers: int = 22
    n_heads: int = 32
    n_kv_heads: int = 4
    ffn_hidden: int = 5632
    seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    grad_checkpoint: bool = False

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(head_dim: int, seq_len: int, theta: float, device=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # q, k: (B, H, T, head_dim); cos, sin: (T, head_dim)
    cos = cos[None, None, :, :].to(q.dtype)
    sin = sin[None, None, :, :].to(q.dtype)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        kv_dim = cfg.n_kv_heads * cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, past_kv=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if past_kv is not None:
            pk, pv = past_kv
            if pk is not None:
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
        new_kv = (k, v) if past_kv is not None else None

        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Full causal mask during training/prefill; a single-token decode
        # step attends to everything in the cache.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(y), new_kv


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, past_kv=None):
        a, new_kv = self.attn(self.attn_norm(x), cos, sin, past_kv)
        x = x + a
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_kv


class SLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        cos, sin = build_rope_cache(cfg.head_dim, cfg.seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scaled init for residual-stream output projections (GPT-2 style).
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if self.cfg.tie_embeddings:
            pass  # lm_head shares storage with tok_embeddings; already counted once
        if non_embedding:
            n -= self.tok_embeddings.weight.numel()
        return n

    def forward(self, idx, targets=None, past_kvs=None, start_pos: int = 0):
        B, T = idx.shape
        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]

        x = self.tok_embeddings(idx)
        new_kvs = [] if past_kvs is not None else None
        for i, layer in enumerate(self.layers):
            past = past_kvs[i] if past_kvs is not None else None
            if self.cfg.grad_checkpoint and self.training:
                x, kv = checkpoint(layer, x, cos, sin, past, use_reentrant=False)
            else:
                x, kv = layer(x, cos, sin, past)
            if new_kvs is not None:
                new_kvs.append(kv)
        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                targets.view(-1),
                ignore_index=-100,
            )
        else:
            logits = self.lm_head(x[:, [-1], :])  # only need the last position
            loss = None
        return logits, loss, new_kvs

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = None,
        stop_ids: set[int] | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling with a KV cache. idx: (1, T) prompt."""
        self.eval()
        past_kvs = [(None, None) for _ in self.layers]
        # Prefill on the full prompt (cropped to context window).
        idx = idx[:, -self.cfg.seq_len :]
        pos = 0
        logits, _, past_kvs = self(idx, past_kvs=past_kvs, start_pos=pos)
        pos = idx.size(1)

        out = idx
        for _ in range(max_new_tokens):
            if pos >= self.cfg.seq_len:
                break
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits[logits < kth] = -float("inf")
                if top_p is not None:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cum = probs.cumsum(dim=-1)
                    mask = cum - probs > top_p
                    sorted_logits[mask] = -float("inf")
                    logits = torch.full_like(logits, -float("inf")).scatter(
                        -1, sorted_idx, sorted_logits
                    )
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_id], dim=1)
            if stop_ids and next_id.item() in stop_ids:
                break
            logits, _, past_kvs = self(next_id, past_kvs=past_kvs, start_pos=pos)
            pos += 1
        return out
