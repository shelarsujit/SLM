"""ChatML-style prompt formatting shared by SFT and inference.

Format:
    <|im_start|>system\n...<|im_end|>\n
    <|im_start|>user\n...<|im_end|>\n
    <|im_start|>assistant\n...<|im_end|>\n
"""

from __future__ import annotations

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EOT = "<|endoftext|>"
DEFAULT_SYSTEM = "You are a helpful assistant."


def render_turn(role: str, content: str) -> str:
    return f"{IM_START}{role}\n{content}{IM_END}\n"


def render_messages(messages: list[dict], add_generation_prompt: bool = False) -> str:
    out = "".join(render_turn(m["role"], m["content"]) for m in messages)
    if add_generation_prompt:
        out += f"{IM_START}assistant\n"
    return out


def encode_for_sft(tokenizer, messages: list[dict], max_len: int):
    """Tokenize a conversation, masking loss to assistant content only.

    Returns (input_ids, labels) as python lists; labels are -100 outside
    assistant responses. Returns None if nothing trainable survives max_len.
    """
    input_ids: list[int] = []
    labels: list[int] = []
    for m in messages:
        prefix = f"{IM_START}{m['role']}\n"
        body = f"{m['content']}{IM_END}\n"
        prefix_ids = tokenizer.encode(prefix).ids
        body_ids = tokenizer.encode(body).ids
        input_ids.extend(prefix_ids + body_ids)
        if m["role"] == "assistant":
            labels.extend([-100] * len(prefix_ids) + body_ids)
        else:
            labels.extend([-100] * (len(prefix_ids) + len(body_ids)))
    eot_id = tokenizer.token_to_id(EOT)
    input_ids.append(eot_id)
    labels.append(eot_id)

    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    if all(l == -100 for l in labels):
        return None
    return input_ids, labels
