"""Train a 32k byte-level BPE tokenizer on a streamed sample of the corpus.

Example:
    python scripts/train_tokenizer.py \
        --dataset HuggingFaceFW/fineweb-edu --name sample-10BT \
        --max-chars 2000000000 --out tokenizer/tokenizer.json
"""

import argparse
import os

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]


def iter_texts(dataset: str, name: str | None, text_key: str, max_chars: int):
    ds = load_dataset(dataset, name=name, split="train", streaming=True)
    seen = 0
    for row in ds:
        text = row[text_key]
        if not text:
            continue
        yield text
        seen += len(text)
        if seen >= max_chars:
            return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--name", default="sample-10BT", help="dataset config name ('' for none)")
    p.add_argument("--text-key", default="text")
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--max-chars", type=int, default=2_000_000_000,
                   help="stop after this many characters (~2GB of text is plenty)")
    p.add_argument("--out", default="tokenizer/tokenizer.json")
    args = p.parse_args()

    tokenizer = Tokenizer(models.BPE(byte_fallback=False))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    name = args.name or None
    print(f"Streaming up to {args.max_chars/1e9:.1f}B chars from {args.dataset} ...")
    tokenizer.train_from_iterator(
        iter_texts(args.dataset, name, args.text_key, args.max_chars), trainer=trainer
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tokenizer.save(args.out)
    print(f"Saved tokenizer ({tokenizer.get_vocab_size()} tokens) to {args.out}")


if __name__ == "__main__":
    main()
