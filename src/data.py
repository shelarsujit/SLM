import torch
from torch.utils.data import IterableDataset, get_worker_info
from datasets import load_dataset
import numpy as np

class PackedStreamingDataset(IterableDataset):
    def __init__(self, tokenizer, dataset_name, subset=None, split="train", 
                 block_size=2048, buffer_size=10000, seed=42, max_tokens=None):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.seed = seed
        self.max_tokens = max_tokens
        
        # Internal state for resumption
        self.samples_yielded = 0

    def __iter__(self):
        worker_info = get_worker_info()
        # Sharding logic: If multiple workers, we must split the stream
        # Note: True sharding on web streams is complex. 
        # A robust strategy is to shard by file if the dataset supports it, 
        # or use different seeds for the shuffle buffer per worker.
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        # Load the dataset in streaming mode
        ds = load_dataset(self.dataset_name, self.subset, split=self.split, streaming=True)
        
        # Shard the dataset if supported by HF (splits by underlying files)
        # This is critical for multi-GPU/multi-worker correctness
        try:
            ds = ds.shard(num_shards=num_workers, index=worker_id)
        except Exception:
            # Fallback: simple skip (less efficient)
            ds = ds.skip(worker_id) 

        # Shuffle buffer
        rng = np.random.default_rng(self.seed + worker_id)
        ds = ds.shuffle(seed=self.seed + worker_id, buffer_size=self.buffer_size)
        
        iterator = iter(ds)
        
        # Packing Logic
        token_buffer =
        
        for item in iterator:
            text = item.get("text", "")
            if not text: 
                continue
                
            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            tokens.append(self.tokenizer.eos_token_id) # EOS delimiter
            token_buffer.extend(tokens)
            
            # Yield full blocks
            while len(token_buffer) >= self.block_size:
                block = token_buffer[:self.block_size]
                token_buffer = token_buffer[self.block_size:]
                
                # Convert to tensor
                input_ids = torch.tensor(block, dtype=torch.long)
                # Labels are typically input_ids (shifted inside model)
                yield {"input_ids": input_ids, "labels": input_ids}
                
                self.samples_yielded += 1
                if self.max_tokens and (self.samples_yielded * self.block_size) >= self.max_tokens:
                    return

    def state_dict(self):
        # Return state for checkpointing
        return {"samples_yielded": self.samples_yielded}

    def load_state_dict(self, state_dict):
        self.samples_yielded = state_dict.get("samples_yielded", 0)
        # Note: Exact stream seeking on resume is extremely slow (O(N)).
        # Practical Strategy: Rely on the `samples_yielded` count to stop training
        # at the right time, but when resuming, we often accept that we might 
        # re-sample some data or we rely on changing the seed for Stage 2.
        # For strict exactness, one would implement a skip loop here.