import yaml
import torch
import os
import time
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW

from model import SLMForCausalLM, SLMConfig
from data import PackedStreamingDataset

def main():
    # 1. Load Configuration
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    # 2. Setup Accelerate
    # Project Config handles automatic checkpoint naming/saving path
    project_config = ProjectConfiguration(
        project_dir=cfg['training']['output_dir'], 
        automatic_checkpoint_naming=True
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
        mixed_precision=cfg['training']['dtype'], # 'bf16'
        log_with="wandb",
        project_config=project_config
    )
    
    # Reproducibility
    set_seed(cfg['data']['seed'])
    
    # 3. Initialize Trackers
    if accelerator.is_main_process:
        accelerator.init_trackers(cfg['training']['run_name'], config=cfg)

    # 4. Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2") # Or custom BPE
    model_config = SLMConfig(**cfg['model'])
    model = SLMForCausalLM(model_config)
    
    # Critical Optimization: torch.compile
    # Reduces overhead of Python loop and fuses ops (RMSNorm/SwiGLU)
    if cfg['training']['compile']:
        model = torch.compile(model, mode="reduce-overhead")

    # 5. Optimizer
    # Fused AdamW is faster on CUDA
    optimizer = AdamW(model.parameters(), lr=cfg['training']['learning_rate'], 
                      weight_decay=cfg['training']['weight_decay'], fused=True)
    
    # 6. Data Pipeline
    train_dataset = PackedStreamingDataset(tokenizer, **cfg['data'])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=cfg['training']['per_device_train_batch_size'],
        num_workers=2, # Careful with streaming datasets and workers!
        pin_memory=True
    )
    
    # 7. Scheduler
    total_batch_size = (cfg['training']['per_device_train_batch_size'] * 
                        accelerator.num_processes * 
                        cfg['training']['gradient_accumulation_steps'])
    max_steps = cfg['training']['max_tokens'] // (cfg['data']['block_size'] * total_batch_size)
    # Correct calculation of steps based on token count
    max_train_steps = int(cfg['training']['max_tokens'] / (cfg['data']['block_size'] * total_batch_size)) * total_batch_size # simplification

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=cfg['training']['warmup_steps'], 
        num_training_steps=max_train_steps
    )

    # 8. Prepare
    # Accelerate wraps model, optimizer, dataloader
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    # 9. Resume Logic
    if cfg['training']['resume_from_checkpoint']:
        accelerator.load_state(cfg['training']['resume_from_checkpoint'])
        accelerator.print(f"Resumed state from {cfg['training']['resume_from_checkpoint']}")

    # 10. Training Loop
    model.train()
    completed_steps = 0
    start_time = time.time()
    
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            outputs = model(batch["input_ids"], labels=batch["labels"])
            loss = outputs["loss"]
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), cfg['training']['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
                
                # Logging
                if completed_steps % cfg['training']['logging_steps'] == 0:
                    throughput = (cfg['data']['block_size'] * total_batch_size) / (time.time() - start_time)
                    accelerator.log({
                        "loss": loss.item(),
                        "lr": scheduler.get_last_lr(),
                        "throughput_tokens_per_sec": throughput
                    }, step=completed_steps)
                    start_time = time.time() # Reset timer
                
                # Checkpointing
                if completed_steps % cfg['training']['save_steps'] == 0:
                    accelerator.save_state() # Saves model, optim, scheduler, rng
                    
        if completed_steps >= max_train_steps:
            break
            
    accelerator.end_training()

if __name__ == "__main__":
    main()