import logging
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from datasets import load_dataset
current_dir = os.path.abspath(os.path.dirname(__file__))
if current_dir in sys.path:
    sys.path.remove(current_dir)
import deepspeed
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Tokenizer,
    OPTForCausalLM,
    GPT2LMHeadModel,
    set_seed
)

# Argument parsing
parser = argparse.ArgumentParser(description='DeepSpeed ImageNet Training with TopK Compression')
# Basic parameters
parser.add_argument('--dataset', default='wikitext-103', type=str, help='dataset name')
parser.add_argument('--model', default='gpt2-large', type=str, help='model architecture')
parser.add_argument('--epochs', default=1, type=int, help='number of epochs to run')
parser.add_argument('--batch-size', default=64, type=int, help='batch size per GPU')
parser.add_argument('--workers', default=8, type=int, help='data loading workers')
parser.add_argument('--seed', type=int, default=42, help='seed for initializing training')
parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
parser.add_argument("--seq_length", type=int, default=512)  
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

# Training parameters
parser.add_argument('--lr', '--learning-rate', default=0.0125, type=float, dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay')

# Checkpoint parameters
parser.add_argument('--ckpt_run', action='store_true', default=False, help='whether to use checkpointing')
parser.add_argument('--ec_run', action='store_true', default=False, help='whether to use ec checkpointing')
parser.add_argument('--test', action='store_true', default=False, help='whether to test')
parser.add_argument('--resume', default=0, type=int, help='data loading workers')
parser.add_argument("--freq", default=10, type=int, help='how many iteration to save a whole checkpoint')
parser.add_argument("--cpu_home", default='', type=str, help='directory to save checkpoints')
parser.add_argument("--fragment_dir", default='', type=str, help='directory to save ec')
parser.add_argument('-k', default=4, type=int, help='number of data elements')
parser.add_argument('-m', default=1, type=int, help='number of parity elements')
parser.add_argument('-ec_type', default="isa_l_rs_vand", help='EC algorithm used')

args = parser.parse_args()
stop_iteration = 100
# args.cpu_home = "/data/ckpt/"

def setlog():
    """Configure logging to output to both file and console"""
    open('Checkpoint_1.log', 'w').close()
    # Create a logger named "IterCheckpoint"
    logger = logging.getLogger("Checkpoint")
    # Output logs to file IterCheckpoint.log
    fh = logging.FileHandler('Checkpoint_1.log', encoding='utf-8')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(filename)s:%(lineno)s %(levelname)s: %(message)s', '%H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # Also output to console
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger

def main():
    """Main entry point for training script"""
    logger = setlog()
    model_path = "" + args.model
    
    # Initialize DeepSpeed distributed training
    deepspeed.init_distributed()
    dist.barrier()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(args.local_rank)
    set_seed(args.seed)  
    logger.info(f"[Rank {rank}/{world_size}] Initialized DeepSpeed")

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token to end-of-sequence token
    logger.info(f"[Rank {rank}/{world_size}] Tokenizer loaded successfully.")
    
    def tokenize_function(examples):
        """Tokenize text examples with truncation and padding"""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.seq_length,
            padding="max_length"
        )
    
    # Load and process wikitext-103 dataset
    dataset = load_dataset("/data/dataset/nlp/transformer/wikitext-103",
                      data_files={
                          "train": "/data/dataset/nlp/transformer/wikitext-103/train.txt",
                          "test": "/data/dataset/nlp/transformer/wikitext-103/test.txt"
                      })
    train_dataset = dataset['train']
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=24
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Use causal language modeling
    )
    train_sampler = DistributedSampler(
        train_dataset,
        shuffle=True,
        num_replicas=world_size,
        rank=rank
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        num_workers=args.workers
    ) 
    if args.test:
        test_dataset = dataset['test']
        test_dataset = test_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            num_proc=24
        )
        test_sampler = DistributedSampler(
            test_dataset,
            shuffle=False,
            num_replicas=world_size,
            rank=rank
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            sampler=test_sampler,
            collate_fn=data_collator,
            num_workers=args.workers
        )
    logger.info(f"[Rank {rank}/{world_size}] Dataset loaded successfully.")

    # Initialize model with gradient checkpointing to save memory
    if args.model.startswith('opt'):
        model = OPTForCausalLM.from_pretrained(model_path)
    else:
        model = GPT2LMHeadModel.from_pretrained(model_path)
    model.gradient_checkpointing_enable()  # Recompute activations
    model.cuda()
    logger.info(f"[Rank {rank}/{world_size}] Model loaded successfully.")
    
    # DeepSpeed initialization (requires ds_config.json)
    ds_config = {
        "train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 5e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 5e-4,
                "warmup_num_steps": 2000
            }
        },
    }
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model, 
        model_parameters=model.parameters(), 
        config=ds_config
    )
    logger.info(f"[Rank {rank}/{world_size}] -----------> epochs:{args.epochs} iterations:{len(train_loader)} batchsize:{args.batch_size} all_b:{args.batch_size*world_size}")
    
    start_time = time.time()
    if args.resume == 1:
        load_base_checkpoint(model, optimizer, logger)
    elif args.resume == 2:
        load_ec_checkpoint(model, optimizer, logger)
    logger.info(f"[Rank {rank}/{world_size}] Load checkpoints use{(time.time() - start_time):.3f}s")

    allstart_time = time.time()
    # Training loop
    for epoch in range(args.epochs):
        model_engine.train()
        train_sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(train_loader):
            end = time.time()
            # Move data to GPU
            inputs = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            # Forward pass
            outputs = model_engine(input_ids=inputs, labels=labels)
            loss = outputs.loss
            # Backward pass and gradient update
            model_engine.backward(loss)
            model_engine.step()

            # Logging (only on master node)
            if rank == 0:
                logger.info(f"[Epoch {epoch}/{args.epochs}] Batch {batch_idx}, Loss: {loss.item()}, Time: {time.time() - end}")

            if batch_idx == stop_iteration:
                break

            if args.test and batch_idx % 50 == 0 and batch_idx != 0:
                test(test_loader, model_engine, logger)
                model_engine.train()
    
    if rank == 0:
        logger.info(f"[Rank {rank}/{world_size}] Batch {stop_iteration}  ———————————————>  ALL Time: {time.time() - allstart_time}")

    # if args.ckpt_run:
    #     optimizer.op_exit()

def load_base_checkpoint(model, optimizer, logger):
    """Load model and optimizer state from a standard checkpoint"""
    filedir = args.cpu_home
    filepath = filedir + '/' + args.model + '_' + args.dataset + '_0_0_full' + '.pth.tar'
    if os.path.isfile(filepath):
        logger.info("loading {}".format(filepath))
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        raise ValueError("No checkpoint found")

def load_ec_checkpoint(model, optimizer, logger):
    """Load model and optimizer state from an erasure-coded checkpoint"""
    return

def test(test_loader, model_engine, logger):
    """Evaluate model on test dataset"""
    for epoch in range(args.epochs):
        model_engine.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):  # Iterate over test set
                inputs = batch["input_ids"].cuda()
                labels = batch["labels"].cuda()
                
                outputs = model_engine(input_ids=inputs, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
            avg_loss = total_loss / len(test_loader)
            logger.info(f"Test Loss: {avg_loss:.4f}")
        model_engine.train()  # Switch back to training mode

if __name__ == "__main__":
    main()