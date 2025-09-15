"""
Training utilities for SimpleGPT
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
from typing import List, Tuple, Optional


class TextDataset(Dataset):
    """Dataset for text training"""
    
    def __init__(self, text: str, tokenizer, block_size: int = 128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Tokenize the text
        self.tokens = tokenizer.encode(text)
        
        # Create overlapping sequences
        self.sequences = []
        for i in range(0, len(self.tokens) - block_size, block_size // 2):
            self.sequences.append(self.tokens[i:i + block_size + 1])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        targets = torch.tensor(sequence[1:], dtype=torch.long)
        return input_ids, targets


class Trainer:
    """Trainer class for SimpleGPT"""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = 'auto',
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 100
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = self._get_device(device)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )
        
        self.warmup_steps = warmup_steps
        self.step = 0
        
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, (input_ids, targets) in enumerate(progress_bar):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            logits, loss = self.model(input_ids, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Learning rate scheduling
            if self.step < self.warmup_steps:
                self.scheduler.step()
            
            self.step += 1
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        return total_loss / num_batches
    
    def train(
        self,
        text: str,
        epochs: int = 10,
        batch_size: int = 4,
        block_size: int = 128,
        save_every: int = 1000,
        save_dir: str = './checkpoints'
    ) -> None:
        """Train the model on text data"""
        
        # Create dataset and dataloader
        dataset = TextDataset(text, self.tokenizer, block_size)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Training on {len(dataset)} sequences")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            avg_loss = self.train_epoch(dataloader)
            print(f"Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
                self.save_checkpoint(save_dir, epoch + 1, avg_loss)
    
    def save_checkpoint(self, save_dir: str, epoch: int, loss: float) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'step': self.step,
            'tokenizer_vocab': self.tokenizer.vocab,
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'd_model': self.model.d_model,
                'n_heads': self.model.n_heads,
                'n_layers': self.model.n_layers,
                'd_ff': self.model.d_ff,
                'max_seq_len': self.model.max_seq_len
            }
        }
        
        filepath = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.step = checkpoint['step']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    def evaluate(self, text: str, block_size: int = 128) -> float:
        """Evaluate the model on text data"""
        self.model.eval()
        
        dataset = TextDataset(text, self.tokenizer, block_size)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, targets in tqdm(dataloader, desc="Evaluating"):
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)
                
                _, loss = self.model(input_ids, targets)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
