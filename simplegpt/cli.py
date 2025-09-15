"""
Command-line interface for SimpleGPT
"""

import argparse
import os
import sys
import torch
from typing import Optional

from .model import SimpleGPT
from .tokenizer import SimpleTokenizer
from .trainer import Trainer
from .inference import GPTInference


def train_model(args):
    """Train a GPT model on text data"""
    print("Loading training data...")
    
    # Read training text
    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Training text length: {len(text):,} characters")
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Save tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_vocab(os.path.join(args.output_dir, 'vocab.txt'))
    
    # Initialize model
    model = SimpleGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps
    )
    
    # Train the model
    trainer.train(
        text=text,
        epochs=args.epochs,
        batch_size=args.batch_size,
        block_size=args.block_size,
        save_every=args.save_every,
        save_dir=args.output_dir
    )
    
    print("Training completed!")


def generate_text(args):
    """Generate text using a trained model"""
    print("Loading model...")
    
    # Load checkpoint to get model config
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_config = checkpoint.get('model_config', {})
    
    # Load tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.load_vocab(args.vocab_file)
    
    # Initialize model with config from checkpoint
    model = SimpleGPT(
        vocab_size=model_config.get('vocab_size', tokenizer.vocab_size),
        d_model=model_config.get('d_model', args.d_model),
        n_heads=model_config.get('n_heads', args.n_heads),
        n_layers=model_config.get('n_layers', args.n_layers),
        d_ff=model_config.get('d_ff', args.d_ff),
        max_seq_len=model_config.get('max_seq_len', args.max_seq_len),
        dropout=0.0  # No dropout during inference
    )
    
    # Initialize inference
    inference = GPTInference(model, tokenizer, device=args.device)
    inference.load_model_from_checkpoint(args.checkpoint)
    
    # Generate text
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = input("Enter a prompt: ")
    
    print(f"\nGenerating text with prompt: '{prompt}'")
    print("-" * 50)
    
    generated_texts = inference.generate_text(
        prompt=prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_samples=args.num_samples
    )
    
    for i, text in enumerate(generated_texts, 1):
        print(f"Sample {i}:")
        print(prompt + text)
        print()


def interactive_chat(args):
    """Interactive chat mode"""
    print("Loading model...")
    
    # Load checkpoint to get model config
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_config = checkpoint.get('model_config', {})
    
    # Load tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.load_vocab(args.vocab_file)
    
    # Initialize model with config from checkpoint
    model = SimpleGPT(
        vocab_size=model_config.get('vocab_size', tokenizer.vocab_size),
        d_model=model_config.get('d_model', args.d_model),
        n_heads=model_config.get('n_heads', args.n_heads),
        n_layers=model_config.get('n_layers', args.n_layers),
        d_ff=model_config.get('d_ff', args.d_ff),
        max_seq_len=model_config.get('max_seq_len', args.max_seq_len),
        dropout=0.0
    )
    
    # Initialize inference
    inference = GPTInference(model, tokenizer, device=args.device)
    inference.load_model_from_checkpoint(args.checkpoint)
    
    print("Interactive chat mode. Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            response = inference.chat(
                message=user_input,
                max_response_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            
            print(f"AI: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='SimpleGPT - Train and use GPT models')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a GPT model')
    train_parser.add_argument('input_file', help='Input text file for training')
    train_parser.add_argument('--output-dir', default='./checkpoints', help='Output directory for checkpoints')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    train_parser.add_argument('--block-size', type=int, default=128, help='Block size for sequences')
    train_parser.add_argument('--d-model', type=int, default=512, help='Model dimension')
    train_parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    train_parser.add_argument('--n-layers', type=int, default=6, help='Number of transformer layers')
    train_parser.add_argument('--d-ff', type=int, default=2048, help='Feed-forward dimension')
    train_parser.add_argument('--max-seq-len', type=int, default=1024, help='Maximum sequence length')
    train_parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    train_parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    train_parser.add_argument('--weight-decay', type=float, default=0.1, help='Weight decay')
    train_parser.add_argument('--warmup-steps', type=int, default=100, help='Warmup steps')
    train_parser.add_argument('--save-every', type=int, default=1000, help='Save checkpoint every N epochs')
    train_parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda, mps)')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate text using a trained model')
    generate_parser.add_argument('checkpoint', help='Path to model checkpoint')
    generate_parser.add_argument('vocab_file', help='Path to vocabulary file')
    generate_parser.add_argument('--prompt', help='Text prompt for generation')
    generate_parser.add_argument('--max-length', type=int, default=100, help='Maximum generation length')
    generate_parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    generate_parser.add_argument('--top-k', type=int, help='Top-k sampling')
    generate_parser.add_argument('--top-p', type=float, help='Top-p (nucleus) sampling')
    generate_parser.add_argument('--num-samples', type=int, default=1, help='Number of samples to generate')
    generate_parser.add_argument('--d-model', type=int, default=512, help='Model dimension')
    generate_parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    generate_parser.add_argument('--n-layers', type=int, default=6, help='Number of transformer layers')
    generate_parser.add_argument('--d-ff', type=int, default=2048, help='Feed-forward dimension')
    generate_parser.add_argument('--max-seq-len', type=int, default=1024, help='Maximum sequence length')
    generate_parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda, mps)')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat mode')
    chat_parser.add_argument('checkpoint', help='Path to model checkpoint')
    chat_parser.add_argument('vocab_file', help='Path to vocabulary file')
    chat_parser.add_argument('--max-length', type=int, default=100, help='Maximum response length')
    chat_parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    chat_parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    chat_parser.add_argument('--top-p', type=float, default=0.9, help='Top-p (nucleus) sampling')
    chat_parser.add_argument('--d-model', type=int, default=512, help='Model dimension')
    chat_parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    chat_parser.add_argument('--n-layers', type=int, default=6, help='Number of transformer layers')
    chat_parser.add_argument('--d-ff', type=int, default=2048, help='Feed-forward dimension')
    chat_parser.add_argument('--max-seq-len', type=int, default=1024, help='Maximum sequence length')
    chat_parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'generate':
        generate_text(args)
    elif args.command == 'chat':
        interactive_chat(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()


