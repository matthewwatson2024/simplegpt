#!/usr/bin/env python3
"""
Demo script for SimpleGPT
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and display the result"""
    print(f"\n{'='*60}")
    print(f"DEMO: {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run SimpleGPT demo"""
    print("🚀 SimpleGPT Demo")
    print("This demo will show you how to train and use a GPT model from scratch!")
    
    # Check device availability
    import torch
    device_info = []
    if torch.cuda.is_available():
        device_info.append("CUDA")
    if torch.backends.mps.is_available():
        device_info.append("MPS (Apple Silicon)")
    device_info.append("CPU")
    
    print(f"Available devices: {', '.join(device_info)}")
    print(f"Auto-selected device: {'MPS' if torch.backends.mps.is_available() else 'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Check if we have a trained model
    if not os.path.exists("checkpoints/checkpoint_epoch_2.pt"):
        print("\n📚 First, let's train a model on the example text...")
        run_command(
            "python main.py train example_text.txt --epochs 2 --batch-size 2 --d-model 128 --n-layers 2",
            "Training a GPT model"
        )
    
    print("\n🎯 Now let's generate some text with different prompts...")
    
    prompts = [
        "The future of AI",
        "Machine learning is",
        "Natural language processing",
        "Deep learning models"
    ]
    
    for prompt in prompts:
        run_command(
            f'python main.py generate checkpoints/checkpoint_epoch_2.pt checkpoints/vocab.txt --prompt "{prompt}" --max-length 30 --temperature 0.7',
            f"Generating text for: '{prompt}'"
        )
    
    print("\n🎉 Demo completed!")
    print("\nYou can now:")
    print("1. Train on your own text: python main.py train your_text.txt")
    print("2. Generate text: python main.py generate checkpoint.pt vocab.txt --prompt 'Your prompt'")
    print("3. Chat interactively: python main.py chat checkpoint.pt vocab.txt")
    print("\nFor more options, run: python main.py --help")


if __name__ == "__main__":
    main()
