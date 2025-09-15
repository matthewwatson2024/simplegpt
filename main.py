#!/usr/bin/env python3
"""
SimpleGPT - A minimal GPT implementation for training and inference

Usage:
    python main.py train <input_file> [options]
    python main.py generate <checkpoint> <vocab_file> [options]
    python main.py chat <checkpoint> <vocab_file> [options]
"""

import torch
from simplegpt.cli import main

if __name__ == '__main__':
    # Show device information
    devices = []
    if torch.cuda.is_available():
        devices.append("CUDA")
    if torch.backends.mps.is_available():
        devices.append("MPS")
    devices.append("CPU")
    
    auto_device = "MPS" if torch.backends.mps.is_available() else "CUDA" if torch.cuda.is_available() else "CPU"
    
    print(f"🔧 Available devices: {', '.join(devices)}")
    print(f"🎯 Auto-selected device: {auto_device}")
    print()
    
    main()
