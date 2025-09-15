# SimpleGPT

A minimal implementation of GPT (Generative Pre-trained Transformer) for training and inference on text data with **GPU acceleration support**.

## Features

- **Character-level tokenization** for simple text processing
- **Multi-head self-attention** mechanism
- **Transformer architecture** with configurable layers
- **Training utilities** with progress tracking and checkpointing
- **Text generation** with temperature, top-k, and top-p sampling
- **Interactive chat mode** for conversational AI
- **Command-line interface** for easy usage
- **GPU acceleration** with CUDA (NVIDIA) and MPS (Apple Silicon) support
- **Automatic device detection** for optimal performance

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd simplegpt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

SimpleGPT automatically detects and uses the best available device (CUDA, MPS, or CPU) for optimal performance.

### 1. Train a Model

Train a GPT model on your text data:

```bash
python main.py train example_text.txt --epochs 5 --batch-size 2
```

The system will automatically use:
- **MPS** on Apple Silicon Macs (fastest)
- **CUDA** on NVIDIA GPUs (Linux/Windows)
- **CPU** as fallback

### 2. Generate Text

Generate text using the trained model:

```bash
python main.py generate checkpoints/checkpoint_epoch_5.pt checkpoints/vocab.txt --prompt "The future of AI"
```

### 3. Interactive Chat

Start an interactive chat session:

```bash
python main.py chat checkpoints/checkpoint_epoch_5.pt checkpoints/vocab.txt
```

### 4. Run Demo

See SimpleGPT in action with the included demo:

```bash
python demo.py
```

## Usage Examples

### Training Options

```bash
# Train with custom parameters
python main.py train my_text.txt \
    --epochs 10 \
    --batch-size 4 \
    --d-model 256 \
    --n-layers 4 \
    --learning-rate 1e-3 \
    --output-dir ./my_model

# Force specific device
python main.py train my_text.txt --device mps    # Apple Silicon GPU
python main.py train my_text.txt --device cuda   # NVIDIA GPU
python main.py train my_text.txt --device cpu    # CPU only
```

### Generation Options

```bash
# Generate with different sampling strategies
python main.py generate model.pt vocab.txt \
    --prompt "Once upon a time" \
    --max-length 200 \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.9 \
    --num-samples 3

# Generate with specific device
python main.py generate model.pt vocab.txt --device mps --prompt "Hello world"
```

## Model Architecture

The SimpleGPT model consists of:

- **Token Embedding**: Maps characters to dense vectors
- **Position Embedding**: Adds positional information
- **Transformer Blocks**: Multi-head attention + feed-forward layers
- **Output Layer**: Projects to vocabulary space

### Configurable Parameters

- `d_model`: Model dimension (default: 512)
- `n_heads`: Number of attention heads (default: 8)
- `n_layers`: Number of transformer layers (default: 6)
- `d_ff`: Feed-forward dimension (default: 2048)
- `max_seq_len`: Maximum sequence length (default: 1024)
- `dropout`: Dropout rate (default: 0.1)

## Training Details

The training process includes:

- **Character-level tokenization** of input text
- **Sliding window** approach for sequence generation
- **Cross-entropy loss** for next-token prediction
- **AdamW optimizer** with learning rate scheduling
- **Gradient clipping** for training stability
- **Automatic checkpointing** during training

## Text Generation

The model supports various sampling strategies:

- **Temperature sampling**: Controls randomness (higher = more random)
- **Top-k sampling**: Keeps only the top k most likely tokens
- **Top-p (nucleus) sampling**: Keeps tokens with cumulative probability ≤ p

## Example Training Data

The repository includes `example_text.txt` with sample content covering:
- Basic language patterns
- Technology and AI topics
- Machine learning concepts
- Natural language processing

## Demo and Testing

### Interactive Demo
Run the included demo to see SimpleGPT in action:
```bash
python demo.py
```

The demo will:
- Show available devices and auto-selection
- Train a small model (if needed)
- Generate text with various prompts
- Demonstrate different sampling strategies

### Command Line Help
Get detailed help for any command:
```bash
python main.py --help                    # General help
python main.py train --help              # Training options
python main.py generate --help           # Generation options
python main.py chat --help               # Chat options
```

## Device Support

SimpleGPT automatically detects and uses the best available device for optimal performance:

### Automatic Device Selection
- **MPS**: Apple Silicon GPUs (macOS) - **Recommended for Mac users**
- **CUDA**: NVIDIA GPUs (Linux/Windows) - **Recommended for Linux/Windows users**
- **CPU**: Fallback for all systems

### Manual Device Selection
```bash
# Force specific device
python main.py train text.txt --device mps    # Apple Silicon GPU
python main.py train text.txt --device cuda   # NVIDIA GPU
python main.py train text.txt --device cpu    # CPU only
python main.py train text.txt --device auto   # Auto-detect (default)
```

### Device Information
The system displays available devices on startup:
```
🔧 Available devices: MPS, CPU
🎯 Auto-selected device: MPS
```

## Performance Notes

### Device Performance
- **Small models** (< 1M parameters): CPU may be faster due to GPU overhead
- **Larger models** (> 1M parameters): GPU acceleration provides significant speedup
- **MPS on Apple Silicon**: ~1.8x faster than CPU for larger models
- **CUDA on NVIDIA**: Similar performance gains on compatible hardware

### Training Performance
- **Training speed**: 75+ iterations/second on MPS vs ~40 on CPU
- **Memory efficiency**: Better memory utilization with GPU acceleration
- **Batch size**: Larger batch sizes benefit more from GPU acceleration

### Optimization Tips
- Use larger models (d_model ≥ 256) to see GPU benefits
- Increase batch size for better GPU utilization
- Monitor memory usage with larger models
- Use `--device auto` for automatic optimal device selection

## System Requirements

### Hardware Requirements
- **CPU**: Any modern processor (Intel/AMD/Apple Silicon)
- **GPU**: Optional but recommended for larger models
  - **NVIDIA**: CUDA-compatible GPU (Linux/Windows)
  - **Apple Silicon**: M1/M2/M3 Macs (macOS)
- **RAM**: 4GB minimum, 8GB+ recommended for larger models
- **Storage**: 1GB for installation, additional space for models

### Software Requirements
- **Python**: 3.8 or higher
- **PyTorch**: 2.0 or higher (automatically installed)
- **Operating System**: macOS, Linux, or Windows

## Limitations

This is a minimal implementation designed for educational purposes:

- **Character-level** tokenization (not subword/word-level)
- **No pre-training** on large datasets
- **Limited context** compared to production models
- **Basic architecture** without advanced optimizations
- **Small vocabulary** compared to modern tokenizers

## Troubleshooting

### Common Issues

**MPS not available on macOS:**
- Ensure you're using PyTorch 2.0+ with MPS support
- Check if you're on Apple Silicon (M1/M2/M3) Mac
- Try `python -c "import torch; print(torch.backends.mps.is_available())"`

**CUDA out of memory:**
- Reduce batch size: `--batch-size 1`
- Reduce model size: `--d-model 128 --n-layers 2`
- Use CPU: `--device cpu`

**Slow training:**
- Use GPU acceleration: `--device auto`
- Increase batch size if memory allows
- Use larger models to see GPU benefits

**Poor text generation:**
- Train for more epochs: `--epochs 10`
- Use more training data
- Adjust temperature: `--temperature 0.7`
- Try different sampling: `--top-k 50 --top-p 0.9`

### Getting Help
- Check the command help: `python main.py --help`
- Run the demo: `python demo.py`
- Ensure all dependencies are installed: `pip install -r requirements.txt`

## Contributing

Feel free to contribute improvements:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Changelog

### Version 0.1.0 (Latest)
- ✅ **MPS Support**: Added Apple Silicon GPU acceleration
- ✅ **Automatic Device Detection**: Smart device selection (CUDA → MPS → CPU)
- ✅ **Enhanced CLI**: Device information display and manual device selection
- ✅ **Performance Improvements**: ~1.8x speedup on Apple Silicon for larger models
- ✅ **Demo Script**: Interactive demonstration of all features
- ✅ **Comprehensive Documentation**: Updated README with device support and troubleshooting

### Features
- Character-level GPT implementation
- Multi-head self-attention mechanism
- Training with progress tracking and checkpointing
- Text generation with multiple sampling strategies
- Interactive chat mode
- Cross-platform GPU acceleration

## License

This project is open source and available under the MIT License.

## Commands For Working With Large Dataset

# Basic streaming training (recommended)
python main.py train large_dataset.txt --dataset-type streaming --batch-size 4 --num-workers 4

# Pre-tokenization workflow (best for very large datasets)
python main.py pretokenize large_dataset.txt --output-dir ./checkpoints
python main.py train large_dataset.txt --pretokenize --batch-size 8 --num-workers 6

# Memory-constrained training with gradient accumulation
python main.py train large_dataset.txt --batch-size 2 --gradient-accumulation-steps 16 --auto-batch-size

# New Commands 2

# Your exact command will now work:
python train_utf8_safe.py --input-file example_text.txt --dataset-type utf8_safe --batch-size 4 --num-workers 4 --epochs 1000000000000000000 --save-every 1

# Or using the main CLI:
python main.py train example_text.txt --dataset-type utf8_safe --batch-size 4 --num-workers 4 --epochs 10 --save-every 1

# For large files with encoding issues:
python train_utf8_safe.py --input-file large_dataset.txt --dataset-type utf8_streaming --batch-size 2 --num-workers 4 --epochs 5 --errors replace
