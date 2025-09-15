"""
Simple tokenizer implementation for GPT
"""

import re
from typing import List, Dict, Tuple


class SimpleTokenizer:
    """A simple character-level tokenizer for GPT training"""
    
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = 0
        
    def build_vocab(self, text: str) -> None:
        """Build vocabulary from text"""
        # Get unique characters
        chars = sorted(list(set(text)))
        
        # Create vocabulary
        self.vocab = {char: i for i, char in enumerate(chars)}
        self.inverse_vocab = {i: char for char, i in self.vocab.items()}
        self.vocab_size = len(chars)
        
        # Add special tokens
        self.vocab['<PAD>'] = self.vocab_size
        self.inverse_vocab[self.vocab_size] = '<PAD>'
        self.vocab_size += 1
        
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        return [self.vocab.get(char, self.vocab['<PAD>']) for char in text]
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        return ''.join([self.inverse_vocab.get(token_id, '<UNK>') for token_id in token_ids])
    
    def save_vocab(self, filepath: str) -> None:
        """Save vocabulary to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for char, token_id in self.vocab.items():
                f.write(f"{char}\t{token_id}\n")
    
    def load_vocab(self, filepath: str) -> None:
        """Load vocabulary from file"""
        self.vocab = {}
        self.inverse_vocab = {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                parts = line.split('\t')
                if len(parts) == 2:
                    char, token_id = parts
                    token_id = int(token_id)
                    self.vocab[char] = token_id
                    self.inverse_vocab[token_id] = char
        
        self.vocab_size = len(self.vocab)
