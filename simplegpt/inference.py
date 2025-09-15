"""
Inference utilities for SimpleGPT
"""

import torch
import torch.nn as nn
from typing import List, Optional
from .tokenizer import SimpleTokenizer


class GPTInference:
    """Inference class for trained GPT models"""
    
    def __init__(self, model: nn.Module, tokenizer: SimpleTokenizer, device: str = 'auto'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = self._get_device(device)
        self.model.to(self.device)
        self.model.eval()
    
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
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        num_samples: int = 1
    ) -> List[str]:
        """
        Generate text from a prompt
        
        Args:
            prompt: Starting text prompt
            max_length: Maximum length to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (keep only top k tokens)
            top_p: Nucleus sampling (keep tokens with cumulative prob <= p)
            num_samples: Number of samples to generate
            
        Returns:
            List of generated text samples
        """
        # Encode the prompt
        prompt_tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(self.device)
        
        generated_samples = []
        
        for _ in range(num_samples):
            # Generate sequence
            generated_ids = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.tokenizer.vocab.get('<PAD>', 0)
            )
            
            # Decode the generated sequence
            generated_tokens = generated_ids[0].cpu().tolist()
            generated_text = self.tokenizer.decode(generated_tokens)
            
            # Remove the original prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
            
            generated_samples.append(generated_text.strip())
        
        return generated_samples
    
    def complete_text(
        self,
        text: str,
        max_completion_length: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> str:
        """
        Complete a given text
        
        Args:
            text: Text to complete
            max_completion_length: Maximum length of completion
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            Completed text
        """
        completions = self.generate_text(
            prompt=text,
            max_length=max_completion_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_samples=1
        )
        
        return text + completions[0]
    
    def chat(
        self,
        message: str,
        max_response_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> str:
        """
        Generate a chat response
        
        Args:
            message: User message
            max_response_length: Maximum response length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            Generated response
        """
        # Add a simple prompt format
        prompt = f"Human: {message}\nAI:"
        
        response = self.generate_text(
            prompt=prompt,
            max_length=max_response_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_samples=1
        )[0]
        
        # Clean up the response
        response = response.replace("Human:", "").replace("AI:", "").strip()
        
        return response
    
    def load_model_from_checkpoint(self, checkpoint_path: str) -> None:
        """Load a trained model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load tokenizer vocab if available
        if 'tokenizer_vocab' in checkpoint:
            self.tokenizer.vocab = checkpoint['tokenizer_vocab']
            self.tokenizer.inverse_vocab = {
                v: k for k, v in self.tokenizer.vocab.items()
            }
            self.tokenizer.vocab_size = len(self.tokenizer.vocab)
        
        print(f"Model loaded from {checkpoint_path}")
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'vocab_size': self.tokenizer.vocab_size,
            'device': str(self.device),
            'model_config': {
                'd_model': getattr(self.model, 'd_model', 'Unknown'),
                'max_seq_len': getattr(self.model, 'max_seq_len', 'Unknown')
            }
        }
