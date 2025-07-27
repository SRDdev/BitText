#!/usr/bin/env python3
"""
BitNet Inference Script - Complete text generation and testing
Handles PyTorch 2.6+ security restrictions and provides full inference capabilities
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import os
import sys
import json
import time
import warnings
from typing import List, Dict, Optional, Tuple
import argparse
from dataclasses import dataclass

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

try:
    from BitText.models.transformer import BitNetTextModel
    from BitText.logs.logger import setup_logger
except ImportError:
    print("Warning: Could not import BitText modules. Using fallback implementations.")
    
    # Fallback logger
    import logging
    def setup_logger(name, log_dir, level):
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s | %(name)s | %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    early_stopping: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 102  # [SEP] token for BERT
    min_length: int = 10

class BitNetInference:
    """BitNet model inference and text generation."""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None, device: str = "auto"):
        """Initialize BitNet inference."""
        self.logger = setup_logger("Inference", "logs", "info")
        self.device = self._setup_device(device)
        
        # Load model and tokenizer
        self.model, self.tokenizer, self.model_config = self._load_model(model_path, config_path)
        self.model.eval()
        
        self.logger.info(f"BitNet model loaded successfully on {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup inference device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        torch_device = torch.device(device)
        self.logger.info(f"Using device: {torch_device}")
        
        if torch_device.type == "cuda":
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        return torch_device
    
    def _load_model(self, model_path: str, config_path: Optional[str]) -> Tuple:
        """Load model with PyTorch 2.6+ compatibility."""
        
        self.logger.info(f"Loading model from: {model_path}")
        
        # Load checkpoint with security fix for PyTorch 2.6+
        try:
            # First try with weights_only=False (trusted source)
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except Exception as e:
            self.logger.warning(f"Failed with weights_only=False: {e}")
            try:
                # Try with numpy safe globals
                import numpy as np
                torch.serialization.add_safe_globals([np.core.multiarray.scalar])
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            except Exception as e2:
                self.logger.error(f"Failed to load checkpoint: {e2}")
                raise RuntimeError(f"Could not load model checkpoint: {e2}")
        
        # Extract configuration
        if 'config' in checkpoint:
            model_config = checkpoint['config']
        elif config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                model_config = json.load(f)
        else:
            # Fallback configuration for your specific model
            model_config = {
                'vocab_size': 30522,
                'd_model': 256,
                'nhead': 4,
                'num_layers': 2,
                'dim_feedforward': 1024,
                'dropout': 0.1,
                'tokenizer': 'bert-base-uncased'
            }
            self.logger.warning("Using fallback configuration")
        
        # Load tokenizer
        tokenizer_name = model_config.get('tokenizer', 'bert-base-uncased')
        self.logger.info(f"Loading tokenizer: {tokenizer_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        # Initialize model - try to import BitNetTextModel, fallback if needed
        vocab_size = len(tokenizer)
        
        try:
            from BitText.models.transformer import BitNetTextModel
            model = BitNetTextModel(
                vocab_size=vocab_size,
                num_layers=model_config.get('num_layers', 2),
                d_model=model_config.get('d_model', 256),
                nhead=model_config.get('nhead', 4),
                dim_feedforward=model_config.get('dim_feedforward', 1024),
                dropout=model_config.get('dropout', 0.1)
            )
        except ImportError:
            self.logger.error("Could not import BitNetTextModel. Please ensure the model file is available.")
            raise
        
        # Load weights - handle different checkpoint formats
        try:
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info("Loaded model from 'model_state_dict'")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                self.logger.info("Loaded model from 'state_dict'")
            else:
                # Assume the entire checkpoint is the state dict
                model.load_state_dict(checkpoint)
                self.logger.info("Loaded model from checkpoint directly")
        except Exception as e:
            self.logger.error(f"Failed to load model weights: {e}")
            self.logger.info("Available keys in checkpoint:", list(checkpoint.keys()) if isinstance(checkpoint, dict) else "Not a dict")
            raise
        
        model = model.to(self.device)
        self.logger.info("Model moved to device successfully")
        
        return model, tokenizer, model_config
    
    def encode_text(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Encode text to input IDs."""
        try:
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=False
            )
            return encoding["input_ids"].to(self.device)
        except Exception as e:
            self.logger.error(f"Failed to encode text: {e}")
            raise
    
    def decode_text(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Failed to decode text: {e}")
            return "<decoding_error>"
    
    def generate_text(self, prompt: str, config: GenerationConfig) -> Dict:
        """Generate text continuation from a prompt."""
        start_time = time.time()
        
        try:
            # Encode prompt
            input_ids = self.encode_text(prompt)
            original_length = input_ids.shape[1]
            
            self.logger.info(f"Generating text with prompt: '{prompt[:50]}...'")
            self.logger.info(f"Original length: {original_length}, Max new tokens: {config.max_new_tokens}")
            
            generated_ids = input_ids.clone()
            
            with torch.no_grad():
                for step in range(config.max_new_tokens):
                    # Forward pass
                    outputs = self.model(generated_ids)
                    
                    # Get logits for the last position
                    next_token_logits = outputs[:, -1, :]
                    
                    # Apply temperature
                    if config.temperature != 1.0:
                        next_token_logits = next_token_logits / config.temperature
                    
                    # Apply repetition penalty
                    if config.repetition_penalty != 1.0:
                        self._apply_repetition_penalty(
                            next_token_logits, generated_ids, config.repetition_penalty
                        )
                    
                    # Apply top-k filtering
                    if config.top_k > 0:
                        next_token_logits = self._top_k_filtering(next_token_logits, config.top_k)
                    
                    # Apply top-p filtering
                    if config.top_p < 1.0:
                        next_token_logits = self._top_p_filtering(next_token_logits, config.top_p)
                    
                    # Sample next token
                    if config.do_sample:
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Add to sequence
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    
                    # Check for early stopping
                    if next_token.item() == config.eos_token_id and config.early_stopping:
                        break
                    
                    # Check minimum length
                    if step + 1 >= config.min_length and next_token.item() == config.eos_token_id:
                        break
            
            # Decode generated text
            full_text = self.decode_text(generated_ids[0])
            generated_text = self.decode_text(generated_ids[0][original_length:])
            
            generation_time = time.time() - start_time
            tokens_generated = generated_ids.shape[1] - original_length
            
            return {
                "prompt": prompt,
                "generated_text": generated_text,
                "full_text": full_text,
                "tokens_generated": tokens_generated,
                "generation_time": generation_time,
                "tokens_per_second": tokens_generated / generation_time if generation_time > 0 else 0,
                "total_length": generated_ids.shape[1]
            }
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return {
                "prompt": prompt,
                "generated_text": f"<generation_error: {str(e)}>",
                "full_text": prompt,
                "tokens_generated": 0,
                "generation_time": time.time() - start_time,
                "tokens_per_second": 0,
                "total_length": 0
            }
    
    def _apply_repetition_penalty(self, logits: torch.Tensor, input_ids: torch.Tensor, penalty: float):
        """Apply repetition penalty to logits."""
        score = torch.gather(logits, 1, input_ids)
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits.scatter_(1, input_ids, score)
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Filter top-k tokens."""
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Filter top-p (nucleus) tokens."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = float('-inf')
        return logits
    
    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity of text under the model."""
        try:
            input_ids = self.encode_text(text)
            
            with torch.no_grad():
                outputs = self.model(input_ids)
                
                # Calculate loss for next token prediction
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                perplexity = torch.exp(loss).item()
            
            return perplexity
        except Exception as e:
            self.logger.error(f"Perplexity calculation failed: {e}")
            return float('inf')
    
    def interactive_chat(self):
        """Interactive chat mode for testing the model."""
        print("\n" + "="*60)
        print("🤖 BitNet Interactive Text Generation")
        print("="*60)
        print("Enter prompts to generate text. Type 'quit' to exit.")
        print("Commands:")
        print("  /temp X    - Set temperature (e.g., /temp 0.8)")
        print("  /tokens X  - Set max tokens (e.g., /tokens 50)")
        print("  /topk X    - Set top-k (e.g., /topk 40)")
        print("  /topp X    - Set top-p (e.g., /topp 0.9)")
        print("  /config    - Show current config")
        print("  /examples  - Show example prompts")
        print("  /test      - Run quick test")
        print("-"*60)
        
        # Default generation config
        gen_config = GenerationConfig(
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        
        while True:
            try:
                user_input = input("\n💭 Prompt: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                # Handle commands
                if user_input.startswith('/'):
                    if user_input.startswith('/temp '):
                        try:
                            gen_config.temperature = float(user_input.split()[1])
                            print(f"🌡️ Temperature set to {gen_config.temperature}")
                        except (IndexError, ValueError):
                            print("❌ Usage: /temp 0.8")
                        continue
                    elif user_input.startswith('/tokens '):
                        try:
                            gen_config.max_new_tokens = int(user_input.split()[1])
                            print(f"🎯 Max tokens set to {gen_config.max_new_tokens}")
                        except (IndexError, ValueError):
                            print("❌ Usage: /tokens 50")
                        continue
                    elif user_input.startswith('/topk '):
                        try:
                            gen_config.top_k = int(user_input.split()[1])
                            print(f"🔝 Top-k set to {gen_config.top_k}")
                        except (IndexError, ValueError):
                            print("❌ Usage: /topk 40")
                        continue
                    elif user_input.startswith('/topp '):
                        try:
                            gen_config.top_p = float(user_input.split()[1])
                            print(f"🎲 Top-p set to {gen_config.top_p}")
                        except (IndexError, ValueError):
                            print("❌ Usage: /topp 0.9")
                        continue
                    elif user_input == '/config':
                        print(f"\n📋 Current Configuration:")
                        print(f"  Temperature: {gen_config.temperature}")
                        print(f"  Max tokens: {gen_config.max_new_tokens}")
                        print(f"  Top-k: {gen_config.top_k}")
                        print(f"  Top-p: {gen_config.top_p}")
                        continue
                    elif user_input == '/examples':
                        self._show_example_prompts()
                        continue
                    elif user_input == '/test':
                        self._run_quick_test(gen_config)
                        continue
                    else:
                        print("❌ Unknown command")
                        continue
                
                if not user_input:
                    continue
                
                # Generate text
                print("🔄 Generating...")
                result = self.generate_text(user_input, gen_config)
                
                # Display results
                print(f"\n🤖 Generated ({result['tokens_generated']} tokens, "
                      f"{result['tokens_per_second']:.1f} tokens/s):")
                print(f"📝 {result['generated_text']}")
                
                # Calculate perplexity if text is long enough
                if len(result['full_text']) > 20:
                    ppl = self.calculate_perplexity(result['full_text'])
                    if ppl != float('inf'):
                        print(f"📊 Perplexity: {ppl:.2f}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                continue
    
    def _show_example_prompts(self):
        """Show example prompts based on WikiText-103 content."""
        examples = [
            "The history of artificial intelligence began",
            "In computer science, machine learning is",
            "The theory of evolution explains",
            "During World War II, many countries",
            "The human brain consists of",
            "Climate change refers to",
            "In mathematics, the concept of",
            "The development of language in humans"
        ]
        
        print("\n📚 Example prompts (based on WikiText-103 content):")
        for i, example in enumerate(examples, 1):
            print(f"  {i}. {example}")
    
    def _run_quick_test(self, gen_config: GenerationConfig):
        """Run a quick test with a simple prompt."""
        test_prompt = "The theory of"
        print(f"\n🧪 Quick test with prompt: '{test_prompt}'")
        
        result = self.generate_text(test_prompt, gen_config)
        print(f"🤖 Result: {result['generated_text']}")
        print(f"📊 Speed: {result['tokens_per_second']:.1f} tokens/s")

def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="BitNet Text Generation Inference")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to trained model checkpoint")
    parser.add_argument("--config_path", type=str, 
                       help="Path to training config file")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run inference on")
    parser.add_argument("--mode", type=str, default="chat",
                       choices=["chat", "generate", "test"],
                       help="Inference mode")
    parser.add_argument("--prompt", type=str,
                       help="Text prompt for generation mode")
    parser.add_argument("--max_tokens", type=int, default=100,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling")
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        print("Available files in directory:")
        model_dir = os.path.dirname(args.model_path)
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith('.pth'):
                    print(f"  📁 {os.path.join(model_dir, f)}")
        return
    
    try:
        # Initialize inference
        print("🚀 Initializing BitNet inference...")
        inference = BitNetInference(
            model_path=args.model_path,
            config_path=args.config_path,
            device=args.device
        )
        
        if args.mode == "chat":
            # Interactive chat mode
            inference.interactive_chat()
            
        elif args.mode == "generate":
            # Single generation mode
            if not args.prompt:
                print("❌ Please provide a prompt with --prompt")
                return
            
            gen_config = GenerationConfig(
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            
            result = inference.generate_text(args.prompt, gen_config)
            
            print(f"\n📝 Prompt: {result['prompt']}")
            print(f"🤖 Generated: {result['generated_text']}")
            print(f"📊 Stats: {result['tokens_generated']} tokens, "
                  f"{result['tokens_per_second']:.1f} tokens/s")
        
        elif args.mode == "test":
            # Quick test mode
            gen_config = GenerationConfig(max_new_tokens=30, temperature=0.8)
            test_prompts = [
                "The history of",
                "In computer science",
                "The theory of evolution"
            ]
            
            print("🧪 Running quick tests...")
            for prompt in test_prompts:
                print(f"\n📝 Testing: '{prompt}'")
                result = inference.generate_text(prompt, gen_config)
                print(f"🤖 Result: {result['generated_text']}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check that the model checkpoint file exists")
        print("2. Ensure BitText modules are in the Python path")
        print("3. Verify PyTorch version compatibility")
        print("4. Try with --device cpu if GPU issues")

if __name__ == "__main__":
    main()