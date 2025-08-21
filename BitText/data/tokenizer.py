#!/usr/bin/env python3
"""
Debug script to test tokenizer loading and vocabulary size.
Run this before training to ensure tokenizer works correctly.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

def test_tokenizer():
    """Test tokenizer loading and basic functionality."""
    
    print("Testing tokenizer loading...")
    
    try:
        from transformers import AutoTokenizer
        print("✓ transformers library imported successfully")
        
        # Test loading BERT tokenizer
        tokenizer_name = "bert-base-uncased"
        print(f"Loading tokenizer: {tokenizer_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print("✓ Tokenizer loaded successfully")
        
        # Check vocabulary size
        vocab_size = len(tokenizer)
        print(f"✓ Vocabulary size: {vocab_size}")
        
        # Check pad token
        print(f"Pad token: {tokenizer.pad_token}")
        print(f"Pad token ID: {tokenizer.pad_token_id}")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"✓ Set pad token to: {tokenizer.pad_token}")
        
        # Test tokenization
        test_text = "Hello world, this is a test."
        tokens = tokenizer.encode_plus(
            test_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        print(f"✓ Test tokenization successful")
        print(f"  Input IDs shape: {tokens['input_ids'].shape}")
        print(f"  Attention mask shape: {tokens['attention_mask'].shape}")
        
        # Test vocabulary access
        print(f"✓ Token for 'hello': {tokenizer.tokenize('hello')}")
        print(f"✓ ID for '[CLS]': {tokenizer.convert_tokens_to_ids('[CLS]')}")
        
        # Test callable interface (newer transformers)
        try:
            tokens_alt = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
            print(f"✓ Callable tokenizer interface works too")
        except Exception as e:
            print(f"Note: Callable interface failed: {e}")
            print("Using encode_plus method instead")
        
        return vocab_size, tokenizer
        
    except ImportError as e:
        print(f"✗ Failed to import transformers: {e}")
        print("Please install transformers: pip install transformers")
        return None, None
        
    except Exception as e:
        print(f"✗ Error testing tokenizer: {e}")
        print(f"Error type: {type(e).__name__}")
        return None, None

def test_model_creation():
    """Test basic model creation with known vocab size."""
    
    print("\nTesting model creation...")
    
    try:
        import torch
        import torch.nn as nn
        print("✓ PyTorch imported successfully")
        
        # Test basic embedding layer creation
        vocab_size = 30522  # BERT vocab size
        d_model = 256
        
        print(f"Creating embedding layer with vocab_size={vocab_size}, d_model={d_model}")
        embedding = nn.Embedding(vocab_size, d_model)
        print("✓ Embedding layer created successfully")
        
        # Test forward pass
        input_ids = torch.randint(0, vocab_size, (2, 10))  # batch_size=2, seq_len=10
        embeddings = embedding(input_ids)
        print(f"✓ Forward pass successful, output shape: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing model creation: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("BitText Tokenizer and Model Debug Test")
    print("=" * 60)
    
    # Test tokenizer
    vocab_size, tokenizer = test_tokenizer()
    
    # Test model creation
    model_ok = test_model_creation()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    
    if vocab_size and model_ok:
        print("✓ All tests passed! Training should work.")
        print(f"✓ Vocabulary size: {vocab_size}")
    else:
        print("✗ Some tests failed. Please fix issues before training.")
        
    print("=" * 60)

if __name__ == "__main__":
    main()