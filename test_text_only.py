#!/usr/bin/env python3
"""
Test script to verify text-only training works with the modified code.
"""
import sys
import os
sys.path.insert(0, 'src')

import torch
from transformers import Gemma3ForConditionalGeneration, AutoTokenizer
from src.dataset.sft_dataset import SupervisedDataset
from src.params import DataArguments

def test_model_loading():
    """Test if we can load the text-only Gemma 270m model."""
    print("Testing model loading...")
    try:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            "google/gemma-3-270m",
            torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True,
        )
        print("✅ Model loaded successfully")
        print(f"Model type: {type(model)}")
        print(f"Has vision_tower: {hasattr(model, 'vision_tower')}")
        print(f"Has multi_modal_projector: {hasattr(model, 'multi_modal_projector')}")
        return model
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None

def test_tokenizer():
    """Test tokenizer loading."""
    print("\nTesting tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
        print("✅ Tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        print(f"❌ Tokenizer loading failed: {e}")
        return None

def test_dataset():
    """Test dataset loading with our German-English data."""
    print("\nTesting dataset...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
        
        # Create mock data args
        data_args = DataArguments()
        data_args.image_folder = "./"
        
        dataset = SupervisedDataset(
            data_path="data.json",
            processor=tokenizer,
            data_args=data_args
        )
        
        print(f"✅ Dataset loaded successfully with {len(dataset)} examples")
        
        # Test first example
        first_example = dataset[0]
        print(f"First example keys: {first_example.keys()}")
        print(f"Input IDs shape: {first_example['input_ids'].shape}")
        print(f"Labels shape: {first_example['labels'].shape}")
        
        return dataset
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return None

def main():
    print("🧪 Testing Text-Only Training Setup")
    print("=" * 50)
    
    model = test_model_loading()
    tokenizer = test_tokenizer()
    dataset = test_dataset()
    
    if model and tokenizer and dataset:
        print("\n🎉 All tests passed! Ready for text-only training.")
        print("\nTo run training, execute:")
        print("PYTHONPATH=$(pwd):$PYTHONPATH ./scripts/finetune.sh")
    else:
        print("\n❌ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
