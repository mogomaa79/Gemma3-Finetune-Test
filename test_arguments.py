#!/usr/bin/env python3
"""
Test argument parsing for debugging
Run this to see what arguments are actually supported
"""

import sys
import os

# Add project root to Python path
project_root = "/kaggle/working/Gemma3-Finetune-Test"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import transformers
    from src.params import DataArguments, ModelArguments, TrainingArguments
    
    print("🔍 Testing argument parser...")
    
    # Create parser
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    # Test with minimal arguments
    test_args = [
        "--model_id", "google/gemma-3-270m-it",
        "--data_path", "data.json",
        "--output_dir", "./test_output",
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", "1",
        "--learning_rate", "2e-5",
        "--logging_steps", "10",
        "--save_steps", "100"
    ]
    
    print(f"Testing with args: {test_args}")
    
    # Save original sys.argv
    original_argv = sys.argv.copy()
    
    try:
        # Replace sys.argv for testing
        sys.argv = ["test_script.py"] + test_args
        
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        print("✅ Basic argument parsing successful!")
        
        # Test with evaluation strategy
        sys.argv = ["test_script.py"] + test_args + ["--evaluation_strategy", "steps"]
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        print("✅ Evaluation strategy argument works!")
        
        # Test with other potentially problematic args
        problematic_args = [
            ["--save_strategy", "steps"],
            ["--lr_scheduler_type", "cosine"],
            ["--warmup_ratio", "0.1"],
            ["--weight_decay", "0.01"],
            ["--fp16"],
            ["--gradient_checkpointing"],
            ["--remove_unused_columns", "False"]
        ]
        
        for arg_pair in problematic_args:
            try:
                sys.argv = ["test_script.py"] + test_args + arg_pair
                model_args, data_args, training_args = parser.parse_args_into_dataclasses()
                print(f"✅ {arg_pair[0]} works!")
            except Exception as e:
                print(f"❌ {arg_pair[0]} failed: {e}")
                
    finally:
        # Restore original sys.argv
        sys.argv = original_argv
        
    print("\n🎉 Argument testing completed!")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
