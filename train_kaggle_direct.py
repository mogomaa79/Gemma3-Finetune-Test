#!/usr/bin/env python3
"""
Direct Python training script for Kaggle
This bypasses shell argument parsing issues
"""

import sys
import os

# Add project root to Python path
project_root = "/kaggle/working/Gemma3-Finetune-Test"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import transformers
from src.params import DataArguments, ModelArguments, TrainingArguments

def main():
    print("🚀 Starting Kaggle training for Gemma3-270M-IT...")
    print(f"📍 Working directory: {os.getcwd()}")
    print(f"🐍 Python path: {sys.path[:3]}...")
    print(f"💾 CUDA memory config: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    
    # Create arguments programmatically to avoid parsing issues
    model_args = ModelArguments(
        model_id="google/gemma-3-270m-it"
    )
    
    data_args = DataArguments(
        data_path="data.json"
    )
    
    training_args = TrainingArguments(
        output_dir="./outputs/gemma3_sft_kaggle",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        save_steps=200,
        save_total_limit=2,
        learning_rate=2e-5,
        logging_steps=10,
        fp16=True,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        
        # LoRA specific arguments
        lora_enable=True,
        freeze_llm=True,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1,
        
        # Quantization arguments
        bits=4,
        double_quant=True,
        quant_type="nf4",
        
        # Model specific
        max_seq_length=1024,
        
        # DeepSpeed
        deepspeed="scripts/zero3_offload.json",
        
        # Disable problematic features
        report_to=None,
    )
    
    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Import and run training function
    from src.train.train_sft import train_with_args
    
    try:
        train_with_args(model_args, data_args, training_args)
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
