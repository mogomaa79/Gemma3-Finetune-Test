#!/bin/bash

# Simple Kaggle training script - uses only basic arguments
# Run this from /kaggle/working/Gemma3-Finetune-Test

# Set up environment
export PYTHONPATH="/kaggle/working/Gemma3-Finetune-Test:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "🚀 Starting simple Kaggle training..."

# Run with minimal arguments to avoid parsing issues
python -m src.train.train_sft \
    --model_id google/gemma-3-270m-it \
    --data_path data.json \
    --output_dir ./outputs/gemma3_sft_simple \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --logging_steps 10 \
    --save_steps 200 \
    --lora_enable \
    --freeze_llm \
    --lora_rank 16 \
    --lora_alpha 32 \
    --bits 4 \
    --max_seq_length 1024 \
    --deepspeed scripts/zero3_offload.json

echo "✅ Simple training completed!"
