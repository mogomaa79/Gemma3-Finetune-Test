#!/bin/bash

# Kaggle-optimized training script for Gemma3-270M-IT (Conservative version)
# This script uses only well-known arguments to avoid parsing errors

# Ensure we're in the project root
cd /kaggle/working/Gemma3-Finetune-Test

# Add project root to Python path
export PYTHONPATH="/kaggle/working/Gemma3-Finetune-Test:$PYTHONPATH"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Model and data configuration
MODEL_NAME="google/gemma-3-270m-it"
DATA_PATH="data.json"
OUTPUT_DIR="./outputs/gemma3_sft_kaggle"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "🚀 Starting Kaggle training for Gemma3-270M-IT..."
echo "📍 Working directory: $(pwd)"
echo "🐍 Python path: $PYTHONPATH"
echo "💾 CUDA memory config: $PYTORCH_CUDA_ALLOC_CONF"

# Conservative training parameters - using only essential arguments
python -m src.train.train_sft \
    --model_id $MODEL_NAME \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --logging_steps 10 \
    --fp16 \
    --dataloader_num_workers 0 \
    --gradient_checkpointing \
    --lora_enable \
    --freeze_llm \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --bits 4 \
    --double_quant \
    --quant_type nf4 \
    --max_seq_length 1024 \
    --deepspeed scripts/zero3_offload.json

echo "✅ Training completed! Check outputs in: $OUTPUT_DIR"
