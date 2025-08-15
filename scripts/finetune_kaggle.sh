#!/bin/bash

# Kaggle-optimized training script for Gemma3-270M-IT
# This script is specifically designed for Kaggle's T4 GPU (15GB VRAM)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Model and data configuration
MODEL_NAME="google/gemma-3-270m-it"
DATA_PATH="data.json"
OUTPUT_DIR="./outputs/gemma3_sft_kaggle"

# Create output directory
mkdir -p $OUTPUT_DIR

# Kaggle-optimized training parameters for T4 GPU
python src/train/train_sft.py \
    --model_id $MODEL_NAME \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --fp16 True \
    --tf32 False \
    --dataloader_num_workers 0 \
    --gradient_checkpointing True \
    --remove_unused_columns False \
    --lora_enable True \
    --freeze_llm True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --bits 4 \
    --double_quant True \
    --quant_type "nf4" \
    --max_seq_length 1024 \
    --report_to none \
    --deepspeed scripts/zero3_offload.json
