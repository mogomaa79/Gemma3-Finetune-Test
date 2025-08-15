#!/bin/bash

MODEL_NAME="google/gemma-3-270m-it"

export PYTHONPATH=src:$PYTHONPATH
# Set PyTorch memory management for better fragmentation handling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Text-only training configuration for maximum memory savings
# This configuration completely bypasses vision components

deepspeed src/train/train_sft.py \
    --use_liger True \
    --deepspeed scripts/zero3_ultra_offload.json \
    --model_id $MODEL_NAME \
    --data_path data.json \
    --image_folder ./ \
    --disable_flash_attn2 True \
    --lora_enable True \
    --vision_lora False \
    --use_dora False \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_namespan_exclude "['lm_head', 'embed_tokens', 'vision_tower', 'multi_modal_projector']" \
    --num_lora_modules 8 \
    --freeze_projector True \
    --freeze_vision_tower True \
    --freeze_llm True \
    --bits 4 \
    --bf16 True \
    --output_dir output/test_text_only \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --dataloader_num_workers 2 \
    --optim "adamw_bnb_8bit" \
    --max_seq_length 16384
