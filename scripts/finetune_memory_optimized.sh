#!/bin/bash

MODEL_NAME="google/gemma-3-270m-it"

export PYTHONPATH=src:$PYTHONPATH
# Set PyTorch memory management for better fragmentation handling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Memory-optimized configuration for Gemma3-270M on limited GPU memory
# This configuration is optimized for GPUs with 16GB VRAM or less

deepspeed src/train/train_sft.py \
    --use_liger True \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path data.json \
    --image_folder ./ \
    --disable_flash_attn2 True \
    --lora_enable True \
    --vision_lora False \
    --use_dora False \
    --lora_rank 32 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --num_lora_modules -1 \
    --freeze_projector False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --bits 4 \
    --bf16 True \
    --output_dir output/test_memory_optimized \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --projector_lr 1e-5 \
    --vision_lr 1e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --dataloader_num_workers 2 \
    --optim "adamw_bnb_8bit" \
    --max_seq_length 32768
