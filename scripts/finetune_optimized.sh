#!/bin/bash

MODEL_NAME="google/gemma-3-270m-it"

export PYTHONPATH=src:$PYTHONPATH

# It is strongly recommended to train Gemma3 models with the `eager` attention implementation instead of `flash_attention_2`
# Optimized configuration for Gemma3-270M model - increased batch sizes for better performance

deepspeed src/train/train_sft.py \
    --use_liger True \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path data.json \
    --image_folder ./ \
    --disable_flash_attn2 True \
    --lora_enable False \
    --freeze_projector False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --bf16 True \
    --output_dir output/test_optimized \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --projector_lr 2e-5 \
    --vision_lr 4e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4
