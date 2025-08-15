#!/bin/bash

MODEL_NAME="google/gemma-3-270m-it"

# It is strongly recommended to train Gemma3 models with the `eager` attention implementation instead of `flash_attention_2`

# The img_projector is not included in the vision_lora. You should set tune_img_projector to True to finetune the img_projector.
# Also, it might be a good idea to set additional lr for the img_projector.

export PYTHONPATH=src:$PYTHONPATH

deepspeed src/train/train_sft.py \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --use_liger True \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path /path/to/your/training/data.json \
    --image_folder /path/to/your/image/folder \
    --disable_flash_attn2 True \
    --freeze_projector False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --bf16 True \
    --output_dir output/lora_vision_test1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --save_steps 1 \
    --save_total_limit 10 \