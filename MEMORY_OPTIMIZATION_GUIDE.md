# Memory Optimization Guide for Gemma3-270M Training

## Problem
CUDA out of memory error occurred with 15.89 GiB GPU capacity, indicating that even the 270M model requires significant memory optimization for training.

## Memory Optimization Strategies

### 1. Immediate Solutions

#### Set Environment Variable (Apply this first!)
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
This helps with memory fragmentation issues.

#### Clear GPU Memory
```python
import torch
torch.cuda.empty_cache()
```

### 2. Training Configuration Options

#### Option A: Memory Optimized (16GB GPU)
```bash
bash scripts/finetune_memory_optimized.sh
```
**Features:**
- 4-bit quantization
- LoRA with rank 32
- Frozen vision tower
- DeepSpeed ZeRO-3 with offloading
- Gradient accumulation steps: 8
- Max sequence length: 32K

#### Option B: Ultra Memory Optimized (12GB GPU)
```bash
bash scripts/finetune_ultra_memory_optimized.sh
```
**Features:**
- 4-bit quantization
- LoRA with rank 16
- Frozen projector and vision tower
- Only 8 LoRA modules
- Gradient accumulation steps: 16
- Max sequence length: 16K

#### Option C: Extreme Memory Optimized (8-12GB GPU)
```bash
bash scripts/finetune_extreme_memory_optimized.sh
```
**Features:**
- 4-bit quantization
- LoRA with rank 8
- Frozen projector and vision tower
- Only 4 LoRA modules
- Gradient accumulation steps: 32
- Max sequence length: 8K

### 3. Manual Memory Optimization Parameters

#### Reduce Batch Size and Sequence Length
```bash
--per_device_train_batch_size 1
--gradient_accumulation_steps 16  # Increase to maintain effective batch size
--max_seq_length 8192             # Reduce from default 131072
```

#### Enable Quantization
```bash
--bits 4                          # Use 4-bit quantization
--optim "adamw_bnb_8bit"         # Use 8-bit optimizer
```

#### Use LoRA Instead of Full Fine-tuning
```bash
--lora_enable True
--lora_rank 8                     # Smaller rank = less memory
--lora_alpha 8
--freeze_llm True                 # Freeze language model
--freeze_vision_tower True        # Freeze vision components
--freeze_projector True           # Freeze projector
```

#### Enable Memory-Saving Features
```bash
--gradient_checkpointing True     # Trade compute for memory
--use_liger True                  # Use optimized kernels
--lazy_preprocess True            # Load data on demand
--dataloader_num_workers 1        # Reduce worker processes
```

#### DeepSpeed Configuration
Use the most aggressive offloading:
```bash
--deepspeed scripts/zero3_ultra_offload.json
```

### 4. Model-Specific Optimizations

#### Reduce Vision Components
```bash
--num_lora_modules 4              # Target only 4 modules
--lora_namespan_exclude "['lm_head', 'embed_tokens', 'vision_tower']"
```

#### Optimize Checkpointing
```bash
--save_strategy "steps"
--save_steps 2000                 # Save less frequently
--save_total_limit 1              # Keep only 1 checkpoint
```

### 5. System-Level Optimizations

#### Before Training
```bash
# Clear system cache
sudo sync && sudo echo 3 > /proc/sys/vm/drop_caches

# Set GPU memory fraction (if using other processes)
export CUDA_VISIBLE_DEVICES=0
```

#### Monitor Memory Usage
```bash
# Watch GPU memory
watch -n 1 nvidia-smi

# Monitor during training
nvidia-smi dmon -s m
```

### 6. Data Preprocessing Optimizations

#### Reduce Data Loading Memory
- Use `--lazy_preprocess True`
- Reduce `--dataloader_num_workers`
- Preprocess images to smaller sizes
- Use fewer frames for video data

### 7. Alternative Approaches

#### If Still Out of Memory:
1. **Use CPU Offloading**: All optimizations above use CPU offloading
2. **Reduce Model Components**: Freeze more components
3. **Use Gradient Accumulation**: Increase steps, reduce batch size
4. **Split Training**: Train language and vision components separately

#### For Development/Testing:
```bash
# Text-only training (no vision)
--freeze_vision_tower True
--freeze_projector True
# Use smaller dataset for testing
```

## Recommended Approach

1. **Start with**: `finetune_memory_optimized.sh`
2. **If OOM**: Try `finetune_ultra_memory_optimized.sh`
3. **If still OOM**: Use `finetune_extreme_memory_optimized.sh`
4. **Last resort**: Further reduce LoRA rank to 4 or use text-only training

## Expected Memory Usage

- **Standard config**: ~14GB VRAM
- **Memory optimized**: ~10GB VRAM  
- **Ultra optimized**: ~8GB VRAM
- **Extreme optimized**: ~6GB VRAM

## Notes

- Training will be slower with heavy optimizations
- LoRA with lower ranks may affect quality
- Gradient accumulation maintains effective batch size
- Monitor training loss to ensure convergence
