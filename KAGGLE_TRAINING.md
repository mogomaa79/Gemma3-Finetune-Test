# Kaggle Training Instructions for Gemma3-270M-IT

## Quick Start for Kaggle

### 1. Setup Environment
```bash
# Navigate to project directory
cd /kaggle/working/Gemma3-Finetune-Test

# Set Python path and memory optimization
export PYTHONPATH="/kaggle/working/Gemma3-Finetune-Test:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install required packages (if needed)
pip install transformers datasets peft deepspeed accelerate bitsandbytes
```

### 2. Validate Setup
```bash
# Check if imports work correctly
python validate_imports.py
```

### 3. Run Training
```bash
# Use the fixed Kaggle script
bash scripts/finetune_kaggle_fixed.sh
```

### 3. Monitor GPU Usage
```bash
# Watch GPU memory usage in another terminal
watch -n 1 nvidia-smi
```

## Kaggle-Specific Optimizations

### Memory Configuration
- **GPU**: T4 (15GB VRAM)
- **Batch Size**: 1 per device
- **Gradient Accumulation**: 8 steps
- **LoRA Rank**: 16 (reduced from 32)
- **4-bit Quantization**: Enabled
- **DeepSpeed ZeRO-3**: With CPU offloading

### Key Features for Kaggle
1. **DeepSpeed ZeRO-3 with Offloading**: Maximizes memory efficiency
2. **4-bit Quantization**: Reduces model memory footprint
3. **LoRA Training**: Only trains adapter weights
4. **No device_map**: Compatible with DeepSpeed ZeRO-3
5. **Gradient Checkpointing**: Trades compute for memory

### Troubleshooting

#### If you get OOM errors:
1. Reduce `per_device_train_batch_size` to 1
2. Increase `gradient_accumulation_steps` to 16
3. Reduce `max_seq_length` to 512
4. Reduce `lora_rank` to 8

#### If training is too slow:
1. Increase `per_device_train_batch_size` to 2
2. Reduce `gradient_accumulation_steps` to 4
3. Disable `gradient_checkpointing`

#### Common Kaggle Issues:
- **TF32 Error**: Fixed (tf32 False in all scripts)
- **Device Map Error**: Fixed (excluded when using DeepSpeed)
- **Memory Error**: Use the provided configurations

### File Structure for Kaggle
```
/kaggle/working/
├── Gemma3-Finetune-Test/
│   ├── data.json                    # Your training data
│   ├── scripts/finetune_kaggle.sh   # Kaggle-optimized script
│   └── outputs/                     # Training outputs
```

### Expected Training Time
- **3 epochs**: ~2-3 hours on Kaggle T4
- **Memory usage**: ~13-14GB VRAM
- **Effective batch size**: 8 (1 × 8 accumulation)

### After Training
```bash
# Check outputs
ls -la outputs/gemma3_sft_kaggle/

# Test the model
python -c "
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
model = Gemma3ForConditionalGeneration.from_pretrained('outputs/gemma3_sft_kaggle')
processor = AutoProcessor.from_pretrained('outputs/gemma3_sft_kaggle')
print('Model loaded successfully!')
"
```
