#!/usr/bin/env python3
"""
Import validation script for Kaggle environment
Run this to check if all imports are working correctly
"""

import sys
import os

# Add project root to Python path
project_root = "/kaggle/working/Gemma3-Finetune-Test"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("🔍 Validating imports for Kaggle environment...")
print(f"📁 Project root: {project_root}")
print(f"🐍 Python path: {sys.path[:3]}...")  # Show first 3 entries

try:
    # Test core imports
    print("\n✅ Testing core imports...")
    from src.params import DataArguments, ModelArguments, TrainingArguments
    print("   ✓ src.params imported successfully")
    
    from src.train.train_utils import get_peft_state_maybe_zero_3
    print("   ✓ src.train.train_utils imported successfully")
    
    from src.train.monkey_patch_forward import replace_gemma3_forward
    print("   ✓ src.train.monkey_patch_forward imported successfully")
    
    from src.trainer import GemmaSFTTrainer
    print("   ✓ src.trainer imported successfully")
    
    from src.dataset import make_supervised_data_module
    print("   ✓ src.dataset imported successfully")
    
    print("\n🎉 All imports successful! Ready for training.")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("\n🔧 Debugging info:")
    print(f"   Current working directory: {os.getcwd()}")
    print(f"   Files in src/: {os.listdir('src') if os.path.exists('src') else 'src directory not found'}")
    print(f"   Files in src/train/: {os.listdir('src/train') if os.path.exists('src/train') else 'src/train directory not found'}")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    sys.exit(1)

print("\n🚀 You can now run: bash scripts/finetune_kaggle_fixed.sh")
