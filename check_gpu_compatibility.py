#!/usr/bin/env python3
"""
GPU Compatibility Check for Gemma3 Training

This script checks your GPU compatibility and recommends optimal training settings.
"""

import torch
import subprocess
import sys

def check_gpu_info():
    """Check GPU information and capabilities"""
    print("🔍 GPU Compatibility Check for Gemma3 Training")
    print("=" * 50)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. GPU training is not possible.")
        return False
    
    # Get GPU information
    gpu_count = torch.cuda.device_count()
    print(f"✅ CUDA is available with {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        major = torch.cuda.get_device_properties(i).major
        minor = torch.cuda.get_device_properties(i).minor
        
        print(f"\n📱 GPU {i}: {gpu_name}")
        print(f"   Memory: {memory_gb:.1f} GB")
        print(f"   Compute Capability: {major}.{minor}")
        
        # Check TF32 support (Ampere = major >= 8)
        tf32_support = major >= 8
        print(f"   TF32 Support: {'✅ Yes' if tf32_support else '❌ No'}")
        
        # Memory recommendations
        if memory_gb >= 24:
            recommended_config = "Standard training (finetune.sh)"
        elif memory_gb >= 16:
            recommended_config = "Memory optimized (finetune_memory_optimized.sh)"
        elif memory_gb >= 12:
            recommended_config = "Ultra memory optimized (finetune_ultra_memory_optimized.sh)"
        elif memory_gb >= 8:
            recommended_config = "Extreme memory optimized (finetune_extreme_memory_optimized.sh)"
        else:
            recommended_config = "Text-only training (finetune_text_only.sh)"
        
        print(f"   Recommended Config: {recommended_config}")
    
    return True

def check_pytorch_version():
    """Check PyTorch version"""
    print(f"\n🐍 PyTorch Version: {torch.__version__}")
    
    # Check if version supports required features
    version_tuple = tuple(map(int, torch.__version__.split('+')[0].split('.')))
    if version_tuple >= (1, 7, 0):
        print("✅ PyTorch version is compatible")
    else:
        print("⚠️  PyTorch version might be too old")

def check_cuda_version():
    """Check CUDA version"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    cuda_version = line.split('release')[1].split(',')[0].strip()
                    print(f"🔧 CUDA Version: {cuda_version}")
                    break
        else:
            print("⚠️  Could not determine CUDA version")
    except FileNotFoundError:
        print("⚠️  nvcc not found in PATH")

def print_recommendations():
    """Print training recommendations"""
    print("\n💡 Training Recommendations:")
    print("=" * 50)
    print("1. Always set: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    print("2. Start with the recommended config based on your GPU memory")
    print("3. If you get OOM errors, try the next more memory-efficient config")
    print("4. Monitor GPU memory usage with: nvidia-smi")
    print("5. For development, consider text-only training first")
    
    print("\n📋 Available Training Scripts:")
    scripts = [
        ("finetune.sh", "Standard training", "24GB+ VRAM"),
        ("finetune_memory_optimized.sh", "Memory optimized", "16GB+ VRAM"),
        ("finetune_ultra_memory_optimized.sh", "Ultra optimized", "12GB+ VRAM"),
        ("finetune_extreme_memory_optimized.sh", "Extreme optimized", "8GB+ VRAM"),
        ("finetune_text_only.sh", "Text-only training", "4GB+ VRAM"),
    ]
    
    for script, description, memory in scripts:
        print(f"   • {script:<35} - {description:<20} ({memory})")

def main():
    """Main function"""
    check_gpu_info()
    check_pytorch_version() 
    check_cuda_version()
    print_recommendations()
    print("\n🚀 Ready to start training! Choose the appropriate script for your setup.")

if __name__ == "__main__":
    main()
