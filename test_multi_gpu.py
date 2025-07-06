#!/usr/bin/env python3
"""
Test Multi-GPU Functionality for CAP
"""

import os
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

def test_multi_gpu():
    """Test multi-GPU functionality"""
    print("🧪 Testing Multi-GPU Functionality")
    print("=" * 50)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"🎮 Found {gpu_count} GPU(s)")
    
    if gpu_count < 2:
        print("⚠️ Less than 2 GPUs available, multi-GPU not possible")
        return False
    
    # Check GPU memory
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        free_memory = torch.cuda.mem_get_info(i)[0] / 1024**3
        print(f"   GPU {i}: {gpu_name} ({free_memory:.1f}GB free / {gpu_memory:.1f}GB total)")
        
        if free_memory < 2.0:  # Less than 2GB free
            print(f"   ⚠️ GPU {i} has low memory ({free_memory:.1f}GB free)")
    
    # Test DataParallel with a simple model
    print("\n🔧 Testing DataParallel...")
    try:
        # Create a simple test model
        test_model = nn.Linear(10, 1).cuda()
        
        # Try to wrap with DataParallel
        if gpu_count >= 2:
            test_model_dp = DataParallel(test_model)
            print("✅ DataParallel test successful")
            
            # Test forward pass
            test_input = torch.randn(32, 10).cuda()
            test_output = test_model_dp(test_input)
            print(f"✅ Forward pass successful: {test_output.shape}")
            
            return True
        else:
            print("❌ Not enough GPUs for DataParallel")
            return False
            
    except Exception as e:
        print(f"❌ DataParallel test failed: {e}")
        return False

def setup_environment():
    """Set up environment for multi-GPU training"""
    print("\n🔧 Setting up environment...")
    
    # NCCL settings for stability
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    
    # PyTorch settings
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    print("✅ Environment variables set")

def main():
    print("🎯 CAP Multi-GPU Test Script")
    print("=" * 50)
    
    # Set up environment
    setup_environment()
    
    # Test multi-GPU functionality
    success = test_multi_gpu()
    
    if success:
        print("\n🎉 Multi-GPU test passed! You can proceed with multi-GPU training.")
        print("\n💡 To run multi-GPU training:")
        print("   python train_et_catp.py")
    else:
        print("\n⚠️ Multi-GPU test failed. Consider using single GPU training.")
        print("\n💡 To run single GPU training:")
        print("   export CUDA_VISIBLE_DEVICES=0")
        print("   python train_et_catp.py")

if __name__ == "__main__":
    main() 