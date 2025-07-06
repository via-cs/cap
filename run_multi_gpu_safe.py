#!/usr/bin/env python3
"""
Safe Multi-GPU Training Script for CAP
Handles NCCL issues and provides fallback options
"""

import os
import sys
import torch
import subprocess
import argparse

def setup_environment():
    """Set up environment variables for stable multi-GPU training"""
    # NCCL settings for stability
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    
    # PyTorch settings
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    print("🔧 Environment variables set for multi-GPU training")

def check_gpu_availability():
    """Check GPU availability and memory"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"🎮 Found {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        free_memory = torch.cuda.mem_get_info(i)[0] / 1024**3
        print(f"   GPU {i}: {gpu_name} ({free_memory:.1f}GB free / {gpu_memory:.1f}GB total)")
    
    return gpu_count >= 2

def run_training_with_fallback():
    """Run training with fallback options if multi-GPU fails"""
    print("🚀 Starting multi-GPU training...")
    
    try:
        # Try multi-GPU training first
        result = subprocess.run([
            sys.executable, 'train_et_catp.py',
            '--config', 'cap/configs/cap/et_cap_trans.yaml'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Multi-GPU training completed successfully!")
            print(result.stdout)
            return True
        else:
            print("⚠️ Multi-GPU training failed, trying single GPU...")
            print("Error:", result.stderr)
            
    except Exception as e:
        print(f"❌ Multi-GPU training failed: {e}")
    
    # Fallback to single GPU
    print("🔄 Falling back to single GPU training...")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    try:
        result = subprocess.run([
            sys.executable, 'train_et_catp.py',
            '--config', 'cap/configs/cap/et_cap_trans.yaml'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Single GPU training completed successfully!")
            print(result.stdout)
            return True
        else:
            print("❌ Single GPU training also failed")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Single GPU training failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Safe Multi-GPU Training for CAP")
    parser.add_argument('--single-gpu', action='store_true', help='Force single GPU training')
    parser.add_argument('--config', type=str, default='cap/configs/cap/et_cap_trans.yaml', help='Config file path')
    args = parser.parse_args()
    
    print("🎯 CAP Multi-GPU Training Script")
    print("=" * 50)
    
    if args.single_gpu:
        print("🔧 Forcing single GPU training...")
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        setup_environment()
        
        if not check_gpu_availability():
            print("❌ Not enough GPUs available, switching to single GPU")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Run training
    success = run_training_with_fallback()
    
    if success:
        print("🎉 Training completed successfully!")
    else:
        print("💥 Training failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 