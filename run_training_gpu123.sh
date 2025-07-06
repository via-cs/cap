#!/bin/bash

# Training Script using GPUs 1, 2, 3 (avoiding GPU 0 which is full)
echo "🚀 Starting CAP Training on GPUs 1, 2, 3"
echo "=========================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=1,2,3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "🔧 Environment Variables Set:"
echo "   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "   NCCL_DEBUG: $NCCL_DEBUG"

echo "📊 Available GPUs (should show 3 GPUs):"
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv

echo ""
echo "🎯 Starting Training..."
python train_et_catp.py --config cap/configs/et_catp_base.yaml

echo "✅ Training completed!" 