#!/bin/bash
# Setup GPU Environment untuk EQTransformer Indonesia Training
# Jalankan: source setup_gpu_eqt.sh

echo "🇮🇩 Setting up GPU environment for EQTransformer Indonesia..."

# Activate EQT environment
conda activate eqt

# Install nvidia packages yang dibutuhkan
echo "📦 Installing NVIDIA packages..."
pip install nvidia-cudnn-cu11==8.6.0.163 nvidia-cublas-cu11==11.10.3.66 nvidia-cuda-runtime-cu11==11.7.99 nvidia-cufft-cu11 nvidia-curand-cu11 nvidia-cusolver-cu11 nvidia-cusparse-cu11

# CUDA Environment Variables untuk EQT environment
export CUDA_HOME=/usr
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/home/mooc_parallel_2021_003/miniconda3/envs/eqt/lib/python3.8/site-packages/nvidia/cuda_runtime/lib:/home/mooc_parallel_2021_003/miniconda3/envs/eqt/lib/python3.8/site-packages/nvidia/cudnn/lib:/home/mooc_parallel_2021_003/miniconda3/envs/eqt/lib/python3.8/site-packages/nvidia/cublas/lib:/home/mooc_parallel_2021_003/miniconda3/envs/eqt/lib/python3.8/site-packages/nvidia/cufft/lib:/home/mooc_parallel_2021_003/miniconda3/envs/eqt/lib/python3.8/site-packages/nvidia/curand/lib:/home/mooc_parallel_2021_003/miniconda3/envs/eqt/lib/python3.8/site-packages/nvidia/cusolver/lib:/home/mooc_parallel_2021_003/miniconda3/envs/eqt/lib/python3.8/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH"

# CUDA NVCC Path untuk libdevice
export CUDA_NVCC_PATH="/home/mooc_parallel_2021_003/miniconda3/envs/eqt/lib/python3.8/site-packages/nvidia/cuda_nvcc"
export PATH="$CUDA_NVCC_PATH/bin:$PATH"

# TensorFlow GPU Optimizations
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1

# Disable XLA JIT compilation untuk avoid libdevice issues
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"

echo "✅ GPU Environment setup complete!"
echo "📋 Current setup:"
echo "   CUDA_HOME: $CUDA_HOME"
echo "   Environment: eqt (Python 3.8)"
echo "   TensorFlow: $(python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null)"
echo "   GPU Status: $(python -c 'import tensorflow as tf; print("Available" if len(tf.config.list_physical_devices("GPU")) > 0 else "Not Available")' 2>/dev/null)"

echo ""
echo "🚀 Ready for EQTransformer Indonesia training with GPU!"
echo "   Untuk debug: cd EQTransformer/core_ind && python train_indonesia.py"
echo "   Quick test: pilih mode 4 (Debug Test)"
echo "   Full training: pilih mode 2/3" 