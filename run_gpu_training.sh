#!/bin/bash
# 🚀 EQTransformer Indonesia GPU Training Launcher

echo "🇮🇩 EQTransformer Indonesia GPU Training 🌋"
echo "========================================"

# Activate conda environment
echo "🔧 Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eqt

# Fix CUDA symbolic links (critical!)
echo "🔧 Fixing CUDA symbolic links..."
NVIDIA_BASE="$CONDA_PREFIX/lib/python3.8/site-packages/nvidia"

# Fix all symbolic links
cd "$NVIDIA_BASE/cuda_runtime/lib/" && ln -sf libcudart.so.11.0 libcudart.so.11 && ln -sf libcudart.so.11.0 libcudart.so
cd "$NVIDIA_BASE/cublas/lib/" && ln -sf libcublas.so.11 libcublas.so && ln -sf libcublasLt.so.11 libcublasLt.so
cd "$NVIDIA_BASE/cufft/lib/" && ln -sf libcufft.so.10 libcufft.so
cd "$NVIDIA_BASE/curand/lib/" && ln -sf libcurand.so.10 libcurand.so
cd "$NVIDIA_BASE/cusolver/lib/" && ln -sf libcusolver.so.11 libcusolver.so
cd "$NVIDIA_BASE/cusparse/lib/" && ln -sf libcusparse.so.11 libcusparse.so
cd "$NVIDIA_BASE/cudnn/lib/" && ln -sf libcudnn.so.8 libcudnn.so

# Setup GPU environment
echo "🔧 Setting up GPU environment..."
export LD_LIBRARY_PATH="$NVIDIA_BASE/cuda_runtime/lib:$NVIDIA_BASE/cudnn/lib:$NVIDIA_BASE/cublas/lib:$NVIDIA_BASE/cufft/lib:$NVIDIA_BASE/curand/lib:$NVIDIA_BASE/cusolver/lib:$NVIDIA_BASE/cusparse/lib:$LD_LIBRARY_PATH"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1

# Quick GPU test
echo "🔍 Quick GPU test..."
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'✅ GPU ready: {len(gpus)} device(s)')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print('❌ No GPU detected')
"

# Change to training directory
cd /home/mooc_parallel_2021_003/new/set-up-jupyterlab-gcp/docker_jupyterlab/jupyter-data/EQTransformer-Indonesia/EQTransformer/core_ind

echo ""
echo "🚀 Starting EQTransformer Indonesia training..."
echo "Available modes:"
echo "1. Quick Test (4 batch, 1 epoch)"
echo "2. Small Training (4 batch, 10 epochs)"
echo "3. Full Training (8 batch, 50 epochs)"
echo "4. Debug Test (1 trace, 1 epoch)"
echo ""

# Default to debug mode if no argument
if [ $# -eq 0 ]; then
    echo "No mode specified, using Debug mode (4)..."
    echo "4" | python train_indonesia.py
else
    echo "$1" | python train_indonesia.py
fi

echo ""
echo "🎉 Training completed!" 