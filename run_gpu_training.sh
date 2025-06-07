#!/bin/bash
# 🚀 EQTransformer Indonesia GPU Training Launcher

# ====================================================================
# 🎯 TRAINING PARAMETERS - EDIT SESUAI KEBUTUHAN
# ====================================================================
BATCH_SIZE=32        # Batch size (recommended: 2-8 for 30k samples)
EPOCHS=2           # Number of epochs
AUGMENTATION=false   # Data augmentation (true/false)
PATIENCE=3          # Early stopping patience (epochs)
COMBINED_DATA=false # Use combined train+valid data (true/false)
DEBUG_TRACES=""     # Debug mode: limit traces (empty for full dataset)

# Uncomment untuk konfigurasi tertentu:
# BATCH_SIZE=8; EPOCHS=50; PATIENCE=10; AUGMENTATION=true    # Production
# BATCH_SIZE=2; EPOCHS=5; PATIENCE=2; AUGMENTATION=false     # Quick Test  
# BATCH_SIZE=1; EPOCHS=1; PATIENCE=1; DEBUG_TRACES=1         # Ultra Debug

echo "🇮🇩 EQTransformer Indonesia GPU Training 🌋"
echo "========================================"
echo "📊 Configuration:"
echo "   Batch Size: $BATCH_SIZE"
echo "   Epochs: $EPOCHS" 
echo "   Augmentation: $AUGMENTATION"
echo "   Patience: $PATIENCE"
echo "   Combined Data: $COMBINED_DATA"
if [ -n "$DEBUG_TRACES" ]; then
    echo "   🐛 Debug Mode: $DEBUG_TRACES traces only"
fi
echo ""

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
echo "🚀 Starting training with custom parameters..."

# Convert bash true/false to Python True/False
AUG_PYTHON=$(if [ "$AUGMENTATION" = "true" ]; then echo "True"; else echo "False"; fi)
COMBINED_PYTHON=$(if [ "$COMBINED_DATA" = "true" ]; then echo "True"; else echo "False"; fi)

# Create Python command with parameters
PYTHON_CMD="
import sys
sys.path.append('.')
from train_indonesia import train_indonesia_eqt

# Training parameters from bash
batch_size = $BATCH_SIZE
epochs = $EPOCHS
augmentation = $AUG_PYTHON
patience = $PATIENCE
combined_data = $COMBINED_PYTHON"

# Add debug traces if specified
if [ -n "$DEBUG_TRACES" ]; then
    PYTHON_CMD="$PYTHON_CMD
debug_traces = $DEBUG_TRACES"
else
    PYTHON_CMD="$PYTHON_CMD
debug_traces = None"
fi

# Add training call
PYTHON_CMD="$PYTHON_CMD

print('🎯 Starting training with parameters:')
print(f'   batch_size={batch_size}')
print(f'   epochs={epochs}')
print(f'   augmentation={augmentation}')
print(f'   patience={patience}')
print(f'   combined_data={combined_data}')
print(f'   debug_traces={debug_traces}')
print()

result = train_indonesia_eqt(
    batch_size=batch_size,
    epochs=epochs,
    augmentation=augmentation,
    patience=patience,
    use_combined_data=combined_data,
    debug_traces=debug_traces
)

if result:
    print('🎉 Training completed successfully!')
else:
    print('❌ Training failed!')
"

# Execute Python training
python -c "$PYTHON_CMD"

echo ""
echo "🎉 Training launcher completed!" 