#!/bin/bash
# 🇮🇩 EQTransformer Indonesia Training From Scratch Script 🌋
# Script untuk training model EQTransformer Indonesia dari awal (from scratch)
# Menggunakan dataset seismik Indonesia tanpa pre-trained weights

echo "🇮🇩 EQTransformer Indonesia Training From Scratch 🌋"
echo "=========================================================="

# =============================================================================
# KONFIGURASI PARAMETER - EDIT SESUAI KEBUTUHAN
# =============================================================================

# Training parameters
BATCH_SIZE=32       # Batch size untuk training (32 optimal untuk GPU)
EPOCHS=10           # Number of epochs
AUGMENTATION=false  # Data augmentation (true/false)
PATIENCE=3          # Early stopping patience
COMBINED_DATA=false # Gunakan combined train+valid (true) atau terpisah (false)
DEBUG_TRACES=""     # Kosong=full data, atau angka untuk debug mode

# =============================================================================
# ADVANCED CONFIGURATION
# =============================================================================

# Model architecture (default sama seperti original EQTransformer)
# CNN_BLOCKS=5      # Jumlah CNN blocks (default di script)
# LSTM_BLOCKS=2     # Jumlah LSTM blocks (default di script)

# Training strategy
# LEARNING_RATE=0.001       # Learning rate (default di script)
# LOSS_WEIGHTS="0.2,0.3,0.5" # Loss weights for detector,P,S (default di script)

# Data augmentation rates (jika AUGMENTATION=true)
# SHIFT_EVENT_R=0.9         # 90% probability shift event
# ADD_NOISE_R=0.3           # 30% probability add noise
# DROP_CHANNEL_R=0.2        # 20% probability drop channel
# SCALE_AMPLITUDE_R=0.5     # 50% probability scale amplitude
# ADD_GAP_R=0.1             # 10% probability add gap

# =============================================================================
# DATASET INFORMATION
# =============================================================================
# Dataset: Indonesia Seismic Dataset
# - Training: 1642 traces (earthquake_local)
# - Validation: 411 traces (earthquake_local) 
# - Input: 30085 samples (300.8s @ 100Hz) - Signature Indonesia
# - Architecture: 7-layer CNN + BiLSTM, 101,395 parameters
# - Training Type: FROM SCRATCH (no pre-trained weights)
# =============================================================================

# Quick presets - uncomment salah satu untuk konfigurasi cepat:
# BATCH_SIZE=16; EPOCHS=50; PATIENCE=5; AUGMENTATION=true    # Conservative
# BATCH_SIZE=32; EPOCHS=100; PATIENCE=7; AUGMENTATION=false  # Balanced  
# BATCH_SIZE=64; EPOCHS=200; PATIENCE=10; AUGMENTATION=false # Aggressive
# BATCH_SIZE=1; EPOCHS=1; PATIENCE=1; DEBUG_TRACES=1         # Ultra Debug

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
    print('🎉 Training from scratch completed successfully!')
else:
    print('❌ Training from scratch failed!')
"

# Execute Python training
python -c "$PYTHON_CMD"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 TRAINING FROM SCRATCH BERHASIL DISELESAIKAN!"
    echo ""
    echo "📊 HASIL TRAINING:"
    echo "   ✅ Model EQTransformer Indonesia berhasil dilatih dari awal"
    echo "   ✅ Model disimpan di direktori models/"
    echo "   ✅ Training history dan loss curves tersedia"
    echo "   ✅ Model siap untuk testing dan evaluasi"
    echo ""
    echo "🔥 MODEL SPECS:"
    echo "   📐 Architecture: 7-layer CNN + BiLSTM (101,395 parameters)"
    echo "   🇮🇩 Dataset: Indonesia Seismic (1642 train + 411 valid traces)"
    echo "   ⚡ Input: 30085 samples (300.8s @ 100Hz)"
    echo "   🎯 Training: FROM SCRATCH (no pre-trained weights)"
    echo ""
    echo "📊 NEXT STEPS:"
    echo "   1. Test model dengan: ./run_test_model.sh"
    echo "   2. Cek hasil di folder models/indonesia_model_TIMESTAMP/"
    echo "   3. Evaluate performance dengan detailed metrics"
else
    echo ""
    echo "❌ TRAINING FROM SCRATCH GAGAL!"
    echo "   Cek log di atas untuk detail error"
    exit 1
fi 