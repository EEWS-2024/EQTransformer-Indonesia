#!/bin/bash

# EQTransformer Fine-tuning Script for Indonesia Data 🌋
# Complete pipeline untuk fine-tuning original model dengan data Indonesia
# Features: GPU setup, plotting, validation testing, comprehensive logging

echo "🌋 EQTransformer Indonesia Fine-tuning Pipeline"
echo "=============================================================="

# =============================================================================
# KONFIGURASI PARAMETER - EDIT SESUAI KEBUTUHAN
# =============================================================================

# Model paths
ORIGINAL_MODEL="models/original_model/EqT_original_model.h5"
CONSERVATIVE_MODEL="models/original_model_conservative/EqT_model_conservative.h5"

# Pilih model mana yang akan di-finetune
MODEL_TYPE="original"  # Change to "conservative" for conservative model

# Fine-tuning parameters
EPOCHS=200
BATCH_SIZE=256
LEARNING_RATE=0.0001
PATIENCE=5
AUGMENTATION="true"

# Detection threshold parameters
P_THRESHOLD=0.05
S_THRESHOLD=0.05
DET_THRESHOLD=0.05

# Data directories
INPUT_DIR="datasets"

# =============================================================================
# SETUP GPU ENVIRONMENT
# =============================================================================

echo ""
echo "🔧 STEP 1: Setting up GPU environment..."
echo "=" * 55

# Activate conda environment
echo "🐍 Activating conda environment 'eqt'..."
eval "$(conda shell.bash hook)"
conda activate eqt

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment 'eqt'"
    echo "   Please create environment first or check environment name"
    exit 1
fi

echo "✅ Conda environment 'eqt' activated"

# Fix CUDA symbolic links
echo "🔧 Fixing CUDA symbolic links..."
NVIDIA_BASE="$CONDA_PREFIX/lib/python3.8/site-packages/nvidia"

if [ ! -d "$NVIDIA_BASE" ]; then
    echo "❌ NVIDIA packages directory not found: $NVIDIA_BASE"
    echo "   Please install GPU dependencies first"
    exit 1
fi

# Fix all CUDA library symbolic links
echo "   Fixing CUDA runtime..."
cd "$NVIDIA_BASE/cuda_runtime/lib/"
ln -sf libcudart.so.11.0 libcudart.so.11 2>/dev/null
ln -sf libcudart.so.11.0 libcudart.so 2>/dev/null

echo "   Fixing cuBLAS..."
cd "$NVIDIA_BASE/cublas/lib/"
ln -sf libcublas.so.11 libcublas.so 2>/dev/null
ln -sf libcublasLt.so.11 libcublasLt.so 2>/dev/null

echo "   Fixing cuFFT..."
cd "$NVIDIA_BASE/cufft/lib/"
ln -sf libcufft.so.10 libcufft.so 2>/dev/null

echo "   Fixing cuRAND..."
cd "$NVIDIA_BASE/curand/lib/"
ln -sf libcurand.so.10 libcurand.so 2>/dev/null

echo "   Fixing cuSOLVER..."
cd "$NVIDIA_BASE/cusolver/lib/"
ln -sf libcusolver.so.11 libcusolver.so 2>/dev/null

echo "   Fixing cuSPARSE..."
cd "$NVIDIA_BASE/cusparse/lib/"
ln -sf libcusparse.so.11 libcusparse.so 2>/dev/null

echo "   Fixing cuDNN..."
cd "$NVIDIA_BASE/cudnn/lib/"
ln -sf libcudnn.so.8 libcudnn.so 2>/dev/null

echo "✅ CUDA symbolic links fixed"

# Setup GPU environment variables
export LD_LIBRARY_PATH="$NVIDIA_BASE/cuda_runtime/lib:$NVIDIA_BASE/cudnn/lib:$NVIDIA_BASE/cublas/lib:$NVIDIA_BASE/cufft/lib:$NVIDIA_BASE/curand/lib:$NVIDIA_BASE/cusolver/lib:$NVIDIA_BASE/cusparse/lib:$LD_LIBRARY_PATH"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1

echo "✅ GPU environment variables set"

# Return to working directory
cd /home/mooc_parallel_2021_003/new/set-up-jupyterlab-gcp/docker_jupyterlab/jupyter-data/EQTransformer-Indonesia

# Test GPU availability
echo "🔍 Testing GPU availability..."
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'GPU devices found: {len(gpus)}')
if gpus:
    print('✅ GPU is ready for fine-tuning!')
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
else:
    print('⚠️  No GPU detected - will use CPU (slower)')
" 2>/dev/null

# =============================================================================
# STEP 2: CHECK PREREQUISITES
# =============================================================================

echo ""
echo "🔍 STEP 2: Checking prerequisites..."
echo "=" * 55

# Determine which model to use
if [ "$MODEL_TYPE" = "conservative" ]; then
    MODEL_PATH="$CONSERVATIVE_MODEL"
    echo "🎯 Using conservative model: $MODEL_PATH"
else
    MODEL_PATH="$ORIGINAL_MODEL"
    echo "🎯 Using original model: $MODEL_PATH"
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    echo "   Please ensure original model is available"
    exit 1
fi

echo "✅ Model found: $MODEL_PATH"

# Check if fine-tuning datasets exist
TRAIN_FINETUNE="$INPUT_DIR/indonesia_finetune_train.hdf5"
VALID_FINETUNE="$INPUT_DIR/indonesia_finetune_valid.hdf5"

if [ ! -f "$TRAIN_FINETUNE" ] || [ ! -f "$VALID_FINETUNE" ]; then
    echo "❌ Fine-tuning datasets not found!"
    echo "   Missing files:"
    [ ! -f "$TRAIN_FINETUNE" ] && echo "     - $TRAIN_FINETUNE"
    [ ! -f "$VALID_FINETUNE" ] && echo "     - $VALID_FINETUNE"
    echo ""
    echo "💡 Run windowing script first:"
    echo "   python create_indonesia_6k_training.py --process_all"
    exit 1
fi

echo "✅ Fine-tuning datasets found:"
echo "   Training: $TRAIN_FINETUNE"
echo "   Validation: $VALID_FINETUNE"

# =============================================================================
# STEP 3: START FINE-TUNING
# =============================================================================

echo ""
echo "🚀 STEP 3: Starting fine-tuning process..."
echo "=" * 55

echo "📋 Fine-tuning configuration:"
echo "   Model: $MODEL_PATH"
echo "   Epochs: $EPOCHS"
echo "   Batch size: $BATCH_SIZE"
echo "   Learning rate: $LEARNING_RATE"
echo "   Early stopping patience: $PATIENCE"
echo "   Data augmentation: $AUGMENTATION"
echo "   P-wave threshold: $P_THRESHOLD"
echo "   S-wave threshold: $S_THRESHOLD"
echo "   Detection threshold: $DET_THRESHOLD"

echo ""
echo "🔥 Starting fine-tuning with GPU acceleration..."
echo "   This will create a timestamped directory in models/"
echo "   All results (plots, models, logs, tests) will be saved there"

START_TIME=$(date +%s)

# Run fine-tuning with all parameters
python -u finetune_original_indonesia.py \
    --model "$MODEL_PATH" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --patience $PATIENCE \
    --p_threshold $P_THRESHOLD \
    --s_threshold $S_THRESHOLD \
    --det_threshold $DET_THRESHOLD \
    $( [ "$AUGMENTATION" = "true" ] && echo "--augmentation" )

FINETUNE_EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# =============================================================================
# STEP 4: RESULTS SUMMARY
# =============================================================================

echo ""
echo "🎊 STEP 4: Fine-tuning pipeline completed!"
echo "=" * 55

if [ $FINETUNE_EXIT_CODE -eq 0 ]; then
    echo "✅ Fine-tuning completed successfully!"
    echo "⏱️  Total duration: $((DURATION/60)) minutes $((DURATION%60)) seconds"
    
    # Find the latest created model directory
    LATEST_MODEL_DIR=$(ls -dt models/finetune_indonesia_* 2>/dev/null | head -1)
    
    if [ -n "$LATEST_MODEL_DIR" ]; then
        echo ""
        echo "📁 Results saved in: $LATEST_MODEL_DIR"
        echo ""
        echo "📊 Files created:"
        echo "   🧠 Models:"
        [ -f "$LATEST_MODEL_DIR/model/best_finetuned_model.h5" ] && echo "      - Best model: $LATEST_MODEL_DIR/model/best_finetuned_model.h5"
        [ -f "$LATEST_MODEL_DIR/model/final_finetuned_model.h5" ] && echo "      - Final model: $LATEST_MODEL_DIR/model/final_finetuned_model.h5"
        
        echo "   📈 Plots:"
        [ -f "$LATEST_MODEL_DIR/plots/training_history_comprehensive.png" ] && echo "      - Comprehensive plots: $LATEST_MODEL_DIR/plots/training_history_comprehensive.png"
        [ -f "$LATEST_MODEL_DIR/plots/loss_curves.png" ] && echo "      - Loss curves: $LATEST_MODEL_DIR/plots/loss_curves.png"
        
        echo "   📋 Logs:"
        [ -f "$LATEST_MODEL_DIR/logs/training_summary.txt" ] && echo "      - Training summary: $LATEST_MODEL_DIR/logs/training_summary.txt"
        [ -f "$LATEST_MODEL_DIR/fine_tuning_complete_info.json" ] && echo "      - Complete info: $LATEST_MODEL_DIR/fine_tuning_complete_info.json"
        
        echo "   🧪 Test Results:"
        [ -f "$LATEST_MODEL_DIR/results/validation_results.json" ] && echo "      - Validation results: $LATEST_MODEL_DIR/results/validation_results.json"
        [ -f "$LATEST_MODEL_DIR/results/prediction_sample.png" ] && echo "      - Prediction plots: $LATEST_MODEL_DIR/results/prediction_sample.png"
        
        echo ""
        echo "🎯 NEXT STEPS:"
        echo "   1. View training curves:"
        echo "      open $LATEST_MODEL_DIR/plots/loss_curves.png"
        echo ""
        echo "   2. Check validation results:"
        echo "      cat $LATEST_MODEL_DIR/results/validation_results.json"
        echo ""
        echo "   3. Use fine-tuned model for predictions:"
        echo "      python your_prediction_script.py --model $LATEST_MODEL_DIR/model/best_finetuned_model.h5"
        echo ""
        echo "   4. Compare with original model performance"
        
        # Display quick summary if available
        if [ -f "$LATEST_MODEL_DIR/fine_tuning_complete_info.json" ]; then
            echo ""
            echo "📊 QUICK SUMMARY:"
            python -c "
import json
try:
    with open('$LATEST_MODEL_DIR/fine_tuning_complete_info.json', 'r') as f:
        info = json.load(f)
    tr = info['training_results']
    ve = info['validation_evaluation']['validation_evaluation']
    print(f'   Final validation loss: {tr[\"final_val_loss\"]:.6f}')
    print(f'   Best validation loss: {tr[\"best_val_loss\"]:.6f}')
    print(f'   Training duration: {tr[\"training_duration_formatted\"]}')
    print(f'   Validation samples tested: {ve[\"total_traces\"]}')
except:
    pass
            " 2>/dev/null
        fi
        
    else
        echo "⚠️  Fine-tuning completed but output directory not found"
    fi
    
else
    echo "❌ Fine-tuning failed with exit code: $FINETUNE_EXIT_CODE"
    echo "   Check the error messages above for troubleshooting"
fi

echo ""
echo "🌋 EQTransformer Indonesia Fine-tuning Pipeline Finished"
echo "==============================================================" 