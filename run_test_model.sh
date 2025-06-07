#!/bin/bash

# EQTransformer Indonesia Model Tester Script 🌋
# Script untuk testing model Indonesia dengan parameter yang mudah diubah

echo "EQTransformer Indonesia Model Tester 🌋"
echo "=================================================="

# =============================================================================
# KONFIGURASI PARAMETER - EDIT SESUAI KEBUTUHAN
# =============================================================================

# Model path (kosong = auto detect model terbaru)
MODEL_PATH=""

# CATATAN PENTING: Original model tidak kompatibel dengan dataset Indonesia!
# Original model: Input (6000, 3) - 60 detik @ 100Hz
# Indonesia dataset: Input (30085, 3) - 300.8 detik @ 100Hz
# Gunakan model Indonesia yang sudah dilatih untuk input shape yang sesuai

# UNTUK ORIGINAL MODEL (TIDAK KOMPATIBEL):
# MODEL_PATH="models/original_model/EqT_original_model.h5"
# MODEL_PATH="models/original_model_conservative/EqT_model_conservative.h5"

# Test dataset paths  
TEST_CSV="../../datasets/indonesia_valid.csv"
TEST_HDF5="../../datasets/indonesia_valid.hdf5"

# Testing parameters
BATCH_SIZE=32

# Generate timestamp for unique output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Auto-detect latest model if not specified
if [ -z "$MODEL_PATH" ]; then
    # Priority: final_model.h5 (full model) over model_weights.h5 (weights only)
    MODEL_PATH=$(find models/ -name "final_model.h5" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -z "$MODEL_PATH" ]; then
        # Fallback to any .h5 file if no final_model.h5 found
        MODEL_PATH=$(find models/ -name "*.h5" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
    fi
    
    echo "🔍 Auto-detected model: $MODEL_PATH"
fi

# Extract model name for output directory
MODEL_BASENAME=$(basename "$MODEL_PATH" .h5)
OUTPUT_DIR="test_results/test_${MODEL_BASENAME}_${TIMESTAMP}"

# =============================================================================
# ADVANCED CONFIGURATION
# =============================================================================

# Jika ingin menggunakan data training untuk test (tidak direkomendasikan)
# TEST_CSV="datasets/indonesia_train.csv"
# TEST_HDF5="datasets/indonesia_train.hdf5"

# Untuk testing dengan batch size berbeda
# BATCH_SIZE=16   # Untuk GPU dengan memory terbatas
# BATCH_SIZE=64   # Untuk GPU dengan memory besar

# =============================================================================
# AUTO-DETECTION DAN VALIDATION
# =============================================================================

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model tidak ditemukan: $MODEL_PATH"
    echo "   Available models:"
    find models/ -name "*.h5" | head -5
    exit 1
fi

echo "📋 KONFIGURASI TEST:"
echo "   Model: $MODEL_PATH"
echo "   Test dataset: Indonesia valid (411 traces, 30085 samples)"
echo "   Input shape: (30085, 3) - Indonesia-trained model compatible"
echo "   Batch size: $BATCH_SIZE"
echo "   Output directory: $OUTPUT_DIR"
echo "   Purpose: Evaluate Indonesia-trained model performance"
echo ""

# =============================================================================
# EXECUTION - JANGAN EDIT BAGIAN INI
# =============================================================================

# Activate conda environment
echo "🔧 Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eqt

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment 'eqt'"
    echo "   Make sure you have created the environment with setup_gpu_eqt.sh"
    exit 1
fi

# Build command
CMD="python EQTransformer/core_ind/test_indonesia_model.py"

# Add parameters
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --test_csv $TEST_CSV"
CMD="$CMD --test_hdf5 $TEST_HDF5"

if [ -n "$MODEL_PATH" ]; then
    CMD="$CMD --model $MODEL_PATH"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
fi

# Show configuration
echo ""
echo "📋 KONFIGURASI TESTING:"
if [ -n "$MODEL_PATH" ]; then
    echo "   Model: $MODEL_PATH"
else
    echo "   Model: Auto-detect terbaru"
fi
echo "   Test CSV: $TEST_CSV"
echo "   Test HDF5: $TEST_HDF5"
echo "   Batch size: $BATCH_SIZE"
if [ -n "$OUTPUT_DIR" ]; then
    echo "   Output: $OUTPUT_DIR"
else
    echo "   Output: Auto-generate"
fi
echo ""

# Confirmation
read -p "🚀 Mulai testing model? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Testing dibatalkan"
    exit 1
fi

echo ""
echo "🚀 MENJALANKAN MODEL TESTING..."
echo "Perintah: $CMD"
echo ""

# Run the command
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 TESTING BERHASIL DISELESAIKAN!"
    echo ""
    echo "📊 HASIL TESTING:"
    echo "   ✅ Model berhasil ditest"
    echo "   ✅ Metrics tersimpan dalam performance_metrics.txt"
    echo "   ✅ Hasil detail tersimpan dalam detailed_test_results.csv"
    echo "   ✅ Visualisasi tersimpan dalam performance_analysis.png"
    echo ""
    echo "💡 TIP: Cek folder test_results/ untuk melihat semua hasil"
else
    echo ""
    echo "❌ TESTING GAGAL!"
    echo "   Cek log di atas untuk detail error"
    exit 1
fi 