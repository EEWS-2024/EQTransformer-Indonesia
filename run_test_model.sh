#!/bin/bash

# 🇮🇩 EQTransformer Indonesia Model Tester Script 🌋
# Script untuk testing model Indonesia dengan parameter yang mudah diubah

echo "🇮🇩 EQTransformer Indonesia Model Tester 🌋"
echo "=================================================="

# =============================================================================
# KONFIGURASI PARAMETER - EDIT SESUAI KEBUTUHAN
# =============================================================================

# Model path (kosong = auto detect model terbaru)
MODEL_PATH=""

# Test dataset paths  
TEST_CSV="../../datasets/indonesia_valid.csv"
TEST_HDF5="../../datasets/indonesia_valid.hdf5"

# Testing parameters
BATCH_SIZE=32

# Output directory (kosong = auto generate dengan timestamp)
OUTPUT_DIR=""

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