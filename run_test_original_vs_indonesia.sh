#!/bin/bash

# Original EQTransformer vs Indonesia Dataset Test Script  
# Clean and modular approach

echo "🌍 Original EQTransformer vs Indonesia Dataset Test"
echo "======================================================="

# =============================================================================
# KONFIGURASI MODEL COMPARISON
# =============================================================================

# Original models to test
ORIGINAL_MODEL="models/original_model/EqT_original_model.h5"
CONSERVATIVE_MODEL="models/original_model_conservative/EqT_model_conservative.h5"

# Indonesia 6K dataset (compatible with original model)
TEST_CSV_6K="datasets/indonesia_valid_6k.csv"
TEST_HDF5_6K="datasets/indonesia_valid_6k.hdf5"

# Testing parameters
BATCH_SIZE=32

# Which model to test (original or conservative)
MODEL_TYPE="original"  # Change to "conservative" for conservative model

# =============================================================================
# MODEL SELECTION
# =============================================================================

# Generate timestamp for unique output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

if [ "$MODEL_TYPE" = "conservative" ]; then
    MODEL_PATH="$CONSERVATIVE_MODEL"
    MODEL_NAME="Conservative EQTransformer"
    OUTPUT_DIR="test_results/original_conservative_vs_indonesia_6k_${TIMESTAMP}"
else
    MODEL_PATH="$ORIGINAL_MODEL"
    MODEL_NAME="Original EQTransformer"
    OUTPUT_DIR="test_results/original_model_vs_indonesia_6k_${TIMESTAMP}"
fi

echo "📋 KONFIGURASI TEST:"
echo "   Model: $MODEL_NAME"
echo "   Model file: $MODEL_PATH"
echo "   Test dataset: Indonesia 6K (358 traces, 6000 samples)"
echo "   Input shape: (6000, 3) - Compatible with original model"
echo "   Output directory: $OUTPUT_DIR"
echo "   Purpose: Compare pre-trained global model vs Indonesia data"
echo ""

# =============================================================================
# VALIDATION CHECKS
# =============================================================================

# Check if original model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    echo "   Make sure the original model files are in models/ directory"
    exit 1
fi

# Check if 6K dataset exists, create if needed
if [ ! -f "$TEST_CSV_6K" ] || [ ! -f "$TEST_HDF5_6K" ]; then
    echo "❌ Indonesia 6K dataset not found!"
    echo "   CSV: $TEST_CSV_6K"
    echo "   HDF5: $TEST_HDF5_6K"
    echo ""
    echo "💡 TIP: Run 'python create_indonesia_6k_dataset.py' first to create the dataset"
    echo "❌ Exiting..."
    exit 1
fi

# Confirmation
read -p "🚀 Start testing $MODEL_NAME dengan Indonesia 6K dataset? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Testing dibatalkan"
    exit 1
fi

# =============================================================================
# EXECUTION
# =============================================================================

# Activate environment
echo "🔧 Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eqt

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment 'eqt'"
    exit 1
fi

# Install required dependencies for original model
echo "🔧 Installing original model dependencies..."
pip install keras-self-attention keras-transformer > /dev/null 2>&1

echo ""
echo "🚀 MENJALANKAN ORIGINAL MODEL TESTING..."
echo "Perintah: python EQTransformer/core_ind/test_original_model.py"
echo ""

# Run the dedicated Python script
python EQTransformer/core_ind/test_original_model.py \
    --model "$MODEL_PATH" \
    --test_csv "$TEST_CSV_6K" \
    --test_hdf5 "$TEST_HDF5_6K" \
    --batch_size $BATCH_SIZE \
    --output_dir "$OUTPUT_DIR"

# Store exit status
TEST_EXIT_STATUS=$?

# Check exit status
if [ $TEST_EXIT_STATUS -eq 0 ]; then
    echo ""
    echo "🎉 ORIGINAL MODEL TESTING BERHASIL!"
    echo ""
    echo "📊 HASIL COMPARISON:"
    echo "   ✅ $MODEL_NAME tested dengan Indonesia 6K dataset"
    echo "   ✅ Results saved to: $OUTPUT_DIR"
    echo "   ✅ Dataset: 358 traces dengan 6000 samples (60s @ 100Hz)"
    echo ""
    echo "🔍 COMPARISON ANALYSIS:"
    echo "   📈 Original Model: Pre-trained global seismic data"
    echo "   Test Data: Indonesia local earthquakes (windowed)"
    echo "   🎯 Purpose: Evaluate global model performance on Indonesia seismology"
    echo ""
    echo "📁 Files generated:"
    echo "   - detailed_test_results.csv: Raw prediction results"
    echo "   - performance_metrics.txt: Comprehensive metrics"
    echo "   - performance_analysis.png: Visualization plots"
    echo ""
    echo "💡 NEXT STEPS:"
    echo "   1. Compare with Indonesia-trained model results"
    echo "   2. Analyze performance differences"
    echo "   3. Consider training Indonesia-specific model for better comparison"
else
    echo ""
    echo "❌ ORIGINAL MODEL TESTING GAGAL!"
    echo "   Exit status: $TEST_EXIT_STATUS"
    echo "   Kemungkinan penyebab:"
    echo "   - Custom objects (SeqSelfAttention, dll) tidak tersedia"
    echo "   - Input shape incompatible (6000 vs 30085 samples)"
    echo "   - Missing dependencies (keras-self-attention, keras-transformer)"
    echo ""
    echo "🔧 SOLUSI:"
    echo "   1. Install dependencies: pip install keras-self-attention keras-transformer"
    echo "   2. Use Indonesia 6K dataset (run create_indonesia_6k_dataset.py first)"
    echo "   3. Or use Indonesia-trained model instead of original model"
    exit 1
fi 