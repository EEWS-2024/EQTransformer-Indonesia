#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EQTransformer Training untuk Dataset Indonesia 🌋

Script training utama untuk melatih EQTransformer dengan data seismik Indonesia
Input: 30085 samples (300.8 detik @ 100Hz) - 5x lebih panjang dari default EQTransformer
"""

import os
import sys

# 🚀 GPU SETUP - HARUS DI AWAL SEBELUM IMPORT TENSORFLOW!
def setup_gpu_environment_early():
    """Setup GPU environment SEBELUM import TensorFlow"""
    print("🔧 Early GPU setup (before TensorFlow import)...")
    
    # Force conda environment path
    conda_prefix = os.environ.get('CONDA_PREFIX', '/home/mooc_parallel_2021_003/miniconda3/envs/eqt')
    if not conda_prefix.endswith('eqt'):
        conda_prefix = '/home/mooc_parallel_2021_003/miniconda3/envs/eqt'
    
    # Set conda environment variables
    os.environ['CONDA_PREFIX'] = conda_prefix
    os.environ['CONDA_DEFAULT_ENV'] = 'eqt'
    
    # Add conda env paths to Python path
    conda_lib_path = os.path.join(conda_prefix, 'lib/python3.8/site-packages')
    if conda_lib_path not in sys.path:
        sys.path.insert(0, conda_lib_path)
    
    # Set NVIDIA CUDA paths
    nvidia_base = os.path.join(conda_prefix, 'lib/python3.8/site-packages/nvidia')
    
    if os.path.exists(nvidia_base):
        nvidia_paths = [
            f'{nvidia_base}/cuda_runtime/lib',
            f'{nvidia_base}/cudnn/lib', 
            f'{nvidia_base}/cublas/lib',
            f'{nvidia_base}/cufft/lib',
            f'{nvidia_base}/curand/lib',
            f'{nvidia_base}/cusolver/lib',
            f'{nvidia_base}/cusparse/lib'
        ]
        
        # Set LD_LIBRARY_PATH SEBELUM import TensorFlow
        existing_path = os.environ.get('LD_LIBRARY_PATH', '')
        new_path = ':'.join(nvidia_paths + [existing_path] if existing_path else nvidia_paths)
        os.environ['LD_LIBRARY_PATH'] = new_path
        
        # Set TensorFlow GPU settings
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        
        print(f"✅ Set LD_LIBRARY_PATH with {len(nvidia_paths)} NVIDIA paths")
        return True
    else:
        print(f"⚠️ NVIDIA libraries not found at: {nvidia_base}")
        return False

# Setup GPU SEBELUM import apa pun yang berhubungan dengan TensorFlow
gpu_setup_success = setup_gpu_environment_early()

import datetime
import pandas as pd
import h5py
import numpy as np
import time
import matplotlib.pyplot as plt
import logging
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Test GPU setelah import TensorFlow
def test_gpu_after_import():
    """Test GPU availability setelah TensorFlow di-import"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        
        if len(gpus) > 0:
            print(f"🚀 GPU berhasil terdeteksi! {len(gpus)} GPU device(s)")
            
            # Enable memory growth untuk semua GPU
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"💾 Memory growth enabled untuk {gpu}")
                except RuntimeError as e:
                    print(f"⚠️ Warning memory growth: {e}")
            
            # Get GPU details
            for i, gpu in enumerate(gpus):
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    device_name = details.get('device_name', 'Unknown GPU')
                    compute_cap = details.get('compute_capability', 'Unknown')
                    print(f"  GPU {i}: {device_name} (Compute: {compute_cap})")
                except Exception as e:
                    print(f"  GPU {i}: {str(gpu)}")
            
            return True
        else:
            print("⚠️ GPU tidak terdeteksi, akan menggunakan CPU")
            if gpu_setup_success:
                print("💡 GPU setup berhasil tapi TensorFlow tetap tidak detect GPU")
                print("🔧 Mungkin perlu restart Python process")
            return False
            
    except Exception as e:
        print(f"⚠️ Error testing GPU: {e}")
        return False

# Test GPU availability
gpu_available = test_gpu_after_import()

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')  # Back to EQTransformer-Indonesia/
eqt_core_dir = os.path.join(current_dir, '..', 'core')  # Dari core_ind ke core
sys.path.append(eqt_core_dir)

# Import EQTransformer modules
from EqT_utils import DataGenerator, cred2
from tensorflow.keras import Input, Model

# Custom Keras callback dengan tqdm
class TqdmCallback(tf.keras.callbacks.Callback):
    """Custom callback untuk progress bar dengan tqdm"""
    
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        print(f"🚀 Memulai training untuk {self.epochs} epochs...")
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_progress = tqdm(
            total=self.params['steps'], 
            desc=f"Epoch {epoch+1}/{self.epochs}",
            unit='batch',
            leave=False
        )
    
    def on_batch_end(self, batch, logs=None):
        # Update progress bar dengan metrics
        if logs:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in logs.items() if isinstance(v, (int, float))])
            self.epoch_progress.set_postfix_str(metrics_str)
        self.epoch_progress.update(1)
    
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_progress.close()
        # Print epoch summary
        if logs:
            metrics_summary = []
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    metrics_summary.append(f"{key}: {value:.4f}")
            print(f"✅ Epoch {epoch+1} selesai - {' | '.join(metrics_summary)}")
        print()  # Spacing

def check_datasets():
    """Check if datasets are available and validate them."""
    train_file = "../../datasets/indonesia_train.csv"
    valid_file = "../../datasets/indonesia_valid.csv"
    
    if not os.path.exists(train_file):
        print(f"❌ Training dataset not found: {train_file}")
        return False, 0, 0
    
    if not os.path.exists(valid_file):
        print(f"❌ Validation dataset not found: {valid_file}")
        return False, 0, 0
        
    try:
        # Load training and validation datasets separately
        train_df = pd.read_csv(train_file)
        valid_df = pd.read_csv(valid_file)
        
        print(f"📊 Dataset Information:")
        print(f"   Training traces: {len(train_df)}")
        print(f"   Validation traces: {len(valid_df)}")
        print(f"   Total traces: {len(train_df) + len(valid_df)}")
        print(f"   Training split: {len(train_df)/(len(train_df) + len(valid_df))*100:.1f}%")
        print(f"   Validation split: {len(valid_df)/(len(train_df) + len(valid_df))*100:.1f}%")
        
        # Validate required columns exist
        required_cols = ['trace_name', 'p_arrival_sample', 's_arrival_sample', 'coda_end_sample', 'trace_category']
        for df_name, df in [("Training", train_df), ("Validation", valid_df)]:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"❌ {df_name} dataset missing columns: {missing_cols}")
                return False, 0, 0
        
        return True, len(train_df), len(valid_df)
        
    except Exception as e:
        print(f"❌ Error reading datasets: {e}")
        return False, 0, 0

def train_indonesia_eqt(
    batch_size=4,
    epochs=10,
    cnn_blocks=3,
    lstm_blocks=1,
    augmentation=True,
    patience=3,
    use_combined_data=False,
    debug_traces=None
):
    """
    Training EQTransformer untuk dataset Indonesia
    
    Parameters:
    -----------
    batch_size : int (default=4)
        Ukuran batch - kecil karena data 5x lebih besar (30085 vs 6000 samples)
    epochs : int (default=10)
        Jumlah epochs training
    cnn_blocks : int (default=3)
        Jumlah residual CNN blocks
    lstm_blocks : int (default=1)  
        Jumlah BiLSTM blocks
    augmentation : bool (default=True)
        Enable/disable data augmentation. Ketika True:
        
        🔄 MEKANISME AUGMENTASI:
        - Batch dibagi 2: separuh original, separuh augmented
        - Total epochs menjadi 2x lebih panjang (lebih banyak variasi data)
        
        🎯 JENIS AUGMENTASI YANG DITERAPKAN:
        Untuk Data Gempa (earthquake_local):
        • shift_event_r (90%): Menggeser posisi P-S arrival dalam waveform
        • add_noise_r (30%): Menambahkan Gaussian noise ke signal  
        • drop_channe_r (20%): Randomly drop channel (simulasi sensor mati)
        • scale_amplitude_r (50%): Scaling amplitude 1-3x (besar/kecil)
        • pre_emphasis: Pre-emphasis filtering untuk enhance frekuensi tinggi
        
        Untuk Data Noise:
        • drop_channe_r (20%): Drop channels randomly
        • add_gap_r (10%): Menambahkan zeros (simulasi data gaps)
        
        ✅ KEUNTUNGAN: Model lebih robust, generalisasi lebih baik
        ⚠️ COST: Training 2x lebih lama, tapi kualitas model meningkat
        
    patience : int (default=3)
        Early stopping patience - berapa epochs menunggu sebelum stop jika validation loss tidak membaik.
        
        🛑 EARLY STOPPING STRATEGY:
        • patience=1: Sangat agresif - stop begitu val_loss naik
        • patience=3: Balanced - tunggu 3 epochs tanpa improvement  
        • patience=5-10: Konservatif - beri kesempatan lebih untuk recovery
        • patience=0: Disable automatic early stopping
        
        ⚙️ IMPLEMENTATION:
        - Keras EarlyStopping: patience epochs tanpa improvement
        - Simple early stopping: langsung stop jika val_loss naik (jika patience <= 2)
        
    use_combined_data : bool (default=False)
        Gabungkan train+valid atau pisah
    debug_traces : int or None (default=None)
        Jika set, hanya gunakan N traces pertama untuk debugging.
        Contoh: debug_traces=10 untuk test cepat
    """
    
    print("TRAINING EQTRANSFORMER INDONESIA 🌋")
    print("="*80)
    
    # Verifikasi dataset
    dataset_ok, train_traces, valid_traces = check_datasets()
    if not dataset_ok:
        return None
    
    # Setup file paths
    if use_combined_data:
        # Gabungkan training + validation
        print("📋 Mode: Combined training + validation data")
        train_hdf5 = "../../datasets/indonesia_train.hdf5"
        train_csv = "../../datasets/indonesia_train.csv"
        
        # Load dan gabungkan CSV
        df_train = pd.read_csv(train_csv)
        df_valid = pd.read_csv("../../datasets/indonesia_valid.csv")
        df_combined = pd.concat([df_train, df_valid], ignore_index=True)
        
        # Buat temporary combined CSV
        combined_csv = "../../datasets/temp_combined.csv"
        df_combined.to_csv(combined_csv, index=False)
        train_csv = combined_csv
        
        all_traces = df_combined['trace_name'].tolist()
        print(f"   Combined traces: {len(all_traces)}")
        
    else:
        # Pisah training dan validation
        print("📋 Mode: Separate training + validation data")
        train_hdf5 = "../../datasets/indonesia_train.hdf5"
        train_csv = "../../datasets/indonesia_train.csv"
        valid_hdf5 = "../../datasets/indonesia_valid.hdf5"
        valid_csv = "../../datasets/indonesia_valid.csv"
        
        df_train = pd.read_csv(train_csv)
        df_valid = pd.read_csv(valid_csv)
        
        # Use all training data
        training_traces = df_train['trace_name'].tolist()
        
        # Use all validation data  
        validation_traces = df_valid['trace_name'].tolist()
        
        print(f"📊 Data splits:")
        print(f"   Training: {len(training_traces)} traces")
        print(f"   Validation: {len(validation_traces)} traces")
        print(f"   Training %: {len(training_traces)/(len(training_traces)+len(validation_traces))*100:.1f}%")
        print(f"   Validation %: {len(validation_traces)/(len(training_traces)+len(validation_traces))*100:.1f}%")
        
        all_traces = training_traces + validation_traces
        
        # Verifikasi traces exist di HDF5
        print(f"🔍 Verifying traces exist in HDF5...")
        try:
            with h5py.File(train_hdf5, 'r') as f:
                missing_traces = []
                for trace in training_traces[:3]:  # Cek 3 traces pertama training
                    if trace not in f['data']:
                        missing_traces.append(trace)
                        
            with h5py.File(valid_hdf5, 'r') as f:
                for trace in validation_traces[:3]:  # Cek 3 traces pertama validation
                    if trace not in f['data']:
                        missing_traces.append(trace)
                        
            if missing_traces:
                print(f"❌ Missing traces di HDF5: {missing_traces[:5]}...")
                return None
            else:
                print("✅ Traces verified in HDF5 files")
        except Exception as e:
            print(f"⚠️ Could not verify HDF5 files: {e}")
    
    # Debug mode: limit traces untuk testing
    if debug_traces:
        print(f"\n🐛 DEBUG MODE: Menggunakan hanya {debug_traces} traces")
        if use_combined_data:
            original_count = len(all_traces)
            all_traces = all_traces[:debug_traces]  # Ambil N traces pertama
            print(f"   Traces dikurangi dari {original_count} → {len(all_traces)}")
        else:
            original_train = len(training_traces)
            original_valid = len(validation_traces)
            training_traces = training_traces[:debug_traces//2]
            validation_traces = validation_traces[:debug_traces//2]
            print(f"   Training traces: {original_train} → {len(training_traces)}")
            print(f"   Validation traces: {original_valid} → {len(validation_traces)}")
            all_traces = training_traces + validation_traces
        print(f"   ⚡ Training akan sangat cepat untuk testing!")
    
    # Model parameters - define before using in generators
    input_dim = (30085, 3)  # Signature Indonesia: 300.8 detik @ 100Hz
    
    # Create data splits
    if use_combined_data:
        # Split combined data 80:20
        train_split = int(0.8 * len(all_traces))
        training_traces = all_traces[:train_split]
        validation_traces = all_traces[train_split:]
        
        print(f"📊 Data splits from combined:")
        print(f"   Training: {len(training_traces)} traces")
        print(f"   Validation: {len(validation_traces)} traces")
        print(f"   Training %: {len(training_traces)/len(all_traces)*100:.1f}%")
        print(f"   Validation %: {len(validation_traces)/len(all_traces)*100:.1f}%")
    
    # Create data generators
    print(f"🔄 Creating data generators...")
    print(f"   Batch size: {batch_size}")
    
    if use_combined_data:
        # Single training generator with split
        training_generator = DataGenerator(
            list_IDs=training_traces,
            file_name=train_hdf5,
            dim=input_dim[0],
            batch_size=batch_size,
            n_channels=3,
            shuffle=True,
            norm_mode='std',
            label_type='triangle',
            augmentation=augmentation,
            add_event_r=0.6,
            shift_event_r=0.9,
            add_noise_r=0.3,
            drop_channe_r=0.2,
            scale_amplitude_r=0.5,
            pre_emphasis=True,
            add_gap_r=0.1
        )
        
        validation_generator = DataGenerator(
            list_IDs=validation_traces,
            file_name=train_hdf5,
            dim=input_dim[0],
            batch_size=batch_size,
            n_channels=3,
            shuffle=False,
            norm_mode='std',
            label_type='triangle',
            augmentation=False,
            pre_emphasis=True
        )
        
    else:
        # Separate training and validation generators
        training_generator = DataGenerator(
            list_IDs=training_traces,
            file_name=train_hdf5,
            dim=input_dim[0],
            batch_size=batch_size,
            n_channels=3,
            shuffle=True,
            norm_mode='std',
            label_type='triangle',
            augmentation=augmentation,
            add_event_r=0.6,
            shift_event_r=0.9,
            add_noise_r=0.3,
            drop_channe_r=0.2,
            scale_amplitude_r=0.5,
            pre_emphasis=True,
            add_gap_r=0.1
        )
        
        validation_generator = DataGenerator(
            list_IDs=validation_traces,
            file_name=valid_hdf5,  # Use separate validation HDF5
            dim=input_dim[0],
            batch_size=batch_size,
            n_channels=3,
            shuffle=False,
            norm_mode='std',
            label_type='triangle',
            augmentation=False,
            pre_emphasis=True
        )
    
    print(f"✅ Data generators ready!")
    print(f"📊 Training batches: {len(training_generator)}")
    print(f"📊 Validation batches: {len(validation_generator)}")
    print(f"📊 Total training samples: {len(training_traces)}")
    print(f"📊 Total validation samples: {len(validation_traces)}")
    
    print(f"\n⚙️ KONFIGURASI TRAINING:")
    print(f"   Input dimension: {input_dim} (300.8 detik)")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Patience: {patience} epochs")
    print(f"   CNN blocks: {cnn_blocks}")
    print(f"   LSTM blocks: {lstm_blocks}")
    print(f"   Augmentation: {augmentation}")
    print(f"   Combined data: {use_combined_data}")
    if debug_traces:
        print(f"   🐛 DEBUG: Limited to {debug_traces} traces")
        print(f"   🎯 Purpose: Quick functionality test")
    
    # Build model
    print(f"\n🏗️ BUILDING MODEL...")
    inp = Input(shape=input_dim, name='input')
    
    if debug_traces:
        print("🐛 DEBUG: ULTRA-MINIMAL MODEL untuk testing...")
        # Model minimal sekali - hanya linear dense layers
        x = Flatten()(inp)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        
        # Minimal outputs untuk 3 tasks
        detector = Dense(input_dim[0], activation='sigmoid', name='detector')(x)
        picker_P = Dense(input_dim[0], activation='sigmoid', name='picker_P')(x)
        picker_S = Dense(input_dim[0], activation='sigmoid', name='picker_S')(x)
        
        model = Model(inputs=inp, outputs=[detector, picker_P, picker_S])
        
        # Compile dengan learning rate tinggi untuk cepat converge
        model.compile(
            optimizer=Adam(learning_rate=0.01),  # LR tinggi untuk test cepat
            loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
            loss_weights=[1.0, 1.0, 1.0]  # Simple weights
        )
        
        print(f"🐛 DEBUG Model: Ultra-minimal architecture")
        print(f"   - Flatten → Dense(128) → Dense(64) → 3 outputs")
        
    else:
        # Simplified CNN model dengan JUMLAH LAYER SAMA seperti original EQTransformer
        print("🏗️ Simplified CNN model (7 layers - same as original)")
        print(f"   Input size: {input_dim[0]} samples")
        print(f"   Using manual architecture with same layer count as cred2()")
        
        x = inp
        
        # 7 CNN layers - SAMA seperti original EQTransformer  
        filters_list = [8, 16, 16, 32, 32, 64, 64]  # Original nb_filters
        kernel_list = [11, 9, 7, 7, 5, 5, 3]        # Original kernel_size
        
        print(f"   Building {len(filters_list)} CNN layers:")
        for i, (filters, kernel) in enumerate(zip(filters_list, kernel_list)):
            print(f"     Layer {i+1}: Conv1D(filters={filters}, kernel={kernel})")
            x = tf.keras.layers.Conv1D(
                filters, 
                kernel, 
                padding='same',           # KEY: padding='same' untuk avoid shape change
                activation='relu', 
                name=f'conv1d_{i+1}'
            )(x)
            
            # Batch normalization untuk layer 1,2,3,4 (seperti residual blocks)
            if i < 4:  
                x = tf.keras.layers.BatchNormalization(name=f'bn_{i+1}')(x)
            
            # Dropout setiap 2 layer
            if i % 2 == 1:
                x = tf.keras.layers.Dropout(0.1, name=f'dropout_{i+1}')(x)
        
        # BiLSTM layer - sama seperti original
        print(f"   Building BiLSTM layer:")
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True), 
            name='bilstm'
        )(x)
        x = tf.keras.layers.Dropout(0.2, name='bilstm_dropout')(x)
        
        # Output layers - ensure same size as input
        print(f"   Building 3 output heads:")
        detector = tf.keras.layers.Conv1D(1, 1, activation='sigmoid', padding='same', name='detector')(x)
        picker_P = tf.keras.layers.Conv1D(1, 1, activation='sigmoid', padding='same', name='picker_P')(x)  
        picker_S = tf.keras.layers.Conv1D(1, 1, activation='sigmoid', padding='same', name='picker_S')(x)
        
        model = tf.keras.Model(inputs=inp, outputs=[detector, picker_P, picker_S])
        
        # Compile model dengan same parameters as original
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
            loss_weights=[0.2, 0.3, 0.5]  # Same as original
        )
        
        print(f"✅ Model created with SAME layer count as original:")
        print(f"   - CNN layers: {len(filters_list)} layers (same as cred2)")
        print(f"   - Filters: {filters_list}")
        print(f"   - Kernels: {kernel_list}")
        print(f"   - BiLSTM: 64 units")
        print(f"   - Total complexity: Same as original EQTransformer")
        print(f"   - Shape safety: Guaranteed {input_dim[0]} → {input_dim[0]}")
        
        # Verify output shapes match input
        test_output = model.predict(np.random.random((1, *input_dim)), verbose=0)
        actual_output_size = test_output[0].shape[1]  # detector output size
        print(f"✅ Shape verification: Input={input_dim[0]}, Output={actual_output_size}")
        
        if actual_output_size == input_dim[0]:
            print(f"🎯 SUCCESS: Shape consistency achieved with same layer count!")
        else:
            print(f"❌ WARNING: Shape mismatch still exists: {input_dim[0]} vs {actual_output_size}")
    
    print(f"✅ Model berhasil dibuat!")
    print(f"   Total parameters: {model.count_params():,}")
    
    # Setup output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to root project models directory instead of current directory
    project_root = os.path.join(current_dir, '..', '..')  # From core_ind to EQTransformer-Indonesia/
    models_dir = os.path.join(project_root, 'models')
    output_dir = os.path.join(models_dir, f"indonesia_model_{timestamp}")
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Output akan disimpan ke: {output_dir}")
    print(f"🗂️ Relative path: models/indonesia_model_{timestamp}")
    
    if debug_traces:
        print(f"🐛 DEBUG: Model akan disimpan langsung ke directory model")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,  # Gunakan parameter patience dari user
            verbose=1,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model_epoch_{epoch:03d}_loss_{val_loss:.4f}.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            factor=0.5,
            patience=max(1, patience // 2),  # LR reduction patience = setengah dari early stopping patience
            min_lr=1e-6,
            verbose=1
        ),
        TqdmCallback()
    ]
    
    print(f"✅ Callbacks setup!")
    print(f"📁 Model checkpoints akan disimpan ke: {output_dir}")
    print(f"📁 Training logs dan results ke: {output_dir}")
    
    # Training
    print(f"\n🚀 MEMULAI TRAINING...")
    print("="*80)
    
    start_time = time.time()
    
    if debug_traces:
        print("🐛 DEBUG: Ultra-simple training mode")
        print("⏱️ Max 2 menit timeout untuk debug...")
        
        # Detailed debugging untuk training loop
        print("\n🔍 DETAILED DEBUGGING INFO:")
        print(f"   Training generator length: {len(training_generator)}")
        print(f"   Validation generator length: {len(validation_generator)}")
        print(f"   Steps per epoch: 1")
        print(f"   Validation steps: 1")
        
        # Test training generator step by step
        print("\n🔍 TESTING TRAINING GENERATOR:")
        try:
            print("   Step 1: Accessing training_generator[0]...")
            train_batch = training_generator[0]
            print(f"   ✅ Got training batch - Input: {train_batch[0]['input'].shape}")
            
            print("   Step 2: Testing model prediction...")
            prediction = model.predict(train_batch[0], verbose=0)
            print(f"   ✅ Model prediction works - Output shapes: {[p.shape for p in prediction]}")
            
            print("   Step 3: Testing loss calculation...")
            loss = model.evaluate(train_batch[0], train_batch[1], verbose=0)
            print(f"   ✅ Loss calculation works - Loss: {loss}")
            
        except Exception as e:
            print(f"   ❌ Error in generator testing: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        print("\n🔍 TESTING VALIDATION GENERATOR:")
        try:
            print("   Step 1: Accessing validation_generator[0]...")
            val_batch = validation_generator[0]
            print(f"   ✅ Got validation batch - Input: {val_batch[0]['input'].shape}")
            
        except Exception as e:
            print(f"   ❌ Error in validation generator: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        print("\n🚀 STARTING ULTRA-SIMPLE TRAINING...")
        print("   Note: Menggunakan steps_per_epoch=1, validation_steps=1")
        
        # Custom training loop dengan detailed logging
        print("\n📊 TRAINING STEP BY STEP:")
        for epoch in range(epochs):
            print(f"\n--- EPOCH {epoch+1}/{epochs} ---")
            
            # Training step
            print("   🔄 Training step...")
            try:
                train_logs = model.train_on_batch(train_batch[0], train_batch[1])
                print(f"   ✅ Training step completed - Loss: {train_logs}")
            except Exception as e:
                print(f"   ❌ Training step failed: {e}")
                import traceback
                traceback.print_exc()
                return None
            
            # Validation step  
            print("   🔄 Validation step...")
            try:
                val_logs = model.test_on_batch(val_batch[0], val_batch[1])
                print(f"   ✅ Validation step completed - Loss: {val_logs}")
            except Exception as e:
                print(f"   ❌ Validation step failed: {e}")
                import traceback
                traceback.print_exc()
                return None
                
            print(f"   📈 Epoch {epoch+1} summary: train_loss={train_logs} val_loss={val_logs}")
        
        # Create fake history for consistency
        history = type('History', (), {})()
        history.history = {
            'loss': [train_logs] * epochs if isinstance(train_logs, (int, float)) else [train_logs[0]] * epochs,
            'val_loss': [val_logs] * epochs if isinstance(val_logs, (int, float)) else [val_logs[0]] * epochs
        }
        
        print("✅ Manual training loop completed successfully!")
        
    else:
        # Enhanced manual training loop dengan tqdm progress bars
        print("🚀 STARTING ENHANCED TRAINING LOOP...")
        
        # Setup logging - DISABLE console handler untuk avoid interference dengan tqdm
        log_file = os.path.join(output_dir, 'training.log')
        
        # Create logger dengan hanya file handler
        logger = logging.getLogger('eqt_training')
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add only file handler (no console output to avoid tqdm interference)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Training strategy: Gunakan semua data untuk training yang optimal
        # Hanya batasi jika debug mode aktif untuk testing cepat
        if debug_traces:
            max_train_steps = min(len(training_generator), 5)   # Limited untuk debug
            max_val_steps = min(len(validation_generator), 2)   # Limited untuk debug
            print(f"🐛 Debug mode: Limited training/validation")
        else:
            max_train_steps = len(training_generator)  # Gunakan SEMUA training data
            max_val_steps = len(validation_generator)   # Gunakan SEMUA validation data
            print(f"📊 Full training: Using all available data")
        
        # Calculate data coverage - GUNAKAN TRACES ASLI, bukan generator.length * batch_size
        total_train_samples = len(training_traces)  # Jumlah traces asli yang digunakan
        total_val_samples = len(validation_traces)   # Jumlah traces asli yang digunakan
        used_train_samples = total_train_samples if not debug_traces else min(total_train_samples, debug_traces)
        used_val_samples = total_val_samples if not debug_traces else min(total_val_samples, debug_traces)
        
        print(f"   Training: {used_train_samples}/{total_train_samples} samples ({used_train_samples/total_train_samples*100:.1f}%)")
        print(f"   Validation: {used_val_samples}/{total_val_samples} samples ({used_val_samples/total_val_samples*100:.1f}%)")
        print(f"   Training batches: {len(training_generator)} (batch_size={training_generator.batch_size})")
        print(f"   Validation batches: {len(validation_generator)} (batch_size={validation_generator.batch_size})")
        
        # Log ke file
        logger.info(f"📊 Training strategy:")
        logger.info(f"   - Max training steps per epoch: {max_train_steps}")
        logger.info(f"   - Max validation steps: {max_val_steps}")
        logger.info(f"   - Total epochs: {epochs}")
        logger.info(f"   - Training coverage: {used_train_samples/total_train_samples*100:.1f}%")
        logger.info(f"   - Validation coverage: {used_val_samples/total_val_samples*100:.1f}%")
        
        history = type('History', (), {})()
        history.history = {
            'loss': [],
            'val_loss': [],
            'epoch_time': []
        }
        
        print(f"📊 Training: {max_train_steps} batches/epoch, Validation: {max_val_steps} batches")
        print(f"🚀 Starting {epochs} epochs training with full data coverage...")
        print()
        
        # Main training loop dengan clean tqdm
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase dengan single clean progress bar
            epoch_train_losses = []
            
            train_desc = f"Epoch {epoch+1}/{epochs} - Training"
            with tqdm(total=max_train_steps, desc=train_desc, unit="batch", 
                     ncols=80, leave=False, dynamic_ncols=False) as train_pbar:
                
                for step in range(max_train_steps):
                    try:
                        batch_idx = step % len(training_generator)
                        train_batch = training_generator[batch_idx]
                        
                        train_loss = model.train_on_batch(train_batch[0], train_batch[1])
                        current_loss = train_loss[0] if isinstance(train_loss, list) else train_loss
                        epoch_train_losses.append(current_loss)
                        
                        # Update progress bar dengan info singkat
                        avg_loss = np.mean(epoch_train_losses)
                        train_pbar.set_postfix_str(f"loss: {current_loss:.4f}, avg: {avg_loss:.4f}")
                        train_pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Training step {step+1} failed: {e}")
                        train_pbar.update(1)
                        continue
            
            # Validation phase dengan single clean progress bar
            epoch_val_losses = []
            
            val_desc = f"Epoch {epoch+1}/{epochs} - Validation"
            with tqdm(total=max_val_steps, desc=val_desc, unit="batch",
                     ncols=80, leave=False, dynamic_ncols=False) as val_pbar:
                
                for step in range(max_val_steps):
                    try:
                        batch_idx = step % len(validation_generator)
                        val_batch = validation_generator[batch_idx]
                        
                        val_loss = model.test_on_batch(val_batch[0], val_batch[1])
                        current_val_loss = val_loss[0] if isinstance(val_loss, list) else val_loss
                        epoch_val_losses.append(current_val_loss)
                        
                        # Update validation progress bar
                        avg_val_loss = np.mean(epoch_val_losses)
                        val_pbar.set_postfix_str(f"val_loss: {current_val_loss:.4f}, avg: {avg_val_loss:.4f}")
                        val_pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Validation step {step+1} failed: {e}")
                        val_pbar.update(1)
                        continue
            
            # Epoch summary - print setelah progress bars selesai
            epoch_time = time.time() - epoch_start_time
            avg_train_loss = np.mean(epoch_train_losses) if epoch_train_losses else 0.0
            avg_val_loss = np.mean(epoch_val_losses) if epoch_val_losses else 0.0
            
            history.history['loss'].append(avg_train_loss)
            history.history['val_loss'].append(avg_val_loss)
            history.history['epoch_time'].append(epoch_time)
            
            # Print clean epoch summary
            print(f"✅ Epoch {epoch+1}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}, time={epoch_time:.1f}s")
            
            # Log ke file
            logger.info(f"Epoch {epoch+1} - Training: {avg_train_loss:.6f}, Validation: {avg_val_loss:.6f}, Time: {epoch_time:.2f}s")
            
            # Note: Early stopping dihandle oleh Keras EarlyStopping callback
            # Tidak perlu manual early stopping yang bisa bikin bug
            logger.info(f"Validation loss trend: {'improving' if len(history.history['val_loss']) < 2 or avg_val_loss < history.history['val_loss'][-2] else 'stable/worsening'}")
        
        print(f"✅ Training completed!")
        logger.info("✅ Enhanced training loop completed successfully!")
        
        # Testing Phase dengan clean progress bar
        print(f"\n🔍 Testing model on validation data...")
        logger.info("🔍 Starting model testing phase...")
        
        test_results = []
        
        # Test pada semua validation data untuk hasil yang akurat
        # Tapi bisa dibatasi jika debug_traces aktif untuk testing cepat
        if debug_traces:
            test_batches = min(len(validation_generator), 5)  # Limited untuk debug
            print(f"🐛 Debug mode: Testing limited to {test_batches} batches")
        else:
            test_batches = len(validation_generator)  # Test SEMUA validation data
            print(f"📊 Testing all validation data: {test_batches} batches")
        
        # Gunakan jumlah traces asli untuk perhitungan akurat
        total_val_samples = len(validation_traces)  # Jumlah traces validation asli
        max_possible_test_samples = test_batches * validation_generator.batch_size
        expected_test_samples = min(total_val_samples, max_possible_test_samples)
        
        print(f"   Total validation samples: {total_val_samples}")
        print(f"   Expected test samples: {expected_test_samples}")
        print(f"   Test batches: {test_batches} (batch_size={validation_generator.batch_size})")
        
        test_desc = "Testing Model"
        with tqdm(total=test_batches, desc=test_desc, unit="batch",
                 ncols=80, leave=False, dynamic_ncols=False) as test_pbar:
            
            for batch_idx in range(test_batches):
                try:
                    val_batch = validation_generator[batch_idx]
                    predictions = model.predict(val_batch[0], verbose=0)
                    
                    # Get ground truth labels
                    true_detector = val_batch[1]['detector']
                    true_picker_P = val_batch[1]['picker_P']  
                    true_picker_S = val_batch[1]['picker_S']
                    
                    # Extract predictions for each sample in batch
                    batch_size = val_batch[0]['input'].shape[0]
                    for sample_idx in range(batch_size):
                        
                        # Predictions
                        pred_detector = predictions[0][sample_idx].flatten()
                        pred_picker_P = predictions[1][sample_idx].flatten()
                        pred_picker_S = predictions[2][sample_idx].flatten()
                        
                        # Ground truth
                        gt_detector = true_detector[sample_idx].flatten()
                        gt_picker_P = true_picker_P[sample_idx].flatten()
                        gt_picker_S = true_picker_S[sample_idx].flatten()
                        
                        # Find peak indices (argmax)
                        predicted_P_index = int(np.argmax(pred_picker_P))
                        predicted_S_index = int(np.argmax(pred_picker_S))
                        
                        # Find ground truth indices (argmax dari gaussian labels)
                        true_P_index = int(np.argmax(gt_picker_P))
                        true_S_index = int(np.argmax(gt_picker_S))
                        
                        # Calculate errors (dalam samples)
                        P_error_samples = predicted_P_index - true_P_index
                        S_error_samples = predicted_S_index - true_S_index
                        
                        # Convert to time (assuming 100 Hz sampling)
                        sampling_rate = 100  # Hz
                        P_error_seconds = P_error_samples / sampling_rate
                        S_error_seconds = S_error_samples / sampling_rate
                        
                        # Check if detection is significant (above threshold)
                        detector_threshold = 0.5
                        P_threshold = 0.3
                        S_threshold = 0.3
                        
                        is_earthquake_detected = np.max(pred_detector) > detector_threshold
                        is_P_detected = np.max(pred_picker_P) > P_threshold  
                        is_S_detected = np.max(pred_picker_S) > S_threshold
                        
                        # Ground truth analysis
                        has_earthquake = np.max(gt_detector) > 0.1  # Ground truth threshold
                        has_P_wave = np.max(gt_picker_P) > 0.1
                        has_S_wave = np.max(gt_picker_S) > 0.1
                        
                        test_results.append({
                            # Identifiers
                            'batch_idx': batch_idx,
                            'sample_idx': sample_idx,
                            
                            # Detection predictions (max values)
                            'detector_max': float(np.max(pred_detector)),
                            'detector_mean': float(np.mean(pred_detector)),
                            'picker_P_max': float(np.max(pred_picker_P)),
                            'picker_P_mean': float(np.mean(pred_picker_P)),
                            'picker_S_max': float(np.max(pred_picker_S)),
                            'picker_S_mean': float(np.mean(pred_picker_S)),
                            
                            # Peak locations (indices)
                            'predicted_P_index': predicted_P_index,
                            'predicted_S_index': predicted_S_index,
                            'true_P_index': true_P_index,
                            'true_S_index': true_S_index,
                            
                            # Time locations (seconds from start)
                            'predicted_P_time_sec': predicted_P_index / sampling_rate,
                            'predicted_S_time_sec': predicted_S_index / sampling_rate,
                            'true_P_time_sec': true_P_index / sampling_rate,
                            'true_S_time_sec': true_S_index / sampling_rate,
                            
                            # Errors
                            'P_error_samples': P_error_samples,
                            'S_error_samples': S_error_samples,
                            'P_error_seconds': P_error_seconds,
                            'S_error_seconds': S_error_seconds,
                            'P_error_abs_seconds': abs(P_error_seconds),
                            'S_error_abs_seconds': abs(S_error_seconds),
                            
                            # Detection flags
                            'is_earthquake_detected': is_earthquake_detected,
                            'is_P_detected': is_P_detected,
                            'is_S_detected': is_S_detected,
                            'has_earthquake': has_earthquake,
                            'has_P_wave': has_P_wave,
                            'has_S_wave': has_S_wave,
                            
                            # Accuracy flags
                            'earthquake_detection_correct': (is_earthquake_detected == has_earthquake),
                            'P_detection_correct': (is_P_detected == has_P_wave),
                            'S_detection_correct': (is_S_detected == has_S_wave)
                        })
                    
                    test_pbar.set_postfix_str(f"samples: {len(test_results)}")
                    test_pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Testing batch {batch_idx} failed: {e}")
                    test_pbar.update(1)
                    continue
        
        # Save detailed test results
        if test_results:
            test_results_dir = os.path.join(output_dir, 'test_results')
            os.makedirs(test_results_dir, exist_ok=True)
            
            test_df = pd.DataFrame(test_results)
            test_csv_path = os.path.join(test_results_dir, 'validation_predictions_detailed.csv')
            test_df.to_csv(test_csv_path, index=False)
            
            # Create summary statistics
            summary_stats = {
                'total_samples': len(test_df),
                'earthquake_detection_accuracy': test_df['earthquake_detection_correct'].mean(),
                'P_detection_accuracy': test_df['P_detection_correct'].mean(),
                'S_detection_accuracy': test_df['S_detection_correct'].mean(),
                'mean_P_error_seconds': test_df['P_error_abs_seconds'].mean(),
                'mean_S_error_seconds': test_df['S_error_abs_seconds'].mean(),
                'median_P_error_seconds': test_df['P_error_abs_seconds'].median(),
                'median_S_error_seconds': test_df['S_error_abs_seconds'].median(),
                'P_error_std_seconds': test_df['P_error_seconds'].std(),
                'S_error_std_seconds': test_df['S_error_seconds'].std()
            }
            
            # Save summary
            summary_df = pd.DataFrame([summary_stats])
            summary_csv_path = os.path.join(test_results_dir, 'test_summary.csv')
            summary_df.to_csv(summary_csv_path, index=False)
            
            logger.info(f"📊 Detailed test results saved to: {test_csv_path}")
            logger.info(f"📊 Test summary saved to: {summary_csv_path}")
            logger.info(f"📊 Total test samples: {len(test_results)}")
            
            # Print key metrics
            print(f"✅ Test results saved: {len(test_results)} samples")
            print(f"📊 Coverage: {len(test_results)}/{total_val_samples} validation samples ({len(test_results)/total_val_samples*100:.1f}%)")
            print(f"📊 Key Metrics:")
            print(f"   🎯 Earthquake Detection Accuracy: {summary_stats['earthquake_detection_accuracy']:.3f}")
            print(f"   🎯 P-wave Detection Accuracy: {summary_stats['P_detection_accuracy']:.3f}")
            print(f"   🎯 S-wave Detection Accuracy: {summary_stats['S_detection_accuracy']:.3f}")
            print(f"   ⏰ Mean P-wave Error: {summary_stats['mean_P_error_seconds']:.3f} seconds")
            print(f"   ⏰ Mean S-wave Error: {summary_stats['mean_S_error_seconds']:.3f} seconds")
        
        # Export Training History - NEW FEATURE!
        print(f"\n📊 EXPORTING TRAINING HISTORY...")
        
        # Save training history to CSV
        history_df = pd.DataFrame({
            'epoch': range(1, len(history.history['loss']) + 1),
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'epoch_time_seconds': history.history['epoch_time']
        })
        
        history_csv_path = os.path.join(output_dir, 'training_history.csv')
        history_df.to_csv(history_csv_path, index=False)
        logger.info(f"📊 Training history saved to: {history_csv_path}")
        
        # Plot loss curves - NEW FEATURE!
        print(f"📈 Creating loss curves plot...")
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Loss curves
        plt.subplot(2, 2, 1)
        plt.plot(history_df['epoch'], history_df['train_loss'], 'b-', label='Training Loss', linewidth=2)
        plt.plot(history_df['epoch'], history_df['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        plt.title('Training & Validation Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Training time per epoch
        plt.subplot(2, 2, 2)
        plt.plot(history_df['epoch'], history_df['epoch_time_seconds'], 'g-', linewidth=2)
        plt.title('Training Time per Epoch', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Loss difference
        plt.subplot(2, 2, 3)
        loss_diff = history_df['val_loss'] - history_df['train_loss']
        plt.plot(history_df['epoch'], loss_diff, 'm-', linewidth=2)
        plt.title('Validation - Training Loss (Overfitting Indicator)', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Difference')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Summary stats
        plt.subplot(2, 2, 4)
        plt.text(0.1, 0.8, f"EQTransformer Indonesia Training", fontsize=16, fontweight='bold')
        plt.text(0.1, 0.7, f"Total Epochs: {len(history_df)}", fontsize=12)
        plt.text(0.1, 0.6, f"Final Train Loss: {history_df['train_loss'].iloc[-1]:.6f}", fontsize=12)
        plt.text(0.1, 0.5, f"Final Val Loss: {history_df['val_loss'].iloc[-1]:.6f}", fontsize=12)
        plt.text(0.1, 0.4, f"Total Time: {sum(history_df['epoch_time_seconds']):.1f}s", fontsize=12)
        plt.text(0.1, 0.3, f"Model: 7-layer CNN + BiLSTM", fontsize=12)
        plt.text(0.1, 0.2, f"Input Size: {input_dim[0]} samples", fontsize=12)
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'loss_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Loss curves plot saved to: {plot_path}")
        print(f"✅ Loss curves saved: {plot_path}")
        
        print(f"\n📋 TRAINING SUMMARY:")
        print(f"   📊 Epochs completed: {len(history_df)}")
        print(f"   🎯 Final training loss: {history_df['train_loss'].iloc[-1]:.6f}")
        print(f"   🎯 Final validation loss: {history_df['val_loss'].iloc[-1]:.6f}")
        print(f"   ⏰ Total training time: {sum(history_df['epoch_time_seconds']):.1f} seconds")
        print(f"   📝 Training log: {log_file}")
        print(f"   📊 History CSV: {history_csv_path}")
        print(f"   📈 Loss plot: {plot_path}")
        if test_results:
            print(f"   🔍 Test results: {test_csv_path}")
        
        logger.info("🎉 All training outputs successfully saved!")
    
    end_time = time.time()
    training_time = (end_time - start_time) / 60
    
    print("="*80)
    print("🎉 TRAINING SELESAI!")
    print(f"⏰ Waktu training: {training_time:.1f} menit")
    
    # Save results
    print(f"\n💾 MENYIMPAN HASIL...")
    
    save_tasks = [
        ("Final model", lambda: model.save(os.path.join(output_dir, 'final_model.h5'))),
        ("Model weights", lambda: model.save_weights(os.path.join(output_dir, 'model_weights.h5'))),
        ("Training history", lambda: np.save(os.path.join(output_dir, 'training_history.npy'), history.history))
    ]
    
    for task_name, task_func in tqdm(save_tasks, desc="Saving results"):
        print(f"💾 {task_name}...")
        task_func()
    
    # Training summary
    print("📝 Writing training summary...")
    with open(os.path.join(output_dir, 'training_summary.txt'), 'w') as f:
        f.write("EQTransformer Indonesia Training Summary 🌋\n")
        f.write("="*60 + "\n")
        f.write(f"Dataset: Indonesia Seismic Data\n")
        f.write(f"Input dimension: {input_dim}\n")
        f.write(f"Training traces: {len(training_traces)}\n")
        f.write(f"Validation traces: {len(validation_traces)}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Epochs completed: {len(history.history['loss'])}\n")
        f.write(f"Training time: {training_time:.1f} minutes\n")
        f.write(f"Final training loss: {history.history['loss'][-1]:.4f}\n")
        f.write(f"Final validation loss: {history.history['val_loss'][-1]:.4f}\n")
        f.write(f"Total parameters: {model.count_params():,}\n")
        f.write(f"CNN blocks: {cnn_blocks}\n")
        f.write(f"LSTM blocks: {lstm_blocks}\n")
        f.write(f"Augmentation: {augmentation}\n")
        f.write(f"Combined data: {use_combined_data}\n")
    
    # Cleanup temporary files
    if use_combined_data and os.path.exists("../../datasets/temp_combined.csv"):
        os.remove("../../datasets/temp_combined.csv")
    
    print(f"✅ Hasil tersimpan di: {output_dir}")
    print("="*80)
    
    # Calculate relative path from project root for user-friendly display
    relative_path = os.path.relpath(output_dir, project_root)
    
    return output_dir, history, model

if __name__ == "__main__":
    """
    EQTransformer Indonesia Training Script 🌋
    
    Script ini dipanggil oleh run_gpu_training.sh dengan parameter yang sudah dikonfigurasi.
    Untuk training langsung, gunakan: ./run_gpu_training.sh
    
    Untuk custom parameters, edit variabel di bagian atas run_gpu_training.sh:
    - BATCH_SIZE: Ukuran batch (2-8 recommended)
    - EPOCHS: Jumlah epochs
    - AUGMENTATION: true/false
    - PATIENCE: Early stopping patience  
    - COMBINED_DATA: true/false
    - DEBUG_TRACES: kosong untuk full data, angka untuk limit traces
    """
    
    print("EQTransformer Indonesia Training 🌋")
    print(f"🖥️ Hardware: {'GPU' if gpu_available else 'CPU'}")
    print("📝 Note: Script ini biasanya dipanggil dari run_gpu_training.sh")
    print("⚙️ Untuk custom parameters, edit run_gpu_training.sh")
    print()
    
    # Fallback menu untuk direct execution (compatibility)
    print("🔧 DIRECT EXECUTION MODE:")
    print("1. Quick Test (4 batch, 1 epoch, patience=2)")
    print("2. Small Training (4 batch, 10 epochs, patience=3)")  
    print("3. Full Training (8 batch, 50 epochs, patience=10)")
    print("4. Debug Test (1 trace, 1 epoch, patience=1)")
    print("5. Custom parameters")
    
    choice = input("\nPilih mode (1/2/3/4/5): ").strip()
    
    if choice == "1":
        result = train_indonesia_eqt(batch_size=4, epochs=1, patience=2, augmentation=False)
    elif choice == "2":
        result = train_indonesia_eqt(batch_size=4, epochs=10, patience=3, augmentation=False)
    elif choice == "3":
        result = train_indonesia_eqt(batch_size=8, epochs=50, patience=10, augmentation=True, use_combined_data=True)
    elif choice == "4":
        print("🐛 DEBUG MODE: Testing dengan 1 trace saja")
        result = train_indonesia_eqt(batch_size=1, epochs=1, patience=1, augmentation=False, debug_traces=1)
    elif choice == "5":
        batch_size = int(input("Batch size (default 4): ") or "4")
        epochs = int(input("Epochs (default 10): ") or "10")
        patience = int(input("Patience (default 3): ") or "3")
        augmentation = input("Augmentation (y/n, default y): ").strip().lower() != 'n'
        combined = input("Use combined data (y/n, default n): ").strip().lower() == 'y'
        debug = input("Debug traces limit (default None): ").strip()
        debug_traces = int(debug) if debug else None
        result = train_indonesia_eqt(
            batch_size=batch_size, 
            epochs=epochs, 
            patience=patience,
            augmentation=augmentation, 
            use_combined_data=combined,
            debug_traces=debug_traces
        )
    else:
        print("Mode default: Small Training")
        result = train_indonesia_eqt(batch_size=4, epochs=10, patience=3, augmentation=False)
    
    if result:
        output_dir, history, model = result
        
        # Calculate paths relative to project root for display
        project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
        relative_output = os.path.relpath(output_dir, project_root)
        
        print(f"\n🎯 MODEL SIAP DIGUNAKAN!")
        print(f"📁 Lokasi output: {relative_output}")
        print(f"🎯 Untuk prediksi, gunakan: {os.path.join(relative_output, 'final_model.h5')}")
        print(f"💾 Training history: {os.path.join(relative_output, 'training_history.csv')}")
        print(f"📈 Loss curves: {os.path.join(relative_output, 'loss_curves.png')}")
    
    print("\nSelesai! 🌋") 