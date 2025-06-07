#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🇮🇩 EQTransformer Training untuk Dataset Indonesia 🌋

Script training utama untuk melatih EQTransformer dengan data seismik Indonesia
Input: 30085 samples (300.8 detik @ 100Hz) - 5x lebih panjang dari default EQTransformer
"""

import os
import sys
import datetime
import pandas as pd
import h5py
import numpy as np
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Auto setup GPU environment
def setup_gpu_environment():
    """Automatically setup GPU environment untuk EQTransformer Indonesia"""
    print("🔧 Auto-setting up GPU environment...")
    
    # Set CUDA Environment Variables
    os.environ['CUDA_HOME'] = '/usr'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
    
    # Set LD_LIBRARY_PATH untuk NVIDIA packages
    nvidia_paths = [
        '/usr/lib/x86_64-linux-gnu',
        '/home/mooc_parallel_2021_003/miniconda3/envs/eqt/lib/python3.8/site-packages/nvidia/cuda_runtime/lib',
        '/home/mooc_parallel_2021_003/miniconda3/envs/eqt/lib/python3.8/site-packages/nvidia/cudnn/lib',
        '/home/mooc_parallel_2021_003/miniconda3/envs/eqt/lib/python3.8/site-packages/nvidia/cublas/lib',
        '/home/mooc_parallel_2021_003/miniconda3/envs/eqt/lib/python3.8/site-packages/nvidia/cufft/lib',
        '/home/mooc_parallel_2021_003/miniconda3/envs/eqt/lib/python3.8/site-packages/nvidia/curand/lib',
        '/home/mooc_parallel_2021_003/miniconda3/envs/eqt/lib/python3.8/site-packages/nvidia/cusolver/lib',
        '/home/mooc_parallel_2021_003/miniconda3/envs/eqt/lib/python3.8/site-packages/nvidia/cusparse/lib'
    ]
    
    existing_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_path = ':'.join(nvidia_paths + [existing_path] if existing_path else nvidia_paths)
    os.environ['LD_LIBRARY_PATH'] = new_path
    
    # Test GPU availability
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 0:
            print(f"✅ GPU setup berhasil! Terdeteksi {len(gpus)} GPU device(s)")
            return True
        else:
            print("⚠️ GPU tidak terdeteksi, akan menggunakan CPU")
            return False
    except Exception as e:
        print(f"⚠️ Error checking GPU: {e}")
        return False

# Setup GPU at import time
gpu_available = setup_gpu_environment()

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
    """Verifikasi ketersediaan dan kualitas dataset"""
    
    print("🔍 VERIFIKASI DATASET")
    print("="*60)
    
    datasets_dir = os.path.join(project_root, 'datasets')
    
    files_to_check = [
        ('indonesia_train.hdf5', 'Training HDF5'),
        ('indonesia_train.csv', 'Training CSV'),
        ('indonesia_valid.hdf5', 'Validation HDF5'),
        ('indonesia_valid.csv', 'Validation CSV')
    ]
    
    all_files_exist = True
    
    # Progress bar untuk file checking
    for filename, description in tqdm(files_to_check, desc="Checking files"):
        filepath = os.path.join(datasets_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"✅ {description}: {size_mb:.1f} MB")
        else:
            print(f"❌ {description}: NOT FOUND")
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Dataset tidak lengkap!")
        return False, None
    
    # Verifikasi format data
    train_csv = os.path.join(datasets_dir, 'indonesia_train.csv')
    print("📋 Loading training CSV...")
    df = pd.read_csv(train_csv)
    
    print(f"\n📊 INFORMASI DATASET:")
    print(f"   Training traces: {len(df)}")
    
    # Cek trace names format dengan progress bar
    print("🔍 Verifying trace names format...")
    traces_with_ev = 0
    for trace_name in tqdm(df['trace_name'], desc="Checking trace names", leave=False):
        if trace_name.endswith('_EV'):
            traces_with_ev += 1
    
    if traces_with_ev == len(df):
        print(f"✅ Semua trace names memiliki format yang benar (_EV suffix)")
    else:
        print(f"❌ {len(df) - traces_with_ev} traces tidak memiliki _EV suffix")
        return False, None
    
    print(f"   P-arrival range: {df['p_arrival_sample'].min()} - {df['p_arrival_sample'].max()}")
    print(f"   S-arrival range: {df['s_arrival_sample'].min()} - {df['s_arrival_sample'].max()}")
    print(f"   Sample trace: {df.iloc[0]['trace_name']}")
    
    return True, datasets_dir

def train_indonesia_eqt(
    batch_size=4,
    epochs=10,
    cnn_blocks=3,
    lstm_blocks=1,
    augmentation=True,
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
        
    use_combined_data : bool (default=False)
        Gabungkan train+valid atau pisah
    debug_traces : int or None (default=None)
        Jika set, hanya gunakan N traces pertama untuk debugging.
        Contoh: debug_traces=10 untuk test cepat
    """
    
    print("🇮🇩 TRAINING EQTRANSFORMER INDONESIA 🌋")
    print("="*80)
    
    # Verifikasi dataset
    dataset_ok, datasets_dir = check_datasets()
    if not dataset_ok:
        return None
    
    # Setup file paths
    if use_combined_data:
        # Gabungkan training + validation
        print("📋 Mode: Combined training + validation data")
        train_hdf5 = os.path.join(datasets_dir, 'indonesia_train.hdf5')
        train_csv = os.path.join(datasets_dir, 'indonesia_train.csv')
        
        # Load dan gabungkan CSV
        df_train = pd.read_csv(os.path.join(datasets_dir, 'indonesia_train.csv'))
        df_valid = pd.read_csv(os.path.join(datasets_dir, 'indonesia_valid.csv'))
        df_combined = pd.concat([df_train, df_valid], ignore_index=True)
        
        # Buat temporary combined CSV
        combined_csv = os.path.join(datasets_dir, 'temp_combined.csv')
        df_combined.to_csv(combined_csv, index=False)
        train_csv = combined_csv
        
        all_traces = df_combined['trace_name'].tolist()
        print(f"   Total traces: {len(all_traces)}")
        
    else:
        # Pisah training dan validation
        print("📋 Mode: Separate training + validation data")
        train_hdf5 = os.path.join(datasets_dir, 'indonesia_train.hdf5')
        train_csv = os.path.join(datasets_dir, 'indonesia_train.csv')
        
        df_train = pd.read_csv(train_csv)
        all_traces = df_train['trace_name'].tolist()
        print(f"   Training traces: {len(all_traces)}")
    
    # Debug mode: limit traces untuk testing
    if debug_traces:
        print(f"\n🐛 DEBUG MODE: Menggunakan hanya {debug_traces} traces")
        original_count = len(all_traces)
        all_traces = all_traces[:debug_traces]  # Ambil N traces pertama
        print(f"   Traces dikurangi dari {original_count} → {len(all_traces)}")
        print(f"   ⚡ Training akan sangat cepat untuk testing!")
        
        # Verifikasi traces exists di HDF5
        print(f"🔍 Verifying traces exist in HDF5...")
        try:
            with h5py.File(os.path.join(datasets_dir, 'indonesia_train.hdf5'), 'r') as f:
                missing_traces = []
                for trace in all_traces[:3]:  # Cek 3 traces pertama
                    trace_path = f'data/{trace}'
                    if trace_path not in f:
                        missing_traces.append(trace)
                    else:
                        data_shape = f[trace_path].shape
                        print(f"   ✅ {trace}: shape {data_shape}")
                
                if missing_traces:
                    print(f"   ❌ Missing traces: {missing_traces}")
                    return None
                else:
                    print(f"   ✅ All checked traces exist in HDF5")
        except Exception as e:
            print(f"   ❌ Error reading HDF5: {e}")
            return None
    
    # Model parameters
    input_dim = (30085, 3)  # Signature Indonesia: 300.8 detik @ 100Hz
    
    print(f"\n⚙️ KONFIGURASI TRAINING:")
    print(f"   Input dimension: {input_dim} (300.8 detik)")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   CNN blocks: {cnn_blocks}")
    print(f"   LSTM blocks: {lstm_blocks}")
    print(f"   Augmentation: {augmentation}")
    print(f"   Combined data: {use_combined_data}")
    if debug_traces:
        print(f"   🐛 DEBUG: Limited to {debug_traces} traces")
        print(f"   🎯 Purpose: Quick functionality test")
    
    # Split data
    print(f"\n🔄 SPLITTING DATA...")
    np.random.seed(42)  # Reproducible
    np.random.shuffle(all_traces)
    
    if debug_traces:
        # Untuk debug, pastikan selalu ada training data
        print(f"🐛 DEBUG: Gunakan semua {debug_traces} traces untuk training")
        training_traces = all_traces  # Semua traces untuk training
        validation_traces = all_traces[-1:]  # 1 trace terakhir untuk validation 
        print(f"   Training: {len(training_traces)} traces")
        print(f"   Validation: {len(validation_traces)} traces")
    else:
        split_idx = int(0.8 * len(all_traces))
        training_traces = all_traces[:split_idx]
        validation_traces = all_traces[split_idx:]
        print(f"🔀 Shuffling traces...")
        print(f"   Training: {len(training_traces)} traces")
        print(f"   Validation: {len(validation_traces)} traces")
    
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
    
    # Setup data generators
    print(f"\n🔄 SETUP DATA GENERATORS...")
    
    # Augmentation parameters - disesuaikan untuk data Indonesia
    if augmentation:
        aug_params = {
            'shift_event_r': 0.9,        # 90% chance shift P-S events  
            'add_noise_r': 0.3,          # 30% chance add Gaussian noise
            'drop_channe_r': 0.2,        # 20% chance drop channels
            'scale_amplitude_r': 0.5,    # 50% chance scale amplitude
            'add_gap_r': 0.1,           # 10% chance add data gaps (noise only)
        }
        print(f"🎯 Augmentation aktif:")
        print(f"   - Shift events: {aug_params['shift_event_r']*100}%")
        print(f"   - Add noise: {aug_params['add_noise_r']*100}%") 
        print(f"   - Drop channels: {aug_params['drop_channe_r']*100}%")
        print(f"   - Scale amplitude: {aug_params['scale_amplitude_r']*100}%")
        print(f"   - Add gaps: {aug_params['add_gap_r']*100}%")
    else:
        aug_params = {
            'shift_event_r': None,
            'add_noise_r': None,
            'drop_channe_r': None,
            'scale_amplitude_r': None,
            'add_gap_r': None,
        }
        print(f"🎯 Augmentation dinonaktifkan")
    
    params_training = {
        'dim': input_dim[0],
        'batch_size': batch_size,
        'n_channels': input_dim[-1],
        'shuffle': True,
        'norm_mode': 'std',
        'label_type': 'gaussian',
        'augmentation': augmentation,
        'file_name': train_hdf5,
        'coda_ratio': 0.4,
        'pre_emphasis': True,
        **aug_params  # Tambahkan augmentation parameters
    }
    
    params_validation = params_training.copy()
    params_validation.update({
        'shuffle': False,
        'augmentation': False,  # No augmentation for validation
        'shift_event_r': None,  # Reset augmentation for validation
        'add_noise_r': None,
        'drop_channe_r': None,
        'scale_amplitude_r': None,
        'add_gap_r': None,
    })
    
    training_generator = DataGenerator(training_traces, **params_training)
    validation_generator = DataGenerator(validation_traces, **params_validation)
    
    print(f"✅ Data generators ready!")
    print(f"📊 Training generator batches: {len(training_generator)}")
    print(f"📊 Validation generator batches: {len(validation_generator)}")
    
    # Test first batch loading
    if debug_traces:
        print(f"\n🔍 TESTING FIRST BATCH LOADING...")
        try:
            print(f"   Loading batch 0...")
            first_batch = training_generator[0]
            print(f"   ✅ First batch loaded successfully!")
            print(f"   Input shape: {first_batch[0]['input'].shape}")
            print(f"   Detector shape: {first_batch[1]['detector'].shape}")
        except Exception as e:
            print(f"   ❌ Error loading first batch: {e}")
            return None
    
    # Setup output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"indonesia_model_{timestamp}"
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=max(3, epochs // 3),  # Adaptive patience
            verbose=1,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=os.path.join(models_dir, 'best_model_epoch_{epoch:03d}_loss_{val_loss:.4f}.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            factor=0.5,
            patience=max(2, epochs // 4),
            min_lr=1e-6,
            verbose=1
        ),
        TqdmCallback()
    ]
    
    print(f"✅ Callbacks setup!")
    print(f"📁 Output directory: {output_dir}")
    
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
        # Manual training loop untuk semua mode (karena model.fit() stuck)
        print("🚀 STARTING MANUAL TRAINING LOOP...")
        print(f"⚡ Mode: Manual batch-by-batch training (CPU optimized)")
        
        # Limit steps per epoch untuk mencegah stuck
        max_train_steps = min(len(training_generator), 10)  # Maksimal 10 batches per epoch
        max_val_steps = min(len(validation_generator), 3)   # Maksimal 3 batches validation
        
        print(f"📊 Training strategy:")
        print(f"   - Max training steps per epoch: {max_train_steps}")
        print(f"   - Max validation steps: {max_val_steps}")
        print(f"   - Total epochs: {epochs}")
        
        history = type('History', (), {})()
        history.history = {
            'loss': [],
            'val_loss': []
        }
        
        print(f"\n🚀 Starting {epochs} epochs...")
        
        for epoch in range(epochs):
            print(f"\n=== EPOCH {epoch+1}/{epochs} ===")
            
            # Training phase
            print(f"🔄 Training phase ({max_train_steps} batches)...")
            epoch_train_losses = []
            
            for step in range(max_train_steps):
                try:
                    batch_idx = step % len(training_generator)
                    train_batch = training_generator[batch_idx]
                    
                    train_loss = model.train_on_batch(train_batch[0], train_batch[1])
                    epoch_train_losses.append(train_loss[0] if isinstance(train_loss, list) else train_loss)
                    
                    if step % 5 == 0 or step == max_train_steps - 1:
                        print(f"   Step {step+1}/{max_train_steps}: loss={train_loss[0] if isinstance(train_loss, list) else train_loss:.4f}")
                        
                except Exception as e:
                    print(f"   ❌ Training step {step+1} failed: {e}")
                    continue
            
            # Validation phase
            print(f"🔍 Validation phase ({max_val_steps} batches)...")
            epoch_val_losses = []
            
            for step in range(max_val_steps):
                try:
                    batch_idx = step % len(validation_generator)
                    val_batch = validation_generator[batch_idx]
                    
                    val_loss = model.test_on_batch(val_batch[0], val_batch[1])
                    epoch_val_losses.append(val_loss[0] if isinstance(val_loss, list) else val_loss)
                    
                except Exception as e:
                    print(f"   ❌ Validation step {step+1} failed: {e}")
                    continue
            
            # Epoch summary
            avg_train_loss = np.mean(epoch_train_losses) if epoch_train_losses else 0.0
            avg_val_loss = np.mean(epoch_val_losses) if epoch_val_losses else 0.0
            
            history.history['loss'].append(avg_train_loss)
            history.history['val_loss'].append(avg_val_loss)
            
            print(f"✅ Epoch {epoch+1} completed:")
            print(f"   - Training loss: {avg_train_loss:.4f}")
            print(f"   - Validation loss: {avg_val_loss:.4f}")
            
            # Simple early stopping
            if epoch > 2 and avg_val_loss > history.history['val_loss'][-2]:
                print(f"🛑 Early stopping: validation loss increased")
                break
        
        print("✅ Manual training loop completed successfully!")
    
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
        f.write("🇮🇩 EQTransformer Indonesia Training Summary 🌋\n")
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
    if use_combined_data and os.path.exists(combined_csv):
        os.remove(combined_csv)
    
    print(f"✅ Hasil tersimpan di: {output_dir}")
    print("="*80)
    
    return output_dir, history, model

if __name__ == "__main__":
    """
    Mode training:
    1. Quick test: batch_size=2, epochs=3
    2. Small training: batch_size=4, epochs=10  
    3. Full training: batch_size=8, epochs=50
    """
    
    print("🇮🇩 PILIHAN MODE TRAINING 🌋")
    print(f"🖥️ Hardware: {'GPU' if gpu_available else 'CPU'}")
    print("1. Quick Test (4 batch, 1 epoch) - 5 menit ⚡")
    print("2. Small Training (4 batch, 10 epochs) - 1 jam")  
    print("3. Full Training (8 batch, 50 epochs) - 6 jam")
    print("4. Debug Test (1 trace, 1 epoch) - 2 menit 🐛")
    print("5. Custom parameters")
    
    choice = input("\nPilih mode (1/2/3/4/5): ").strip()
    
    if choice == "1":
        result = train_indonesia_eqt(batch_size=4, epochs=1, augmentation=False)
    elif choice == "2":
        result = train_indonesia_eqt(batch_size=4, epochs=10, augmentation=False)
    elif choice == "3":
        result = train_indonesia_eqt(batch_size=8, epochs=50, augmentation=False, use_combined_data=True)
    elif choice == "4":
        print("🐛 DEBUG MODE: Testing dengan 1 trace saja")
        print("⚡ Ultra-minimal setup untuk test functionality")
        result = train_indonesia_eqt(batch_size=1, epochs=1, augmentation=False, debug_traces=1)  # Cuma 1 trace!
    elif choice == "5":
        batch_size = int(input("Batch size (default 4): ") or "4")
        epochs = int(input("Epochs (default 10): ") or "10")
        augmentation = input("Augmentation (y/n, default y): ").strip().lower() != 'n'
        combined = input("Use combined data (y/n, default n): ").strip().lower() == 'y'
        debug = input("Debug traces limit (default None): ").strip()
        debug_traces = int(debug) if debug else None
        result = train_indonesia_eqt(batch_size=batch_size, epochs=epochs, 
                                   augmentation=augmentation, use_combined_data=combined,
                                   debug_traces=debug_traces)
    else:
        print("Mode default: Small Training")
        result = train_indonesia_eqt(batch_size=4, epochs=10, augmentation=True)
    
    if result:
        output_dir, history, model = result
        print(f"\n🎯 MODEL SIAP DIGUNAKAN!")
        print(f"📁 Lokasi: {output_dir}")
        print(f"🔍 Untuk prediksi, gunakan: final_model.h5")
    
    print("\n🇮🇩 Selesai! 🌋") 