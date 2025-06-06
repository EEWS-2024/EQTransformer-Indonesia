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

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')  # Back to EQTransformer-Indonesia/
eqt_core_dir = os.path.join(current_dir, '..', 'core')  # Dari core_ind ke core
sys.path.append(eqt_core_dir)

# Import EQTransformer modules
from EqT_utils import DataGenerator, cred2
from tensorflow.keras import Input

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
    
    print("🔀 Shuffling traces...")
    all_traces_shuffled = []
    for trace in tqdm(all_traces, desc="Shuffling", leave=False):
        all_traces_shuffled.append(trace)
    np.random.shuffle(all_traces_shuffled)
    
    split_idx = int(0.8 * len(all_traces_shuffled))
    training_traces = all_traces_shuffled[:split_idx]
    validation_traces = all_traces_shuffled[split_idx:]
    
    print(f"   Training: {len(training_traces)} traces")
    print(f"   Validation: {len(validation_traces)} traces")
    
    # Build model
    print(f"\n🏗️ BUILDING MODEL...")
    print("⚙️ Initializing architecture...")
    inp = Input(shape=input_dim, name='input')
    print("🧠 Creating EQTransformer network...")
    model = cred2(
        nb_filters=[8, 16, 16, 32, 32, 64, 64],
        kernel_size=[11, 9, 7, 7, 5, 5, 3],
        padding='same',
        activationf='relu',
        cnn_blocks=cnn_blocks,
        BiLSTM_blocks=lstm_blocks,
        drop_rate=0.2,
        loss_weights=[0.2, 0.3, 0.5],  # detector, picker_P, picker_S
        loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy']
    )(inp)
    
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
    
    history = model.fit(
        training_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        workers=1,
        use_multiprocessing=False  # Safer for debugging
    )
    
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
    print("1. Quick Test (2 batch, 3 epochs) - 10 menit")
    print("2. Small Training (4 batch, 10 epochs) - 1 jam")  
    print("3. Full Training (8 batch, 50 epochs) - 6 jam")
    print("4. Debug Test (10 traces, 2 epochs) - 2 menit ⚡")
    print("5. Custom parameters")
    
    choice = input("\nPilih mode (1/2/3/4/5): ").strip()
    
    if choice == "1":
        result = train_indonesia_eqt(batch_size=64, epochs=1, augmentation=False)
    elif choice == "2":
        result = train_indonesia_eqt(batch_size=4, epochs=10, augmentation=True)
    elif choice == "3":
        result = train_indonesia_eqt(batch_size=8, epochs=50, augmentation=True, use_combined_data=True)
    elif choice == "4":
        print("🐛 DEBUG MODE: Testing dengan 10 traces saja")
        result = train_indonesia_eqt(batch_size=2, epochs=2, augmentation=False, debug_traces=10)
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