#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EQTransformer Fine-tuning Script for Indonesia Data 🌋

Fine-tune original EQTransformer model with Indonesia data:
1. Load pre-trained original model (6000 samples input)
2. Use windowed Indonesia dataset (30000→6000 samples)
3. Freeze encoder weights (feature extraction)
4. Only update decoder weights (task-specific adaptation)

Strategy: Transfer learning - encoder learns universal seismic features,
decoder adapts to Indonesian seismic characteristics.
"""

import os
import sys
import numpy as np
import pandas as pd

# Suppress TensorFlow warnings and verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Only show errors

import h5py
from datetime import datetime
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add EQTransformer modules to path
sys.path.append('.')
sys.path.append('EQTransformer')

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from EQTransformer.core.EqT_utils import f1

class IndonesiaDataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for Indonesia fine-tuning dataset"""
    
    def __init__(self, list_IDs, hdf5_file, csv_file, batch_size=16, 
                 dim=6000, n_channels=3, shuffle=True, augmentation=False):
        self.list_IDs = list_IDs
        self.hdf5_file = hdf5_file
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augmentation = augmentation
        
        # Load CSV metadata
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['trace_name'].isin(list_IDs)]
        self.df = self.df.set_index('trace_name')
        
        print(f"📊 DataGenerator initialized:")
        print(f"   Traces: {len(self.list_IDs)}")
        print(f"   Batch size: {batch_size}")
        print(f"   Augmentation: {augmentation}")
        
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # Initialize arrays
        X = np.empty((self.batch_size, self.dim, self.n_channels))
        y_det = np.empty((self.batch_size, self.dim, 1))  # Detection
        y_p = np.empty((self.batch_size, self.dim, 1))    # P picking
        y_s = np.empty((self.batch_size, self.dim, 1))    # S picking

        # Read HDF5 file
        with h5py.File(self.hdf5_file, 'r') as f:
            for i, trace_id in enumerate(list_IDs_temp):
                try:
                    # Get waveform data from HDF5
                    if 'data' in f:
                        waveform = f['data'][trace_id][:]
                    else:
                        waveform = f[trace_id][:]
                    
                    # Ensure correct shape (6000, 3)
                    if waveform.shape != (self.dim, self.n_channels):
                        print(f"⚠️ Shape mismatch for {trace_id}: {waveform.shape}")
                        # Pad or truncate if needed
                        if waveform.shape[0] < self.dim:
                            pad_width = ((0, self.dim - waveform.shape[0]), (0, 0))
                            waveform = np.pad(waveform, pad_width, mode='constant')
                        else:
                            waveform = waveform[:self.dim, :]
                    
                    # Normalize waveform
                    for ch in range(self.n_channels):
                        if np.std(waveform[:, ch]) != 0:
                            waveform[:, ch] = (waveform[:, ch] - np.mean(waveform[:, ch])) / np.std(waveform[:, ch])
                    
                    X[i, :, :] = waveform
                    
                    # Get labels from CSV
                    row = self.df.loc[trace_id]
                    p_arrival = int(row['p_arrival_sample'])
                    s_arrival = int(row['s_arrival_sample'])
                    
                    # Create triangle labels (similar to EQTransformer)
                    y_det_trace = self._create_triangle_label(p_arrival, s_arrival)
                    y_p_trace = self._create_triangle_label(p_arrival)
                    y_s_trace = self._create_triangle_label(s_arrival)
                    
                    y_det[i, :, 0] = y_det_trace
                    y_p[i, :, 0] = y_p_trace
                    y_s[i, :, 0] = y_s_trace
                    
                except Exception as e:
                    print(f"❌ Error processing {trace_id}: {e}")
                    # Fill with zeros if error
                    X[i, :, :] = 0
                    y_det[i, :, 0] = 0
                    y_p[i, :, 0] = 0
                    y_s[i, :, 0] = 0

        return X, [y_det, y_p, y_s]
    
    def _create_triangle_label(self, arrival_p, arrival_s=None, width=20):
        """Create triangle-shaped label around arrival time(s)"""
        label = np.zeros(self.dim)
        
        if arrival_s is None:
            # Single arrival (P or S only)
            arrival = arrival_p
            if 0 <= arrival < self.dim:
                self._add_triangle_peak(label, arrival, width)
        else:
            # Detection label combines P and S
            if 0 <= arrival_p < self.dim:
                self._add_triangle_peak(label, arrival_p, width)
            if 0 <= arrival_s < self.dim:
                self._add_triangle_peak(label, arrival_s, width)
        
        return label
    
    def _add_triangle_peak(self, label, arrival, width):
        """Add a triangle peak to existing label"""
        start = max(0, arrival - width)
        end = min(self.dim, arrival + width + 1)
        
        # Left side of triangle
        for i in range(start, arrival):
            if i >= 0:
                triangle_val = (i - start) / (arrival - start) if arrival > start else 0
                label[i] = max(label[i], triangle_val)  # Take maximum to combine peaks
        
        # Peak
        if arrival < self.dim:
            label[arrival] = 1.0
            
        # Right side of triangle
        for i in range(arrival + 1, end):
            if i < self.dim:
                triangle_val = (end - i) / (end - arrival) if end > arrival else 0
                label[i] = max(label[i], triangle_val)  # Take maximum to combine peaks

class RealTimeProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback untuk menampilkan progress real-time"""
    
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n📈 Epoch {epoch + 1} starting...", flush=True)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"✅ Epoch {epoch + 1} completed:", flush=True)
        print(f"   Training Loss: {logs.get('loss', 0):.4f}", flush=True)
        print(f"   Validation Loss: {logs.get('val_loss', 0):.4f}", flush=True)
        print(f"   Learning Rate: {logs.get('lr', 0):.6f}", flush=True)
        
        # Show improvement
        if epoch > 0:
            val_loss = logs.get('val_loss', float('inf'))
            if hasattr(self, 'best_val_loss'):
                if val_loss < self.best_val_loss:
                    print(f"   🎯 Validation loss improved: {self.best_val_loss:.4f} → {val_loss:.4f}", flush=True)
                    self.best_val_loss = val_loss
                else:
                    print(f"   ⚠️  Validation loss: {val_loss:.4f} (no improvement)", flush=True)
            else:
                self.best_val_loss = val_loss
        else:
            self.best_val_loss = logs.get('val_loss', float('inf'))
    
    def on_batch_end(self, batch, logs=None):
        # Show progress every 5 batches
        if (batch + 1) % 5 == 0:
            logs = logs or {}
            print(f"   Batch {batch + 1}: loss={logs.get('loss', 0):.4f}", flush=True)

def load_original_model(model_path):
    """Load original EQTransformer model with custom objects"""
    print(f"📥 Loading original model: {model_path}")
    
    # Import external packages first
    try:
        from keras_self_attention import SeqSelfAttention
        from keras_transformer import FeedForward
        from keras_layer_normalization import LayerNormalization
        print("✅ External keras packages loaded successfully")
        
        # Custom objects with external packages
        custom_objects = {
            'f1': f1,
            'SeqSelfAttention': SeqSelfAttention,
            'FeedForward': FeedForward,
            'LayerNormalization': LayerNormalization
        }
        
        # Try loading model
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        print(f"✅ Model loaded successfully with external packages")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shapes: {[output.shape for output in model.outputs]}")
        
        return model
        
    except ImportError as e:
        print(f"⚠️ External keras packages not available: {e}")
        
    except Exception as e:
        print(f"❌ Failed to load with external packages: {e}")
        
    # Fallback: Try with EQTransformer internal custom objects
    try:
        from EQTransformer.core.EqT_utils import SeqSelfAttention, FeedForward, LayerNormalization
        print("✅ Using EQTransformer internal custom objects")
        
        custom_objects = {
            'f1': f1,
            'SeqSelfAttention': SeqSelfAttention,
            'FeedForward': FeedForward,
            'LayerNormalization': LayerNormalization
        }
        
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        print(f"✅ Model loaded successfully with internal custom objects")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shapes: {[output.shape for output in model.outputs]}")
        
        return model
        
    except Exception as e:
        print(f"❌ Failed to load with internal custom objects: {e}")
    
    # Last resort: Try minimal custom objects
    try:
        print("⚠️ Trying minimal custom objects")
        custom_objects = {'f1': f1}
        
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        print(f"✅ Model loaded successfully with minimal custom objects")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shapes: {[output.shape for output in model.outputs]}")
        
        return model
        
    except Exception as e:
        print(f"❌ Failed to load with minimal custom objects: {e}")
        
    raise RuntimeError("Failed to load model with any configuration")

def freeze_encoder_layers(model, freeze_pattern="encoder"):
    """Freeze encoder layers, keep decoder trainable"""
    print(f"🔒 Freezing encoder layers...")
    
    frozen_count = 0
    trainable_count = 0
    
    for layer in model.layers:
        layer_name = layer.name.lower()
        
        # Freeze ENCODER-related layers (feature extraction part)
        if any(pattern in layer_name for pattern in [
            # Main encoder CNN layers
            'conv1d_',          # Numbered CNN layers in encoder (conv1d_1, conv1d_2, etc)
            'max_pooling1d',    # Pooling layers in encoder
            'batch_normalization', # BN layers in encoder
            
            # Encoder transformer blocks
            'attentiond0',      # First transformer in encoder  
            'attentiond',       # Second transformer in encoder
            
            # Encoder bidirectional LSTM and CNN blocks
            'bidirectional',    # BiLSTM layers in encoder
        ]) and not any(decoder_pattern in layer_name for decoder_pattern in [
            'decoder',          # Don't freeze decoder layers
            'picker',           # Don't freeze picker layers
            'plstm',           # Don't freeze P-specific LSTM
            'slstm',           # Don't freeze S-specific LSTM
            'attentionp',       # Don't freeze P attention
            'attentions',       # Don't freeze S attention
        ]):
            layer.trainable = False
            frozen_count += 1
            print(f"   🔒 Frozen: {layer.name}")
        
        # Keep DECODER layers trainable
        else:
            layer.trainable = True
            trainable_count += 1
            print(f"   🔓 Trainable: {layer.name}")
    
    print(f"\n📊 Layer freeze summary:")
    print(f"   Frozen layers: {frozen_count}")
    print(f"   Trainable layers: {trainable_count}")
    print(f"   Total layers: {len(model.layers)}")
    
    # Count trainable parameters more accurately
    trainable_params = 0
    total_params = 0
    
    for layer in model.layers:
        if hasattr(layer, 'count_params'):
            layer_params = layer.count_params()
            total_params += layer_params
            if layer.trainable:
                trainable_params += layer_params
    
    if total_params == 0:
        total_params = model.count_params()
        trainable_params = sum([
            np.prod(var.shape) for var in model.trainable_variables
        ])
    
    print(f"\n🔢 Parameter summary:")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable ratio: {trainable_params/total_params*100:.1f}%")
    
    return model

def prepare_indonesia_data(data_dir="datasets"):
    """Prepare windowed Indonesia dataset"""
    print(f"📊 Preparing Indonesia fine-tuning dataset...")
    
    # Check if fine-tuning dataset exists
    train_hdf5 = f"{data_dir}/indonesia_finetune_train.hdf5"
    train_csv = f"{data_dir}/indonesia_finetune_train.csv"
    valid_hdf5 = f"{data_dir}/indonesia_finetune_valid.hdf5"
    valid_csv = f"{data_dir}/indonesia_finetune_valid.csv"
    
    missing_files = []
    for file_path in [train_hdf5, train_csv, valid_hdf5, valid_csv]:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing fine-tuning dataset files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print(f"\n💡 Run windowing script first:")
        print(f"   python create_indonesia_6k_training.py --process_all")
        raise FileNotFoundError("Fine-tuning dataset not found")
    
    # Load training data info
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    
    print(f"✅ Indonesia fine-tuning dataset ready:")
    print(f"   Training samples: {len(train_df)}")
    print(f"   Validation samples: {len(valid_df)}")
    print(f"   Input shape: (6000, 3) - compatible with original model")
    
    return {
        'train_hdf5': train_hdf5,
        'train_csv': train_csv,
        'train_traces': train_df['trace_name'].tolist(),
        'valid_hdf5': valid_hdf5,
        'valid_csv': valid_csv,
        'valid_traces': valid_df['trace_name'].tolist()
    }

def create_data_generators(data_info, batch_size=16, augmentation=True):
    """Create data generators for training"""
    print(f"🔄 Creating data generators...")
    
    # Training generator with augmentation
    train_generator = IndonesiaDataGenerator(
        list_IDs=data_info['train_traces'],
        hdf5_file=data_info['train_hdf5'],
        csv_file=data_info['train_csv'],
        batch_size=batch_size,
        dim=6000,
        n_channels=3,
        shuffle=True,
        augmentation=augmentation
    )
    
    # Validation generator without augmentation
    valid_generator = IndonesiaDataGenerator(
        list_IDs=data_info['valid_traces'],
        hdf5_file=data_info['valid_hdf5'],
        csv_file=data_info['valid_csv'],
        batch_size=batch_size,
        dim=6000,
        n_channels=3,
        shuffle=False,
        augmentation=False
    )
    
    print(f"✅ Data generators created:")
    print(f"   Training batches: {len(train_generator)}")
    print(f"   Validation batches: {len(valid_generator)}")
    print(f"   Batch size: {batch_size}")
    print(f"   Augmentation: {augmentation}")
    
    return train_generator, valid_generator

def plot_training_history(history, output_dir):
    """Create training history plots"""
    print(f"📊 Creating training plots...")
    
    # Create plots directory
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive training plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('EQTransformer Fine-tuning Training History', fontsize=16, fontweight='bold')
    
    # Plot 1: Total Loss
    axes[0, 0].plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Detection Loss
    if 'detector_loss' in history.history:
        axes[0, 1].plot(history.history['detector_loss'], 'b-', label='Training', linewidth=2)
        axes[0, 1].plot(history.history['val_detector_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_title('Detection Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: P-Picker Loss
    if 'picker_P_loss' in history.history:
        axes[1, 0].plot(history.history['picker_P_loss'], 'b-', label='Training', linewidth=2)
        axes[1, 0].plot(history.history['val_picker_P_loss'], 'r-', label='Validation', linewidth=2)
        axes[1, 0].set_title('P-Wave Picker Loss', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: S-Picker Loss
    if 'picker_S_loss' in history.history:
        axes[1, 1].plot(history.history['picker_S_loss'], 'b-', label='Training', linewidth=2)
        axes[1, 1].plot(history.history['val_picker_S_loss'], 'r-', label='Validation', linewidth=2)
        axes[1, 1].set_title('S-Wave Picker Loss', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comprehensive plot
    plot_path = plots_dir / "training_history_comprehensive.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create simple loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    simple_plot_path = plots_dir / "loss_curves.png"
    plt.savefig(simple_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training plots saved:")
    print(f"   📊 Comprehensive: {plot_path}")
    print(f"   📈 Simple: {simple_plot_path}")
    
    return plot_path, simple_plot_path

def test_model_on_validation(model, data_info, output_dir, batch_size=16, 
                            p_threshold=0.1, s_threshold=0.1, det_threshold=0.1):
    """Test fine-tuned model on validation data"""
    print(f"🧪 Testing fine-tuned model on validation data...")
    
    # Create results directory
    results_dir = Path(output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create validation data generator
    valid_generator = IndonesiaDataGenerator(
        list_IDs=data_info['valid_traces'],
        hdf5_file=data_info['valid_hdf5'],
        csv_file=data_info['valid_csv'],
        batch_size=batch_size,
        dim=6000,
        n_channels=3,
        shuffle=False,
        augmentation=False
    )
    
    # Evaluate model
    print(f"📊 Evaluating on {len(data_info['valid_traces'])} validation traces...")
    evaluation_results = model.evaluate(valid_generator, verbose=0)  # Silent evaluation
    
    # Create evaluation report
    metrics_names = model.metrics_names
    evaluation_report = {
        "validation_evaluation": {
            "total_traces": len(data_info['valid_traces']),
            "batch_size": batch_size,
            "evaluation_date": datetime.now().isoformat()
        },
        "metrics": {}
    }
    
    for i, metric_name in enumerate(metrics_names):
        evaluation_report["metrics"][metric_name] = float(evaluation_results[i])
        print(f"   {metric_name}: {evaluation_results[i]:.6f}")
    
    # Create detailed predictions CSV
    print(f"📝 Creating detailed predictions CSV...")
    detailed_predictions = create_detailed_predictions_csv(
        model, valid_generator, data_info, results_dir, p_threshold, s_threshold, det_threshold
    )
    
    # Predict on sample batch for detailed analysis
    print(f"🔍 Making predictions on sample validation data...")
    sample_batch = valid_generator[0]
    X_sample, y_true_sample = sample_batch
    
    predictions = model.predict(X_sample, verbose=0)
    pred_det, pred_p, pred_s = predictions
    
    # Calculate sample statistics
    sample_stats = {
        "sample_predictions": {
            "batch_size": len(X_sample),
            "detection_accuracy": float(np.mean(np.abs(pred_det - y_true_sample[0]) < 0.1)),
            "p_picking_accuracy": float(np.mean(np.abs(pred_p - y_true_sample[1]) < 0.1)),
            "s_picking_accuracy": float(np.mean(np.abs(pred_s - y_true_sample[2]) < 0.1)),
            "mean_detection_confidence": float(np.mean(pred_det)),
            "mean_p_confidence": float(np.mean(pred_p)),
            "mean_s_confidence": float(np.mean(pred_s))
        }
    }
    
    evaluation_report.update(sample_stats)
    
    # Save evaluation results
    results_file = results_dir / "validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    
    # Create prediction visualization for first sample
    plt.figure(figsize=(15, 10))
    
    # Plot waveform
    plt.subplot(4, 1, 1)
    for i in range(3):
        plt.plot(X_sample[0, :, i], label=f'Channel {i+1}')
    plt.title('Input Waveform (First Validation Sample)', fontweight='bold')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot detection
    plt.subplot(4, 1, 2)
    plt.plot(y_true_sample[0][0, :, 0], 'g-', label='True Detection', linewidth=2)
    plt.plot(pred_det[0, :, 0], 'r--', label='Predicted Detection', linewidth=2)
    plt.title('Detection Results', fontweight='bold')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot P-picking
    plt.subplot(4, 1, 3)
    plt.plot(y_true_sample[1][0, :, 0], 'g-', label='True P-Picking', linewidth=2)
    plt.plot(pred_p[0, :, 0], 'b--', label='Predicted P-Picking', linewidth=2)
    plt.title('P-Wave Picking Results', fontweight='bold')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot S-picking
    plt.subplot(4, 1, 4)
    plt.plot(y_true_sample[2][0, :, 0], 'g-', label='True S-Picking', linewidth=2)
    plt.plot(pred_s[0, :, 0], 'm--', label='Predicted S-Picking', linewidth=2)
    plt.title('S-Wave Picking Results', fontweight='bold')
    plt.xlabel('Sample')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    prediction_plot = results_dir / "prediction_sample.png"
    plt.savefig(prediction_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Validation testing completed:")
    print(f"   📊 Results: {results_file}")
    print(f"   📈 Predictions plot: {prediction_plot}")
    print(f"   📋 Detailed CSV: {results_dir / 'detailed_predictions.csv'}")
    print(f"   🎯 Overall validation loss: {evaluation_results[0]:.6f}")
    
    return evaluation_report

def create_detailed_predictions_csv(model, data_generator, data_info, results_dir, p_threshold=0.1, s_threshold=0.1, det_threshold=0.1):
    """Create detailed CSV with prediction results for each trace"""
    print(f"📋 Creating detailed predictions for {len(data_info['valid_traces'])} traces...")
    
    # Load metadata
    df_meta = pd.read_csv(data_info['valid_csv'])
    df_meta = df_meta.set_index('trace_name')
    
    # Initialize results list
    results = []
    
    # Process each batch
    total_batches = len(data_generator)
    for batch_idx in range(total_batches):
        # Simple progress without tqdm
        print(f"   Processing batch {batch_idx + 1}/{total_batches} ({(batch_idx+1)/total_batches*100:.1f}%)...")
        
        # Get batch data
        X_batch, y_true_batch = data_generator[batch_idx]
        
        # Get trace IDs for this batch
        start_idx = batch_idx * data_generator.batch_size
        end_idx = min(start_idx + data_generator.batch_size, len(data_info['valid_traces']))
        batch_trace_ids = data_info['valid_traces'][start_idx:end_idx]
        
        # Make predictions
        predictions = model.predict(X_batch, verbose=0)
        pred_det, pred_p, pred_s = predictions
        
        # Process each trace in batch
        for i, trace_id in enumerate(batch_trace_ids):
            if i >= len(X_batch):  # Skip if batch is incomplete
                continue
                
            # Get metadata
            meta = df_meta.loc[trace_id]
            
            # Get true arrivals
            true_p_arrival = int(meta['p_arrival_sample'])
            true_s_arrival = int(meta['s_arrival_sample'])
            
            # Find predicted arrivals (peak detection)
            pred_p_curve = pred_p[i, :, 0]
            pred_s_curve = pred_s[i, :, 0]
            pred_det_curve = pred_det[i, :, 0]
            
            # Find peaks with minimum prominence
            pred_p_peak = find_peak_arrival(pred_p_curve, threshold=p_threshold)
            pred_s_peak = find_peak_arrival(pred_s_curve, threshold=s_threshold)
            pred_det_peak = find_peak_arrival(pred_det_curve, threshold=det_threshold)
            
            # Calculate confidence scores (max probability)
            p_confidence = float(np.max(pred_p_curve))
            s_confidence = float(np.max(pred_s_curve))
            det_confidence = float(np.max(pred_det_curve))
            
            # Calculate arrival errors
            p_error = abs(pred_p_peak - true_p_arrival) if pred_p_peak is not None else None
            s_error = abs(pred_s_peak - true_s_arrival) if pred_s_peak is not None else None
            
            # Calculate accuracy (within tolerance)
            p_accurate_10 = p_error is not None and p_error <= 10  # Within 10 samples
            p_accurate_20 = p_error is not None and p_error <= 20  # Within 20 samples
            s_accurate_10 = s_error is not None and s_error <= 10
            s_accurate_20 = s_error is not None and s_error <= 20
            
            # Store results with correct column names
            result = {
                'trace_name': trace_id,
                'station_code': meta.get('station_code', ''),
                'network_code': meta.get('network_code', ''),
                'source_magnitude': meta.get('source_magnitude', ''),
                'source_distance_km': meta.get('source_distance_km', ''),
                'snr_db': meta.get('snr_db', ''),
                
                # True arrivals
                'true_p_arrival_sample': true_p_arrival,
                'true_s_arrival_sample': true_s_arrival,
                'true_s_p_diff': true_s_arrival - true_p_arrival,
                
                # Predicted arrivals
                'pred_p_arrival_sample': pred_p_peak if pred_p_peak is not None else '',
                'pred_s_arrival_sample': pred_s_peak if pred_s_peak is not None else '',
                'pred_det_peak_sample': pred_det_peak if pred_det_peak is not None else '',
                'pred_s_p_diff': pred_s_peak - pred_p_peak if pred_s_peak is not None and pred_p_peak is not None else '',
                
                # Confidence scores (0-1)
                'p_confidence': f"{p_confidence:.4f}",
                's_confidence': f"{s_confidence:.4f}",
                'det_confidence': f"{det_confidence:.4f}",
                
                # Prediction errors (samples)
                'p_error_samples': p_error if p_error is not None else '',
                's_error_samples': s_error if s_error is not None else '',
                
                # Accuracy flags
                'p_accurate_10_samples': p_accurate_10,
                'p_accurate_20_samples': p_accurate_20,
                's_accurate_10_samples': s_accurate_10,
                's_accurate_20_samples': s_accurate_20,
                
                # Detection status
                'p_detected': pred_p_peak is not None,
                's_detected': pred_s_peak is not None,
                'event_detected': pred_det_peak is not None,
                
                # Quality metrics
                'prediction_quality': calculate_prediction_quality(p_error, s_error, p_confidence, s_confidence)
            }
            
            results.append(result)
    
    # Create DataFrame and save
    df_results = pd.DataFrame(results)
    
    # Add summary statistics
    summary_stats = {
        'total_traces': len(df_results),
        'p_detection_rate': f"{(df_results['p_detected'].sum() / len(df_results) * 100):.1f}%",
        's_detection_rate': f"{(df_results['s_detected'].sum() / len(df_results) * 100):.1f}%",
        'event_detection_rate': f"{(df_results['event_detected'].sum() / len(df_results) * 100):.1f}%",
        'p_accuracy_10_samples': f"{(df_results['p_accurate_10_samples'].sum() / len(df_results) * 100):.1f}%",
        'p_accuracy_20_samples': f"{(df_results['p_accurate_20_samples'].sum() / len(df_results) * 100):.1f}%",
        's_accuracy_10_samples': f"{(df_results['s_accurate_10_samples'].sum() / len(df_results) * 100):.1f}%",
        's_accuracy_20_samples': f"{(df_results['s_accurate_20_samples'].sum() / len(df_results) * 100):.1f}%",
        'mean_p_confidence': f"{df_results['p_confidence'].astype(float).mean():.3f}",
        'mean_s_confidence': f"{df_results['s_confidence'].astype(float).mean():.3f}",
        'mean_det_confidence': f"{df_results['det_confidence'].astype(float).mean():.3f}"
    }
    
    # Save detailed results
    csv_file = results_dir / "detailed_predictions.csv"
    df_results.to_csv(csv_file, index=False)
    
    # Save summary statistics
    summary_file = results_dir / "prediction_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"✅ Detailed predictions saved:")
    print(f"   📋 CSV: {csv_file} ({len(df_results)} traces)")
    print(f"   📊 Summary: {summary_file}")
    print(f"   🎯 P-wave detection rate: {summary_stats['p_detection_rate']}")
    print(f"   🎯 S-wave detection rate: {summary_stats['s_detection_rate']}")
    print(f"   🎯 P-wave accuracy (±10 samples): {summary_stats['p_accuracy_10_samples']}")
    print(f"   🎯 S-wave accuracy (±10 samples): {summary_stats['s_accuracy_10_samples']}")
    
    return df_results

def find_peak_arrival(prediction_curve, threshold=0.1, min_distance=20):
    """Find the most prominent peak in prediction curve"""
    # Find all peaks above threshold
    peaks = []
    for i in range(min_distance, len(prediction_curve) - min_distance):
        if (prediction_curve[i] > threshold and 
            prediction_curve[i] > prediction_curve[i-1] and 
            prediction_curve[i] > prediction_curve[i+1]):
            peaks.append((i, prediction_curve[i]))
    
    if not peaks:
        return None
    
    # Return the highest peak
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks[0][0]

def calculate_prediction_quality(p_error, s_error, p_conf, s_conf):
    """Calculate overall prediction quality score"""
    score = 0
    
    # Confidence component (0-40 points)
    score += min(40, (p_conf + s_conf) * 20)
    
    # Accuracy component (0-60 points)
    if p_error is not None:
        if p_error <= 5:
            score += 30  # Excellent P accuracy
        elif p_error <= 10:
            score += 20  # Good P accuracy
        elif p_error <= 20:
            score += 10  # Fair P accuracy
    
    if s_error is not None:
        if s_error <= 5:
            score += 30  # Excellent S accuracy
        elif s_error <= 10:
            score += 20  # Good S accuracy
        elif s_error <= 20:
            score += 10  # Fair S accuracy
    
    # Convert to letter grade
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"

def setup_training_callbacks(output_dir, patience=5):
    """Setup training callbacks"""
    print(f"⚙️ Setting up training callbacks...")
    
    # Create output directories
    models_dir = Path(output_dir) / "model"
    logs_dir = Path(output_dir) / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=str(models_dir / "best_finetuned_model.h5"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience//2,
            min_lr=1e-6,
            verbose=1
        ),
        
        # Real-time progress callback
        RealTimeProgressCallback()
    ]
    
    print(f"✅ Callbacks configured:")
    print(f"   📦 Model: {models_dir / 'best_finetuned_model.h5'}")
    print(f"   📋 Logs: {logs_dir}")
    print(f"   ⏰ Early stopping patience: {patience}")
    print(f"   📉 LR reduction patience: {patience//2}")
    
    return callbacks

def fine_tune_model(
    original_model_path,
    output_dir,
    epochs=10,
    batch_size=16,
    learning_rate=0.0001,  # Lower LR for fine-tuning
    patience=5,
    augmentation=True,
    p_threshold=0.1,
    s_threshold=0.1,
    det_threshold=0.1
):
    """Main fine-tuning function"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"models/finetune_indonesia_{timestamp}"
    
    print(f"🌋 EQTransformer Fine-tuning for Indonesia Data")
    print(f"=" * 60)
    print(f"Original model: {original_model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Strategy: Freeze encoder, train decoder only")
    print(f"Detection thresholds: P={p_threshold}, S={s_threshold}, Det={det_threshold}")
    
    # 1. Load original model
    model = load_original_model(original_model_path)
    
    # 2. Freeze encoder layers
    model = freeze_encoder_layers(model)
    
    # 3. Prepare data
    data_info = prepare_indonesia_data()
    
    # 4. Create data generators
    train_gen, valid_gen = create_data_generators(
        data_info, batch_size, augmentation
    )
    
    # 5. Setup callbacks
    callbacks = setup_training_callbacks(output_dir, patience)
    
    # 6. Compile model with lower learning rate
    print(f"⚙️ Compiling model for fine-tuning...")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
        loss_weights=[0.2, 0.3, 0.5],  # Same as original
        metrics=[f1]
    )
    
    print(f"✅ Model compiled with learning rate: {learning_rate}")
    
    # 7. Train model
    print(f"\n🚀 Starting fine-tuning...", flush=True)
    print(f"   Epochs: {epochs}", flush=True)
    print(f"   Batch size: {batch_size}", flush=True)
    print(f"   Learning rate: {learning_rate}", flush=True)
    print(f"   Training samples: {len(data_info['train_traces'])}", flush=True)
    print(f"   Validation samples: {len(data_info['valid_traces'])}", flush=True)
    print(f"   P-wave threshold: {p_threshold}", flush=True)
    print(f"   S-wave threshold: {s_threshold}", flush=True)
    print(f"   Detection threshold: {det_threshold}", flush=True)
    
    start_time = datetime.now()
    
    # Force flush stdout untuk real-time output
    sys.stdout.flush()
    
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1  # Kembali ke progress bar dengan custom callback
    )
    
    end_time = datetime.now()
    training_duration = end_time - start_time
    
    # 8. Save final model
    final_model_path = Path(output_dir) / "model" / "final_finetuned_model.h5"
    model.save(str(final_model_path))
    
    print(f"\n🎉 Fine-tuning completed!")
    print(f"   Duration: {training_duration}")
    print(f"   Best model: {Path(output_dir) / 'model' / 'best_finetuned_model.h5'}")
    print(f"   Final model: {final_model_path}")
    
    # 9. Create training plots
    plot_comprehensive, plot_simple = plot_training_history(history, output_dir)
    
    # 10. Test model on validation data
    evaluation_report = test_model_on_validation(
        model, data_info, output_dir, batch_size, 
        p_threshold, s_threshold, det_threshold
    )
    
    # 11. Save comprehensive training info
    training_info = {
        "fine_tuning_info": {
            "timestamp": timestamp,
            "original_model": original_model_path,
            "output_directory": output_dir,
            "strategy": "freeze_encoder_train_decoder"
        },
        "model_config": {
            "input_shape": "(6000, 3)",
            "encoder_frozen": True,
            "decoder_trainable": True,
            "learning_rate": learning_rate,
            "epochs_trained": len(history.history['loss'])
        },
        "dataset_config": {
            "train_samples": len(data_info['train_traces']),
            "valid_samples": len(data_info['valid_traces']),
            "batch_size": batch_size,
            "augmentation": augmentation,
            "windowing": "30000→6000 samples (optimal P/S centered)"
        },
        "threshold_config": {
            "p_threshold": p_threshold,
            "s_threshold": s_threshold,
            "det_threshold": det_threshold,
            "description": "Thresholds used for peak detection during validation testing"
        },
        "training_results": {
            "final_train_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]),
            "best_val_loss": float(min(history.history['val_loss'])),
            "training_duration_seconds": training_duration.total_seconds(),
            "training_duration_formatted": str(training_duration)
        },
        "files_created": {
            "best_model": str(Path(output_dir) / 'model' / 'best_finetuned_model.h5'),
            "final_model": str(final_model_path),
            "comprehensive_plot": str(plot_comprehensive),
            "simple_plot": str(plot_simple),
            "validation_results": str(Path(output_dir) / 'results' / 'validation_results.json'),
            "prediction_plot": str(Path(output_dir) / 'results' / 'prediction_sample.png')
        }
    }
    
    # Add validation results to training info
    training_info["validation_evaluation"] = evaluation_report
    
    info_file = Path(output_dir) / "fine_tuning_complete_info.json"
    with open(info_file, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    # Create summary log
    logs_dir = Path(output_dir) / "logs"
    summary_log = logs_dir / "training_summary.txt"
    with open(summary_log, 'w') as f:
        f.write("EQTransformer Fine-tuning Summary Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Training Duration: {training_duration}\n")
        f.write(f"Detection Thresholds: P={p_threshold}, S={s_threshold}, Det={det_threshold}\n")
        f.write(f"Final Training Loss: {training_info['training_results']['final_train_loss']:.6f}\n")
        f.write(f"Final Validation Loss: {training_info['training_results']['final_val_loss']:.6f}\n")
        f.write(f"Best Validation Loss: {training_info['training_results']['best_val_loss']:.6f}\n")
        f.write(f"Validation Samples Tested: {evaluation_report['validation_evaluation']['total_traces']}\n")
        f.write(f"\nFiles Created:\n")
        for file_type, file_path in training_info['files_created'].items():
            f.write(f"  - {file_type}: {file_path}\n")
    
    print(f"📊 Complete training info saved: {info_file}")
    print(f"📋 Training summary: {summary_log}")
    
    print(f"\n🎊 Fine-tuning pipeline completed successfully!")
    print(f"📁 All results saved in: {output_dir}")
    print(f"📁 Directory structure:")
    print(f"   📦 {output_dir}/model/ - Saved models")
    print(f"   📊 {output_dir}/plots/ - Training curves")
    print(f"   📋 {output_dir}/logs/ - Training logs")
    print(f"   🧪 {output_dir}/results/ - Validation test results")
    
    return model, history, output_dir

def main():
    parser = argparse.ArgumentParser(description='Fine-tune EQTransformer original model with Indonesia data')
    parser.add_argument('--model', required=True, help='Path to original EQTransformer model')
    parser.add_argument('--output_dir', default=None, help='Output directory (auto-generated if not specified)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--augmentation', action='store_true', default=True, help='Enable data augmentation')
    parser.add_argument('--no_augmentation', action='store_false', dest='augmentation', help='Disable data augmentation')
    parser.add_argument('--p_threshold', type=float, default=0.1, help='P-wave detection threshold')
    parser.add_argument('--s_threshold', type=float, default=0.1, help='S-wave detection threshold')
    parser.add_argument('--det_threshold', type=float, default=0.1, help='Earthquake detection threshold')
    
    args = parser.parse_args()
    
    # Validate model file
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)
    
    # Run fine-tuning
    model, history, output_dir = fine_tune_model(
        original_model_path=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        augmentation=args.augmentation,
        p_threshold=args.p_threshold,
        s_threshold=args.s_threshold,
        det_threshold=args.det_threshold
    )

if __name__ == "__main__":
    main() 