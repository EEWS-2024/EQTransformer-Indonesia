#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EQTransformer Indonesia Model Tester 🌋

Script untuk testing model EQTransformer Indonesia yang sudah di-training
Input: Model path dan test dataset
Output: Comprehensive evaluation results
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add EQT core to path
current_dir = os.path.dirname(os.path.abspath(__file__))
eqt_core_dir = os.path.join(current_dir, '..', 'core')
sys.path.append(eqt_core_dir)

# Import TensorFlow after path setup
import tensorflow as tf
from tensorflow.keras.models import load_model
from EqT_utils import DataGeneratorTest

def setup_gpu():
    """Setup GPU if available"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 GPU detected: {len(gpus)} device(s)")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return True
        else:
            print("💻 Using CPU")
            return False
    except Exception as e:
        print(f"⚠️ GPU setup error: {e}")
        return False

def find_latest_model(models_dir="../../models"):
    """Find the latest trained model"""
    models_path = os.path.join(current_dir, models_dir)
    
    if not os.path.exists(models_path):
        print(f"❌ Models directory not found: {models_path}")
        return None
    
    # Find all model directories
    model_dirs = [d for d in os.listdir(models_path) 
                  if d.startswith('indonesia_model_') and os.path.isdir(os.path.join(models_path, d))]
    
    if not model_dirs:
        print(f"❌ No Indonesia models found in: {models_path}")
        return None
    
    # Sort by timestamp (newest first)
    model_dirs.sort(reverse=True)
    latest_model_dir = model_dirs[0]
    
    # Check for final_model.h5
    model_file = os.path.join(models_path, latest_model_dir, 'final_model.h5')
    if os.path.exists(model_file):
        print(f"✅ Found latest model: {latest_model_dir}")
        return model_file
    else:
        print(f"❌ final_model.h5 not found in: {latest_model_dir}")
        return None

def load_test_data(test_csv="../../datasets/indonesia_valid.csv", test_hdf5="../../datasets/indonesia_valid.hdf5"):
    """Load test dataset"""
    test_csv_path = os.path.join(current_dir, test_csv)
    test_hdf5_path = os.path.join(current_dir, test_hdf5)
    
    if not os.path.exists(test_csv_path):
        print(f"❌ Test CSV not found: {test_csv_path}")
        return None, None, 0
    
    if not os.path.exists(test_hdf5_path):
        print(f"❌ Test HDF5 not found: {test_hdf5_path}")
        return None, None, 0
    
    # Load CSV
    try:
        df = pd.read_csv(test_csv_path)
        test_traces = df['trace_name'].tolist()
        
        print(f"📊 Test Dataset Info:")
        print(f"   CSV file: {os.path.basename(test_csv_path)}")
        print(f"   HDF5 file: {os.path.basename(test_hdf5_path)}")
        print(f"   Total test traces: {len(test_traces)}")
        
        # Show trace categories distribution
        if 'trace_category' in df.columns:
            category_counts = df['trace_category'].value_counts()
            print(f"   Trace categories:")
            for category, count in category_counts.items():
                print(f"     {category}: {count} traces ({count/len(df)*100:.1f}%)")
        
        return test_hdf5_path, test_traces, len(test_traces)
        
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None, None, 0

def create_test_generator(test_hdf5, test_traces, batch_size=32, input_dim=30085):
    """Create data generator for testing"""
    try:
        test_generator = DataGeneratorTest(
            list_IDs=test_traces,
            file_name=test_hdf5,
            dim=input_dim,  # Indonesia signature: 30085 samples
            batch_size=batch_size,
            n_channels=3,
            norm_mode='std'
        )
        
        print(f"✅ Test generator created:")
        print(f"   Test batches: {len(test_generator)}")
        print(f"   Batch size: {batch_size}")
        print(f"   Input dimension: {input_dim}")
        
        return test_generator
        
    except Exception as e:
        print(f"❌ Error creating test generator: {e}")
        return None

def analyze_predictions(predictions, test_traces, test_hdf5, output_dir, sampling_rate=100):
    """Analyze model predictions comprehensively"""
    print("\n🔍 ANALYZING PREDICTIONS...")
    
    detector_preds, picker_P_preds, picker_S_preds = predictions
    
    results = []
    
    # Analyze each trace
    print("📊 Processing predictions for each trace...")
    with h5py.File(test_hdf5, 'r') as f:
        for i, trace_name in enumerate(tqdm(test_traces, desc="Analyzing traces")):
            if i >= detector_preds.shape[0]:
                break
                
            try:
                # Get predictions for this trace
                pred_detector = detector_preds[i].flatten()
                pred_picker_P = picker_P_preds[i].flatten()
                pred_picker_S = picker_S_preds[i].flatten()
                
                # Get ground truth from HDF5
                dataset = f.get('data/' + trace_name)
                if dataset is None:
                    continue
                    
                trace_category = dataset.attrs.get('trace_category', 'unknown')
                
                # Get ground truth arrivals if available
                true_P_sample = dataset.attrs.get('p_arrival_sample', None)
                true_S_sample = dataset.attrs.get('s_arrival_sample', None)
                snr_db = dataset.attrs.get('snr_db', None)
                
                # Find predicted peaks
                pred_P_sample = int(np.argmax(pred_picker_P))
                pred_S_sample = int(np.argmax(pred_picker_S))
                
                # Calculate detection metrics
                detector_max = float(np.max(pred_detector))
                detector_mean = float(np.mean(pred_detector))
                
                picker_P_max = float(np.max(pred_picker_P))
                picker_P_mean = float(np.mean(pred_picker_P))
                
                picker_S_max = float(np.max(pred_picker_S))
                picker_S_mean = float(np.mean(pred_picker_S))
                
                # Time conversions
                pred_P_time = pred_P_sample / sampling_rate
                pred_S_time = pred_S_sample / sampling_rate
                
                # Calculate errors if ground truth available
                P_error_samples = None
                S_error_samples = None
                P_error_seconds = None
                S_error_seconds = None
                
                if true_P_sample is not None:
                    P_error_samples = pred_P_sample - true_P_sample
                    P_error_seconds = P_error_samples / sampling_rate
                    
                if true_S_sample is not None:
                    S_error_samples = pred_S_sample - true_S_sample
                    S_error_seconds = S_error_samples / sampling_rate
                
                # Detection thresholds
                is_earthquake_detected = detector_max > 0.5
                is_P_detected = picker_P_max > 0.3
                is_S_detected = picker_S_max > 0.3
                
                # Ground truth flags
                has_earthquake = trace_category == 'earthquake_local'
                has_P_wave = true_P_sample is not None and true_P_sample > 0
                has_S_wave = true_S_sample is not None and true_S_sample > 0
                
                results.append({
                    # Trace info
                    'trace_name': trace_name,
                    'trace_category': trace_category,
                    'snr_db': snr_db,
                    
                    # Ground truth
                    'true_P_sample': true_P_sample,
                    'true_S_sample': true_S_sample,
                    'has_earthquake': has_earthquake,
                    'has_P_wave': has_P_wave,
                    'has_S_wave': has_S_wave,
                    
                    # Predictions
                    'pred_P_sample': pred_P_sample,
                    'pred_S_sample': pred_S_sample,
                    'pred_P_time_sec': pred_P_time,
                    'pred_S_time_sec': pred_S_time,
                    
                    # Detection metrics
                    'detector_max': detector_max,
                    'detector_mean': detector_mean,
                    'picker_P_max': picker_P_max,
                    'picker_P_mean': picker_P_mean,
                    'picker_S_max': picker_S_max,
                    'picker_S_mean': picker_S_mean,
                    
                    # Detection flags
                    'is_earthquake_detected': is_earthquake_detected,
                    'is_P_detected': is_P_detected,
                    'is_S_detected': is_S_detected,
                    
                    # Errors
                    'P_error_samples': P_error_samples,
                    'S_error_samples': S_error_samples,
                    'P_error_seconds': P_error_seconds,
                    'S_error_seconds': S_error_seconds
                })
                
            except Exception as e:
                print(f"⚠️ Error processing trace {trace_name}: {e}")
                continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("❌ No valid results to analyze!")
        return None
    
    # Save detailed results
    results_csv = os.path.join(output_dir, 'detailed_test_results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"💾 Detailed results saved: {results_csv}")
    
    return results_df

def calculate_performance_metrics(results_df, output_dir):
    """Calculate comprehensive performance metrics"""
    print("\n📊 CALCULATING PERFORMANCE METRICS...")
    
    metrics = {}
    
    # Filter valid data
    valid_results = results_df.dropna(subset=['P_error_seconds', 'S_error_seconds'])
    earthquake_results = results_df[results_df['has_earthquake'] == True]
    noise_results = results_df[results_df['has_earthquake'] == False]
    
    print(f"📈 Dataset breakdown:")
    print(f"   Total traces analyzed: {len(results_df)}")
    print(f"   Earthquake traces: {len(earthquake_results)}")
    print(f"   Noise traces: {len(noise_results)}")
    print(f"   Valid P/S error data: {len(valid_results)}")
    
    # Detection Accuracy
    if len(results_df) > 0:
        # Earthquake detection
        earthquake_detection_acc = results_df['is_earthquake_detected'].eq(results_df['has_earthquake']).mean()
        
        # P-wave detection (only for earthquake traces)
        if len(earthquake_results) > 0:
            P_detection_acc = earthquake_results['is_P_detected'].eq(earthquake_results['has_P_wave']).mean()
            S_detection_acc = earthquake_results['is_S_detected'].eq(earthquake_results['has_S_wave']).mean()
        else:
            P_detection_acc = S_detection_acc = 0.0
        
        metrics['detection'] = {
            'earthquake_detection_accuracy': earthquake_detection_acc,
            'P_wave_detection_accuracy': P_detection_acc,
            'S_wave_detection_accuracy': S_detection_acc
        }
    
    # Timing Accuracy (for valid P/S data)
    if len(valid_results) > 0:
        P_errors = valid_results['P_error_seconds'].abs()
        S_errors = valid_results['S_error_seconds'].abs()
        
        metrics['timing_accuracy'] = {
            'P_mean_error_sec': P_errors.mean(),
            'P_median_error_sec': P_errors.median(),
            'P_std_error_sec': valid_results['P_error_seconds'].std(),
            'P_95th_percentile_error_sec': P_errors.quantile(0.95),
            
            'S_mean_error_sec': S_errors.mean(),
            'S_median_error_sec': S_errors.median(),
            'S_std_error_sec': valid_results['S_error_seconds'].std(),
            'S_95th_percentile_error_sec': S_errors.quantile(0.95),
        }
        
        # Accuracy within thresholds
        P_acc_1s = (P_errors <= 1.0).mean()
        P_acc_0_5s = (P_errors <= 0.5).mean()
        P_acc_0_1s = (P_errors <= 0.1).mean()
        
        S_acc_1s = (S_errors <= 1.0).mean()
        S_acc_0_5s = (S_errors <= 0.5).mean()
        S_acc_0_1s = (S_errors <= 0.1).mean()
        
        metrics['timing_thresholds'] = {
            'P_accuracy_within_1s': P_acc_1s,
            'P_accuracy_within_0.5s': P_acc_0_5s,
            'P_accuracy_within_0.1s': P_acc_0_1s,
            'S_accuracy_within_1s': S_acc_1s,
            'S_accuracy_within_0.5s': S_acc_0_5s,
            'S_accuracy_within_0.1s': S_acc_0_1s,
        }
    
    # Detection probability statistics
    metrics['detection_probabilities'] = {
        'detector_mean_prob': results_df['detector_max'].mean(),
        'detector_std_prob': results_df['detector_max'].std(),
        'picker_P_mean_prob': results_df['picker_P_max'].mean(),
        'picker_P_std_prob': results_df['picker_P_max'].std(),
        'picker_S_mean_prob': results_df['picker_S_max'].mean(),
        'picker_S_std_prob': results_df['picker_S_max'].std(),
    }
    
    # SNR-based analysis (if available)
    if 'snr_db' in results_df.columns and not results_df['snr_db'].isna().all():
        snr_data = valid_results.dropna(subset=['snr_db'])
        if len(snr_data) > 0:
            # Bin by SNR ranges
            snr_bins = [(-np.inf, 10), (10, 20), (20, 30), (30, np.inf)]
            snr_analysis = {}
            
            for snr_min, snr_max in snr_bins:
                if snr_max == np.inf:
                    bin_name = f"SNR_{snr_min}+"
                    bin_data = snr_data[snr_data['snr_db'] >= snr_min]
                else:
                    bin_name = f"SNR_{snr_min}-{snr_max}"
                    bin_data = snr_data[(snr_data['snr_db'] >= snr_min) & (snr_data['snr_db'] < snr_max)]
                
                if len(bin_data) > 0:
                    snr_analysis[bin_name] = {
                        'count': len(bin_data),
                        'P_mean_error': bin_data['P_error_seconds'].abs().mean(),
                        'S_mean_error': bin_data['S_error_seconds'].abs().mean(),
                        'detection_accuracy': bin_data['is_earthquake_detected'].eq(bin_data['has_earthquake']).mean()
                    }
            
            metrics['snr_analysis'] = snr_analysis
    
    # Save metrics
    metrics_file = os.path.join(output_dir, 'performance_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("EQTransformer Indonesia Model Performance Metrics 🌋\n")
        f.write("=" * 70 + "\n\n")
        
        if 'detection' in metrics:
            f.write("🎯 DETECTION ACCURACY:\n")
            for key, value in metrics['detection'].items():
                f.write(f"   {key}: {value:.4f}\n")
            f.write("\n")
        
        if 'timing_accuracy' in metrics:
            f.write("⏰ TIMING ACCURACY:\n")
            for key, value in metrics['timing_accuracy'].items():
                f.write(f"   {key}: {value:.4f}\n")
            f.write("\n")
            
        if 'timing_thresholds' in metrics:
            f.write("🎯 TIMING THRESHOLDS:\n")
            for key, value in metrics['timing_thresholds'].items():
                f.write(f"   {key}: {value:.4f}\n")
            f.write("\n")
        
        if 'detection_probabilities' in metrics:
            f.write("📊 DETECTION PROBABILITIES:\n")
            for key, value in metrics['detection_probabilities'].items():
                f.write(f"   {key}: {value:.4f}\n")
            f.write("\n")
        
        if 'snr_analysis' in metrics:
            f.write("📈 SNR-BASED ANALYSIS:\n")
            for bin_name, bin_metrics in metrics['snr_analysis'].items():
                f.write(f"   {bin_name}:\n")
                for key, value in bin_metrics.items():
                    if key == 'count':
                        f.write(f"     {key}: {value}\n")
                    else:
                        f.write(f"     {key}: {value:.4f}\n")
                f.write("\n")
    
    print(f"💾 Performance metrics saved: {metrics_file}")
    
    return metrics

def create_visualizations(results_df, metrics, output_dir):
    """Create comprehensive visualizations"""
    print("\n📈 CREATING VISUALIZATIONS...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplot figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('EQTransformer Indonesia Model Performance Analysis 🌋', fontsize=16, fontweight='bold')
    
    # 1. Detection Probabilities Distribution
    ax1 = axes[0, 0]
    valid_detector = results_df['detector_max'].dropna()
    valid_P = results_df['picker_P_max'].dropna()
    valid_S = results_df['picker_S_max'].dropna()
    
    ax1.hist(valid_detector, bins=30, alpha=0.7, label='Detector', color='red')
    ax1.hist(valid_P, bins=30, alpha=0.7, label='P-picker', color='blue')
    ax1.hist(valid_S, bins=30, alpha=0.7, label='S-picker', color='green')
    ax1.set_xlabel('Maximum Probability')
    ax1.set_ylabel('Count')
    ax1.set_title('Detection Probability Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. P-wave Error Distribution
    ax2 = axes[0, 1]
    valid_P_errors = results_df['P_error_seconds'].dropna()
    if len(valid_P_errors) > 0:
        ax2.hist(valid_P_errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', alpha=0.8, label='Perfect')
        ax2.set_xlabel('P-wave Error (seconds)')
        ax2.set_ylabel('Count')
        ax2.set_title('P-wave Timing Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. S-wave Error Distribution  
    ax3 = axes[0, 2]
    valid_S_errors = results_df['S_error_seconds'].dropna()
    if len(valid_S_errors) > 0:
        ax3.hist(valid_S_errors, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(0, color='red', linestyle='--', alpha=0.8, label='Perfect')
        ax3.set_xlabel('S-wave Error (seconds)')
        ax3.set_ylabel('Count')
        ax3.set_title('S-wave Timing Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Detection Accuracy by Category
    ax4 = axes[1, 0]
    if 'detection' in metrics:
        categories = ['Earthquake\nDetection', 'P-wave\nDetection', 'S-wave\nDetection']
        accuracies = [
            metrics['detection']['earthquake_detection_accuracy'],
            metrics['detection']['P_wave_detection_accuracy'],
            metrics['detection']['S_wave_detection_accuracy']
        ]
        colors = ['red', 'blue', 'green']
        bars = ax4.bar(categories, accuracies, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Detection Accuracy by Type')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        # Add accuracy values on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Timing Accuracy Thresholds
    ax5 = axes[1, 1]
    if 'timing_thresholds' in metrics:
        thresholds = ['1.0s', '0.5s', '0.1s']
        P_accs = [
            metrics['timing_thresholds']['P_accuracy_within_1s'],
            metrics['timing_thresholds']['P_accuracy_within_0.5s'],
            metrics['timing_thresholds']['P_accuracy_within_0.1s']
        ]
        S_accs = [
            metrics['timing_thresholds']['S_accuracy_within_1s'],
            metrics['timing_thresholds']['S_accuracy_within_0.5s'],
            metrics['timing_thresholds']['S_accuracy_within_0.1s']
        ]
        
        x = np.arange(len(thresholds))
        width = 0.35
        
        ax5.bar(x - width/2, P_accs, width, label='P-wave', color='blue', alpha=0.7)
        ax5.bar(x + width/2, S_accs, width, label='S-wave', color='green', alpha=0.7)
        
        ax5.set_xlabel('Error Threshold')
        ax5.set_ylabel('Accuracy')
        ax5.set_title('Timing Accuracy within Thresholds')
        ax5.set_xticks(x)
        ax5.set_xticklabels(thresholds)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Performance Summary Text
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = "📊 PERFORMANCE SUMMARY\n\n"
    
    if 'detection' in metrics:
        summary_text += f"🎯 Detection Accuracy:\n"
        summary_text += f"  Earthquake: {metrics['detection']['earthquake_detection_accuracy']:.3f}\n"
        summary_text += f"  P-wave: {metrics['detection']['P_wave_detection_accuracy']:.3f}\n"
        summary_text += f"  S-wave: {metrics['detection']['S_wave_detection_accuracy']:.3f}\n\n"
    
    if 'timing_accuracy' in metrics:
        summary_text += f"⏰ Timing Errors:\n"
        summary_text += f"  P-wave: {metrics['timing_accuracy']['P_mean_error_sec']:.3f}s\n"
        summary_text += f"  S-wave: {metrics['timing_accuracy']['S_mean_error_sec']:.3f}s\n\n"
    
    if 'timing_thresholds' in metrics:
        summary_text += f"🎯 Within 0.5s:\n"
        summary_text += f"  P-wave: {metrics['timing_thresholds']['P_accuracy_within_0.5s']:.3f}\n"
        summary_text += f"  S-wave: {metrics['timing_thresholds']['S_accuracy_within_0.5s']:.3f}\n\n"
    
    summary_text += f"📈 Dataset:\n"
    summary_text += f"  Total traces: {len(results_df)}\n"
    earthquake_count = len(results_df[results_df['has_earthquake'] == True])
    summary_text += f"  Earthquakes: {earthquake_count}\n"
    summary_text += f"  Noise: {len(results_df) - earthquake_count}\n"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save visualization
    viz_file = os.path.join(output_dir, 'performance_analysis.png')
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Visualizations saved: {viz_file}")

def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description='EQTransformer Indonesia Model Tester 🌋')
    parser.add_argument('--model', type=str, help='Path to model file (.h5)')
    parser.add_argument('--test_csv', type=str, default='../../datasets/indonesia_valid.csv', 
                       help='Path to test CSV file')
    parser.add_argument('--test_hdf5', type=str, default='../../datasets/indonesia_valid.hdf5',
                       help='Path to test HDF5 file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    
    args = parser.parse_args()
    
    print("EQTransformer Indonesia Model Tester 🌋")
    print("=" * 60)
    
    # Setup GPU
    gpu_available = setup_gpu()
    
    # Find model if not specified
    if args.model:
        model_path = args.model
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return
    else:
        print("🔍 Looking for latest trained model...")
        model_path = find_latest_model()
        if not model_path:
            return
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(current_dir, f"../../test_results/test_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Load test data
    print(f"\n📊 Loading test data...")
    test_hdf5, test_traces, num_traces = load_test_data(args.test_csv, args.test_hdf5)
    if not test_hdf5:
        return
    
    # Create test generator
    print(f"\n🔄 Creating test generator...")
    test_generator = create_test_generator(test_hdf5, test_traces, args.batch_size)
    if not test_generator:
        return
    
    # Load model
    print(f"\n🤖 Loading model...")
    print(f"   Model path: {model_path}")
    
    try:
        start_time = time.time()
        model = load_model(model_path)
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        print(f"   Model parameters: {model.count_params():,}")
        
        # Model summary
        input_shape = model.input_shape
        output_shapes = [output.shape for output in model.outputs]
        print(f"   Input shape: {input_shape}")
        print(f"   Output shapes: {output_shapes}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Run predictions
    print(f"\n🚀 Running predictions...")
    print(f"   Test batches: {len(test_generator)}")
    print(f"   Expected samples: {len(test_generator) * args.batch_size}")
    
    try:
        start_time = time.time()
        
        predictions = model.predict(test_generator, verbose=1)
        
        prediction_time = time.time() - start_time
        
        print(f"✅ Predictions completed in {prediction_time:.2f} seconds")
        print(f"   Prediction shapes: {[pred.shape for pred in predictions]}")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return
    
    # Analyze results
    results_df = analyze_predictions(predictions, test_traces, test_hdf5, output_dir)
    if results_df is None:
        return
    
    # Calculate metrics
    metrics = calculate_performance_metrics(results_df, output_dir)
    
    # Create visualizations
    create_visualizations(results_df, metrics, output_dir)
    
    # Final summary
    print(f"\n🎉 TESTING COMPLETED SUCCESSFULLY!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Files generated:")
    print(f"   - detailed_test_results.csv: Raw prediction results")
    print(f"   - performance_metrics.txt: Comprehensive metrics")
    print(f"   - performance_analysis.png: Visualization plots")
    
    # Print key metrics
    if 'detection' in metrics:
        print(f"\n🎯 KEY PERFORMANCE METRICS:")
        print(f"   Earthquake Detection: {metrics['detection']['earthquake_detection_accuracy']:.3f}")
        print(f"   P-wave Detection: {metrics['detection']['P_wave_detection_accuracy']:.3f}")
        print(f"   S-wave Detection: {metrics['detection']['S_wave_detection_accuracy']:.3f}")
    
    if 'timing_accuracy' in metrics:
        print(f"   P-wave Timing Error: {metrics['timing_accuracy']['P_mean_error_sec']:.3f}s")
        print(f"   S-wave Timing Error: {metrics['timing_accuracy']['S_mean_error_sec']:.3f}s")

if __name__ == "__main__":
    main() 