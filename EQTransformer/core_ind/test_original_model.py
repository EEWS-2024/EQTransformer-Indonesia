#!/usr/bin/env python3
"""
EQTransformer Original Model Tester
Test original EQTransformer model with Indonesia 6K dataset
"""

import sys
import os
import numpy as np
import pandas as pd
import time
import argparse

# Add paths
sys.path.append('.')
sys.path.append('..')

# Import TensorFlow first
import tensorflow as tf

# Import testing modules
from test_indonesia_model import *

def create_custom_objects():
    """Create custom objects dictionary for original model"""
    custom_objects = {}
    
    # Try to import custom layers
    try:
        from keras_self_attention import SeqSelfAttention
        custom_objects['SeqSelfAttention'] = SeqSelfAttention
        print("✅ SeqSelfAttention imported")
    except ImportError as e:
        print(f"❌ SeqSelfAttention not available: {e}")
    
    try:
        from keras_transformer import FeedForward
        custom_objects['FeedForward'] = FeedForward
        print("✅ FeedForward imported")
    except ImportError:
        try:
            from keras_position_wise_feed_forward import FeedForward
            custom_objects['FeedForward'] = FeedForward
            print("✅ FeedForward (alt) imported")
        except ImportError as e:
            print(f"❌ FeedForward not available: {e}")
    
    # Add custom metrics
    def f1_score(y_true, y_pred):
        """F1 score metric"""
        def recall(y_true, y_pred):
            true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
            possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
            return recall

        def precision(y_true, y_pred):
            true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
            predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
            return precision

        precision_val = precision(y_true, y_pred)
        recall_val = recall(y_true, y_pred)
        return 2*((precision_val*recall_val)/(precision_val+recall_val+tf.keras.backend.epsilon()))
    
    # Add metrics to custom objects
    custom_objects['f1'] = f1_score
    custom_objects['precision'] = f1_score
    custom_objects['recall'] = f1_score
    
    return custom_objects

def main():
    """Main function for original model testing"""
    parser = argparse.ArgumentParser(description='Test Original EQTransformer with Indonesia 6K Dataset')
    parser.add_argument('--model', type=str, required=True, help='Path to original model')
    parser.add_argument('--test_csv', type=str, required=True, help='Test CSV file (6K dataset)')
    parser.add_argument('--test_hdf5', type=str, required=True, help='Test HDF5 file (6K dataset)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    print("🌍 Original EQTransformer vs Indonesia 6K Dataset - FULL ANALYSIS")
    print("=" * 70)
    
    # Setup custom objects for original model
    print("🔧 Setting up custom objects for original model...")
    custom_objects = create_custom_objects()
    print(f"✅ Custom objects created: {len(custom_objects)} objects")
    
    print("✅ Test modules imported")
    
    # Setup GPU
    gpu_available = setup_gpu()
    
    # Load test data
    print(f"\n📊 Loading Indonesia 6K test data...")
    if not os.path.exists(args.test_csv):
        print(f"❌ Test CSV not found: {args.test_csv}")
        return
        
    if not os.path.exists(args.test_hdf5):
        print(f"❌ Test HDF5 not found: {args.test_hdf5}")
        return
    
    # Load traces
    df = pd.read_csv(args.test_csv)
    test_traces = df['trace_name'].tolist()
    print(f"✅ Test Dataset Info:")
    print(f"   CSV file: {os.path.basename(args.test_csv)}")
    print(f"   HDF5 file: {os.path.basename(args.test_hdf5)}")
    print(f"   Total test traces: {len(test_traces)}")
    
    # Show trace categories distribution
    if 'trace_category' in df.columns:
        category_counts = df['trace_category'].value_counts()
        print(f"   Trace categories:")
        for category, count in category_counts.items():
            print(f"     {category}: {count} traces ({count/len(df)*100:.1f}%)")
    
    # Load model with custom objects
    print(f"\n🤖 Loading original EQTransformer model...")
    
    # Override load_model function to use custom objects
    original_load_model = load_model
    def load_model_override(model_path, **kwargs):
        return original_load_model(model_path, custom_objects=custom_objects, **kwargs)
    
    # Monkey patch load_model in globals
    globals()['load_model'] = load_model_override
    
    try:
        start_time = time.time()
        model = load_model_override(args.model)
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
    
    # Create test generator for 6K data
    print(f"\n🔄 Creating test generator...")
    test_generator = create_test_generator(args.test_hdf5, test_traces, args.batch_size, input_dim=6000)
    if not test_generator:
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}")
    
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
    
    # Analyze results using full analysis from test_indonesia_model.py
    results_df = analyze_predictions(predictions, test_traces, args.test_hdf5, args.output_dir)
    if results_df is None:
        return
    
    # Calculate metrics
    metrics = calculate_performance_metrics(results_df, args.output_dir)
    
    # Create visualizations
    create_visualizations(results_df, metrics, args.output_dir)
    
    # Final summary
    print(f"\n🎉 ORIGINAL MODEL TESTING COMPLETED WITH FULL ANALYSIS!")
    print(f"📁 Results saved to: {args.output_dir}")
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