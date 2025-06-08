#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indonesia Fine-tuning Dataset Creator 🌋

Convert Indonesia training and validation datasets from 30085 samples to 6000 samples
for fine-tuning with original EQTransformer model.

Input: 
- datasets/indonesia_train_dataset.hdf5 (30085 samples)
- datasets/indonesia_valid_dataset.hdf5 (30085 samples)

Output:
- datasets/indonesia_finetune_train.hdf5 (6000 samples)
- datasets/indonesia_finetune_valid.hdf5 (6000 samples)
- + corresponding CSV files
"""

import os
import sys
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
import argparse
from pathlib import Path

def create_optimal_window(trace_data, p_arrival, s_arrival, window_size=6000):
    """
    Create optimal window that includes P and S arrivals when possible
    
    Strategy:
    1. If both P and S exist, center window between them
    2. If only P exists, center on P with margin for S
    3. If only S exists, place P expected position in window
    4. Fallback to centered window
    
    Returns:
    - windowed_data: (6000, 3) array
    - window_offset: start sample of the window
    - adjusted_p, adjusted_s: new arrival times relative to window
    - valid: boolean indicating if arrivals are within window
    """
    original_length = trace_data.shape[0]
    
    # Strategy untuk optimal windowing
    if p_arrival is not None and s_arrival is not None:
        # Jika ada P dan S, center window antara P dan S
        center_point = (p_arrival + s_arrival) // 2
        start_sample = center_point - window_size // 2
        
        # Pastikan P arrival tidak terlalu dekat dengan awal window
        min_p_offset = 1000  # 10 detik buffer untuk P
        if (p_arrival - start_sample) < min_p_offset:
            start_sample = p_arrival - min_p_offset
            
    elif p_arrival is not None:
        # Hanya ada P arrival, center dengan margin untuk S
        # Asumsi S datang 20-50 detik setelah P
        expected_s_delay = 3000  # 30 detik delay rata-rata
        center_for_s = p_arrival + expected_s_delay // 2
        start_sample = center_for_s - window_size // 2
        
        # Pastikan P tidak terlalu dekat awal
        min_p_offset = 1000  # 10 detik buffer
        if (p_arrival - start_sample) < min_p_offset:
            start_sample = p_arrival - min_p_offset
            
    elif s_arrival is not None:
        # Hanya ada S arrival, posisikan agar expected P dalam window
        # Asumsi P datang 20-50 detik sebelum S
        expected_p_delay = 3000  # 30 detik sebelum S
        expected_p = s_arrival - expected_p_delay
        start_sample = expected_p - 1000  # 10 detik buffer sebelum expected P
        
    else:
        # Fallback ke centered window
        start_sample = (original_length - window_size) // 2
    
    # Pastikan window dalam bounds
    if start_sample < 0:
        start_sample = 0
    elif start_sample + window_size > original_length:
        start_sample = original_length - window_size
    
    # Extract window
    end_sample = start_sample + window_size
    windowed_data = trace_data[start_sample:end_sample, :]
    
    # Adjust ground truth relative to window
    adjusted_p = None
    adjusted_s = None
    valid = True
    
    if p_arrival is not None:
        adjusted_p = p_arrival - start_sample
        if adjusted_p < 0 or adjusted_p >= window_size:
            # P arrival outside window - mark as invalid
            valid = False
            adjusted_p = None
    
    if s_arrival is not None:
        adjusted_s = s_arrival - start_sample
        if adjusted_s < 0 or adjusted_s >= window_size:
            # S arrival outside window - still valid if P is valid
            adjusted_s = None
    
    return windowed_data, start_sample, adjusted_p, adjusted_s, valid

def process_dataset(input_hdf5, input_csv, output_hdf5, output_csv, dataset_name="dataset"):
    """
    Process entire dataset to create windowed version for fine-tuning
    """
    print(f"🔄 Processing {dataset_name}:")
    print(f"   Input: {input_hdf5} -> {output_hdf5}")
    print(f"   Window size: 6000 samples (60 seconds @ 100Hz)")
    print(f"   Strategy: Optimal P/S centered windowing")
    
    # Load CSV metadata
    df = pd.read_csv(input_csv)
    print(f"   Total traces: {len(df)}")
    
    # Prepare output data
    output_metadata = []
    valid_traces = 0
    invalid_traces = 0
    windowing_stats = {
        'both_arrivals': 0,
        'p_only': 0,
        's_only': 0,
        'no_arrivals': 0,
        'p_preserved': 0,
        's_preserved': 0
    }
    
    # Process each trace
    with h5py.File(input_hdf5, 'r') as input_f:
        with h5py.File(output_hdf5, 'w') as output_f:
            # Create data group
            data_group = output_f.create_group('data')
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_name}"):
                trace_name = row['trace_name']
                
                try:
                    # Load trace data
                    dataset = input_f.get('data/' + trace_name)
                    if dataset is None:
                        print(f"⚠️ Trace not found: {trace_name}")
                        invalid_traces += 1
                        continue
                    
                    trace_data = np.array(dataset)
                    
                    # Get ground truth arrivals
                    p_arrival = dataset.attrs.get('p_arrival_sample', None)
                    s_arrival = dataset.attrs.get('s_arrival_sample', None)
                    
                    # Convert to int if not None
                    if p_arrival is not None:
                        p_arrival = int(p_arrival)
                    if s_arrival is not None:
                        s_arrival = int(s_arrival)
                    
                    # Track arrival statistics
                    if p_arrival is not None and s_arrival is not None:
                        windowing_stats['both_arrivals'] += 1
                    elif p_arrival is not None:
                        windowing_stats['p_only'] += 1
                    elif s_arrival is not None:
                        windowing_stats['s_only'] += 1
                    else:
                        windowing_stats['no_arrivals'] += 1
                    
                    # Create optimal window
                    windowed_data, window_offset, adjusted_p, adjusted_s, valid = create_optimal_window(
                        trace_data, p_arrival, s_arrival, window_size=6000
                    )
                    
                    # Only keep traces where at least P arrival is valid or no arrivals (noise)
                    if valid or (p_arrival is None and s_arrival is None):
                        # Create output dataset
                        output_dataset = data_group.create_dataset(
                            trace_name, 
                            data=windowed_data,
                            dtype=np.float32,
                            compression='gzip'
                        )
                        
                        # Copy original attributes
                        for attr_name, attr_value in dataset.attrs.items():
                            if attr_name == 'p_arrival_sample':
                                if adjusted_p is not None:
                                    output_dataset.attrs[attr_name] = adjusted_p
                                    windowing_stats['p_preserved'] += 1
                            elif attr_name == 's_arrival_sample':
                                if adjusted_s is not None:
                                    output_dataset.attrs[attr_name] = adjusted_s
                                    windowing_stats['s_preserved'] += 1
                            else:
                                output_dataset.attrs[attr_name] = attr_value
                        
                        # Add windowing metadata
                        output_dataset.attrs['original_length'] = trace_data.shape[0]
                        output_dataset.attrs['window_offset'] = window_offset
                        output_dataset.attrs['window_size'] = 6000
                        output_dataset.attrs['windowing_strategy'] = 'optimal_ps_centered'
                        
                        # Update CSV metadata for window
                        output_row = row.copy()
                        if adjusted_p is not None:
                            output_row['p_arrival_sample'] = adjusted_p
                        if adjusted_s is not None:
                            output_row['s_arrival_sample'] = adjusted_s
                        
                        output_metadata.append(output_row)
                        valid_traces += 1
                        
                    else:
                        invalid_traces += 1
                        
                except Exception as e:
                    print(f"❌ Error processing {trace_name}: {e}")
                    invalid_traces += 1
                    continue
    
    # Create output CSV
    if output_metadata:
        output_df = pd.DataFrame(output_metadata)
        output_df.to_csv(output_csv, index=False)
    
    # Print statistics
    print(f"\n✅ {dataset_name.title()} processing completed:")
    print(f"   Valid traces: {valid_traces}")
    print(f"   Invalid traces: {invalid_traces}")
    print(f"   Success rate: {valid_traces/(valid_traces+invalid_traces)*100:.1f}%")
    
    print(f"\n📊 Windowing statistics:")
    print(f"   Both P&S arrivals: {windowing_stats['both_arrivals']}")
    print(f"   P only: {windowing_stats['p_only']}")
    print(f"   S only: {windowing_stats['s_only']}")
    print(f"   No arrivals: {windowing_stats['no_arrivals']}")
    print(f"   P preserved: {windowing_stats['p_preserved']}")
    print(f"   S preserved: {windowing_stats['s_preserved']}")
    
    print(f"\n📁 Output files:")
    print(f"     HDF5: {output_hdf5}")
    print(f"     CSV: {output_csv}")
    
    return valid_traces, invalid_traces, windowing_stats

def main():
    parser = argparse.ArgumentParser(description='Create Indonesia Fine-tuning Dataset')
    parser.add_argument('--input_dir', default='datasets', 
                       help='Input directory containing 30K datasets')
    parser.add_argument('--output_dir', default='datasets',
                       help='Output directory for fine-tuning datasets')
    parser.add_argument('--process_train', action='store_true',
                       help='Process training dataset')
    parser.add_argument('--process_valid', action='store_true',
                       help='Process validation dataset')
    parser.add_argument('--process_all', action='store_true',
                       help='Process both training and validation datasets')
    
    args = parser.parse_args()
    
    print("Indonesia Fine-tuning Dataset Creator 🌋")
    print("=" * 55)
    print("Purpose: Create fine-tuning datasets for original EQTransformer")
    
    # Determine what to process
    process_train = args.process_train or args.process_all
    process_valid = args.process_valid or args.process_all
    
    if not (process_train or process_valid):
        print("❌ Please specify --process_train, --process_valid, or --process_all")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_valid = 0
    total_invalid = 0
    
    # Process training dataset
    if process_train:
        print(f"\n{'='*55}")
        print(f"PROCESSING TRAINING DATASET")
        print(f"{'='*55}")
        
        train_input_hdf5 = f"{args.input_dir}/indonesia_train.hdf5"
        train_input_csv = f"{args.input_dir}/indonesia_train.csv"
        train_output_hdf5 = f"{args.output_dir}/indonesia_finetune_train.hdf5"
        train_output_csv = f"{args.output_dir}/indonesia_finetune_train.csv"
        
        # Check input files
        if not os.path.exists(train_input_hdf5):
            print(f"❌ Training HDF5 not found: {train_input_hdf5}")
            return
        if not os.path.exists(train_input_csv):
            print(f"❌ Training CSV not found: {train_input_csv}")
            return
        
        # Process training data
        valid_count, invalid_count, train_stats = process_dataset(
            train_input_hdf5, train_input_csv,
            train_output_hdf5, train_output_csv,
            "training dataset"
        )
        
        total_valid += valid_count
        total_invalid += invalid_count
    
    # Process validation dataset
    if process_valid:
        print(f"\n{'='*55}")
        print(f"PROCESSING VALIDATION DATASET")
        print(f"{'='*55}")
        
        valid_input_hdf5 = f"{args.input_dir}/indonesia_valid.hdf5"
        valid_input_csv = f"{args.input_dir}/indonesia_valid.csv"
        valid_output_hdf5 = f"{args.output_dir}/indonesia_finetune_valid.hdf5"
        valid_output_csv = f"{args.output_dir}/indonesia_finetune_valid.csv"
        
        # Check input files
        if not os.path.exists(valid_input_hdf5):
            print(f"❌ Validation HDF5 not found: {valid_input_hdf5}")
            return
        if not os.path.exists(valid_input_csv):
            print(f"❌ Validation CSV not found: {valid_input_csv}")
            return
        
        # Process validation data
        valid_count, invalid_count, valid_stats = process_dataset(
            valid_input_hdf5, valid_input_csv,
            valid_output_hdf5, valid_output_csv,
            "validation dataset"
        )
        
        total_valid += valid_count
        total_invalid += invalid_count
    
    # Final summary
    if total_valid > 0:
        print(f"\n🎉 DATASET CREATION SUCCESSFUL!")
        print(f"{'='*55}")
        print(f"📊 Overall Statistics:")
        print(f"   Total valid traces: {total_valid}")
        print(f"   Total invalid traces: {total_invalid}")
        print(f"   Overall success rate: {total_valid/(total_valid+total_invalid)*100:.1f}%")
        
        print(f"\n🚀 Ready for fine-tuning:")
        print(f"   Input shape: (6000, 3) - compatible with original EQTransformer")
        print(f"   Windowing: Optimal P/S centered strategy")
        print(f"   Ground truth: Preserved and adjusted for windows")
        
        print(f"\n💡 Next steps:")
        print(f"   1. Run fine-tuning script:")
        print(f"      python finetune_original_indonesia.py --model models/original_model/EqT_original_model.h5")
        print(f"   2. Compare fine-tuned vs original model performance")
        print(f"   3. Evaluate on test dataset")
        
    else:
        print(f"\n❌ No valid traces created!")
        print(f"   Check input datasets and windowing parameters")

if __name__ == "__main__":
    main() 