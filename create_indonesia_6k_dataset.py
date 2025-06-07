#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indonesia 6K Dataset Creator 🌋

Script untuk membuat dataset Indonesia dengan 6000 samples melalui windowing
dari dataset original 30085 samples untuk kompatibilitas dengan original EQTransformer model.

Input: indonesia_valid.hdf5 (30085 samples)
Output: indonesia_valid_6k.hdf5 (6000 samples) + indonesia_valid_6k.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
import argparse

def window_trace_data(trace_data, window_size=6000, strategy='centered'):
    """
    Window trace data from 30085 to 6000 samples
    
    Parameters:
    - trace_data: numpy array (30085, 3)
    - window_size: target window size (6000)
    - strategy: 'centered', 'event_centered', or 'start'
    
    Returns:
    - windowed_data: numpy array (6000, 3)
    - window_offset: start sample of the window
    """
    original_length = trace_data.shape[0]
    
    if strategy == 'centered':
        # Centered window - ambil dari tengah
        start_sample = (original_length - window_size) // 2
        
    elif strategy == 'start':
        # Window dari awal
        start_sample = 0
        
    else:  # event_centered - default ke centered jika tidak ada info event
        start_sample = (original_length - window_size) // 2
    
    end_sample = start_sample + window_size
    
    # Pastikan tidak melewati batas
    if end_sample > original_length:
        end_sample = original_length
        start_sample = end_sample - window_size
    
    if start_sample < 0:
        start_sample = 0
        end_sample = window_size
    
    windowed_data = trace_data[start_sample:end_sample, :]
    
    return windowed_data, start_sample

def adjust_ground_truth(p_arrival, s_arrival, window_offset, window_size=6000, sampling_rate=100):
    """
    Adjust ground truth arrival times for windowed data
    
    Parameters:
    - p_arrival, s_arrival: original arrival sample numbers
    - window_offset: start sample of the window
    - window_size: size of the window
    - sampling_rate: sampling rate
    
    Returns:
    - adjusted_p, adjusted_s: new arrival times relative to window
    - valid: boolean indicating if arrivals are within window
    """
    adjusted_p = None
    adjusted_s = None
    valid = True
    
    if p_arrival is not None:
        adjusted_p = p_arrival - window_offset
        if adjusted_p < 0 or adjusted_p >= window_size:
            # P arrival outside window
            valid = False
            adjusted_p = None
    
    if s_arrival is not None:
        adjusted_s = s_arrival - window_offset
        if adjusted_s < 0 or adjusted_s >= window_size:
            # S arrival outside window - masih bisa valid jika P arrival ada
            adjusted_s = None
    
    return adjusted_p, adjusted_s, valid

def create_optimal_window(trace_data, p_arrival, s_arrival, window_size=6000):
    """
    Create optimal window that includes P and S arrivals when possible
    
    Returns:
    - windowed_data, window_offset, adjusted_p, adjusted_s, valid
    """
    original_length = trace_data.shape[0]
    
    # Jika ada P dan S arrival, center window di antara keduanya
    if p_arrival is not None and s_arrival is not None:
        center_point = (p_arrival + s_arrival) // 2
        start_sample = center_point - window_size // 2
        
    elif p_arrival is not None:
        # Center di P arrival
        start_sample = p_arrival - window_size // 2
        
    elif s_arrival is not None:
        # Center di S arrival
        start_sample = s_arrival - window_size // 2
        
    else:
        # Fallback ke centered window
        start_sample = (original_length - window_size) // 2
    
    # Pastikan window dalam bounds
    if start_sample < 0:
        start_sample = 0
    elif start_sample + window_size > original_length:
        start_sample = original_length - window_size
    
    end_sample = start_sample + window_size
    windowed_data = trace_data[start_sample:end_sample, :]
    
    # Adjust ground truth
    adjusted_p, adjusted_s, valid = adjust_ground_truth(
        p_arrival, s_arrival, start_sample, window_size
    )
    
    return windowed_data, start_sample, adjusted_p, adjusted_s, valid

def process_dataset(input_hdf5, input_csv, output_hdf5, output_csv, window_size=6000, strategy='optimal'):
    """
    Process entire dataset to create windowed version
    """
    print(f"🔄 Processing Indonesia dataset:")
    print(f"   Input: {input_hdf5} -> {output_hdf5}")
    print(f"   Window size: {window_size} samples (60 seconds @ 100Hz)")
    print(f"   Strategy: {strategy}")
    
    # Load CSV metadata
    df = pd.read_csv(input_csv)
    print(f"   Total traces: {len(df)}")
    
    # Prepare output data
    output_data = []
    valid_traces = 0
    invalid_traces = 0
    
    # Process each trace
    with h5py.File(input_hdf5, 'r') as input_f:
        with h5py.File(output_hdf5, 'w') as output_f:
            # Create data group
            data_group = output_f.create_group('data')
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing traces"):
                trace_name = row['trace_name']
                
                try:
                    # Load trace data
                    dataset = input_f.get('data/' + trace_name)
                    if dataset is None:
                        print(f"⚠️ Trace not found: {trace_name}")
                        continue
                    
                    trace_data = np.array(dataset)
                    
                    # Get ground truth
                    p_arrival = dataset.attrs.get('p_arrival_sample', None)
                    s_arrival = dataset.attrs.get('s_arrival_sample', None)
                    
                    # Convert to int if not None
                    if p_arrival is not None:
                        p_arrival = int(p_arrival)
                    if s_arrival is not None:
                        s_arrival = int(s_arrival)
                    
                    # Create window based on strategy
                    if strategy == 'optimal':
                        windowed_data, window_offset, adjusted_p, adjusted_s, valid = create_optimal_window(
                            trace_data, p_arrival, s_arrival, window_size
                        )
                    else:
                        windowed_data, window_offset = window_trace_data(
                            trace_data, window_size, strategy
                        )
                        adjusted_p, adjusted_s, valid = adjust_ground_truth(
                            p_arrival, s_arrival, window_offset, window_size
                        )
                    
                    # Only keep traces where at least P arrival is valid
                    if valid and adjusted_p is not None:
                        # Create output dataset
                        output_dataset = data_group.create_dataset(trace_name, data=windowed_data)
                        
                        # Copy and adjust attributes
                        for attr_name, attr_value in dataset.attrs.items():
                            if attr_name == 'p_arrival_sample':
                                if adjusted_p is not None:
                                    output_dataset.attrs[attr_name] = adjusted_p
                            elif attr_name == 's_arrival_sample':
                                if adjusted_s is not None:
                                    output_dataset.attrs[attr_name] = adjusted_s
                            else:
                                output_dataset.attrs[attr_name] = attr_value
                        
                        # Add windowing info
                        output_dataset.attrs['original_length'] = trace_data.shape[0]
                        output_dataset.attrs['window_offset'] = window_offset
                        output_dataset.attrs['window_size'] = window_size
                        
                        # Add to output CSV data
                        output_row = row.copy()
                        output_data.append(output_row)
                        valid_traces += 1
                        
                    else:
                        invalid_traces += 1
                        print(f"⚠️ Skipping {trace_name}: arrivals outside window")
                        
                except Exception as e:
                    print(f"❌ Error processing {trace_name}: {e}")
                    invalid_traces += 1
                    continue
    
    # Create output CSV
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Dataset processing completed:")
    print(f"   Valid traces: {valid_traces}")
    print(f"   Invalid traces: {invalid_traces}")
    print(f"   Success rate: {valid_traces/(valid_traces+invalid_traces)*100:.1f}%")
    print(f"   Output files:")
    print(f"     HDF5: {output_hdf5}")
    print(f"     CSV: {output_csv}")
    
    return valid_traces, invalid_traces

def main():
    parser = argparse.ArgumentParser(description='Create Indonesia 6K Dataset for Original Model')
    parser.add_argument('--input_hdf5', default='datasets/indonesia_valid.hdf5', 
                       help='Input HDF5 file (30085 samples)')
    parser.add_argument('--input_csv', default='datasets/indonesia_valid.csv',
                       help='Input CSV file')
    parser.add_argument('--output_hdf5', default='datasets/indonesia_valid_6k.hdf5',
                       help='Output HDF5 file (6000 samples)')
    parser.add_argument('--output_csv', default='datasets/indonesia_valid_6k.csv',
                       help='Output CSV file')
    parser.add_argument('--window_size', type=int, default=6000,
                       help='Window size (default: 6000 for original model)')
    parser.add_argument('--strategy', choices=['optimal', 'centered', 'start'], 
                       default='optimal',
                       help='Windowing strategy')
    
    args = parser.parse_args()
    
    print("Indonesia 6K Dataset Creator 🌋")
    print("=" * 50)
    
    # Check input files
    if not os.path.exists(args.input_hdf5):
        print(f"❌ Input HDF5 not found: {args.input_hdf5}")
        return
    
    if not os.path.exists(args.input_csv):
        print(f"❌ Input CSV not found: {args.input_csv}")
        return
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_hdf5), exist_ok=True)
    
    # Process dataset
    valid_count, invalid_count = process_dataset(
        args.input_hdf5, args.input_csv,
        args.output_hdf5, args.output_csv,
        args.window_size, args.strategy
    )
    
    if valid_count > 0:
        print(f"\n🎉 Dataset creation successful!")
        print(f"📊 Ready for original model testing:")
        print(f"   Window size: {args.window_size} samples (60s @ 100Hz)")
        print(f"   Compatible with original EQTransformer input shape")
        print(f"   Ground truth preserved and adjusted for windows")
    else:
        print(f"\n❌ No valid traces created!")

if __name__ == "__main__":
    main() 