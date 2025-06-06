#!/usr/bin/env python3
"""
Script untuk menganalisis format data EQTransformer
Menganalisis file HDF5, NPY, dan CSV dalam ModelsAndSampleData/
"""

import h5py
import numpy as np
import pandas as pd
import os
from datetime import datetime

def analyze_hdf5_file(filepath):
    """Analisis file HDF5 EQTransformer"""
    print(f"\n{'='*60}")
    print(f"ANALISIS FILE HDF5: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    with h5py.File(filepath, 'r') as f:
        print(f"📁 Root keys: {list(f.keys())}")
        
        # Analisis grup 'data'
        if 'data' in f.keys():
            data_group = f['data']
            sample_names = list(data_group.keys())
            print(f"📊 Total samples: {len(sample_names)}")
            print(f"🔍 Sample names (first 5): {sample_names[:5]}")
            
            # Analisis sample pertama secara detail
            if sample_names:
                first_sample = sample_names[0]
                sample_data = data_group[first_sample]
                
                print(f"\n--- DETAIL SAMPLE: {first_sample} ---")
                print(f"📏 Shape: {sample_data.shape}")
                print(f"🔢 Dtype: {sample_data.dtype}")
                print(f"📈 Min value: {np.min(sample_data):.6f}")
                print(f"📈 Max value: {np.max(sample_data):.6f}")
                print(f"📊 Mean: {np.mean(sample_data):.6f}")
                print(f"📊 Std: {np.std(sample_data):.6f}")
                
                # Analisis per komponen jika 2D
                if len(sample_data.shape) == 2:
                    n_samples, n_components = sample_data.shape
                    print(f"⚡ Jumlah samples: {n_samples}")
                    print(f"🔧 Jumlah komponen: {n_components} (biasanya E, N, Z)")
                    
                    for i in range(min(3, n_components)):
                        comp_data = sample_data[:, i]
                        print(f"   Komponen {i}: min={np.min(comp_data):.4f}, max={np.max(comp_data):.4f}, mean={np.mean(comp_data):.4f}")
                
                # Cek 5 sample untuk konsistensi
                print(f"\n--- KONSISTENSI SHAPE (5 sample pertama) ---")
                for i, name in enumerate(sample_names[:5]):
                    shape = data_group[name].shape
                    print(f"   {name}: {shape}")

def analyze_npy_file(filepath):
    """Analisis file NPY"""
    print(f"\n{'='*60}")
    print(f"ANALISIS FILE NPY: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    data = np.load(filepath)
    print(f"📏 Shape: {data.shape}")
    print(f"🔢 Dtype: {data.dtype}")
    
    if data.ndim == 1:
        print(f"📊 Total items: {len(data)}")
        print(f"🔍 First 10 items: {data[:10]}")
        print(f"🔍 Last 10 items: {data[-10:]}")
    elif data.ndim == 2:
        print(f"📊 Dimensions: {data.shape[0]} x {data.shape[1]}")
        print(f"📈 Min: {np.min(data):.6f}")
        print(f"📈 Max: {np.max(data):.6f}")
        print(f"📊 Mean: {np.mean(data):.6f}")

def analyze_csv_file(filepath):
    """Analisis file CSV"""
    print(f"\n{'='*60}")
    print(f"ANALISIS FILE CSV: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    df = pd.read_csv(filepath)
    print(f"📏 Shape: {df.shape}")
    print(f"📊 Columns: {list(df.columns)}")
    
    # Informasi kolom penting
    important_cols = ['p_arrival_sample', 's_arrival_sample', 'trace_name', 'source_magnitude', 'trace_category']
    for col in important_cols:
        if col in df.columns:
            print(f"🔍 {col}: min={df[col].min()}, max={df[col].max()}, unique={df[col].nunique()}")
    
    # Sample data
    print(f"\n--- SAMPLE DATA (3 baris pertama) ---")
    print(df.head(3).to_string())

def main():
    """Fungsi utama untuk menganalisis semua file"""
    print("🌏 ANALISIS FORMAT DATA EQTransformer")
    print(f"⏰ Waktu analisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Daftar file yang akan dianalisis
    files_to_analyze = [
        ('100samples.hdf5', analyze_hdf5_file),
        ('100samples.csv', analyze_csv_file),
        ('test.npy', analyze_npy_file)
    ]
    
    for filename, analyzer_func in files_to_analyze:
        filepath = filename
        if os.path.exists(filepath):
            try:
                analyzer_func(filepath)
            except Exception as e:
                print(f"❌ Error analyzing {filename}: {e}")
        else:
            print(f"⚠️  File not found: {filename}")
    
    print(f"\n{'='*60}")
    print("✅ ANALISIS SELESAI")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 