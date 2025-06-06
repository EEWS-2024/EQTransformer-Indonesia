#!/usr/bin/env python3
"""
Analisis struktur data EQTransformer - Version yang diperbaiki
"""

import h5py
import numpy as np
import pandas as pd
import os
from datetime import datetime

def analyze_hdf5_structure():
    """Analisis struktur HDF5 file EQTransformer"""
    print(f"\n{'='*80}")
    print(f"🔬 STRUKTUR DATA EQTRANSFORMER HDF5")
    print(f"{'='*80}")
    
    with h5py.File('100samples.hdf5', 'r') as f:
        print(f"📁 Keys: {list(f.keys())}")
        
        data_group = f['data']
        sample_names = list(data_group.keys())
        
        print(f"\n📊 INFORMASI UMUM:")
        print(f"   Total samples: {len(sample_names)}")
        print(f"   File size: {os.path.getsize('100samples.hdf5') / 1024 / 1024:.2f} MB")
        
        # Analisis first sample
        sample_name = sample_names[0]
        sample_data = np.array(data_group[sample_name])
        
        print(f"\n🔍 CONTOH SAMPLE: {sample_name}")
        print(f"   Shape: {sample_data.shape}")
        print(f"   Dtype: {sample_data.dtype}")
        print(f"   Duration: {sample_data.shape[0]/100:.1f} detik @ 100Hz")
        print(f"   Components: {sample_data.shape[1]} (E, N, Z)")
        print(f"   Value range: {sample_data.min():.1f} to {sample_data.max():.1f}")
        print(f"   Mean: {sample_data.mean():.4f}")
        print(f"   Std: {sample_data.std():.2f}")
        
        # Analisis per channel
        print(f"\n📊 ANALISIS PER CHANNEL:")
        channels = ['E (East)', 'N (North)', 'Z (Vertical)']
        for i in range(3):
            channel_data = sample_data[:, i]
            print(f"   {channels[i]}:")
            print(f"      Range: {channel_data.min():.1f} to {channel_data.max():.1f}")
            print(f"      RMS: {np.sqrt(np.mean(channel_data**2)):.2f}")
        
        # Cek konsistensi shape untuk beberapa sample
        print(f"\n🔄 KONSISTENSI SHAPE (sample pertama 5):")
        for i, name in enumerate(sample_names[:5]):
            shape = data_group[name].shape
            print(f"   {i+1}. {name}: {shape}")

def analyze_csv_metadata():
    """Analisis metadata CSV"""
    print(f"\n{'='*80}")
    print(f"📋 METADATA CSV")
    print(f"{'='*80}")
    
    df = pd.read_csv('100samples.csv')
    
    print(f"📊 DIMENSI:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {len(df.columns)}")
    
    # Info kolom penting
    key_columns = {
        'p_arrival_sample': 'P-wave arrival (samples)',
        's_arrival_sample': 'S-wave arrival (samples)', 
        'source_magnitude': 'Earthquake magnitude',
        'source_depth_km': 'Source depth (km)',
        'source_distance_km': 'Distance (km)',
        'trace_category': 'Event type'
    }
    
    print(f"\n🔑 KOLOM KUNCI:")
    for col, desc in key_columns.items():
        if col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                print(f"   {col}:")
                print(f"      {desc}")
                print(f"      Range: {df[col].min():.2f} - {df[col].max():.2f}")
                print(f"      Mean: {df[col].mean():.2f}")
            else:
                print(f"   {col}: {df[col].unique()}")
    
    # Analisis timing
    if 'p_arrival_sample' in df.columns and 's_arrival_sample' in df.columns:
        p_times = df['p_arrival_sample'] / 100  # Convert to seconds
        s_times = df['s_arrival_sample'] / 100
        ps_diff = s_times - p_times
        
        print(f"\n⏱️  TIMING ANALYSIS:")
        print(f"   P-wave arrival: {p_times.min():.1f} - {p_times.max():.1f} detik")
        print(f"   S-wave arrival: {s_times.min():.1f} - {s_times.max():.1f} detik")
        print(f"   P-S interval: {ps_diff.min():.1f} - {ps_diff.max():.1f} detik")
        print(f"   Average P-S: {ps_diff.mean():.1f} detik")

def analyze_test_npy():
    """Analisis test.npy file"""
    print(f"\n{'='*80}")
    print(f"🧪 TEST SET NPY")
    print(f"{'='*80}")
    
    test_names = np.load('test.npy')
    
    print(f"📊 INFO:")
    print(f"   Total test traces: {len(test_names)}")
    print(f"   Data type: {test_names.dtype}")
    
    # Extract network codes
    networks = [name.split('.')[1] if '.' in name else 'Unknown' for name in test_names]
    unique_networks = set(networks)
    
    print(f"\n🌐 NETWORK CODES (top 10):")
    from collections import Counter
    network_counts = Counter(networks)
    for net, count in network_counts.most_common(10):
        print(f"   {net}: {count} traces ({100*count/len(test_names):.1f}%)")
    
    # Show some examples
    print(f"\n🔍 CONTOH NAMA TRACE:")
    for i in range(min(10, len(test_names))):
        print(f"   {i+1}. {test_names[i]}")

def compare_formats():
    """Perbandingan dengan dataset Indonesia"""
    print(f"\n{'='*80}")
    print(f"🔄 PERBANDINGAN FORMAT DATA")
    print(f"{'='*80}")
    
    print(f"""
📊 EQTRANSFORMER (Global):
   • Format: HDF5 + CSV metadata
   • Window: 60 detik (6000 samples @ 100Hz)
   • Channels: 3 (E, N, Z)
   • Data type: float32 (raw counts)
   • P-arrival: 4-9 detik dari awal
   • S-arrival: 7-22 detik dari awal
   • Region: Amerika Serikat (California)
   • Network: TA (Transportable Array)
   
🇮🇩 DATASET INDONESIA:
   • Format: NPZ + CSV metadata
   • Window: Variable length (rata-rata ~50-120 detik)
   • Channels: 3 (E, N, Z)
   • Data type: float32 (normalized)
   • P-arrival: Variable timing
   • S-arrival: Variable timing
   • Region: Indonesia
   • Network: Multiple (IA, GE, IU, dll)
   
🔑 PERBEDAAN UTAMA:
   • EQT: Fixed 60s window, raw counts
   • Indonesia: Variable window, normalized data
   • EQT: Single network (TA)
   • Indonesia: Multiple networks
   • EQT: California geology
   • Indonesia: Subduction zone seismicity
""")

def main():
    """Main analysis function"""
    print("🌏 ANALISIS STRUKTUR DATA EQTRANSFORMER")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if files exist
    if os.path.exists('100samples.hdf5'):
        analyze_hdf5_structure()
    else:
        print("❌ File 100samples.hdf5 tidak ditemukan")
    
    if os.path.exists('100samples.csv'):
        analyze_csv_metadata()
    else:
        print("❌ File 100samples.csv tidak ditemukan")
    
    if os.path.exists('test.npy'):
        analyze_test_npy()
    else:
        print("❌ File test.npy tidak ditemukan")
    
    compare_formats()
    
    print(f"\n{'='*80}")
    print("✅ ANALISIS STRUKTUR SELESAI")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 