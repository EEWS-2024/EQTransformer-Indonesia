#!/usr/bin/env python3
"""
Analisis mendalam struktur data EQTransformer
Membandingkan dengan format dataset Indonesia
"""

import h5py
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json

def deep_analyze_hdf5(filepath):
    """Analisis mendalam file HDF5 EQTransformer"""
    print(f"\n{'='*80}")
    print(f"🔬 ANALISIS MENDALAM HDF5: {os.path.basename(filepath)}")
    print(f"{'='*80}")
    
    with h5py.File(filepath, 'r') as f:
        print(f"📁 Struktur Root: {list(f.keys())}")
        
        if 'data' in f.keys():
            data_group = f['data']
            sample_names = list(data_group.keys())
            
            print(f"\n🎯 INFO UMUM:")
            print(f"   📊 Total samples: {len(sample_names)}")
            print(f"   📏 File size: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
            
            # Analisis distribusi shape
            shapes = []
            dtypes = []
            min_vals = []
            max_vals = []
            means = []
            stds = []
            
            print(f"\n⏳ Menganalisis semua {len(sample_names)} samples...")
            
            for i, name in enumerate(sample_names[:20]):  # Analisis 20 sample pertama
                sample_data = np.array(data_group[name])  # Convert to numpy array
                shapes.append(sample_data.shape)
                dtypes.append(str(sample_data.dtype))
                min_vals.append(np.min(sample_data))
                max_vals.append(np.max(sample_data))
                means.append(np.mean(sample_data))
                stds.append(np.std(sample_data))
                
                if i % 10 == 0:
                    print(f"   ✅ Processed {i+1}/{min(20, len(sample_names))} samples")
            
            # Statistik distribusi
            print(f"\n📈 DISTRIBUSI DATA (20 samples pertama):")
            print(f"   📏 Shapes: {set(shapes)}")
            print(f"   🔢 Dtypes: {set(dtypes)}")
            print(f"   📊 Min values: min={min(min_vals):.2f}, max={max(min_vals):.2f}")
            print(f"   📊 Max values: min={min(max_vals):.2f}, max={max(max_vals):.2f}")
            print(f"   📊 Means: min={min(means):.2f}, max={max(means):.2f}")
            print(f"   📊 Stds: min={min(stds):.2f}, max={max(stds):.2f}")
            
            # Analisis satu sample secara detail
            first_sample = sample_names[0]
            sample_data = np.array(data_group[first_sample])  # Convert to numpy array
            
            print(f"\n🔍 ANALISIS DETAIL SAMPLE: {first_sample}")
            print(f"   📏 Shape: {sample_data.shape}")
            print(f"   ⏱️  Duration: {sample_data.shape[0]/100:.1f} detik (asumsi 100 Hz)")
            print(f"   🔧 Komponen: {sample_data.shape[1]} (E, N, Z)")
            
            # Analisis per komponen
            for i in range(sample_data.shape[1]):
                comp = sample_data[:, i]
                print(f"   📊 Komponen {i}: shape={comp.shape}, min={np.min(comp):.2f}, max={np.max(comp):.2f}")
                print(f"      📈 Mean={np.mean(comp):.4f}, Std={np.std(comp):.2f}")
                print(f"      🔢 Non-zero: {np.count_nonzero(comp)}/{len(comp)} ({100*np.count_nonzero(comp)/len(comp):.1f}%)")
            
            # Cek apakah data sudah dinormalisasi
            print(f"\n🎚️  STATUS NORMALISASI:")
            all_data = sample_data.flatten()
            print(f"   📊 Global min: {np.min(all_data):.6f}")
            print(f"   📊 Global max: {np.max(all_data):.6f}")
            print(f"   📊 Global mean: {np.mean(all_data):.6f}")
            print(f"   📊 Global std: {np.std(all_data):.6f}")
            
            if np.abs(np.mean(all_data)) < 1000 and np.std(all_data) < 10000:
                print("   ✅ Data kemungkinan sudah dinormalisasi/difilter")
            else:
                print("   ⚠️  Data masih dalam bentuk raw counts")

def analyze_csv_metadata(filepath):
    """Analisis mendalam metadata CSV"""
    print(f"\n{'='*80}")
    print(f"📋 ANALISIS MENDALAM CSV: {os.path.basename(filepath)}")
    print(f"{'='*80}")
    
    df = pd.read_csv(filepath)
    
    print(f"📊 DIMENSI DATA:")
    print(f"   📏 Shape: {df.shape}")
    print(f"   📁 Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    print(f"\n📝 KOLOM DAN TIPE DATA:")
    for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes)):
        print(f"   {i+1:2d}. {col:<35} | {str(dtype):<10}")
    
    # Analisis kolom kunci untuk seismology
    seismic_cols = {
        'p_arrival_sample': 'Indeks kedatangan gelombang P',
        's_arrival_sample': 'Indeks kedatangan gelombang S', 
        'source_magnitude': 'Magnitudo gempa',
        'source_depth_km': 'Kedalaman sumber gempa',
        'source_distance_km': 'Jarak episentral',
        'snr_db': 'Signal-to-noise ratio'
    }
    
    print(f"\n🌊 ANALISIS PARAMETER SEISMIK:")
    for col, desc in seismic_cols.items():
        if col in df.columns:
            values = df[col].dropna()
            print(f"   📊 {col}:")
            print(f"      📝 {desc}")
            print(f"      📈 Range: {values.min():.2f} - {values.max():.2f}")
            print(f"      📊 Mean: {values.mean():.2f}, Std: {values.std():.2f}")
            print(f"      📋 Valid values: {len(values)}/{len(df)} ({100*len(values)/len(df):.1f}%)")
    
    # Analisis interval P-S
    if 'p_arrival_sample' in df.columns and 's_arrival_sample' in df.columns:
        ps_interval = df['s_arrival_sample'] - df['p_arrival_sample']
        ps_interval_sec = ps_interval / 100  # Asumsi 100 Hz
        
        print(f"\n⏱️  ANALISIS INTERVAL P-S:")
        print(f"   📊 Interval samples: {ps_interval.min():.0f} - {ps_interval.max():.0f}")
        print(f"   ⏰ Interval seconds: {ps_interval_sec.min():.1f} - {ps_interval_sec.max():.1f}")
        print(f"   📈 Mean interval: {ps_interval_sec.mean():.1f}s")
    
    # Analisis distribusi magnitude
    if 'source_magnitude' in df.columns:
        mag = df['source_magnitude'].dropna()
        print(f"\n🏔️  DISTRIBUSI MAGNITUDO:")
        print(f"   📊 Range: M{mag.min():.1f} - M{mag.max():.1f}")
        print(f"   📈 Mean: M{mag.mean():.1f}")
        
        # Binning magnitude
        mag_bins = [0, 2, 3, 4, 5, 10]
        mag_counts = pd.cut(mag, mag_bins).value_counts().sort_index()
        for interval, count in mag_counts.items():
            if count > 0:
                print(f"   📊 {interval}: {count} events ({100*count/len(mag):.1f}%)")

def analyze_npy_testset(filepath):
    """Analisis file test set NPY"""
    print(f"\n{'='*80}")
    print(f"🧪 ANALISIS TEST SET: {os.path.basename(filepath)}")
    print(f"{'='*80}")
    
    data = np.load(filepath)
    
    print(f"📊 INFO UMUM:")
    print(f"   📏 Shape: {data.shape}")
    print(f"   🔢 Dtype: {data.dtype}")
    print(f"   📁 Size: {data.nbytes / 1024:.2f} KB")
    
    # Analisis pola penamaan
    print(f"\n🏷️  ANALISIS POLA PENAMAAN:")
    
    # Extract network codes
    networks = [name.split('.')[1] if '.' in name else 'Unknown' for name in data]
    network_counts = pd.Series(networks).value_counts()
    
    print(f"   🌐 Network codes (top 10):")
    for net, count in network_counts.head(10).items():
        print(f"      {net}: {count} traces ({100*count/len(data):.1f}%)")
    
    # Extract years
    years = []
    for name in data:
        try:
            if '_' in name:
                year_part = name.split('_')[1][:4]
                if year_part.isdigit():
                    years.append(int(year_part))
        except:
            pass
    
    if years:
        year_counts = pd.Series(years).value_counts().sort_index()
        print(f"\n📅 DISTRIBUSI TAHUN:")
        for year, count in year_counts.items():
            print(f"   {year}: {count} traces ({100*count/len(data):.1f}%)")
    
    # Extract event types
    event_types = [name.split('_')[-1] if '_' in name else 'Unknown' for name in data]
    type_counts = pd.Series(event_types).value_counts()
    
    print(f"\n🏷️  TIPE EVENT:")
    for etype, count in type_counts.items():
        print(f"   {etype}: {count} traces ({100*count/len(data):.1f}%)")

def compare_with_indonesia_dataset():
    """Perbandingan dengan dataset Indonesia"""
    print(f"\n{'='*80}")
    print(f"🔄 PERBANDINGAN DENGAN DATASET INDONESIA")
    print(f"{'='*80}")
    
    # Load Indonesia dataset info
    indonesia_csv = "../dataset_indonesia/npz_padded_summary.json"
    if os.path.exists(indonesia_csv):
        with open(indonesia_csv, 'r') as f:
            indonesia_data = json.load(f)
        
        print(f"📊 PERBANDINGAN UKURAN DATA:")
        print(f"   🌏 EQTransformer (global): 100 samples")
        print(f"   🇮🇩 Indonesia dataset: {indonesia_data['total_files']} samples")
        
        print(f"\n📏 PERBANDINGAN PANJANG WAVEFORM:")
        print(f"   🌏 EQTransformer: 6000 samples (60 detik @ 100Hz)")
        print(f"   🇮🇩 Indonesia: {indonesia_data['data_length_stats']['mean']:.0f} samples (~{indonesia_data['data_length_stats']['mean']/100:.0f} detik @ 100Hz)")
        
        print(f"\n📍 PERBANDINGAN DISTRIBUSI STASIUN:")
        print(f"   🌏 EQTransformer: 1 stasiun (109C)")
        print(f"   🇮🇩 Indonesia: {len(indonesia_data['station_distribution'])} stasiun")
        
        print(f"\n⏱️  PERBANDINGAN INTERVAL P-S:")
        print(f"   🌏 EQTransformer: ~12-22 detik (berdasarkan CSV)")
        print(f"   🇮🇩 Indonesia: {indonesia_data.get('p_index_stats', {}).get('mean', 0)/100:.1f} - {indonesia_data.get('s_index_stats', {}).get('mean', 0)/100:.1f} detik")
    else:
        print("⚠️  File dataset Indonesia tidak ditemukan untuk perbandingan")

def generate_summary_report():
    """Generate laporan ringkasan"""
    print(f"\n{'='*80}")
    print(f"📝 RINGKASAN TEMUAN")
    print(f"{'='*80}")
    
    print(f"""
🎯 FORMAT DATA EQTRANSFORMER:

📊 STRUKTUR FILE:
   • HDF5: Berisi waveform data dalam grup 'data'
   • CSV: Metadata lengkap dengan 35 kolom
   • NPY: Daftar nama file untuk test set

📏 DIMENSI WAVEFORM:
   • Shape: (6000, 3) = 60 detik @ 100 Hz
   • Komponen: E, N, Z (3 komponen seismik)
   • Data type: float32
   • Range: ~±50,000 (raw counts, belum normalized)

🏷️  LABELING SYSTEM:
   • P arrival: 400-900 samples (4-9 detik dari awal)
   • S arrival: 723-2198 samples (7-22 detik dari awal)
   • Manual picking dengan weight 0.5-0.75

🌍 COVERAGE:
   • Geographic: Amerika Serikat (California)
   • Network: TA (Transportable Array)
   • Magnitude: M0.87 - M4.3
   • Event type: earthquake_local

⚡ KARAKTERISTIK KUNCI:
   • Window length: 60 detik (lebih pendek dari Indonesia)
   • Pre-event time: ~4-9 detik
   • Post-S time: ~40-50 detik
   • Sampling rate: 100 Hz konsisten
   • Data format: Raw seismic counts (belum normalized)
""")

def main():
    """Fungsi utama analisis mendalam"""
    print("🔬 ANALISIS MENDALAM DATA EQTRANSFORMER")
    print(f"⏰ Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Analisis file-file EQTransformer
    if os.path.exists('100samples.hdf5'):
        deep_analyze_hdf5('100samples.hdf5')
    
    if os.path.exists('100samples.csv'):
        analyze_csv_metadata('100samples.csv')
    
    if os.path.exists('test.npy'):
        analyze_npy_testset('test.npy')
    
    # Perbandingan dengan dataset Indonesia
    compare_with_indonesia_dataset()
    
    # Generate summary
    generate_summary_report()
    
    print(f"\n{'='*80}")
    print("✅ ANALISIS MENDALAM SELESAI")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 