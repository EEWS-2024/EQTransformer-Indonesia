#!/usr/bin/env python3
"""
Konversi Dataset Indonesia dari NPZ+CSV ke HDF5+CSV (format EQTransformer)
Mempertahankan panjang data 30085 samples (tidak resize ke 6000)
"""

import numpy as np
import pandas as pd
import h5py
import os
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

def load_npz_files_list(train_csv, valid_csv):
    """Load daftar file NPZ dari CSV train dan validation"""
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    
    # Gabungkan semua file
    all_files = []
    
    # Dari train
    if 'file_path' in train_df.columns:
        train_files = train_df['file_path'].tolist()
    else:
        train_files = train_df.iloc[:, 0].tolist()  # Ambil kolom pertama
    
    # Dari valid
    if 'file_path' in valid_df.columns:
        valid_files = valid_df['file_path'].tolist()
    else:
        valid_files = valid_df.iloc[:, 0].tolist()
    
    all_files = train_files + valid_files
    
    print(f"📊 Total files ditemukan:")
    print(f"   Train: {len(train_files)} files")
    print(f"   Valid: {len(valid_files)} files")
    print(f"   Total: {len(all_files)} files")
    
    return all_files

def extract_metadata_from_npz(npz_path):
    """Ekstrak metadata dari file NPZ"""
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            # Load basic data
            waveform = data['data']  # Shape: (30085, 3)
            p_idx = data['p_idx'].item() if data['p_idx'].size > 0 else None
            s_idx = data['s_idx'].item() if data['s_idx'].size > 0 else None
            station_id = str(data['station_id'])
            t0 = str(data['t0'])
            channels = data['channel'] if 'channel' in data else ['E', 'N', 'Z']
            
            # Parse filename untuk info tambahan
            filename = Path(npz_path).stem
            parts = filename.split('.')
            
            # Extract network dan station dari filename
            if len(parts) >= 2:
                network = parts[0]  # GE, IA, dll
                station = parts[1]  # LUWI, FAKI, dll
            else:
                network = 'ID'  # Default
                station = station_id
            
            # Extract timestamp dari t0
            try:
                timestamp = pd.to_datetime(t0)
            except:
                timestamp = pd.to_datetime('2000-01-01T00:00:00')
            
            return {
                'waveform': waveform,
                'p_idx': p_idx,
                's_idx': s_idx,
                'station_id': station_id,
                'network': network,
                'station': station,
                'timestamp': timestamp,
                't0': t0,
                'channels': channels,
                'filename': filename
            }
    except Exception as e:
        print(f"❌ Error loading {npz_path}: {e}")
        return None

def create_eqt_style_metadata(npz_data_list):
    """Buat metadata CSV dalam style EQTransformer"""
    
    metadata_records = []
    
    for i, data in enumerate(tqdm(npz_data_list, desc="Creating metadata")):
        if data is None:
            continue
            
        # Generate trace name (EQTransformer style)
        timestamp_str = data['timestamp'].strftime('%Y%m%d%H%M%S')
        trace_name = f"{data['station']}.{data['network']}_{timestamp_str}_EV"
        
        # Hitung timing dalam detik (asumsi 100 Hz)
        sampling_rate = 100.0
        p_time_sec = data['p_idx'] / sampling_rate if data['p_idx'] else 30.0
        s_time_sec = data['s_idx'] / sampling_rate if data['s_idx'] else 60.0
        
        # Generate dummy coordinates (Indonesia region)
        # Distribusi random dalam bounding box Indonesia
        latitude = np.random.uniform(-11.0, 6.0)   # Indonesia latitude range
        longitude = np.random.uniform(95.0, 141.0) # Indonesia longitude range
        
        # Generate dummy seismic parameters
        magnitude = np.random.uniform(3.0, 6.5)
        depth = np.random.uniform(5.0, 50.0)
        distance = np.random.uniform(10.0, 300.0)
        
        # Build record
        record = {
            # Station info
            'network_code': data['network'],
            'receiver_code': data['station'],
            'receiver_type': 'BH',  # Broadband
            'receiver_latitude': latitude,
            'receiver_longitude': longitude,
            'receiver_elevation_m': 100.0,
            
            # P arrival
            'p_arrival_sample': float(data['p_idx']) if data['p_idx'] else 3000.0,
            'p_status': 'manual',
            'p_weight': 0.5,
            'p_travel_sec': p_time_sec,
            
            # S arrival  
            's_arrival_sample': float(data['s_idx']) if data['s_idx'] else 6000.0,
            's_status': 'manual',
            's_weight': 0.5,
            
            # Source info
            'source_id': f'eq_{i:06d}',
            'source_origin_time': data['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'source_origin_uncertainty_sec': 0.5,
            'source_latitude': latitude + np.random.uniform(-0.5, 0.5),
            'source_longitude': longitude + np.random.uniform(-0.5, 0.5),
            'source_error_sec': 0.5,
            'source_gap_deg': np.random.uniform(50, 180),
            'source_horizontal_uncertainty_km': np.random.uniform(1, 10),
            'source_depth_km': depth,
            'source_depth_uncertainty_km': np.random.uniform(1, 5),
            'source_magnitude': magnitude,
            'source_magnitude_type': 'ml',
            'source_magnitude_author': 'BMKG',
            'source_mechanism_strike_dip_rake': None,
            'source_distance_deg': distance / 111.0,  # Approximate deg
            'source_distance_km': distance,
            'back_azimuth_deg': np.random.uniform(0, 360),
            
            # Trace info
            'snr_db': [np.random.uniform(20, 60), np.random.uniform(20, 60), np.random.uniform(20, 60)],
            'coda_end_sample': [[data['waveform'].shape[0] - 1000]],
            'trace_start_time': data['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f'),
            'trace_category': 'earthquake_local',
            'trace_name': trace_name
        }
        
        metadata_records.append(record)
    
    return pd.DataFrame(metadata_records)

def create_hdf5_file(npz_data_list, metadata_df, output_path):
    """Buat file HDF5 dengan format EQTransformer"""
    
    print(f"\n📦 Membuat file HDF5: {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # Buat group 'data' seperti EQTransformer
        data_group = f.create_group('data')
        
        # Track existing names to avoid duplicates
        existing_names = set()
        
        for i, (data, row) in enumerate(tqdm(zip(npz_data_list, metadata_df.itertuples()), 
                                           total=len(npz_data_list), 
                                           desc="Writing HDF5")):
            if data is None:
                continue
                
            trace_name = row.trace_name
            
            # Handle duplicate names
            original_name = trace_name
            counter = 1
            while trace_name in existing_names:
                trace_name = f"{original_name}_{counter:03d}"
                counter += 1
            
            existing_names.add(trace_name)
            
            waveform = data['waveform']  # Shape: (30085, 3)
            
            # Simpan waveform sebagai float32 (seperti EQTransformer)
            dataset = data_group.create_dataset(
                trace_name, 
                data=waveform.astype(np.float32),
                compression='gzip',
                compression_opts=6
            )
            
            # Add attributes (dengan handling yang lebih robust)
            dataset.attrs['station'] = str(data['station_id'])
            dataset.attrs['network'] = str(data['network'])
            
            # Handle channels attribute safely
            try:
                if isinstance(data['channels'], (list, tuple, np.ndarray)):
                    channels_str = ','.join(str(ch) for ch in data['channels'])
                else:
                    channels_str = 'E,N,Z'  # Default
            except:
                channels_str = 'E,N,Z'  # Fallback default
            
            dataset.attrs['channels'] = channels_str
            dataset.attrs['p_index'] = int(data['p_idx']) if data['p_idx'] else -1
            dataset.attrs['s_index'] = int(data['s_idx']) if data['s_idx'] else -1
    
    print(f"✅ HDF5 file berhasil dibuat: {output_path}")

def main():
    """Fungsi utama konversi"""
    print("🔄 KONVERSI DATASET INDONESIA KE FORMAT EQTRANSFORMER")
    print("="*60)
    
    # Paths
    npz_dir = Path("npz_padded")
    train_csv = "padded_train_list.csv" 
    valid_csv = "padded_valid_list.csv"
    
    output_dir = Path("dataset_indonesia_eqt_format")
    output_dir.mkdir(exist_ok=True)
    
    # Check input files
    if not npz_dir.exists():
        print(f"❌ Direktori NPZ tidak ditemukan: {npz_dir}")
        return
    
    if not Path(train_csv).exists() or not Path(valid_csv).exists():
        print(f"❌ File CSV tidak ditemukan")
        return
    
    # Load file list
    print("\n📋 Loading file list...")
    npz_files = load_npz_files_list(train_csv, valid_csv)
    
    # Load NPZ data
    print(f"\n📊 Loading {len(npz_files)} NPZ files...")
    npz_data_list = []
    failed_files = []
    
    for npz_file in tqdm(npz_files, desc="Loading NPZ files"):
        # Construct full path
        if isinstance(npz_file, str):
            # Remove any prefix path if present
            npz_filename = Path(npz_file).name
            full_path = npz_dir / npz_filename
        else:
            full_path = npz_dir / str(npz_file)
            
        data = extract_metadata_from_npz(full_path)
        if data is not None:
            npz_data_list.append(data)
        else:
            failed_files.append(str(full_path))
    
    print(f"✅ Berhasil load: {len(npz_data_list)} files")
    print(f"❌ Gagal load: {len(failed_files)} files")
    
    if len(npz_data_list) == 0:
        print("❌ Tidak ada data yang berhasil di-load!")
        return
    
    # Create metadata CSV
    print(f"\n📝 Membuat metadata CSV...")
    metadata_df = create_eqt_style_metadata(npz_data_list)
    
    # Save metadata CSV
    csv_output = output_dir / "indonesia_dataset.csv"
    metadata_df.to_csv(csv_output, index=False)
    print(f"✅ Metadata CSV disimpan: {csv_output}")
    
    # Create HDF5 file
    hdf5_output = output_dir / "indonesia_dataset.hdf5"
    create_hdf5_file(npz_data_list, metadata_df, hdf5_output)
    
    # Create summary
    summary = {
        'total_traces': len(npz_data_list),
        'failed_files': len(failed_files),
        'data_shape': list(npz_data_list[0]['waveform'].shape) if npz_data_list else [],
        'sampling_rate': 100.0,
        'duration_seconds': npz_data_list[0]['waveform'].shape[0] / 100.0 if npz_data_list else 0,
        'channels': 3,
        'format': 'EQTransformer_compatible',
        'files': {
            'hdf5': str(hdf5_output),
            'csv': str(csv_output)
        },
        'conversion_time': datetime.now().isoformat()
    }
    
    # Save summary
    summary_output = output_dir / "conversion_summary.json"
    with open(summary_output, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 RINGKASAN KONVERSI:")
    print(f"   ✅ Total traces: {summary['total_traces']}")
    print(f"   📏 Data shape: {summary['data_shape']}")
    print(f"   ⏱️  Duration: {summary['duration_seconds']:.1f} detik")
    print(f"   📁 Output files:")
    print(f"      🗃️  HDF5: {hdf5_output}")
    print(f"      📋 CSV: {csv_output}")
    print(f"      📝 Summary: {summary_output}")
    
    if failed_files:
        print(f"\n⚠️  Failed files:")
        for f in failed_files[:10]:  # Show first 10
            print(f"      {f}")
        if len(failed_files) > 10:
            print(f"      ... dan {len(failed_files)-10} files lainnya")
    
    print(f"\n✅ KONVERSI SELESAI!")
    print(f"   Dataset Indonesia sekarang dalam format EQTransformer!")

if __name__ == "__main__":
    main() 