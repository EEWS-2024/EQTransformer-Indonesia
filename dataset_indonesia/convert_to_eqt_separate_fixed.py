import numpy as np
import pandas as pd
import h5py
import os
from datetime import datetime
import json

def load_npz_file(filepath):
    """Load data from NPZ file"""
    try:
        data = np.load(filepath)
        
        # Fields berdasarkan struktur NPZ Indonesia yang sebenarnya
        waveform = data['data']  # field 'data' yang berisi waveform (30085, 3)
        p_idx = data['p_idx']
        s_idx = data['s_idx']
        station_id = str(data['station_id'])
        t0 = str(data['t0'])
        channel = str(data['channel'])
        channel_type = str(data['channel_type']) if 'channel_type' in data else channel[:2]
        
        # Extract network dan station dari station_id atau filename
        # Format biasanya: GE.STATION atau NETWORK.STATION
        if '.' in station_id:
            parts = station_id.split('.')
            network = parts[0] 
            station = parts[1] if len(parts) > 1 else station_id
        else:
            # Fallback: extract dari filename
            filename = os.path.basename(filepath).replace('.npz', '')
            if filename.startswith(('GE.', 'IA.', 'II.')):
                network = filename.split('.')[0]
                station = filename.split('.')[1]
            else:
                network = 'GE'  # default
                station = station_id
        
        return {
            'waveform': waveform,
            'p_idx': p_idx,
            's_idx': s_idx,
            'station': station,
            'network': network,
            'location': '',  # tidak ada di data
            'channel': channel,
            'starttime': t0,
            'station_id': station_id
        }
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return None

def create_eqt_format_separate(npz_dir, train_list_path, valid_list_path):
    """Convert Indonesian dataset to EQTransformer format with separate train/validation files"""
    
    # Read train and validation lists
    train_df = pd.read_csv(train_list_path)
    valid_df = pd.read_csv(valid_list_path)
    
    train_files = train_df['fname'].tolist()
    valid_files = valid_df['fname'].tolist()
    
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(valid_files)}")
    
    # Process training data
    print("\nProcessing training data...")
    train_metadata, train_count = process_dataset(npz_dir, train_files, "train")
    
    # Process validation data  
    print("\nProcessing validation data...")
    valid_metadata, valid_count = process_dataset(npz_dir, valid_files, "validation")
    
    # Create training HDF5 file
    if train_count > 0:
        print(f"\nCreating training HDF5 file with {train_count} samples...")
        create_hdf5_file(npz_dir, train_files, "indonesia_train_dataset.hdf5")
        
        # Save training CSV metadata
        train_csv_df = pd.DataFrame(train_metadata)
        train_csv_df.to_csv('indonesia_train_dataset.csv', index=False)
        print("Training CSV metadata saved to: indonesia_train_dataset.csv")
    else:
        print("No valid training data found!")
    
    # Create validation HDF5 file
    if valid_count > 0:
        print(f"\nCreating validation HDF5 file with {valid_count} samples...")
        create_hdf5_file(npz_dir, valid_files, "indonesia_valid_dataset.hdf5")
        
        # Save validation CSV metadata
        valid_csv_df = pd.DataFrame(valid_metadata)
        valid_csv_df.to_csv('indonesia_valid_dataset.csv', index=False)
        print("Validation CSV metadata saved to: indonesia_valid_dataset.csv")
    else:
        print("No valid validation data found!")
    
    # Save conversion summary
    summary = {
        'total_train_files': len(train_files),
        'successful_train_conversions': train_count,
        'total_valid_files': len(valid_files), 
        'successful_valid_conversions': valid_count,
        'train_hdf5_file': 'indonesia_train_dataset.hdf5' if train_count > 0 else None,
        'train_csv_file': 'indonesia_train_dataset.csv' if train_count > 0 else None,
        'valid_hdf5_file': 'indonesia_valid_dataset.hdf5' if valid_count > 0 else None,
        'valid_csv_file': 'indonesia_valid_dataset.csv' if valid_count > 0 else None,
        'conversion_time': datetime.now().isoformat()
    }
    
    with open('conversion_summary_separate.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nConversion Summary:")
    print(f"Training: {train_count}/{len(train_files)} files successfully converted")
    print(f"Validation: {valid_count}/{len(valid_files)} files successfully converted")
    print("Summary saved to: conversion_summary_separate.json")
    
    return summary

def process_dataset(npz_dir, file_list, dataset_type):
    """Process a list of files and extract metadata"""
    metadata_list = []
    successful_count = 0
    
    for i, filename in enumerate(file_list):
        if i % 100 == 0:
            print(f"Processing {dataset_type} file {i+1}/{len(file_list)}: {filename}")
        
        filepath = os.path.join(npz_dir, filename)
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        npz_data = load_npz_file(filepath)
        if npz_data is None:
            continue
        
        # Extract waveform info
        waveform = npz_data['waveform']
        if waveform.shape != (30085, 3):
            print(f"Unexpected waveform shape {waveform.shape} for {filename}")
            continue
        
        # Create trace name (unique identifier)
        trace_name = f"{npz_data['network']}.{npz_data['station']}.{npz_data['location']}.{npz_data['channel']}.{filename.replace('.npz', '')}"
        
        # Handle p_idx and s_idx yang mungkin berbentuk array
        p_idx = npz_data['p_idx']
        s_idx = npz_data['s_idx']
        
        # Convert to scalar jika berbentuk array
        if hasattr(p_idx, 'shape') and p_idx.shape != ():
            p_idx = p_idx.item() if p_idx.size == 1 else p_idx[0]
        if hasattr(s_idx, 'shape') and s_idx.shape != ():
            s_idx = s_idx.item() if s_idx.size == 1 else s_idx[0]
        
        # Create metadata entry following EQTransformer format
        metadata_entry = {
            'trace_name': trace_name,
            'station_code': npz_data['station'],
            'instrument_type': npz_data['channel'][:2],  # BH, HL, etc.
            'station_channels': npz_data['channel'],
            'network_code': npz_data['network'],
            'station_latitude': get_station_latitude(npz_data['station']),
            'station_longitude': get_station_longitude(npz_data['station']),
            'station_elevation_km': 0.1,  # Default elevation
            'trace_start_time': npz_data['starttime'],
            'source_id': filename.replace('.npz', ''),
            'source_origin_time': npz_data['starttime'],
            'source_origin_uncertainty_sec': 0.1,
            'source_latitude': get_event_latitude(npz_data['station']),
            'source_longitude': get_event_longitude(npz_data['station']),
            'source_error_sec': 0.1,
            'source_gap_deg': 45.0,
            'source_horizontal_uncertainty_km': 1.0,
            'source_depth_km': 10.0,
            'source_depth_uncertainty_km': 2.0,
            'source_magnitude': 4.5,
            'source_magnitude_type': 'ML',
            'source_magnitude_author': 'Indonesia_Network',
            'source_mechanism_strike_dip_rake': '0_90_0',
            'source_distance_deg': calculate_distance_deg(
                get_station_latitude(npz_data['station']),
                get_station_longitude(npz_data['station']),
                get_event_latitude(npz_data['station']),
                get_event_longitude(npz_data['station'])
            ),
            'source_distance_km': calculate_distance_deg(
                get_station_latitude(npz_data['station']),
                get_station_longitude(npz_data['station']),
                get_event_latitude(npz_data['station']),
                get_event_longitude(npz_data['station'])
            ) * 111.0,  # Convert degrees to km
            'back_azimuth_deg': 180.0,
            'snr_db': 15.0,
            'coda_end_sample': 30085,
            'trace_category': 'earthquake_local',
            'trace_polarity': 'positive',
            'p_arrival_sample': int(p_idx),
            'p_status': 'manual',
            'p_weight': 1.0,
            'p_travel_sec': int(p_idx) / 100.0,  # Convert sample to seconds (100 Hz)
            's_arrival_sample': int(s_idx),
            's_status': 'manual', 
            's_weight': 1.0,
            's_travel_sec': int(s_idx) / 100.0,  # Convert sample to seconds
            'receiver_code': npz_data['station'],
            'receiver_type': 'HH',
            'receiver_latitude': get_station_latitude(npz_data['station']),
            'receiver_longitude': get_station_longitude(npz_data['station']),
            'receiver_elevation_km': 0.1
        }
        
        metadata_list.append(metadata_entry)
        successful_count += 1
    
    return metadata_list, successful_count

def create_hdf5_file(npz_dir, file_list, output_filename):
    """Create HDF5 file for the given file list"""
    successful_traces = []
    failed_files = []
    
    # First pass: collect all valid waveforms
    for filename in file_list:
        filepath = os.path.join(npz_dir, filename)
        if not os.path.exists(filepath):
            failed_files.append(f"File not found: {filename}")
            continue
            
        npz_data = load_npz_file(filepath)
        if npz_data is None:
            failed_files.append(f"Could not load: {filename}")
            continue
        
        waveform = npz_data['waveform']
        if waveform.shape != (30085, 3):
            failed_files.append(f"Wrong shape {waveform.shape}: {filename}")
            continue
        
        # Create trace name
        trace_name = f"{npz_data['network']}.{npz_data['station']}.{npz_data['location']}.{npz_data['channel']}.{filename.replace('.npz', '')}"
        
        successful_traces.append({
            'trace_name': trace_name,
            'waveform': waveform,
            'filename': filename
        })
    
    print(f"Successfully loaded {len(successful_traces)} traces")
    if failed_files:
        print(f"Failed to load {len(failed_files)} files")
    
    if len(successful_traces) == 0:
        print("No valid traces found! HDF5 file not created.")
        return
    
    # Create HDF5 file
    with h5py.File(output_filename, 'w') as hdf5_file:
        # Create 'data' group to match EQTransformer format
        data_group = hdf5_file.create_group('data')
        
        for trace_info in successful_traces:
            # Create dataset for each trace with the trace name as key
            trace_name = trace_info['trace_name']
            waveform = trace_info['waveform']
            
            # Store waveform data (30085, 3) - keep original length
            dataset = data_group.create_dataset(
                trace_name, 
                data=waveform,
                dtype=np.float32,
                compression='gzip'
            )
            
            # Add attributes
            dataset.attrs['network'] = trace_info['trace_name'].split('.')[0]
            dataset.attrs['station'] = trace_info['trace_name'].split('.')[1]
            dataset.attrs['channels'] = 'ENZ'
            dataset.attrs['sampling_rate'] = 100.0
            dataset.attrs['units'] = 'counts'
    
    print(f"HDF5 file created: {output_filename}")
    print(f"Contains {len(successful_traces)} traces")

def get_station_latitude(station):
    """Get station latitude based on station code"""
    # Indonesia station coordinates (example coordinates)
    coords = {
        'GSI': -0.69, 'UGM': -7.77, 'FAKI': -3.83, 'SMRI': -2.94, 'TNTI': -0.38,
        'BKNI': 0.45, 'SOEI': -5.51, 'LHMI': 5.31, 'SANI': -8.50, 'PLAI': -3.37,
        'LUWI': -2.55, 'MMRI': 1.48, 'JAGI': -6.83, 'BNDI': -6.20, 'MNAI': 0.51,
        'PMBI': -2.79, 'TOLI2': -1.05, 'SAUI': -6.21, 'BBJI': -6.13, 'GENI': -3.52,
        'BKB': -8.52, 'LHMI': 5.31
    }
    return coords.get(station, -5.0)  # Default to central Indonesia

def get_station_longitude(station):
    """Get station longitude based on station code"""
    coords = {
        'GSI': 127.64, 'UGM': 110.37, 'FAKI': 102.25, 'SMRI': 117.42, 'TNTI': 124.91,
        'BKNI': 125.48, 'SOEI': 95.32, 'LHMI': 95.98, 'SANI': 116.50, 'PLAI': 102.23,
        'LUWI': 120.30, 'MMRI': 124.84, 'JAGI': 110.45, 'BNDI': 106.85, 'MNAI': 123.61,
        'PMBI': 108.24, 'TOLI2': 120.79, 'SAUI': 106.85, 'BBJI': 106.85, 'GENI': 134.19,
        'BKB': 116.41, 'LHMI': 95.98
    }
    return coords.get(station, 110.0)  # Default to central Indonesia

def get_event_latitude(station):
    """Get approximate event latitude (offset from station)"""
    return get_station_latitude(station) + np.random.uniform(-0.5, 0.5)

def get_event_longitude(station):
    """Get approximate event longitude (offset from station)"""
    return get_station_longitude(station) + np.random.uniform(-0.5, 0.5)

def calculate_distance_deg(lat1, lon1, lat2, lon2):
    """Calculate distance in degrees between two points"""
    return np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

if __name__ == "__main__":
    # Configuration
    npz_directory = "dataset_indonesia/npz_padded"
    train_list_file = "dataset_indonesia/padded_train_list.csv"
    valid_list_file = "dataset_indonesia/padded_valid_list.csv"
    
    # Check if directories exist
    if not os.path.exists(npz_directory):
        print(f"Error: NPZ directory not found: {npz_directory}")
        exit(1)
        
    if not os.path.exists(train_list_file):
        print(f"Error: Training list file not found: {train_list_file}")
        exit(1)
        
    if not os.path.exists(valid_list_file):
        print(f"Error: Validation list file not found: {valid_list_file}")
        exit(1)
    
    # Run conversion
    print("Starting conversion to EQTransformer format (separate train/validation)...")
    summary = create_eqt_format_separate(npz_directory, train_list_file, valid_list_file)
    
    print("\n" + "="*60)
    print("CONVERSION COMPLETED!")
    print("="*60)
    print(f"Training files: {summary['successful_train_conversions']}/{summary['total_train_files']}")
    print(f"Validation files: {summary['successful_valid_conversions']}/{summary['total_valid_files']}")
    print("\nOutput files created:")
    if summary['train_hdf5_file']:
        print(f"- {summary['train_hdf5_file']} (training waveforms)")
        print(f"- {summary['train_csv_file']} (training metadata)")
    if summary['valid_hdf5_file']:
        print(f"- {summary['valid_hdf5_file']} (validation waveforms)")
        print(f"- {summary['valid_csv_file']} (validation metadata)")
    print(f"- conversion_summary_separate.json (conversion details)") 