#!/usr/bin/env python3
"""
Validasi hasil konversi dataset Indonesia ke format EQTransformer
Memastikan kompatibilitas dengan EQTransformer API
"""

import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def validate_hdf5_structure(hdf5_path):
    """Validasi struktur HDF5 file"""
    print(f"🔍 Validasi HDF5: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Check root keys
        root_keys = list(f.keys())
        print(f"   📁 Root keys: {root_keys}")
        
        if 'data' not in root_keys:
            print("   ❌ Missing 'data' group!")
            return False
        
        data_group = f['data']
        sample_names = list(data_group.keys())
        
        print(f"   📊 Total samples: {len(sample_names)}")
        print(f"   🔍 Sample names (first 3): {sample_names[:3]}")
        
        # Check first sample
        if len(sample_names) > 0:
            first_sample = data_group[sample_names[0]]
            print(f"   📏 Sample shape: {first_sample.shape}")
            print(f"   🔢 Sample dtype: {first_sample.dtype}")
            
            # Check attributes
            attrs = dict(first_sample.attrs)
            print(f"   📝 Sample attributes: {list(attrs.keys())}")
            
            return True
        else:
            print("   ❌ No samples found!")
            return False

def validate_csv_structure(csv_path):
    """Validasi struktur CSV metadata"""
    print(f"\n📋 Validasi CSV: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"   📊 Shape: {df.shape}")
    print(f"   📝 Columns: {len(df.columns)}")
    
    # Check required columns (based on EQTransformer format)
    required_cols = [
        'network_code', 'receiver_code', 'receiver_type',
        'receiver_latitude', 'receiver_longitude',
        'p_arrival_sample', 's_arrival_sample',
        'source_magnitude', 'source_depth_km',
        'trace_category', 'trace_name'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"   ❌ Missing columns: {missing_cols}")
        return False
    else:
        print(f"   ✅ All required columns present")
    
    # Check data ranges
    print(f"\n   📊 Data ranges:")
    print(f"      P arrival: {df['p_arrival_sample'].min():.0f} - {df['p_arrival_sample'].max():.0f}")
    print(f"      S arrival: {df['s_arrival_sample'].min():.0f} - {df['s_arrival_sample'].max():.0f}")
    print(f"      Magnitude: {df['source_magnitude'].min():.1f} - {df['source_magnitude'].max():.1f}")
    print(f"      Depth: {df['source_depth_km'].min():.1f} - {df['source_depth_km'].max():.1f}")
    
    # Check trace categories
    categories = df['trace_category'].unique()
    print(f"      Categories: {categories}")
    
    return True

def test_compatibility_with_eqt(hdf5_path, csv_path):
    """Test kompatibilitas dengan EQTransformer API"""
    print(f"\n🧪 Testing EQTransformer compatibility...")
    
    try:
        # Test loading like EQTransformer does
        with h5py.File(hdf5_path, 'r') as f:
            data_group = f['data']
            sample_names = list(data_group.keys())
            
            # Test loading a sample
            if len(sample_names) > 0:
                sample_name = sample_names[0]
                waveform = np.array(data_group[sample_name])
                
                print(f"   ✅ Successfully loaded waveform: {waveform.shape}")
                print(f"   📊 Data range: {waveform.min():.2f} to {waveform.max():.2f}")
                print(f"   📈 Data mean: {waveform.mean():.6f}")
                print(f"   📉 Data std: {waveform.std():.6f}")
                
                # Test metadata loading
                df = pd.read_csv(csv_path)
                trace_info = df[df['trace_name'] == sample_name]
                
                if len(trace_info) > 0:
                    row = trace_info.iloc[0]
                    print(f"   ✅ Found metadata for trace: {sample_name}")
                    print(f"   📍 Station: {row['receiver_code']}")
                    print(f"   🌐 Network: {row['network_code']}")
                    print(f"   📊 P/S picks: {row['p_arrival_sample']:.0f}/{row['s_arrival_sample']:.0f}")
                    
                    return True
                else:
                    print(f"   ❌ No metadata found for trace: {sample_name}")
                    return False
        
    except Exception as e:
        print(f"   ❌ Compatibility test failed: {e}")
        return False

def create_sample_visualization(hdf5_path, csv_path, output_dir):
    """Buat visualisasi sampel untuk validasi"""
    print(f"\n📈 Creating sample visualization...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    with h5py.File(hdf5_path, 'r') as f:
        data_group = f['data']
        sample_names = list(data_group.keys())
        
        # Load metadata
        df = pd.read_csv(csv_path)
        
        # Plot first 3 samples
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Dataset Indonesia - EQTransformer Format Validation', fontsize=16)
        
        for i in range(min(3, len(sample_names))):
            sample_name = sample_names[i]
            waveform = np.array(data_group[sample_name])
            
            # Get metadata
            trace_info = df[df['trace_name'] == sample_name].iloc[0]
            p_idx = int(trace_info['p_arrival_sample'])
            s_idx = int(trace_info['s_arrival_sample'])
            
            # Time axis
            time = np.arange(len(waveform)) / 100.0  # 100 Hz
            
            channels = ['E', 'N', 'Z']
            for j in range(3):
                ax = axes[i, j]
                ax.plot(time, waveform[:, j], 'k-', linewidth=0.5)
                ax.axvline(p_idx/100.0, color='red', linestyle='--', label='P-pick')
                ax.axvline(s_idx/100.0, color='blue', linestyle='--', label='S-pick')
                ax.set_title(f'{sample_name} - {channels[j]}')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.grid(True, alpha=0.3)
                if i == 0 and j == 0:
                    ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / 'validation_samples.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved visualization: {plot_path}")
        plt.close()

def generate_validation_report(hdf5_path, csv_path):
    """Generate laporan validasi lengkap"""
    print(f"\n📝 Generating validation report...")
    
    report = {
        'validation_time': pd.Timestamp.now().isoformat(),
        'files': {
            'hdf5': str(hdf5_path),
            'csv': str(csv_path)
        }
    }
    
    # HDF5 validation
    with h5py.File(hdf5_path, 'r') as f:
        data_group = f['data']
        sample_names = list(data_group.keys())
        
        if len(sample_names) > 0:
            first_sample = np.array(data_group[sample_names[0]])
            report['hdf5_validation'] = {
                'total_samples': len(sample_names),
                'sample_shape': list(first_sample.shape),
                'sample_dtype': str(first_sample.dtype),
                'data_range': [float(first_sample.min()), float(first_sample.max())],
                'file_size_mb': Path(hdf5_path).stat().st_size / 1024 / 1024
            }
    
    # CSV validation
    df = pd.read_csv(csv_path)
    report['csv_validation'] = {
        'shape': list(df.shape),
        'columns': list(df.columns),
        'p_arrival_range': [float(df['p_arrival_sample'].min()), float(df['p_arrival_sample'].max())],
        's_arrival_range': [float(df['s_arrival_sample'].min()), float(df['s_arrival_sample'].max())],
        'magnitude_range': [float(df['source_magnitude'].min()), float(df['source_magnitude'].max())],
        'unique_stations': len(df['receiver_code'].unique()),
        'unique_networks': len(df['network_code'].unique())
    }
    
    return report

def main():
    """Fungsi utama validasi"""
    print("🔍 VALIDASI DATASET INDONESIA - FORMAT EQTRANSFORMER")
    print("="*60)
    
    # Input paths
    dataset_dir = Path("dataset_indonesia_eqt_format")
    hdf5_path = dataset_dir / "indonesia_dataset.hdf5"
    csv_path = dataset_dir / "indonesia_dataset.csv"
    
    # Check if files exist
    if not hdf5_path.exists():
        print(f"❌ HDF5 file not found: {hdf5_path}")
        return
    
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    # Run validations
    hdf5_valid = validate_hdf5_structure(hdf5_path)
    csv_valid = validate_csv_structure(csv_path)
    
    if hdf5_valid and csv_valid:
        compat_valid = test_compatibility_with_eqt(hdf5_path, csv_path)
        
        if compat_valid:
            # Create visualization
            create_sample_visualization(hdf5_path, csv_path, dataset_dir)
            
            # Generate report
            report = generate_validation_report(hdf5_path, csv_path)
            
            # Save report
            report_path = dataset_dir / "validation_report.json"
            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\n✅ VALIDASI BERHASIL!")
            print(f"   📄 Report saved: {report_path}")
            print(f"   📊 Dataset siap digunakan dengan EQTransformer!")
            
        else:
            print(f"\n❌ COMPATIBILITY TEST FAILED!")
    else:
        print(f"\n❌ VALIDATION FAILED!")

if __name__ == "__main__":
    main() 