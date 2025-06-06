import h5py
import pandas as pd
import numpy as np

def validate_eqt_format(hdf5_file, csv_file, dataset_name):
    """Validate EQTransformer format for separate datasets"""
    print(f"\n=== Validating {dataset_name} Dataset ===")
     
    # Validate HDF5 file
    print(f"\n1. HDF5 File: {hdf5_file}")
    try:
        with h5py.File(hdf5_file, 'r') as f:
            print(f"   - Root groups: {list(f.keys())}")
            
            if 'data' in f:
                data_group = f['data']
                trace_names = list(data_group.keys())
                print(f"   - Number of traces: {len(trace_names)}")
                
                if len(trace_names) > 0:
                    # Check first trace
                    first_trace = data_group[trace_names[0]]
                    print(f"   - First trace shape: {first_trace.shape}")
                    print(f"   - First trace dtype: {first_trace.dtype}")
                    print(f"   - Attributes: {dict(first_trace.attrs)}")
                    
                    # Check a few more traces for consistency
                    for i, trace_name in enumerate(trace_names[:3]):
                        trace = data_group[trace_name]
                        print(f"   - Trace {i+1} ({trace_name}): shape={trace.shape}, dtype={trace.dtype}")
            else:
                print("   - ERROR: 'data' group not found!")
                
    except Exception as e:
        print(f"   - ERROR reading HDF5: {str(e)}")
    
    # Validate CSV file
    print(f"\n2. CSV File: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        print(f"   - Number of rows: {len(df)}")
        print(f"   - Number of columns: {len(df.columns)}")
        print(f"   - Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
        
        # Check required EQTransformer columns
        required_cols = ['trace_name', 'p_arrival_sample', 's_arrival_sample', 
                        'station_code', 'network_code']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"   - WARNING: Missing required columns: {missing_cols}")
        else:
            print("   - All required columns present")
        
        # Show sample data
        print(f"\n   Sample data:")
        print(f"   - Trace names (first 3):")
        for i in range(min(3, len(df))):
            print(f"     {i+1}. {df.iloc[i]['trace_name']}")
        
        print(f"   - P/S arrival statistics:")
        print(f"     P arrivals - min: {df['p_arrival_sample'].min()}, max: {df['p_arrival_sample'].max()}, mean: {df['p_arrival_sample'].mean():.1f}")
        print(f"     S arrivals - min: {df['s_arrival_sample'].min()}, max: {df['s_arrival_sample'].max()}, mean: {df['s_arrival_sample'].mean():.1f}")
        
        # Check station distribution
        station_counts = df['station_code'].value_counts()
        print(f"   - Number of unique stations: {len(station_counts)}")
        print(f"   - Top 5 stations: {station_counts.head().to_dict()}")
        
        # Check network distribution
        network_counts = df['network_code'].value_counts()
        print(f"   - Networks: {network_counts.to_dict()}")
        
    except Exception as e:
        print(f"   - ERROR reading CSV: {str(e)}")

def compare_datasets():
    """Compare training and validation datasets"""
    print(f"\n=== Dataset Comparison ===")
    
    try:
        train_df = pd.read_csv('indonesia_train_dataset.csv')
        valid_df = pd.read_csv('indonesia_valid_dataset.csv')
        
        print(f"Training dataset size: {len(train_df)}")
        print(f"Validation dataset size: {len(valid_df)}")
        print(f"Total dataset size: {len(train_df) + len(valid_df)}")
        
        # Check for overlap
        train_traces = set(train_df['trace_name'])
        valid_traces = set(valid_df['trace_name'])
        overlap = train_traces.intersection(valid_traces)
        
        if overlap:
            print(f"WARNING: {len(overlap)} traces appear in both datasets!")
            print(f"Overlapping traces: {list(overlap)[:5]}...")
        else:
            print("✓ No overlap between training and validation datasets")
        
        # Compare station distributions
        train_stations = set(train_df['station_code'])
        valid_stations = set(valid_df['station_code'])
        common_stations = train_stations.intersection(valid_stations)
        
        print(f"Stations in training: {len(train_stations)}")
        print(f"Stations in validation: {len(valid_stations)}")
        print(f"Common stations: {len(common_stations)}")
        
    except Exception as e:
        print(f"ERROR comparing datasets: {str(e)}")

if __name__ == "__main__":
    print("EQTransformer Format Validation - Separate Datasets")
    print("=" * 60)
    
    # Validate training dataset
    validate_eqt_format('indonesia_train_dataset.hdf5', 'indonesia_train_dataset.csv', 'Training')
    
    # Validate validation dataset
    validate_eqt_format('indonesia_valid_dataset.hdf5', 'indonesia_valid_dataset.csv', 'Validation')
    
    # Compare datasets
    compare_datasets()
    
    print(f"\n" + "=" * 60)
    print("VALIDATION COMPLETED!")
    print("=" * 60) 