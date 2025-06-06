# EQTransformer - ModelsAndSampleData

## 📋 Deskripsi

Folder ini berisi data contoh dan model yang telah dilatih untuk EQTransformer, sebuah sistem berbasis deep learning untuk deteksi otomatis gempa bumi dan pemilihan fase P & S dari data seismik.

## 📁 Struktur Data

### 🗃️ File yang Tersedia

```
ModelsAndSampleData/
├── 100samples.hdf5          # Data waveform seismik (HDF5)
├── 100samples.csv           # Metadata dan label (CSV)  
├── test.npy                 # Daftar nama file untuk testing
├── EqT_model.h5            # Model EQTransformer yang sudah dilatih
└── README.md               # Dokumentasi ini
```

## 🔬 Format Data

### 1. **Data Waveform (100samples.hdf5)**

**Format:** HDF5 dengan struktur hierarkis
- **Grup utama:** `'data'`
- **Total samples:** 100 trace seismik
- **Ukuran file:** ~7.22 MB

**Struktur Waveform:**
```python
Shape: (6000, 3)          # 60 detik @ 100 Hz, 3 komponen
Dtype: float32            # Raw seismic counts
Channels: [E, N, Z]       # East, North, Vertical
```

**Karakteristik Data:**
- **Durasi:** 60 detik (fixed window)
- **Sampling rate:** 100 Hz
- **Range amplitudo:** ±50,000 counts (raw, belum normalized)
- **Format nama:** `STASIUN.NETWORK_YYYYMMDDHHMMSS_EV`

**Contoh akses data:**
```python
import h5py
import numpy as np

with h5py.File('100samples.hdf5', 'r') as f:
    sample_names = list(f['data'].keys())
    waveform = np.array(f['data'][sample_names[0]])  # Shape: (6000, 3)
```

### 2. **Metadata (100samples.csv)**

**Dimensi:** 100 baris × 35 kolom
**Ukuran:** ~141 KB

**Kolom Kunci:**

| Kolom | Deskripsi | Range | Rata-rata |
|-------|-----------|--------|-----------|
| `p_arrival_sample` | Kedatangan P-wave (index) | 400-900 | 673 |
| `s_arrival_sample` | Kedatangan S-wave (index) | 723-2198 | 1497 |
| `source_magnitude` | Magnitudo gempa | 0.87-4.3 | 2.4 |
| `source_depth_km` | Kedalaman sumber | 0.45-17.16 km | 9.52 km |
| `source_distance_km` | Jarak episentral | 18-110 km | 69 km |
| `trace_category` | Tipe event | `earthquake_local` | - |

**Timing Analysis:**
- **P-wave arrival:** 4.0 - 9.0 detik dari awal trace
- **S-wave arrival:** 7.2 - 22.0 detik dari awal trace  
- **P-S interval:** 2.7 - 13.0 detik (rata-rata 8.2 detik)

### 3. **Test Set (test.npy)**

**Format:** NumPy array berisi nama file
- **Total entries:** 126,566 trace names
- **Data type:** String (Unicode 26 karakter)
- **Coverage:** Multiple networks (CI, AZ, GO, AV, BK, dll)

**Format nama file:**
```
STASIUN.NETWORK_YYYYMMDDHHMMSS_TYPE
```
Dimana:
- `STASIUN`: Kode stasiun seismik
- `NETWORK`: Kode jaringan seismik
- `YYYYMMDDHHMMSS`: Timestamp event
- `TYPE`: EV (earthquake) atau NO (noise)

## 🌍 Karakteristik Dataset

### **Geographic Coverage**
- **Region:** Amerika Serikat (terutama California)
- **Network utama:** TA (Transportable Array)  
- **Stasiun contoh:** 109C (32.89°N, -117.11°W)

### **Seismic Characteristics**
- **Event type:** Local earthquakes
- **Magnitude range:** M0.87 - M4.3
- **Depth range:** 0.45 - 17.16 km
- **Distance range:** 18 - 110 km

### **Data Quality**
- **Label method:** Manual picking
- **P/S weight:** 0.5 (manual quality)
- **Data status:** Pre-processed, filtered
- **SNR:** ~37-65 dB range

## 🔧 Penggunaan Data

### **Loading Waveform Data**

```python
import h5py
import numpy as np
import pandas as pd

# Load waveform
def load_waveform(filename='100samples.hdf5'):
    with h5py.File(filename, 'r') as f:
        sample_names = list(f['data'].keys())
        waveforms = {}
        for name in sample_names:
            waveforms[name] = np.array(f['data'][name])
    return waveforms

# Load metadata
def load_metadata(filename='100samples.csv'):
    return pd.read_csv(filename)

# Example usage
waveforms = load_waveform()
metadata = load_metadata()

# Get first sample
first_name = list(waveforms.keys())[0]
waveform = waveforms[first_name]  # Shape: (6000, 3)
trace_info = metadata[metadata['trace_name'] == first_name]
```

### **Ekstraksi P & S Picks**

```python
def get_picks(trace_name, metadata):
    """Ekstrak timing P dan S picks"""
    row = metadata[metadata['trace_name'] == trace_name].iloc[0]
    
    p_sample = int(row['p_arrival_sample'])
    s_sample = int(row['s_arrival_sample'])
    
    p_time = p_sample / 100.0  # Convert to seconds
    s_time = s_sample / 100.0
    
    return {
        'p_pick': p_time,
        's_pick': s_time,
        'ps_interval': s_time - p_time,
        'magnitude': row['source_magnitude'],
        'distance': row['source_distance_km']
    }
```

### **Visualisasi Data**

```python
import matplotlib.pyplot as plt

def plot_seismogram(waveform, picks=None, sample_rate=100):
    """Plot 3-component seismogram"""
    time = np.arange(len(waveform)) / sample_rate
    channels = ['E', 'N', 'Z']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    for i, (ax, channel) in enumerate(zip(axes, channels)):
        ax.plot(time, waveform[:, i], 'k-', linewidth=0.5)
        ax.set_ylabel(f'{channel} (counts)')
        ax.grid(True, alpha=0.3)
        
        if picks:
            ax.axvline(picks['p_pick'], color='red', 
                      linestyle='--', label='P-pick')
            ax.axvline(picks['s_pick'], color='blue', 
                      linestyle='--', label='S-pick')
            if i == 0:  # Add legend only to first subplot
                ax.legend()
    
    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    return fig
```

## 📊 Perbandingan dengan Dataset Indonesia

| Aspek | EQTransformer (Global) | Dataset Indonesia |
|-------|----------------------|-------------------|
| **Format** | HDF5 + CSV | NPZ + CSV |
| **Window** | 60s (fixed) | Variable (50-120s) |
| **Sampling** | 100 Hz | 100 Hz |
| **Normalisasi** | Raw counts | Normalized |
| **Region** | California, USA | Indonesia |
| **Networks** | TA (single) | Multiple (IA, GE, IU) |
| **Geology** | Transform fault | Subduction zone |
| **Event types** | Local earthquakes | Regional + local |

## 🚀 Implementasi Training

### **Data Preparation untuk Transfer Learning**

```python
def prepare_indonesian_format(eqt_waveform, indonesian_length=6000):
    """
    Konversi format EQTransformer ke Indonesian dataset format
    """
    # EQTransformer already 6000 samples, no need to resize
    if eqt_waveform.shape[0] != indonesian_length:
        # Resize if needed
        from scipy.interpolate import interp1d
        old_time = np.linspace(0, 1, eqt_waveform.shape[0])
        new_time = np.linspace(0, 1, indonesian_length)
        
        resized = np.zeros((indonesian_length, 3))
        for i in range(3):
            f = interp1d(old_time, eqt_waveform[:, i], kind='linear')
            resized[:, i] = f(new_time)
        return resized
    
    return eqt_waveform

def normalize_waveform(waveform):
    """Normalize ke format Indonesia (zero mean, unit variance)"""
    normalized = np.zeros_like(waveform)
    for i in range(waveform.shape[1]):
        channel = waveform[:, i]
        normalized[:, i] = (channel - np.mean(channel)) / np.std(channel)
    return normalized
```

## 📚 Referensi

1. **EQTransformer Paper:** Mousavi, S.M., Ellsworth, W.L., Zhu, W. et al. (2020). Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking. Nature Communications 11, 3952.

2. **Model Architecture:** Attention-based Transformer untuk sequence-to-sequence learning pada data seismik

3. **Training Data:** STEAD (STanford EArthquake Dataset) - 1.2M annotated seismic traces

## 📞 Kontak & Support

Untuk pertanyaan teknis atau troubleshooting, silakan merujuk ke:
- **EQTransformer GitHub:** https://github.com/smousavi05/EQTransformer
- **Dokumentasi lengkap:** Lihat folder `examples/` untuk tutorial penggunaan
- **Dataset Indonesia:** Lihat folder `dataset_indonesia/` untuk data lokal

---

**Catatan:** Data ini dimaksudkan untuk penelitian dan pengembangan sistem deteksi gempa berbasis AI. Untuk penggunaan operasional, diperlukan validasi tambahan dengan data lokal Indonesia. 