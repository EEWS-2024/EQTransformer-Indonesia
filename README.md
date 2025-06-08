# Implementasi EQTransformer untuk Data Seismik Indonesia

Implementasi komprehensif model deep learning EQTransformer untuk data seismik Indonesia, termasuk training dari awal, fine-tuning transfer learning, dan evaluasi komparatif model.

## Ringkasan Eksekutif

Proyek ini mengimplementasikan dan mengevaluasi arsitektur EQTransformer untuk deteksi event seismik dan phase picking di Indonesia. Tiga pendekatan berbeda telah diimplementasikan dan dibandingkan:

1. **Training dari awal** dengan data Indonesia
2. **Fine-tuning transfer learning** dari model global pre-trained
3. **Pengetesan model original** dengan data Indonesia

## Daftar Isi

- [Gambaran Proyek](#gambaran-proyek)
- [Persiapan Data](#persiapan-data)
- [Pendekatan Implementasi](#pendekatan-implementasi)
- [Arsitektur Teknis](#arsitektur-teknis)
- [Pengaturan Eksperimen](#pengaturan-eksperimen)
- [Hasil dan Performa](#hasil-dan-performa)
- [Instruksi Penggunaan](#instruksi-penggunaan)
- [Struktur File](#struktur-file)
- [Dependensi](#dependensi)
- [Detail Teknis](#detail-teknis)

## Gambaran Proyek

### Tujuan
- Mengadaptasi arsitektur EQTransformer untuk karakteristik seismik Indonesia
- Mengevaluasi performa model pada data gempa bumi Indonesia

### Karakteristik Dataset
- **Data original**: 30.000 sampel per trace (5 menit pada 100 Hz)
- **Data windowed**: 6.000 sampel per trace (1 menit pada 100 Hz)
- **Training set**: 1.642 trace setelah windowing
- **Validation set**: 411 trace setelah windowing
- **Data seismik tiga komponen**: East, North, Vertical

## Persiapan Data

### Pemrosesan Dataset Awal

Dataset seismik Indonesia memerlukan preprocessing ekstensif agar kompatibel dengan EQTransformer:

```python
# Struktur dataset original
indonesia_train.hdf5    # 30.085 trace × 30.000 sampel
indonesia_valid.hdf5    # 30.085 trace × 30.000 sampel

# Struktur target untuk kompatibilitas EQTransformer  
indonesia_finetune_train.hdf5  # 1.642 trace × 6.000 sampel
indonesia_finetune_valid.hdf5  # 411 trace × 6.000 sampel
```

### Strategi Windowing

Implementasi algoritma windowing optimal untuk mempertahankan informasi arrival P dan S:

1. **P dan S ada**: Pusatkan window di antara arrival P dan S
2. **Hanya P**: Pusatkan dengan margin untuk expected S-wave arrival
3. **Hanya S**: Posisikan untuk menyertakan expected P-wave arrival
4. **Tidak ada arrival**: Aplikasikan centered windowing

#### Implementasi Teknis
```bash
python create_indonesia_6k_training.py --process_all
```

Parameter kunci:
- Ukuran window: 6.000 sampel (60 detik)
- Buffer P-wave: 1.000 sampel (10 detik)
- Expected S-P delay: 3.000 sampel (30 detik)

## Pendekatan Implementasi

### Pendekatan 1: Training dari Awal

Training arsitektur EQTransformer sepenuhnya pada data Indonesia tanpa pre-trained weights.

**Konfigurasi:**
- Epochs: 200
- Batch size: 4
- Learning rate: 0.001
- Arsitektur: Standard EQTransformer (CNN + BiLSTM + Transformer)

**Script Utama:**
- `EQTransformer/core_ind/train_indonesia_final.py`
- Custom data generator untuk format Indonesia
- Triangle peak labeling untuk P/S arrivals

### Pendekatan 2: Fine-tuning Transfer Learning

Fine-tuning model EQTransformer pre-trained dengan data Indonesia menggunakan strategi encoder freezing.

**Konfigurasi:**
- Base model: EqT_original_model.h5 (pre-trained pada data global)
- Frozen layers: 71 encoder layers (feature extraction)
- Trainable layers: 77 decoder layers (adaptasi task-specific)

**Parameter Training:**
- Epochs: 200 (dengan early stopping)
- Batch size: 256
- Learning rate: 0.0001 (lebih rendah untuk fine-tuning)
- Early stopping patience: 5

**Script Utama:**
- `finetune_original_indonesia.py`
- `run_finetune_indonesia.sh`



## Arsitektur Teknis

### Gambaran Arsitektur EQTransformer

```
Input (6000, 3) → CNN Encoder → BiLSTM → Transformer → CNN Decoder → 3 Output
                     ↓             ↓          ↓            ↓
                  Feature      Sequential   Global     Detection
                 Extraction   Modeling   Attention   & Picking
```

### Strategi Layer Freezing (Transfer Learning)

**Frozen Encoder Layers:**
- Convolutional layers (conv1d_*)
- Pooling layers (max_pooling1d_*)
- Batch normalization layers
- Bidirectional LSTM layers
- Attention layers (attentiond0, attentiond)

**Trainable Decoder Layers:**
- Upsampling layers (up_sampling1d_*)
- Cropping layers (cropping1d_*)
- Detection output layer
- P-wave picker output layer
- S-wave picker output layer

### Custom Data Generator

Implementasi specialized data generator untuk data seismik Indonesia:

```python
class IndonesiaDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, hdf5_file, csv_file, batch_size=16, 
                 dim=6000, n_channels=3, shuffle=True, augmentation=False):
        # Initialize parameters
        
    def _create_triangle_label(self, arrival_p, arrival_s=None, width=20):
        # Create triangle-shaped labels around P/S arrivals
        
    def __data_generation(self, list_IDs_temp):
        # Generate batches with normalization and label creation
```

## Pengaturan Eksperimen

### Konfigurasi Hardware
- GPU: NVIDIA Tesla T4 (16GB VRAM)
- CPU: Intel Xeon (multiple cores)
- Memory: 64GB RAM
- Environment: Google Cloud Platform

### Software Stack
- Python 3.8
- TensorFlow 2.x
- Keras
- NumPy, Pandas, H5py
- Matplotlib untuk visualisasi

### Konfigurasi Training

#### Training dari Awal
```bash
python EQTransformer/core_ind/train_indonesia_final.py \
    --batch_size 4 \
    --epochs 200 \
    --learning_rate 0.001 \
    --cnn_blocks 3 \
    --lstm_blocks 1 \
    --augmentation \
    --patience 10
```

#### Fine-tuning
```bash
python finetune_original_indonesia.py \
    --model models/original_model/EqT_original_model.h5 \
    --epochs 200 \
    --batch_size 256 \
    --learning_rate 0.0001 \
    --patience 5 \
    --p_threshold 0.05 \
    --s_threshold 0.05 \
    --det_threshold 0.05
```

## Instruksi Penggunaan


### Quick Start

#### Pipeline Fine-tuning Lengkap
```bash
chmod +x run_finetune_indonesia.sh
./run_finetune_indonesia.sh
```

#### Eksekusi Manual

1. **Persiapan Data:**
```bash
python create_indonesia_6k_training.py --process_all
```

2. **Fine-tuning:**
```bash
python finetune_original_indonesia.py \
    --model models/original_model/EqT_original_model.h5 \
    --epochs 15 \
    --batch_size 16
```

3. **Testing:**
```bash
python test_model_performance.py \
    --model path/to/fine_tuned_model.h5 \
    --test_data datasets/indonesia_finetune_valid.hdf5
```

### Opsi Konfigurasi

#### Pemilihan Model
- `models/original_model/EqT_original_model.h5`: Model pre-trained standar
- `models/original_model_conservative/EqT_model_conservative.h5`: Varian konservatif

#### Parameter Training
- `--epochs`: Jumlah training epochs (10-200)
- `--batch_size`: Batch size (4-256, tergantung GPU memory)
- `--learning_rate`: Learning rate (0.0001-0.001)
- `--patience`: Early stopping patience (3-10)

#### Threshold Deteksi
- `--p_threshold`: Threshold deteksi P-wave (0.05-0.2)
- `--s_threshold`: Threshold deteksi S-wave (0.05-0.2)
- `--det_threshold`: Threshold deteksi earthquake (0.05-0.2)

## Struktur File

```
EQTransformer-Indonesia/
├── README.md                              # Dokumentasi komprehensif ini
├── finetune_original_indonesia.py         # Implementasi fine-tuning utama
├── create_indonesia_6k_training.py        # Windowing dan persiapan data
├── run_finetune_indonesia.sh             # Otomasi pipeline lengkap
│
├── datasets/                              # Direktori dataset
│   ├── indonesia_train_dataset.hdf5       # Data training original (30K sampel)
│   ├── indonesia_valid_dataset.hdf5       # Data validation original (30K sampel)
│   ├── indonesia_finetune_train.hdf5      # Data training windowed (6K sampel)
│   ├── indonesia_finetune_train.csv       # Metadata training
│   ├── indonesia_finetune_valid.hdf5      # Data validation windowed (6K sampel)
│   └── indonesia_finetune_valid.csv       # Metadata validation
│
├── models/                                # Penyimpanan model
│   ├── original_model/
│   │   ├── EqT_original_model.h5          # Model global pre-trained
│   │   └── EqT_model_conservative.h5      # Varian konservatif
│   └── finetune_indonesia_YYYYMMDD_HHMMSS/ # Hasil fine-tuning
│       ├── model/
│       │   ├── best_finetuned_model.h5    # Model terbaik (lowest validation loss)
│       │   └── final_finetuned_model.h5   # Model epoch final
│       ├── plots/
│       │   ├── training_history_comprehensive.png
│       │   └── loss_curves.png
│       ├── logs/
│       │   └── training_summary.txt
│       ├── results/
│       │   ├── detailed_predictions.csv   # Prediksi per-trace
│       │   ├── prediction_summary.json    # Statistik ringkasan
│       │   └── prediction_sample.png      # Visualisasi sampel prediksi
│       └── fine_tuning_complete_info.json # Metadata training lengkap
│
├── EQTransformer/                         # Implementasi core EQTransformer
│   ├── core/                             # Modul EQTransformer original
│   └── core_ind/                         # Implementasi spesifik Indonesia
│       ├── train_indonesia_final.py      # Training from-scratch
│       └── test_indonesia_model.py       # Utilitas testing model
│
└── documentation/                        # Dokumentasi tambahan
    └── training_logs/                    # Log training historis
```

## Detail Teknis

### Algoritma Peak Detection

Implementasi custom peak detection untuk identifikasi arrival time:

```python
def find_peak_arrival(prediction_curve, threshold=0.1, min_distance=20):
    """Find the most prominent peak in prediction curve"""
    peaks = []
    for i in range(min_distance, len(prediction_curve) - min_distance):
        if (prediction_curve[i] > threshold and 
            prediction_curve[i] > prediction_curve[i-1] and 
            prediction_curve[i] > prediction_curve[i+1]):
            peaks.append((i, prediction_curve[i]))
    
    if not peaks:
        return None
    
    # Return the highest peak
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks[0][0]
```

### Konfigurasi Loss Function

Multi-task learning dengan weighted loss functions:

```python
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
    loss_weights=[0.2, 0.3, 0.5],  # [detection, P-picking, S-picking]
    metrics=[f1]
)
```

### Data Augmentation

Diterapkan selama training untuk meningkatkan generalisasi:
- Amplitude scaling
- Time shifting (dalam constraints arrival)
- Gaussian noise addition

### Optimisasi GPU

Implementasi training memory-efficient:
- Dynamic GPU memory growth
- Optimisasi batch size berdasarkan available memory
- Dukungan mixed precision training

## Analisis Performa

### Metrik Akurasi

**Definisi:**
- **Detection Rate**: Persentase event yang terdeteksi di atas threshold
- **Accuracy (±N sampel)**: Persentase deteksi dalam N sampel dari true arrival
- **Confidence Score**: Nilai probabilitas maksimum dalam kurva prediksi

