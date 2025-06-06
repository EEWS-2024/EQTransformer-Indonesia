 python check_waveform_stats.py 
=== Descriptive Statistics ===
          duration  sampling_rate       samples
count  2053.000000         2053.0   2053.000000
mean    300.801481          100.0  30081.148076
std       0.015549            0.0      1.554874
min     300.750000          100.0  30076.000000
25%     300.790000          100.0  30080.000000
50%     300.790000          100.0  30080.000000
75%     300.810000          100.0  30082.000000
max     300.850000          100.0  30086.000000

=== Total Files per Station ===
   station  total_files
0     TNTI          265
1     FAKI          248
2      UGM          190
3     LUWI          177
4     PLAI          153
5     JAGI          126
6     SOEI          115
7     SANI          103
8    TOLI2           98
9     MNAI           94
10    MMRI           85
11    SAUI           82
12     GSI           74
13    BNDI           67
14    SMRI           49
15    LHMI           44
16    BBJI           37
17    BKNI           26
18    GENI            9
19     BKB            7
20    PMBI            4

=== Total Files per Bandcode (2 huruf pertama channel) ===
  bandcode  total_files
0       BH          632
1       BL          583
2       SH          500
3       HL          137
4       HH          135
5       SL           66

=== Total Unique MiniSEED Files: 2053 ===

=== Reading: GE.LUWI.BH.gfz2019velu.mseed ===
3 Trace(s) in Stream:
GE.LUWI..BHE | 2019-10-29T06:45:26.719538Z - 2019-10-29T06:50:27.509538Z | 100.0 Hz, 30080 samples
GE.LUWI..BHN | 2019-10-29T06:45:26.719538Z - 2019-10-29T06:50:27.509538Z | 100.0 Hz, 30080 samples
GE.LUWI..BHZ | 2019-10-29T06:45:26.719538Z - 2019-10-29T06:50:27.509538Z | 100.0 Hz, 30080 samples

=== Reading: GE.UGM.HL.gfz2009ptvf.mseed ===
3 Trace(s) in Stream:
GE.UGM..HLE | 2009-08-13T10:48:13.658393Z - 2009-08-13T10:53:14.468393Z | 100.0 Hz, 30082 samples
GE.UGM..HLN | 2009-08-13T10:48:13.658393Z - 2009-08-13T10:53:14.468393Z | 100.0 Hz, 30082 samples
GE.UGM..HLZ | 2009-08-13T10:48:13.658393Z - 2009-08-13T10:53:14.468393Z | 100.0 Hz, 30082 samples

=== Reading: GE.MMRI.SH.gfz2012ugap.mseed ===
3 Trace(s) in Stream:
GE.MMRI..SHE | 2012-10-14T21:49:28.408393Z - 2012-10-14T21:54:29.228393Z | 100.0 Hz, 30083 samples
GE.MMRI..SHN | 2012-10-14T21:49:28.408393Z - 2012-10-14T21:54:29.228393Z | 100.0 Hz, 30083 samples
GE.MMRI..SHZ | 2012-10-14T21:49:28.408393Z - 2012-10-14T21:54:29.228393Z | 100.0 Hz, 30083 samples


python check_waveform_npz_stats.py 
=== ANALISIS FILE NPZ PADDED UNTUK PHASENET ===
Direktori: ./npz_padded
============================================================
Total file NPZ: 2053

Memproses file NPZ...
Analyzing NPZ files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2053/2053 [00:12<00:00, 164.03it/s]

=== STATISTIK DESKRIPTIF DATA ===
       data_length  num_components      p_index       s_index  p_s_interval     data_min     data_max     data_mean     data_std   data_range
count       2053.0          2053.0  2053.000000   2053.000000   2053.000000  2053.000000  2053.000000  2.053000e+03  2053.000000  2053.000000
mean       30085.0             3.0  3000.736483   6599.663906   3598.927423    -0.991560     0.990654 -5.401760e-08     0.104230     1.982214
std            0.0             0.0     0.578378   2495.676880   2495.714461     0.031694     0.033426  3.323626e-06     0.057207     0.044317
min        30085.0             3.0  3000.000000   3286.000000    285.000000    -1.000000     0.629093 -1.008113e-04     0.027689     1.440245
25%        30085.0             3.0  3000.000000   4813.000000   1812.000000    -1.000000     1.000000 -5.681518e-16     0.063697     1.999305
50%        30085.0             3.0  3001.000000   5976.000000   2976.000000    -1.000000     1.000000  2.420214e-19     0.088900     2.000000
75%        30085.0             3.0  3001.000000   7806.000000   4805.000000    -1.000000     1.000000  4.323728e-15     0.125702     2.000000
max        30085.0             3.0  3002.000000  27082.000000  24081.000000    -0.440245     1.000000  2.789397e-05     0.464627     2.000000

=== DISTRIBUSI PANJANG DATA ===
data_length
30085    2053
Name: count, dtype: int64

=== DISTRIBUSI PER STASIUN ===
station
TNTI     265
FAKI     248
UGM      190
LUWI     177
PLAI     153
JAGI     126
SOEI     115
SANI     103
TOLI2     98
MNAI      94
MMRI      85
SAUI      82
GSI       74
BNDI      67
SMRI      49
LHMI      44
BBJI      37
BKNI      26
GENI       9
BKB        7
PMBI       4
Name: count, dtype: int64

=== DISTRIBUSI PER CHANNEL TYPE ===
channel_type
BH    632
BL    583
SH    500
HL    137
HH    135
SL     66
Name: count, dtype: int64

=== DISTRIBUSI ORIGINAL VS AUGMENTED ===
Original data: 632
Augmented data: 1421

=== ANALISIS P DAN S INDICES ===
File dengan P index valid: 2053
File dengan S index valid: 2053
File dengan P dan S index valid: 2053
P index - Min: 3000, Max: 3002, Mean: 3000.7
S index - Min: 3286, Max: 27082, Mean: 6599.7
P-S interval - Min: 285, Max: 24081, Mean: 3598.9

=== VALIDASI PADDING (P INDEX >= 3000) ===
File dengan P index < 3000: 0
File dengan P index >= 3000: 2053
✅ Semua file memiliki P index >= 3000 (padding berhasil)

=== KONSISTENSI SHAPE DATA ===
Distribusi shape data:
  (30085, 3): 2053 files
✅ Semua file memiliki shape yang konsisten

=== KUALITAS DATA ===
File dengan data semua nol: 0
File dengan range data sangat kecil: 0
File dengan range data outlier: 507

=== MENYIMPAN HASIL ANALISIS ===
Statistik detail disimpan ke: ./npz_padded_stats.csv
Ringkasan disimpan ke: ./npz_padded_summary.json

=== MEMBUAT VISUALISASI ===
Visualisasi disimpan di: ./figures

=== ANALISIS SELESAI ===

Analisis selesai! Total 2053 file NPZ dianalisis.
File output:
- npz_padded_stats.csv: Statistik detail per file
- npz_padded_summary.json: Ringkasan statistik
- figures/: Visualisasi hasil analisis