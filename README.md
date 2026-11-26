# AirScope — Beijing PM2.5 Machine Learning Lab

Aplikasi pembelajaran Machine Learning berbasis Streamlit yang mempraktikkan materi buku *Implementasi Algoritma Machine Learning Berbasis Web dengan Framework Streamlit* (Abu Tholib, 2023), BAB 1–7, menggunakan dataset Beijing PM2.5.

## Isi proyek

```text
Beijing_PM25_Cupertino/
├── app.py                         # Aplikasi Streamlit dan antarmuka Cupertino-inspired
├── ml_core.py                     # Preprocessing, training, evaluasi, K-Means, inference helpers
├── requirements.txt
├── README.md
├── run_windows.bat                # Instalasi + jalankan di Windows
├── .streamlit/config.toml
├── data/
│   └── PRSA_data_2010.1.1-2014.12.31.csv
├── outputs/
│   ├── model_baseline_8000.csv
│   └── kmeans_k_scores_3500.csv
└── notebooks/
    └── Beijing_PM25_ML_Lab.ipynb   # Workbook praktik BAB 1–7
```

## Fitur aplikasi

- **Overview:** KPI dataset, rata-rata PM2.5 per tahun, distribusi kelas target.
- **Data Explorer:** filter tahun, tabel observasi, histogram PM2.5, pola per jam, korelasi, serta media playground untuk `st.image()`, `st.audio()`, dan `st.video()`.
- **Data Preparation:** cek missing value, data reduction, feature engineering, encoding, scaling, download data bersih.
- **Model Lab:** SVM, Decision Tree (entropy), Random Forest, dan Gaussian Naïve Bayes; train/test split, accuracy, precision, recall, F1-score, classification report, confusion matrix, feature importance (untuk tree-based model), download pipeline PKL dan hasil metrik CSV.
- **K-Means Clustering:** standardisasi fitur numerik, eksperimen K=2–8, elbow/inertia, silhouette score, scatter plot, profil cluster, download model dan assignment.
- **Live Prediction:** prediksi satu observasi melalui form atau prediksi banyak observasi melalui upload CSV.
- **Chapter Guide:** pemetaan praktik ke semua BAB 1–7.

## 1. Persiapan lingkungan (Windows)

Disarankan menggunakan Python 3.10–3.12.

```powershell
# Masuk ke direktori proyek
cd Beijing_PM25_Cupertino

# Membuat virtual environment
py -m venv .venv

# Aktivasi virtual environment (PowerShell)
.\.venv\Scripts\Activate.ps1

# Jika memakai Command Prompt:
# .venv\Scripts\activate.bat

# Instalasi seluruh library
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Menjalankan aplikasi

```powershell
streamlit run app.py
```

Streamlit akan menampilkan alamat lokal, umumnya `http://localhost:8501`. Di Windows, Anda juga dapat menjalankan `run_windows.bat` untuk membuat environment, memasang dependensi, dan menjalankan aplikasi.

## 3. Publikasi online (opsional)

Untuk mendapatkan URL publik, unggah folder proyek ke repository GitHub, lalu hubungkan repository tersebut ke Streamlit Community Cloud dan pilih `app.py` sebagai main file. Pastikan file CSV pada folder `data/` ikut diunggah. URL publik tidak dibuat otomatis oleh paket proyek ini.

## 4. Alur praktik yang direkomendasikan

1. Buka **Overview** untuk memahami dataset dan tujuan proyek.
2. Buka **Data Explorer** untuk melihat struktur, tren, distribusi, dan korelasi.
3. Buka **Data Preparation** untuk mengamati missing values, penghapusan baris target kosong, data reduction, feature engineering, dan preprocessing.
4. Buka **Model Lab**. Pilih satu atau semua algoritma, atur jumlah sampel dan proporsi test, lalu latih model.
5. Bandingkan metrik dan confusion matrix. Gunakan feature importance untuk Decision Tree / Random Forest.
6. Buka **K-Means Clustering** untuk mencoba nilai K dan membandingkan elbow serta silhouette score.
7. Buka **Live Prediction** untuk uji prediksi individual atau batch.
8. Gunakan **Chapter Guide** sebagai checklist penuntasan materi buku.

## Dataset dan definisi target

Dataset yang digunakan: `PRSA_data_2010.1.1-2014.12.31.csv` (43.824 observasi per jam, 2010–2014). Variabel sumber utama meliputi `pm2.5`, `DEWP`, `TEMP`, `PRES`, `cbwd`, `Iws`, `Is`, dan `Ir`, serta informasi tanggal/waktu.

Untuk klasifikasi, `pm2.5` diubah menjadi target tiga kelas pembelajaran:

- **Rendah:** PM2.5 ≤ 35 µg/m³
- **Sedang:** 35 < PM2.5 ≤ 75 µg/m³
- **Tinggi:** PM2.5 > 75 µg/m³

Batas tersebut dipakai sebagai label praktikum agar algoritma klasifikasi dapat dibandingkan. **Batas ini bukan kategori AQI resmi atau dasar rekomendasi kesehatan.** Baris dengan nilai target `pm2.5` kosong tidak bisa dipakai untuk supervised training. Pada file ini terdapat 2.067 nilai `pm2.5` yang kosong, sehingga tersisa 41.757 baris berlabel.

Untuk menghindari target leakage, kolom `pm2.5` hanya dipakai untuk membentuk `y`; kolom tersebut tidak pernah menjadi fitur prediktor `X`. Prediksi memakai fitur waktu/cuaca: `year`, `DEWP`, `TEMP`, `PRES`, `Iws`, `Is`, `Ir`, encoding siklik `hour` dan `month`, serta `cbwd`.

## Rincian metodologi

- **Missing values:** pemeriksaan dengan `isna().sum()`. Baris target kosong dikeluarkan dari data supervised. Imputer median/modus tetap berada di dalam pipeline untuk menjaga konsistensi input.
- **Data reduction:** `No` (nomor urut) dan `day` tidak dipakai sebagai prediktor.
- **Encoding:** One-Hot Encoding pada `cbwd`.
- **Scaling:** StandardScaler untuk fitur numerik pada model klasifikasi; StandardScaler terpisah untuk K-Means.
- **Train-test split:** default 70:30 (mengikuti contoh BAB 3 dan BAB 5); opsi 80:20 tersedia untuk mereplikasi contoh BAB 7.
- **Evaluasi supervised:** accuracy, weighted precision, weighted recall, weighted F1-score, classification report, confusion matrix.
- **Evaluasi clustering:** inertia dan silhouette score untuk K=2 sampai 8.
- **Model persistence:** pipeline yang mencakup preprocessing dan estimator dapat diunduh dalam format `.pkl`.
- **Decision Tree note:** implementasi memakai `DecisionTreeClassifier(criterion="entropy")`, sesuai contoh kode pada BAB 4. Ini merupakan pohon keputusan berbasis entropy di scikit-learn; tidak mengklaim identik sepenuhnya dengan implementasi C4.5 berbasis gain ratio.

## Snapshot baseline (untuk referensi awal)

Run lokal dengan 8.000 baris sampel, split 70:30, `random_state=42`, 100 pohon Random Forest, dan kedalaman pohon 12 menghasilkan metrik berikut. Ini baseline pembelajaran, bukan jaminan hasil pada setiap versi library atau konfigurasi.

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| Random Forest | 67.54% | 66.94% |
| SVM | 64.83% | 65.30% |
| Decision Tree (Entropy) | 60.38% | 60.93% |
| Gaussian Naïve Bayes | 56.71% | 51.68% |

File hasil baseline dan skor K-Means tersedia di folder `outputs/`.

## Catatan desain dan batasan

Antarmuka menggunakan pendekatan **Cupertino-inspired**: warna terang, typography sistem bergaya SF Pro melalui font stack sistem, kartu rounded, translucent/glass-like panels, soft shadows, serta animasi transisi ringan. Font SF Pro asli tidak dibundel; browser memakai system font yang tersedia dan fallback.

Aplikasi ini adalah **alat pembelajaran**, bukan sistem pemantauan kualitas udara resmi. Prediksi bukan pengukuran aktual, tidak tervalidasi sebagai sistem operasional, dan tidak dimaksudkan untuk keputusan kesehatan atau kebijakan.

### Fitur pada rujukan Telco Churn

Rujukan yang diberikan dipakai sebagai arah untuk membuat alur dashboard yang mudah dijelajahi: ringkasan metrik, eksplorasi data, pipeline/preprocessing, perbandingan model, evaluasi visual, dan prediksi interaktif. Situs rujukan tidak berhasil dimuat melalui pemeriksaan otomatis saat proyek ini dibuat, sehingga implementasi ini tidak mengklaim menyalin layout atau setiap detail UI-nya.

## Referensi utama

Abu Tholib. (2023). *Implementasi Algoritma Machine Learning Berbasis Web dengan Framework Streamlit*. Pustaka Nurja. ISBN 978-623-6757-74-1.
