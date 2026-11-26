# AERIS — Beijing PM2.5 Machine Learning Web App

Aplikasi web interaktif berbasis **Streamlit** untuk eksplorasi data kualitas udara Beijing, preprocessing, eksperimen machine learning, evaluasi, clustering, dan prediksi konsentrasi PM2.5.

## 1. Anggota kelompok

- **Farel Nofri Landa** — 241011401676
- **Septian Cahaya Irwanto** — 241011402116
- **Muhammad Rojak Ardiansyah** — 241011400688

## 2. Isi aplikasi

| Halaman | Fitur |
|---|---|
| Overview | Ringkasan dataset, metrik utama, tren bulanan PM2.5, distribusi target, dan alur proyek |
| Dataset & EDA | Preview data, tipe data, missing value, histogram, scatter plot, dan correlation heatmap |
| Preprocessing | Laporan pembersihan, pembentukan timestamp, rekayasa fitur siklikal, imputasi, encoding, dan preview data siap model |
| Regression Lab | Training dan perbandingan Linear Regression, LinearSVR, Decision Tree, Random Forest, Extra Trees, dan HistGradientBoosting; evaluasi MAE, RMSE, R²; residual plot; feature importance; simpan/download model |
| Klasifikasi | SVM/LinearSVC, Decision Tree, Random Forest, dan Gaussian Naive Bayes; accuracy, precision, recall, F1 macro, confusion matrix, classification report |
| K-Means Clustering | Evaluasi K=2–7 menggunakan silhouette score, pembentukan cluster, profil cluster, visualisasi, dan download hasil |
| Prediksi | Prediksi PM2.5 melalui form manual atau batch CSV, serta download template input |

## 3. Alur machine learning

1. Membaca dataset CSV.
2. Memeriksa ukuran data, tipe kolom, distribusi target, dan nilai kosong.
3. Memvalidasi kolom dan membuat timestamp dari `year`, `month`, `day`, dan `hour`.
4. Menghapus baris tanpa target `pm2.5`, timestamp tidak valid, dan baris duplikat.
5. Membuat fitur siklikal jam dan bulan (`sin`/`cos`) serta `dayofweek`.
6. Melakukan imputasi median untuk fitur numerik dan modus untuk fitur kategori; `cbwd` diubah dengan One-Hot Encoding.
7. Melakukan scaling untuk model yang membutuhkannya.
8. Memisahkan data train dan test secara **kronologis**. Data periode paling akhir menjadi data test.
9. Melatih model, melakukan prediksi, dan menghitung metrik evaluasi.
10. Menyimpan model terpilih menggunakan Joblib atau mengunduhnya dalam format pickle.
11. Menggunakan model yang telah dilatih untuk prediksi manual atau batch.

## 4. Penyesuaian algoritma dengan dataset

Kolom target `pm2.5` merupakan nilai numerik kontinu, sehingga tugas utama proyek adalah **regresi**. Karena itu:

- SVM pada tugas regresi direalisasikan sebagai **LinearSVR**.
- Decision Tree dan Random Forest menggunakan versi **Regressor**.
- Ditambahkan Linear Regression sebagai baseline serta Extra Trees dan HistGradientBoosting sebagai pembanding.
- **K-Means** digunakan untuk analisis pola kelompok tanpa label.
- Untuk mempraktikkan algoritma klasifikasi pada modul, tersedia halaman klasifikasi tambahan. Nilai PM2.5 dibagi menjadi tiga kelas kuantil berdasarkan data training, lalu diuji dengan LinearSVC, Decision Tree, Random Forest, dan Gaussian Naive Bayes.

**Catatan penting:** kelas “Relatif rendah/sedang/tinggi” pada halaman klasifikasi adalah kategori relatif yang dibentuk dari kuantil dataset training. Kategori tersebut bukan ambang regulasi ISPU/AQI dan bukan pengganti hasil regresi PM2.5.

Pembagian data dibuat kronologis (bukan acak) karena setiap baris memiliki urutan waktu per jam. Pendekatan ini lebih sesuai untuk menguji kemampuan model pada periode setelah data training serta mengurangi risiko kebocoran informasi waktu.

## 5. Ringkasan dataset bawaan

File: `data/PRSA_data_2010.1.1-2014.12.31.csv`

- Periode: 1 Januari 2010 sampai 31 Desember 2014.
- Observasi mentah: 43.824 baris.
- Kolom: 13.
- Target: `pm2.5`.
- Nilai target kosong pada dataset asli: 2.067 baris.
- Observasi dengan target tersedia setelah pembersihan: 41.757 baris.
- Kolom `No` merupakan nomor urut dan tidak digunakan sebagai fitur prediksi.

Nilai `pm2.5` yang kosong dikeluarkan dari data supervised learning karena tidak memiliki label target untuk dipelajari. Fitur yang kosong, apabila ditemukan pada dataset lain, ditangani di dalam pipeline menggunakan imputasi.

## 6. Instalasi dan menjalankan aplikasi

### Windows

Buka Command Prompt atau terminal VS Code, lalu masuk ke folder proyek:

```bash
cd Beijing_PM25_Streamlit
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
cd Beijing_PM25_Streamlit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Setelah perintah dijalankan, Streamlit akan membuka aplikasi di browser. Jika tidak terbuka otomatis, akses alamat lokal yang ditampilkan di terminal (biasanya `http://localhost:8501`).

## 7. Cara menggunakan aplikasi

1. Mulai dari halaman **Overview** untuk melihat gambaran umum proyek.
2. Buka **Dataset & EDA** untuk mengecek data mentah dan karakteristik target.
3. Buka **Preprocessing** untuk mempelajari proses pembersihan serta pembentukan fitur.
4. Buka **Regression Lab**, pilih algoritma, lalu tekan **Train & evaluasi model regresi**.
5. Bandingkan MAE, RMSE, dan R². RMSE yang lebih rendah dan R² yang lebih tinggi umumnya lebih baik untuk model pada data uji yang sama.
6. Simpan atau download model terbaik.
7. Buka **Prediksi** untuk memasukkan kondisi cuaca secara manual atau mengunggah CSV batch.
8. Gunakan halaman **Klasifikasi** dan **K-Means Clustering** untuk eksperimen supervised classification dan unsupervised learning.

## 8. Format CSV untuk prediksi batch

Kolom wajib:

```text
year,month,day,hour,DEWP,TEMP,PRES,cbwd,Iws,Is,Ir
```

Kolom `pm2.5` tidak diperlukan untuk file prediksi. Halaman Prediksi menyediakan tombol untuk mengunduh template CSV.

## 9. Struktur folder

```text
Beijing_PM25_Streamlit/
├── app.py
├── ml_core.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── data/
│   └── PRSA_data_2010.1.1-2014.12.31.csv
└── models/
    └── (model hasil training disimpan di sini)
```

## 10. Catatan tampilan

UI menggunakan gaya Cupertino-inspired: layout lapang, rounded corners, tipografi system/SF Pro stack, panel translucent dengan blur, bayangan lembut, serta transisi ringan. Pada perangkat Windows yang tidak memiliki SF Pro, browser menggunakan font sistem yang tersedia sebagai fallback.
