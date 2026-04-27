# UTS Machine Learning: Klasifikasi Citrus (Orange vs Grapefruit) 🍊🍇

Repositori ini berisi proyek pembuatan model *Machine Learning* untuk memenuhi Tugas Ujian Tengah Semester (UTS). Tujuan dari proyek ini adalah membangun dan membandingkan tiga algoritma klasifikasi—**Decision Tree, Naive Bayes, dan Support Vector Machine (SVM)**—untuk mengidentifikasi apakah sebuah buah merupakan jeruk (*orange*) atau anggur (*grapefruit*) berdasarkan fitur fisik dan warnanya.

## Informasi Dataset
Dataset yang digunakan diunduh dari [Kaggle: Oranges vs Grapefruit](https://www.kaggle.com/datasets/joshmcadams/oranges-vs-grapefruit). Data terdiri dari 10.000 baris dengan fitur sebagai berikut:
* `diameter` : Diameter buah.
* `weight` : Berat buah.
* `red`, `green`, `blue` : Intensitas warna RGB buah.
* `name` : Target klasifikasi (*orange* / *grapefruit*).

---

## Tahapan Pembuatan Model

Berikut adalah langkah-langkah sistematis yang dilakukan dalam pembuatan model hingga evaluasi, sesuai dengan kode yang dilampirkan:

### 1. Pengumpulan dan Pemahaman Data
Melakukan *load* dataset `citrus.csv` menggunakan library `pandas` dan menampilkan ringkasan statistik (mean, standar deviasi, min, max) untuk memahami rentang nilai dan kualitas data.

### 2. Eksplorasi Data (Exploratory Data Analysis / EDA)
* **Distribusi Data:** Memvisualisasikan distribusi fitur (seperti *diameter*) menggunakan *histogram* untuk membandingkan sebaran antara kelas jeruk dan anggur.
* **Matriks Korelasi:** Membuat *heatmap* korelasi untuk melihat seberapa kuat hubungan antar-variabel independen maupun hubungannya terhadap kelas target.

### 3. Pra-pemrosesan Data (Data Preprocessing)
* **Label Encoding:** Mengonversi nilai target tekstual ('orange' dan 'grapefruit') menjadi nilai biner numerik (1 dan 0) menggunakan `LabelEncoder`.
* **Feature Scaling:** Menstandarisasi nilai fitur menggunakan `StandardScaler`. Ini sangat penting agar algoritma berbasis jarak (seperti SVM) dan distribusi probabilitas (seperti Naive Bayes) tidak bias terhadap skala fitur yang lebih besar.

### 4. Pembagian Data (Data Splitting)
Dataset dibagi menjadi dua set: **Data Latih (Train)** dan **Data Uji (Test)** dengan rasio **75:25** (`test_size=0.25`). Data latih digunakan untuk melatih model, sementara data uji digunakan untuk simulasi evaluasi yang adil.

### 5. Inisialisasi dan Pelatihan Model
Tiga model dilatih secara komparatif menggunakan data latih yang sama:
* **Decision Tree:** Algoritma yang membagi data berdasarkan aturan berstruktur pohon.
* **Gaussian Naive Bayes:** Algoritma berbasis teorema Bayes dengan asumsi distribusi normal pada fitur kontinu.
* **Support Vector Machine (SVM):** Model yang mencari *hyperplane* dengan margin maksimal untuk memisahkan kedua kelas.

### 6. Evaluasi dan Perbandingan Model
Model dievaluasi menggunakan Data Uji dengan mengeluarkan matriks berikut:
* **Classification Report:** Memuat nilai *Accuracy*, *Precision*, *Recall*, dan *F1-Score*.
* **Confusion Matrix:** Memvisualisasikan tabel prediksi *True Positives/Negatives* dan *False Positives/Negatives*.
* **Kurva Perbandingan:** Menampilkan grafik **Precision-Recall Curve** dan **ROC Curve (AUC)** untuk melihat keandalan model secara visual.

---

## 7. Output Program

**Eksplorasi Data (EDA)**
<p align="center">
  <img src="https://github.com/user-attachments/assets/d0b3b0e5-2d4f-4b72-97b2-a0288812b0bc" width="45%" alt="Dataset Info 1" />
  <img src="https://github.com/user-attachments/assets/8a5d1361-a659-4651-a58e-e87ba5c33d2e" width="45%" alt="Dataset Info 2" />
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/1e27d8e7-c5e8-4c23-b747-ac1ecfcb0bb2" width="45%" alt="Distribusi" />
  <img src="https://github.com/user-attachments/assets/4707653f-f207-4344-ab77-57d6743d4400" width="45%" alt="Korelasi" />
</p>

**Hasil Klasifikasi & Confusion Matrix**
<p align="center">
  <img src="https://github.com/user-attachments/assets/eacda84d-6c5c-48c8-bb4a-11b26be7d0c4" width="45%" alt="Report 1" />
  <img src="https://github.com/user-attachments/assets/f48e12ae-15ba-4b66-a57e-752dd5e22092" width="45%" alt="Confusion Matrix 1" />
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/2a1bf967-3544-456e-a5e9-b3784071ae37" width="45%" alt="Confusion Matrix 2" />
  <img src="https://github.com/user-attachments/assets/6905cff3-aedf-4e0a-a61e-f72af7460284" width="45%" alt="Confusion Matrix 3" />
</p>

**Kurva Perbandingan**
<p align="center">
  <img src="https://github.com/user-attachments/assets/133c97e5-8446-4f9c-ba44-80c988bddc83" width="45%" alt="PR Curve" />
  <img src="https://github.com/user-attachments/assets/2f0b7202-b44a-4a67-996b-0c577255edad" width="45%" alt="ROC Curve" />
</p>

---

## Cara Menjalankan Program (Visual Studio Code)
1. Pastikan ekstensi **Jupyter** sudah terinstal di VS Code.
2. Install library yang dibutuhkan: `pip install pandas numpy matplotlib seaborn scikit-learn`.
3. Buka file `main.py`.
4. Klik tombol **"Run Cell"** di atas tulisan `#%%` secara berurutan dari atas ke bawah untuk melihat tabel interaktif dan grafik evaluasinya.

---

## Kesimpulan Hasil Evaluasi
*(Silahkan isi bagian ini setelah kamu me-run kodenya)*
Berdasarkan hasil perbandingan dari tahapan evaluasi di atas, model **[Tuliskan model terbaik di sini, misal: Support Vector Machine]** memberikan performa paling tinggi dengan akurasi sebesar **[Tuliskan angka]%**, *F1-Score* sebesar **[Tuliskan angka]**, dan nilai Area Under Curve (AUC) tertinggi pada *ROC Curve*.

---
**Penulis:**
Muhammad Tibia Nugraha
*Teknik Informatika, Universitas Islam Negeri Sunan Gunung Djati*
