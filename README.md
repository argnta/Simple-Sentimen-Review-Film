# Simple Sentiment Review (IMDB Film Review Analysis)

Aplikasi analisis sentimen berbasis Python untuk mengklasifikasikan review film dari dataset IMDB menjadi sentimen positif atau negatif. Proyek ini menggunakan metode Machine Learning tradisional **Logistic Regression** yang dikombinasikan dengan pembobotan teks **TF-IDF Vectorizer**.

Proyek ini dibagi menjadi dua bagian utama: proses pelatihan model (`main.py`) dan antarmuka interaktif berbasis Command Line Interface / CLI (`test.py`).

## Fitur Utama

- **Pembersihan Teks Tingkat Lanjut (Text Preprocessing):**
  - Menghapus tag HTML (seperti `<br />`).
  - Menghapus tanda baca secara selektif.
  - Mengubah teks menjadi huruf kecil (*case folding*).
  - Normalisasi teks angka spesifik (`10` menjadi `ratingten`, `100` menjadi `ratinghundred`).
  - Menghapus *stopwords* dengan tetap mempertahankan kata negasi penting (seperti `not`, `no`, `never`, `neither`, `nor`) untuk menjaga konteks sentimen kalimat.
  - Proses *stemming* menggunakan `SnowballStemmer` dari NLTK untuk mengubah kata ke bentuk dasarnya.
- **Ekstraksi Fitur:** Menggunakan `TfidfVectorizer` dengan kombinasi *unigram* dan *bigram* (`ngram_range=(1, 2)`) serta pola token kustom `(?u)\b\w+\b` agar karakter alphanumeric dapat terindeks dengan baik sebagai kesatuan fitur.
- **Model Efisien:** Menggunakan `LogisticRegression` dengan konvergensi maksimum hingga 1000 iterasi untuk performa klasifikasi yang cepat dan akurat pada data berdimensi tinggi (*sparse matrix*).
- **Portabilitas Model:** Menyimpan model dan vektor pembobot ke dalam file serialized `.pkl` menggunakan modul `pickle`, sehingga siap digunakan kapan saja tanpa perlu melatih ulang dari awal.
- **Interactive CLI:** Aplikasi interaktif untuk menguji review film kustom secara langsung melalui terminal secara *real-time*.

## Struktur Direktori

```text
├── .gitignore                  # File konfigurasi pengabaian Git
├── IMDBDataset.csv             # Dataset mentah dari IMDB (isi data review & sentiment)
├── main.py                     # Skrip untuk pemrosesan data, pelatihan model, dan evaluasi
├── model_sentimen.pkl          # Model Logistic Regression yang sudah dilatih (Generated)
├── vectorizer.pkl              # Vektor TF-IDF yang sudah dilatih (Generated)
├── test.py                     # Skrip antarmuka CLI untuk pengujian interaktif
└── README.md                   # Dokumentasi proyek

```


## Persyaratan Sistem & Instalasi

### 1. Prasyarat

Pastikan Anda sudah menginstal **Python 3.8+** di sistem Anda.

### 2. Kloning Repositori

```bash
git clone [https://github.com/argnta/Simple-Sentimen-Review-Film](https://github.com/argnta/Simple-Sentimen-Review-Film)
cd simple-sentiment-review

```

### 3. Setup Virtual Environment

Direkomendasikan untuk menggunakan virtual environment agar manajemen dependensi proyek lebih rapi:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / MacOS
python3 -m venv .venv
source .venv/bin/activate

```

### 4. Instalasi Library

Instal beberapa library pihak ketiga yang dibutuhkan oleh proyek ini:

```bash
pip install pandas scikit-learn nltk

```

### 5. Unduh Resource NLTK

Aplikasi ini membutuhkan korpus *stopwords* dari NLTK. Jalankan perintah ini di dalam terminal atau via interpreter Python Anda:

```python
import nltk
nltk.download('stopwords')

```

---

## Cara Penggunaan

### 1. Tahap Pelatihan Model (`main.py`)

Jika Anda ingin melatih ulang model atau memperbarui pembobotan menggunakan dataset `IMDBDataset.csv`, jalankan perintah berikut:

```bash
python main.py

```

**Proses yang terjadi di dalam skrip:**

* Memuat dataset dan melakukan pembersihan teks (*preprocessing*) pada kolom review.
* Membagi data secara acak menggunakan *stratified split* (75% training, 25% testing) untuk menjaga keseimbangan proporsi kelas sentimen.
* Melakukan transformasi teks ke bentuk matriks TF-IDF.
* Melatih model Logistic Regression dan menampilkan skor akurasi akhir di terminal.
* Menyimpan objek hasil latih ke file `model_sentimen.pkl` dan `vectorizer.pkl`.

### 2. Tahap Pengujian / Inferensi Interaktif (`test.py`)

Untuk mencoba menganalisis sentimen dari kalimat Anda sendiri secara langsung, jalankan perintah:

```bash
python test.py

```

**Contoh Alur Interaksi di Terminal:**

```text
==================================================
  APLIKASI ANALISIS SENTIMEN REVIEW FILM (IMDB)  
  Ketik 'exit' pada input review untuk keluar.   
==================================================

Masukkan Review Film: The movie was absolutely fantastic! The acting was superb.
--------------------------------------------------
Hasil Analisis: SENTIMEN POSITIVE
--------------------------------------------------

Masukkan Review Film: I completely wasted my time. The plot made no sense at all.
--------------------------------------------------
Hasil Analisis: SENTIMEN NEGATIVE
--------------------------------------------------

Masukkan Review Film: exit

Bye!!

```

---

## Lisensi

Proyek ini dibuat untuk tujuan pembelajaran dan pengembangan diri di bidang Natural Language Processing (NLP). Anda bebas menggunakan, memodifikasi, dan mendistribusikan kode ini.


